package com.cvcoolz.androidcamera

import android.content.Context
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.SurfaceTexture
import android.hardware.camera2.*
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.hardware.camera2.params.OutputConfiguration
import android.hardware.camera2.params.SessionConfiguration
import android.media.ImageReader
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.util.Range
import android.util.Size
import android.view.Surface
import android.view.WindowManager
import java.util.concurrent.Executor
import java.util.concurrent.Semaphore
import java.util.concurrent.TimeUnit

class CameraController(
    private val context: Context,
    private val onFrame: (ByteArray) -> Unit
) {
    companion object {
        private const val TAG = "CameraController"
    }

    private val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var imageReader: ImageReader? = null
    private var backgroundThread: HandlerThread? = null
    private var backgroundHandler: Handler? = null
    private val openCloseLock = Semaphore(1)

    private lateinit var cameraId: String
    private lateinit var characteristics: CameraCharacteristics

    // Sensor info
    var exposureRange: Range<Long> = Range(0L, 0L); private set
    var isoRange: Range<Int> = Range(0, 0); private set
    var minFocusDistance: Float = 0f; private set
    var maxDigitalZoom: Float = 1f; private set
    var exposureCompensationRange: Range<Int> = Range(0, 0); private set
    var exposureCompensationStep: Float = 0f; private set
    var availableResolutions: List<Size> = emptyList(); private set
    var sensorActiveRect: Rect = Rect(); private set
    var sensorOrientation: Int = 0; private set

    // Current settings
    var currentWidth = 1280; private set
    var currentHeight = 720; private set
    private var aeMode = CameraMetadata.CONTROL_AE_MODE_ON
    private var manualExposureTime: Long? = null
    private var manualISO: Int? = null
    private var exposureCompensation: Int = 0
    private var afMode = CameraMetadata.CONTROL_AF_MODE_CONTINUOUS_PICTURE
    private var manualFocusDistance: Float? = null
    private var zoomRatio: Float = 1.0f
    private var awbMode = CameraMetadata.CONTROL_AWB_MODE_AUTO
    private var jpegQuality: Int = 80
    private var fpsRange: Range<Int>? = null

    private var previewSurface: Surface? = null
    private var surfaceTextureRef: SurfaceTexture? = null

    // Auto values read from capture results
    @Volatile var lastAutoExposureNs: Long = 0L; private set
    @Volatile var lastAutoISO: Int = 0; private set
    @Volatile var lastAutoFocusDistance: Float = 0f; private set

    fun open(surfaceTexture: SurfaceTexture?) {
        startBackgroundThread()
        cameraId = findBackCamera()
        characteristics = cameraManager.getCameraCharacteristics(cameraId)
        readCapabilities()

        val bestSize = pickClosestSize(currentWidth, currentHeight)
        currentWidth = bestSize.width
        currentHeight = bestSize.height

        surfaceTextureRef = surfaceTexture
        if (surfaceTexture != null) {
            surfaceTexture.setDefaultBufferSize(currentWidth, currentHeight)
            previewSurface = Surface(surfaceTexture)
        }

        if (!openCloseLock.tryAcquire(3, TimeUnit.SECONDS)) {
            throw RuntimeException("Timed out waiting to open camera")
        }

        try {
            cameraManager.openCamera(cameraId, object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    openCloseLock.release()
                    cameraDevice = camera
                    createCaptureSession()
                }

                override fun onDisconnected(camera: CameraDevice) {
                    openCloseLock.release()
                    camera.close()
                    cameraDevice = null
                }

                override fun onError(camera: CameraDevice, error: Int) {
                    openCloseLock.release()
                    camera.close()
                    cameraDevice = null
                    Log.e(TAG, "Camera error: $error")
                }
            }, backgroundHandler)
        } catch (e: SecurityException) {
            openCloseLock.release()
            Log.e(TAG, "Camera permission not granted", e)
        }
    }

    fun close() {
        try {
            openCloseLock.tryAcquire(3, TimeUnit.SECONDS)
            captureSession?.close()
            captureSession = null
            cameraDevice?.close()
            cameraDevice = null
            imageReader?.close()
            imageReader = null
            previewSurface?.release()
            previewSurface = null
        } finally {
            openCloseLock.release()
        }
        stopBackgroundThread()
    }

    private fun findBackCamera(): String {
        for (id in cameraManager.cameraIdList) {
            val chars = cameraManager.getCameraCharacteristics(id)
            val facing = chars.get(CameraCharacteristics.LENS_FACING)
            if (facing == CameraCharacteristics.LENS_FACING_BACK) return id
        }
        return cameraManager.cameraIdList[0]
    }

    private fun readCapabilities() {
        val map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)!!

        exposureRange = characteristics.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
            ?: Range(0L, 0L)
        isoRange = characteristics.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)
            ?: Range(0, 0)
        minFocusDistance = characteristics.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE)
            ?: 0f
        maxDigitalZoom = characteristics.get(CameraCharacteristics.SCALER_AVAILABLE_MAX_DIGITAL_ZOOM)
            ?: 1f
        sensorActiveRect = characteristics.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
            ?: Rect()
        sensorOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 0

        val compRange = characteristics.get(CameraCharacteristics.CONTROL_AE_COMPENSATION_RANGE)
        exposureCompensationRange = compRange ?: Range(0, 0)
        val compStep = characteristics.get(CameraCharacteristics.CONTROL_AE_COMPENSATION_STEP)
        exposureCompensationStep = compStep?.toFloat() ?: 0f

        availableResolutions = map.getOutputSizes(ImageFormat.JPEG)
            ?.sortedByDescending { it.width * it.height }
            ?: emptyList()

        // Pick FPS range: prefer 30-30 to force consistent framerate and limit motion blur
        val fpsRanges = characteristics.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES)
        fpsRange = fpsRanges?.firstOrNull { it.lower == 30 && it.upper == 30 }
            ?: fpsRanges?.filter { it.upper >= 30 }?.minByOrNull { it.upper - it.lower }
            ?: fpsRanges?.maxByOrNull { it.upper }
    }

    private fun pickClosestSize(targetW: Int, targetH: Int): Size {
        if (availableResolutions.isEmpty()) return Size(targetW, targetH)
        return availableResolutions.minByOrNull {
            val dw = it.width - targetW
            val dh = it.height - targetH
            dw * dw + dh * dh
        } ?: Size(targetW, targetH)
    }

    private fun createCaptureSession() {
        val camera = cameraDevice ?: return

        imageReader?.close()
        imageReader = ImageReader.newInstance(currentWidth, currentHeight, ImageFormat.JPEG, 3).apply {
            setOnImageAvailableListener({ reader ->
                val image = reader.acquireLatestImage() ?: return@setOnImageAvailableListener
                try {
                    val buffer = image.planes[0].buffer
                    val bytes = ByteArray(buffer.remaining())
                    buffer.get(bytes)
                    onFrame(bytes)
                } finally {
                    image.close()
                }
            }, backgroundHandler)
        }

        val surfaces = mutableListOf(imageReader!!.surface)
        previewSurface?.let { surfaces.add(it) }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            val outputs = surfaces.map { OutputConfiguration(it) }
            val executor = Executor { command -> backgroundHandler?.post(command) }
            val config = SessionConfiguration(
                SessionConfiguration.SESSION_REGULAR,
                outputs,
                executor,
                object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(session: CameraCaptureSession) {
                        captureSession = session
                        updateRepeatingRequest()
                    }

                    override fun onConfigureFailed(session: CameraCaptureSession) {
                        Log.e(TAG, "Session configuration failed")
                    }
                }
            )
            camera.createCaptureSession(config)
        } else {
            @Suppress("DEPRECATION")
            camera.createCaptureSession(surfaces,
                object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(session: CameraCaptureSession) {
                        captureSession = session
                        updateRepeatingRequest()
                    }

                    override fun onConfigureFailed(session: CameraCaptureSession) {
                        Log.e(TAG, "Session configuration failed")
                    }
                }, backgroundHandler)
        }
    }

    private fun updateRepeatingRequest() {
        val camera = cameraDevice ?: return
        val session = captureSession ?: return

        try {
            val builder = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                addTarget(imageReader!!.surface)
                previewSurface?.let { addTarget(it) }

                // AE mode
                set(CaptureRequest.CONTROL_AE_MODE, aeMode)
                if (aeMode == CameraMetadata.CONTROL_AE_MODE_OFF) {
                    manualExposureTime?.let { set(CaptureRequest.SENSOR_EXPOSURE_TIME, it) }
                    manualISO?.let { set(CaptureRequest.SENSOR_SENSITIVITY, it) }
                } else {
                    set(CaptureRequest.CONTROL_AE_EXPOSURE_COMPENSATION, exposureCompensation)
                }

                // FPS
                fpsRange?.let { set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, it) }

                // AF mode
                set(CaptureRequest.CONTROL_AF_MODE, afMode)
                if (afMode == CameraMetadata.CONTROL_AF_MODE_OFF) {
                    manualFocusDistance?.let { set(CaptureRequest.LENS_FOCUS_DISTANCE, it) }
                }

                // Zoom
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                    set(CaptureRequest.CONTROL_ZOOM_RATIO, zoomRatio)
                } else {
                    val cropW = (sensorActiveRect.width() / zoomRatio).toInt()
                    val cropH = (sensorActiveRect.height() / zoomRatio).toInt()
                    val cropX = (sensorActiveRect.width() - cropW) / 2
                    val cropY = (sensorActiveRect.height() - cropH) / 2
                    set(CaptureRequest.SCALER_CROP_REGION, Rect(cropX, cropY, cropX + cropW, cropY + cropH))
                }

                // AWB
                set(CaptureRequest.CONTROL_AWB_MODE, awbMode)

                // JPEG quality and orientation
                set(CaptureRequest.JPEG_QUALITY, jpegQuality.toByte())
                set(CaptureRequest.JPEG_ORIENTATION, getJpegOrientation())
            }

            session.setRepeatingRequest(builder.build(), captureCallback, backgroundHandler)
        } catch (e: CameraAccessException) {
            Log.e(TAG, "Failed to set repeating request", e)
        }
    }

    // --- Public control methods ---

    fun setExposureTimeNs(ns: Long) {
        aeMode = CameraMetadata.CONTROL_AE_MODE_OFF
        manualExposureTime = ns.coerceIn(exposureRange.lower, exposureRange.upper)
        if (manualISO == null) manualISO = isoRange.lower
        updateRepeatingRequest()
    }

    fun setISO(iso: Int) {
        aeMode = CameraMetadata.CONTROL_AE_MODE_OFF
        manualISO = iso.coerceIn(isoRange.lower, isoRange.upper)
        if (manualExposureTime == null) manualExposureTime = 16_666_666L // ~1/60s default
        updateRepeatingRequest()
    }

    fun setExposureCompensation(ev: Int) {
        aeMode = CameraMetadata.CONTROL_AE_MODE_ON
        exposureCompensation = ev.coerceIn(exposureCompensationRange.lower, exposureCompensationRange.upper)
        manualExposureTime = null
        manualISO = null
        updateRepeatingRequest()
    }

    fun setAutoExposure() {
        aeMode = CameraMetadata.CONTROL_AE_MODE_ON
        exposureCompensation = 0
        manualExposureTime = null
        manualISO = null
        updateRepeatingRequest()
    }

    fun setFocusDistance(diopters: Float) {
        afMode = CameraMetadata.CONTROL_AF_MODE_OFF
        manualFocusDistance = diopters.coerceIn(0f, minFocusDistance)
        updateRepeatingRequest()
    }

    fun setAutoFocus() {
        afMode = CameraMetadata.CONTROL_AF_MODE_CONTINUOUS_PICTURE
        manualFocusDistance = null
        updateRepeatingRequest()
    }

    fun setZoom(ratio: Float) {
        zoomRatio = ratio.coerceIn(1.0f, maxDigitalZoom)
        updateRepeatingRequest()
    }

    fun setWhiteBalance(mode: String) {
        awbMode = when (mode.lowercase()) {
            "auto" -> CameraMetadata.CONTROL_AWB_MODE_AUTO
            "incandescent" -> CameraMetadata.CONTROL_AWB_MODE_INCANDESCENT
            "fluorescent" -> CameraMetadata.CONTROL_AWB_MODE_FLUORESCENT
            "daylight" -> CameraMetadata.CONTROL_AWB_MODE_DAYLIGHT
            "cloudy" -> CameraMetadata.CONTROL_AWB_MODE_CLOUDY_DAYLIGHT
            "shade" -> CameraMetadata.CONTROL_AWB_MODE_SHADE
            "warm_fluorescent" -> CameraMetadata.CONTROL_AWB_MODE_WARM_FLUORESCENT
            "twilight" -> CameraMetadata.CONTROL_AWB_MODE_TWILIGHT
            "off" -> CameraMetadata.CONTROL_AWB_MODE_OFF
            else -> CameraMetadata.CONTROL_AWB_MODE_AUTO
        }
        updateRepeatingRequest()
    }

    fun setJpegQuality(quality: Int) {
        jpegQuality = quality.coerceIn(1, 100)
        updateRepeatingRequest()
    }

    fun setFps(min: Int, max: Int) {
        fpsRange = Range(min, max)
        updateRepeatingRequest()
    }

    fun setResolution(width: Int, height: Int, surfaceTexture: SurfaceTexture? = null) {
        val best = pickClosestSize(width, height)
        if (best.width == currentWidth && best.height == currentHeight) return
        currentWidth = best.width
        currentHeight = best.height

        captureSession?.close()
        captureSession = null

        val st = surfaceTexture ?: surfaceTextureRef
        if (st != null) {
            st.setDefaultBufferSize(currentWidth, currentHeight)
            previewSurface?.release()
            previewSurface = Surface(st)
        }

        createCaptureSession()
    }

    fun getStatus(): Map<String, Any> = mapOf(
        "resolution" to "${currentWidth}x${currentHeight}",
        "ae_mode" to if (aeMode == CameraMetadata.CONTROL_AE_MODE_ON) "auto" else "manual",
        "exposure_time_ns" to (manualExposureTime ?: -1),
        "iso" to (manualISO ?: -1),
        "exposure_compensation" to exposureCompensation,
        "af_mode" to if (afMode == CameraMetadata.CONTROL_AF_MODE_OFF) "manual" else "auto",
        "focus_distance" to (manualFocusDistance ?: -1f),
        "zoom" to zoomRatio,
        "awb_mode" to awbMode,
        "jpeg_quality" to jpegQuality,
        "fps_range" to (fpsRange?.let { "${it.lower}-${it.upper}" } ?: "unknown")
    )

    fun getCapabilities(): Map<String, Any> = mapOf(
        "exposure_range_ns" to listOf(exposureRange.lower, exposureRange.upper),
        "iso_range" to listOf(isoRange.lower, isoRange.upper),
        "min_focus_distance" to minFocusDistance,
        "max_zoom" to maxDigitalZoom,
        "exposure_compensation_range" to listOf(exposureCompensationRange.lower, exposureCompensationRange.upper),
        "exposure_compensation_step" to exposureCompensationStep,
        "available_resolutions" to availableResolutions.map { "${it.width}x${it.height}" },
        "awb_modes" to listOf("auto", "incandescent", "fluorescent", "daylight", "cloudy", "shade", "warm_fluorescent", "twilight", "off")
    )

    private val captureCallback = object : CameraCaptureSession.CaptureCallback() {
        override fun onCaptureCompleted(
            session: CameraCaptureSession,
            request: CaptureRequest,
            result: TotalCaptureResult
        ) {
            result.get(CaptureResult.SENSOR_EXPOSURE_TIME)?.let { lastAutoExposureNs = it }
            result.get(CaptureResult.SENSOR_SENSITIVITY)?.let { lastAutoISO = it }
            result.get(CaptureResult.LENS_FOCUS_DISTANCE)?.let { lastAutoFocusDistance = it }
        }
    }

    fun getAutoValues(): Map<String, Any> = mapOf(
        "exposure_time_ns" to lastAutoExposureNs,
        "iso" to lastAutoISO,
        "focus_distance" to lastAutoFocusDistance
    )

    fun lockFromAuto() {
        // Lock exposure to whatever auto chose
        aeMode = CameraMetadata.CONTROL_AE_MODE_OFF
        manualExposureTime = lastAutoExposureNs
        manualISO = lastAutoISO
        // Lock focus to whatever auto chose
        afMode = CameraMetadata.CONTROL_AF_MODE_OFF
        manualFocusDistance = lastAutoFocusDistance
        updateRepeatingRequest()
    }

    private fun getJpegOrientation(): Int {
        // Sensor natively outputs landscape. For landscape activity we want no rotation.
        return 0
    }

    private fun startBackgroundThread() {
        backgroundThread = HandlerThread("CameraBackground").also { it.start() }
        backgroundHandler = Handler(backgroundThread!!.looper)
    }

    private fun stopBackgroundThread() {
        backgroundThread?.quitSafely()
        try {
            backgroundThread?.join()
        } catch (_: InterruptedException) {}
        backgroundThread = null
        backgroundHandler = null
    }
}
