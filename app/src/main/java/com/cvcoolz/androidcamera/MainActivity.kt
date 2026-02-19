package com.cvcoolz.androidcamera

import android.Manifest
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.graphics.Matrix
import android.graphics.SurfaceTexture
import android.os.Bundle
import android.os.IBinder
import android.util.Log
import android.view.Surface
import android.view.TextureView
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.json.JSONObject
import java.net.Inet4Address
import java.net.NetworkInterface

class MainActivity : AppCompatActivity() {
    companion object {
        private const val TAG = "MainActivity"
        private const val REQUEST_CAMERA = 100
        private const val STREAM_PORT = 5000
        private const val CONTROL_PORT = 5001
    }

    private lateinit var previewView: TextureView
    private lateinit var toggleButton: Button
    private lateinit var statusText: TextView

    private var streamService: StreamService? = null
    private var serviceBound = false

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
            streamService = (binder as StreamService.LocalBinder).service
            serviceBound = true
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            streamService = null
            serviceBound = false
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        toggleButton = findViewById(R.id.toggleButton)
        statusText = findViewById(R.id.statusText)

        // Bind to service
        Intent(this, StreamService::class.java).also {
            bindService(it, serviceConnection, Context.BIND_AUTO_CREATE)
        }

        toggleButton.setOnClickListener {
            val svc = streamService ?: return@setOnClickListener
            if (svc.isStreaming) stopStreaming() else startStreaming()
        }

        previewView.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
                if (!hasCameraPermission()) {
                    requestCameraPermission()
                }
            }
            override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) {}
            override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean = true
            override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {}
        }
    }

    private fun startStreaming() {
        if (!hasCameraPermission()) {
            requestCameraPermission()
            return
        }
        val svc = streamService ?: return
        val surfaceTexture = previewView.surfaceTexture // nullable — headless if null

        // Start as foreground service first (must be called from foreground activity)
        Intent(this, StreamService::class.java).also {
            startForegroundService(it)
        }

        svc.startStreaming(surfaceTexture) { cmd -> handleCommand(cmd) }

        svc.cameraController?.let { cam ->
            configurePreviewTransform(cam.currentWidth, cam.currentHeight, cam.sensorOrientation)
        }

        toggleButton.text = "Stop"
        val ip = getDeviceIp()
        statusText.text = "Streaming on $ip:$STREAM_PORT | Control: $ip:$CONTROL_PORT"
        Log.i(TAG, "Streaming started - $ip:$STREAM_PORT")
    }

    private fun stopStreaming() {
        streamService?.stopStreaming()
        toggleButton.text = "Start"
        statusText.text = "Stopped"
        Log.i(TAG, "Streaming stopped")
    }

    @Suppress("DEPRECATION")
    private fun configurePreviewTransform(previewWidth: Int, previewHeight: Int, sensorOrientation: Int) {
        val viewWidth = previewView.width.toFloat()
        val viewHeight = previewView.height.toFloat()
        if (viewWidth == 0f || viewHeight == 0f) return

        val displayRotation = when (windowManager.defaultDisplay.rotation) {
            Surface.ROTATION_0 -> 0
            Surface.ROTATION_90 -> 90
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_270 -> 270
            else -> 0
        }

        val matrix = Matrix()
        val centerX = viewWidth / 2f
        val centerY = viewHeight / 2f

        val rotateDegrees = (sensorOrientation - displayRotation + 360) % 360

        if (rotateDegrees == 90 || rotateDegrees == 270) {
            matrix.postRotate(rotateDegrees.toFloat(), centerX, centerY)
            val scaleX = viewWidth / viewHeight
            val scaleY = viewHeight / viewWidth
            matrix.postScale(scaleX, scaleY, centerX, centerY)
        } else if (rotateDegrees == 180) {
            matrix.postRotate(180f, centerX, centerY)
        }

        previewView.setTransform(matrix)
    }

    private fun handleCommand(cmd: JSONObject): JSONObject {
        val camera = streamService?.cameraController
            ?: return error("Camera not active")

        return try {
            when (cmd.getString("command")) {
                "set_exposure_time" -> {
                    camera.setExposureTimeNs(cmd.getLong("value"))
                    ok("Exposure time set")
                }
                "set_iso" -> {
                    camera.setISO(cmd.getInt("value"))
                    ok("ISO set")
                }
                "set_exposure_compensation" -> {
                    camera.setExposureCompensation(cmd.getInt("value"))
                    ok("Exposure compensation set")
                }
                "set_auto_exposure" -> {
                    camera.setAutoExposure()
                    ok("Auto exposure enabled")
                }
                "set_focus" -> {
                    camera.setFocusDistance(cmd.getDouble("value").toFloat())
                    ok("Focus distance set")
                }
                "set_auto_focus" -> {
                    camera.setAutoFocus()
                    ok("Auto focus enabled")
                }
                "set_zoom" -> {
                    camera.setZoom(cmd.getDouble("value").toFloat())
                    ok("Zoom set")
                }
                "set_white_balance" -> {
                    camera.setWhiteBalance(cmd.getString("value"))
                    ok("White balance set")
                }
                "set_resolution" -> {
                    camera.setResolution(cmd.getInt("width"), cmd.getInt("height"))
                    ok("Resolution set to ${camera.currentWidth}x${camera.currentHeight}")
                }
                "set_jpeg_quality" -> {
                    camera.setJpegQuality(cmd.getInt("value"))
                    ok("JPEG quality set")
                }
                "set_fps" -> {
                    camera.setFps(cmd.getInt("min"), cmd.getInt("max"))
                    ok("FPS range set")
                }
                "get_status" -> {
                    JSONObject().apply {
                        put("status", "ok")
                        put("streaming", streamService?.isStreaming ?: false)
                        put("clients", streamService?.streamingServer?.clientCount ?: 0)
                        camera.getStatus().forEach { (k, v) -> put(k, v) }
                    }
                }
                "get_capabilities" -> {
                    JSONObject().apply {
                        put("status", "ok")
                        camera.getCapabilities().forEach { (k, v) -> put(k, v) }
                    }
                }
                "get_auto_values" -> {
                    JSONObject().apply {
                        put("status", "ok")
                        camera.getAutoValues().forEach { (k, v) -> put(k, v) }
                    }
                }
                "lock_from_auto" -> {
                    camera.lockFromAuto()
                    val vals = camera.getAutoValues()
                    JSONObject().apply {
                        put("status", "ok")
                        put("message", "Locked from auto values")
                        vals.forEach { (k, v) -> put(k, v) }
                    }
                }
                "help" -> ControlServer.helpText()
                else -> error("Unknown command: ${cmd.optString("command")}")
            }
        } catch (e: Exception) {
            error(e.message ?: "Command failed")
        }
    }

    private fun ok(msg: String) = JSONObject().apply {
        put("status", "ok")
        put("message", msg)
    }

    private fun error(msg: String) = JSONObject().apply {
        put("status", "error")
        put("message", msg)
    }

    private fun getDeviceIp(): String {
        try {
            val interfaces = NetworkInterface.getNetworkInterfaces()
            while (interfaces.hasMoreElements()) {
                val iface = interfaces.nextElement()
                if (iface.isLoopback || !iface.isUp) continue
                val addresses = iface.inetAddresses
                while (addresses.hasMoreElements()) {
                    val addr = addresses.nextElement()
                    if (addr is Inet4Address && !addr.isLoopbackAddress) {
                        return addr.hostAddress ?: "unknown"
                    }
                }
            }
        } catch (_: Exception) {}
        return "unknown"
    }

    private fun hasCameraPermission() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

    private fun requestCameraPermission() {
        ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), REQUEST_CAMERA)
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CAMERA) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "Camera permission granted. Press Start.", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show()
            }
        }
    }

    override fun onDestroy() {
        if (serviceBound) {
            unbindService(serviceConnection)
            serviceBound = false
        }
        super.onDestroy()
    }
}
