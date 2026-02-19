package com.cvcoolz.androidcamera

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.graphics.SurfaceTexture
import android.os.Binder
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import org.json.JSONObject

class StreamService : Service() {
    companion object {
        private const val TAG = "StreamService"
        private const val CHANNEL_ID = "cv_camera_stream"
        private const val NOTIFICATION_ID = 1
        private const val STREAM_PORT = 5000
        private const val CONTROL_PORT = 5001
    }

    inner class LocalBinder : Binder() {
        val service: StreamService get() = this@StreamService
    }

    private val binder = LocalBinder()

    var cameraController: CameraController? = null; private set
    var streamingServer: StreamingServer? = null; private set
    var controlServer: ControlServer? = null; private set
    private var wakeLock: PowerManager.WakeLock? = null

    var isStreaming = false; private set

    override fun onBind(intent: Intent?): IBinder = binder

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
    }

    fun startStreaming(surfaceTexture: SurfaceTexture?, commandHandler: (JSONObject) -> JSONObject) {
        if (isStreaming) return

        // Start foreground service with notification
        val notification = buildNotification()
        startForeground(NOTIFICATION_ID, notification)

        acquireWakeLock()

        streamingServer = StreamingServer(STREAM_PORT).also { it.start() }

        cameraController = CameraController(this) { jpegData ->
            streamingServer?.sendFrame(jpegData)
        }.also { it.open(surfaceTexture) }

        controlServer = ControlServer(CONTROL_PORT, commandHandler).also { it.start() }

        isStreaming = true
        Log.i(TAG, "Streaming started")
    }

    fun stopStreaming() {
        controlServer?.stop()
        controlServer = null

        cameraController?.close()
        cameraController = null

        streamingServer?.stop()
        streamingServer = null

        releaseWakeLock()

        isStreaming = false
        stopForeground(STOP_FOREGROUND_REMOVE)
        Log.i(TAG, "Streaming stopped")
    }

    private fun acquireWakeLock() {
        val pm = getSystemService(POWER_SERVICE) as PowerManager
        wakeLock = pm.newWakeLock(
            PowerManager.PARTIAL_WAKE_LOCK,
            "CVCamera::StreamWakeLock"
        ).apply { acquire() }
    }

    private fun releaseWakeLock() {
        wakeLock?.let { if (it.isHeld) it.release() }
        wakeLock = null
    }

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID,
            "Camera Stream",
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            description = "Camera streaming service"
        }
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(channel)
    }

    private fun buildNotification(): Notification {
        val intent = Intent(this, MainActivity::class.java)
        val pendingIntent = PendingIntent.getActivity(
            this, 0, intent, PendingIntent.FLAG_IMMUTABLE
        )

        return Notification.Builder(this, CHANNEL_ID)
            .setContentTitle("CV Camera")
            .setContentText("Streaming on port $STREAM_PORT")
            .setSmallIcon(android.R.drawable.ic_menu_camera)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .build()
    }

    override fun onDestroy() {
        stopStreaming()
        super.onDestroy()
    }
}
