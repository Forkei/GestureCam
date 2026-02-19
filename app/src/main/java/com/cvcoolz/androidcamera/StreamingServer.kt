package com.cvcoolz.androidcamera

import android.util.Log
import java.io.OutputStream
import java.net.ServerSocket
import java.net.Socket
import java.nio.ByteBuffer
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.atomic.AtomicReference
import kotlin.concurrent.thread

class StreamingServer(private val port: Int = 5000) {
    companion object {
        private const val TAG = "StreamingServer"
    }

    private var serverSocket: ServerSocket? = null
    private val clients = CopyOnWriteArrayList<ClientConnection>()
    private val latestFrame = AtomicReference<ByteArray?>(null)

    @Volatile
    var isRunning = false; private set

    val clientCount: Int get() = clients.size

    private class ClientConnection(val socket: Socket) {
        val output: OutputStream = socket.getOutputStream()
    }

    fun start() {
        if (isRunning) return
        isRunning = true

        // Accept thread
        thread(name = "StreamAccept") {
            try {
                serverSocket = ServerSocket(port).also {
                    it.reuseAddress = true
                }
                Log.i(TAG, "Streaming server listening on port $port")

                while (isRunning) {
                    try {
                        val client = serverSocket?.accept() ?: break
                        client.tcpNoDelay = true
                        client.sendBufferSize = 512 * 1024
                        client.setSoLinger(true, 0)
                        clients.add(ClientConnection(client))
                        Log.i(TAG, "Client connected: ${client.inetAddress} (${clients.size} total)")
                    } catch (e: Exception) {
                        if (isRunning) Log.w(TAG, "Accept error", e)
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Server error", e)
            }
        }

        // Sender thread — decoupled from camera capture
        thread(name = "StreamSender") {
            var lastSent: ByteArray? = null
            while (isRunning) {
                val frame = latestFrame.getAndSet(null)
                if (frame != null && frame !== lastSent && clients.isNotEmpty()) {
                    lastSent = frame
                    broadcastFrame(frame)
                } else {
                    Thread.sleep(1) // yield briefly when no new frame
                }
            }
        }
    }

    fun sendFrame(jpegData: ByteArray) {
        // Non-blocking: just store the latest frame
        latestFrame.set(jpegData)
    }

    private fun broadcastFrame(jpegData: ByteArray) {
        val header = ByteBuffer.allocate(4).putInt(jpegData.size).array()
        val dead = mutableListOf<ClientConnection>()

        for (client in clients) {
            try {
                client.output.write(header)
                client.output.write(jpegData)
                client.output.flush()
            } catch (e: Exception) {
                dead.add(client)
                try { client.socket.close() } catch (_: Exception) {}
            }
        }

        if (dead.isNotEmpty()) {
            clients.removeAll(dead.toSet())
            Log.i(TAG, "Removed ${dead.size} dead client(s), ${clients.size} remaining")
        }
    }

    fun stop() {
        isRunning = false
        clients.forEach { try { it.socket.close() } catch (_: Exception) {} }
        clients.clear()
        try { serverSocket?.close() } catch (_: Exception) {}
        serverSocket = null
        Log.i(TAG, "Streaming server stopped")
    }
}
