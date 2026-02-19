package com.cvcoolz.androidcamera

import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.PrintWriter
import java.net.ServerSocket
import java.net.Socket
import kotlin.concurrent.thread

class ControlServer(
    private val port: Int = 5001,
    private val onCommand: (JSONObject) -> JSONObject
) {
    private var serverSocket: ServerSocket? = null

    @Volatile
    var isRunning = false; private set

    fun start() {
        if (isRunning) return
        isRunning = true

        thread(name = "ControlServer") {
            try {
                serverSocket = ServerSocket(port).also {
                    it.reuseAddress = true
                }
                Log.i(TAG, "Control server listening on port $port")

                while (isRunning) {
                    try {
                        val client = serverSocket?.accept() ?: break
                        thread(name = "ControlClient-${client.inetAddress}") {
                            handleClient(client)
                        }
                    } catch (e: Exception) {
                        if (isRunning) Log.w(TAG, "Accept error", e)
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Server error", e)
            }
        }
    }

    private fun handleClient(client: Socket) {
        try {
            val reader = BufferedReader(InputStreamReader(client.getInputStream()))
            val writer = PrintWriter(client.getOutputStream(), true)
            Log.i(TAG, "Control client connected: ${client.inetAddress}")

            while (isRunning && !client.isClosed) {
                val line = reader.readLine() ?: break
                if (line.isBlank()) continue

                try {
                    val command = JSONObject(line)
                    val response = onCommand(command)
                    writer.println(response.toString())
                } catch (e: Exception) {
                    val error = JSONObject().apply {
                        put("status", "error")
                        put("message", e.message ?: "Unknown error")
                    }
                    writer.println(error.toString())
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Client handler error", e)
        } finally {
            try { client.close() } catch (_: Exception) {}
            Log.i(TAG, "Control client disconnected")
        }
    }

    fun stop() {
        isRunning = false
        try { serverSocket?.close() } catch (_: Exception) {}
        serverSocket = null
        Log.i(TAG, "Control server stopped")
    }

    companion object {
        private const val TAG = "ControlServer"

        fun helpText(): JSONObject = JSONObject().apply {
            put("status", "ok")
            put("commands", JSONArray().apply {
                put(cmdHelp("set_exposure_time", "Set manual exposure in nanoseconds", "value: long"))
                put(cmdHelp("set_iso", "Set manual ISO sensitivity", "value: int"))
                put(cmdHelp("set_exposure_compensation", "Set AE exposure compensation (EV)", "value: int"))
                put(cmdHelp("set_auto_exposure", "Enable auto exposure", ""))
                put(cmdHelp("set_focus", "Set manual focus distance (diopters)", "value: float"))
                put(cmdHelp("set_auto_focus", "Enable auto focus", ""))
                put(cmdHelp("set_zoom", "Set zoom ratio", "value: float"))
                put(cmdHelp("set_white_balance", "Set white balance mode", "value: string"))
                put(cmdHelp("set_resolution", "Set capture resolution", "width: int, height: int"))
                put(cmdHelp("set_jpeg_quality", "Set JPEG quality 1-100", "value: int"))
                put(cmdHelp("set_fps", "Set target FPS range", "min: int, max: int"))
                put(cmdHelp("get_status", "Get current camera settings", ""))
                put(cmdHelp("get_capabilities", "Get camera capabilities/ranges", ""))
                put(cmdHelp("help", "Show this help", ""))
            })
        }

        private fun cmdHelp(cmd: String, desc: String, params: String) = JSONObject().apply {
            put("command", cmd)
            put("description", desc)
            if (params.isNotEmpty()) put("params", params)
        }
    }
}
