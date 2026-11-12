package com.example.activitycollector.wear

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.BatteryManager
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.os.PowerManager
import android.os.SystemClock
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone

/**
 * Records raw accelerometer and gyroscope events to a CSV file while the
 * watch is worn, including with the screen off.
 *
 * Every row is one real sensor event with the sensor's own hardware
 * timestamp. Nothing is resampled, merged or filled in: gaps in the data stay
 * visible as gaps, so the checker script can report them.
 *
 * File format (one file per session, in the app's external files dir):
 *   # metadata lines starting with '#'
 *   sensor,timestamp_ns,x,y,z
 *   A,<SensorEvent.timestamp>,<x>,<y>,<z>     accelerometer, m/s²
 *   G,<SensorEvent.timestamp>,<x>,<y>,<z>     gyroscope, rad/s
 */
class RecordingService : Service(), SensorEventListener {

    /** Live numbers shown on the recorder screen. All are measured, none estimated. */
    data class Stats(
        val label: String,
        val file: File,
        val startedAtElapsedMs: Long,
        val accelEvents: Long = 0,
        val gyroEvents: Long = 0,
        val bytesWritten: Long = 0,
        val batteryAtStartPct: Int? = null,
        val batteryNowPct: Int? = null,
    )

    companion object {
        private const val TAG = "RecordingService"
        private const val CHANNEL_ID = "recording"
        private const val NOTIFICATION_ID = 41
        private const val EXTRA_LABEL = "label"

        /** 20 ms = 50 Hz, the rate the fall model uses. */
        const val SAMPLING_PERIOD_US = 20_000

        private val _stats = MutableStateFlow<Stats?>(null)
        /** Null while not recording. */
        val stats: StateFlow<Stats?> = _stats.asStateFlow()

        fun start(context: Context, label: String) {
            val intent = Intent(context, RecordingService::class.java).putExtra(EXTRA_LABEL, label)
            context.startForegroundService(intent)
        }

        fun stop(context: Context) {
            context.stopService(Intent(context, RecordingService::class.java))
        }

        fun recordingsDir(context: Context): File =
            File(context.getExternalFilesDir(null), "recordings").apply { mkdirs() }
    }

    private lateinit var sensorManager: SensorManager
    private var sensorThread: HandlerThread? = null
    private var writer: BufferedWriter? = null
    private var wakeLock: PowerManager.WakeLock? = null
    private val uiHandler by lazy { Handler(mainLooper) }

    // Written only on the sensor thread.
    @Volatile private var accelCount = 0L
    @Volatile private var gyroCount = 0L
    @Volatile private var bytes = 0L

    private val statsTicker = object : Runnable {
        override fun run() {
            publishStats()
            uiHandler.postDelayed(this, 1000L)
        }
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (_stats.value != null) return START_NOT_STICKY // already recording
        val label = intent?.getStringExtra(EXTRA_LABEL) ?: run { stopSelf(); return START_NOT_STICKY }

        ServiceCompat.startForeground(
            this,
            NOTIFICATION_ID,
            buildNotification(label),
            if (Build.VERSION.SDK_INT >= 34) ServiceInfo.FOREGROUND_SERVICE_TYPE_HEALTH else 0,
        )

        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        val accel = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        val gyro = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        if (accel == null) {
            Log.e(TAG, "No accelerometer; cannot record")
            stopSelf()
            return START_NOT_STICKY
        }

        val file = newSessionFile(label)
        writer = BufferedWriter(FileWriter(file), 64 * 1024).also { w ->
            writeHeader(w, label, accel, gyro)
        }

        // A partial wake lock keeps the CPU draining the sensor FIFO while the
        // screen is off. Without it, samples can be silently overwritten.
        wakeLock = (getSystemService(POWER_SERVICE) as PowerManager)
            .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "FallGuard:Recording")
            .apply { acquire() }

        val thread = HandlerThread("sensor-recorder").apply { start() }
        sensorThread = thread
        val handler = Handler(thread.looper)
        sensorManager.registerListener(this, accel, SAMPLING_PERIOD_US, handler)
        gyro?.let { sensorManager.registerListener(this, it, SAMPLING_PERIOD_US, handler) }

        _stats.value = Stats(
            label = label,
            file = file,
            startedAtElapsedMs = SystemClock.elapsedRealtime(),
            batteryAtStartPct = batteryPct(),
            batteryNowPct = batteryPct(),
        )
        uiHandler.postDelayed(statsTicker, 1000L)
        Log.i(TAG, "Recording '$label' to ${file.absolutePath}")
        return START_NOT_STICKY
    }

    override fun onSensorChanged(event: SensorEvent) {
        val w = writer ?: return
        val tag = when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> { accelCount++; 'A' }
            Sensor.TYPE_GYROSCOPE -> { gyroCount++; 'G' }
            else -> return
        }
        val line = "$tag,${event.timestamp},${event.values[0]},${event.values[1]},${event.values[2]}\n"
        w.write(line)
        bytes += line.length
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    override fun onDestroy() {
        uiHandler.removeCallbacks(statsTicker)
        if (::sensorManager.isInitialized) sensorManager.unregisterListener(this)
        // Close the file on the sensor thread so no event is written after close.
        val thread = sensorThread
        if (thread != null) {
            Handler(thread.looper).post {
                closeWriter()
                thread.quitSafely()
            }
        } else {
            closeWriter()
        }
        wakeLock?.let { if (it.isHeld) it.release() }
        _stats.value = null
        super.onDestroy()
    }

    private fun closeWriter() {
        try {
            writer?.flush()
            writer?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to close recording: ${e.message}")
        }
        writer = null
    }

    private fun publishStats() {
        val current = _stats.value ?: return
        sensorThread?.let { t -> Handler(t.looper).post { writer?.flush() } }
        _stats.value = current.copy(
            accelEvents = accelCount,
            gyroEvents = gyroCount,
            bytesWritten = bytes,
            batteryNowPct = batteryPct(),
        )
    }

    private fun newSessionFile(label: String): File {
        val stamp = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())
        val safeLabel = label.lowercase(Locale.US).replace(Regex("[^a-z0-9]+"), "-").trim('-')
        return File(recordingsDir(this), "session_${stamp}_$safeLabel.csv")
    }

    private fun writeHeader(w: BufferedWriter, label: String, accel: Sensor, gyro: Sensor?) {
        val iso = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSZ", Locale.US).apply {
            timeZone = TimeZone.getDefault()
        }.format(Date())
        w.write("# fallguard_recording_version=1\n")
        w.write("# label=$label\n")
        w.write("# started_at=$iso\n")
        w.write("# device=${Build.MANUFACTURER} ${Build.MODEL} (API ${Build.VERSION.SDK_INT})\n")
        w.write("# accelerometer=${accel.vendor} ${accel.name}\n")
        w.write("# gyroscope=${gyro?.let { "${it.vendor} ${it.name}" } ?: "none"}\n")
        w.write("# requested_sampling_period_us=$SAMPLING_PERIOD_US\n")
        w.write("# timestamp_clock=SensorEvent.timestamp (ns, elapsedRealtimeNanos base)\n")
        w.write("# started_elapsed_ns=${SystemClock.elapsedRealtimeNanos()}\n")
        w.write("sensor,timestamp_ns,x,y,z\n")
    }

    private fun batteryPct(): Int? {
        val bm = getSystemService(BATTERY_SERVICE) as BatteryManager
        val pct = bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
        return if (pct in 0..100) pct else null
    }

    private fun buildNotification(label: String): Notification {
        val nm = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        nm.createNotificationChannel(
            NotificationChannel(CHANNEL_ID, "Recording", NotificationManager.IMPORTANCE_LOW)
        )
        val open = PendingIntent.getActivity(
            this, 0,
            Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
            PendingIntent.FLAG_IMMUTABLE,
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_media_play)
            .setContentTitle("FallGuard recording")
            .setContentText("Label: $label")
            .setOngoing(true)
            .setContentIntent(open)
            .setCategory(NotificationCompat.CATEGORY_WORKOUT)
            .build()
    }
}
