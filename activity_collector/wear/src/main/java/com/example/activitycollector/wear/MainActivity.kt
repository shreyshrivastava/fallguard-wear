package com.example.activitycollector.wear

import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.os.SystemClock
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.wear.compose.foundation.lazy.ScalingLazyColumn
import androidx.wear.compose.foundation.lazy.rememberScalingLazyListState
import androidx.wear.compose.material.MaterialTheme
import androidx.wear.compose.material.PositionIndicator
import androidx.wear.compose.material.Scaffold
import androidx.wear.compose.material.Text
import androidx.wear.compose.material.TimeText
import com.example.activitycollector.SensorBuffer
import kotlin.math.sqrt

/**
 * Phase 1 screen: shows live accelerometer and gyroscope readings from the
 * watch's own sensors at the detector's 50 Hz rate. Every number on screen is
 * a measured value; anything not yet measured shows a dash.
 *
 * Sensors are read only while this screen is visible. Continuous background
 * monitoring moves to a foreground service in Phase 4.
 */
class MainActivity : ComponentActivity(), SensorEventListener {

    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null

    /** Same 2-second, 6-channel window the detector reads on the phone. */
    private val buffer = SensorBuffer(capacity = 100)

    private val accel = mutableStateOf<FloatArray?>(null)
    private val gyro = mutableStateOf<FloatArray?>(null)
    private val accelRateHz = mutableStateOf<Int?>(null)
    private val buffered = mutableStateOf(0)

    private var rateWindowStart = 0L
    private var rateWindowCount = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

        setContent {
            LiveSensorScreen(
                hasAccelerometer = accelerometer != null,
                hasGyroscope = gyroscope != null,
                accel = accel.value,
                gyro = gyro.value,
                accelRateHz = accelRateHz.value,
                buffered = buffered.value,
                capacity = buffer.capacity,
            )
        }
    }

    override fun onResume() {
        super.onResume()
        rateWindowStart = SystemClock.elapsedRealtime()
        rateWindowCount = 0
        accelerometer?.let { sensorManager.registerListener(this, it, SAMPLING_PERIOD_US) }
        gyroscope?.let { sensorManager.registerListener(this, it, SAMPLING_PERIOD_US) }
    }

    override fun onPause() {
        sensorManager.unregisterListener(this)
        accelRateHz.value = null
        super.onPause()
    }

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_GYROSCOPE -> {
                buffer.latestGx = event.values[0]
                buffer.latestGy = event.values[1]
                buffer.latestGz = event.values[2]
                gyro.value = event.values.copyOf(3)
            }
            Sensor.TYPE_ACCELEROMETER -> {
                buffer.push(event.values[0], event.values[1], event.values[2])
                accel.value = event.values.copyOf(3)
                buffered.value = buffer.size
                countForRate()
            }
        }
    }

    /** Measures the real accelerometer event rate over each 1-second window. */
    private fun countForRate() {
        rateWindowCount++
        val now = SystemClock.elapsedRealtime()
        val elapsed = now - rateWindowStart
        if (elapsed >= 1000L) {
            accelRateHz.value = (rateWindowCount * 1000L / elapsed).toInt()
            rateWindowStart = now
            rateWindowCount = 0
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    companion object {
        /** 20 ms = 50 Hz, the rate the fall model was trained on. */
        private const val SAMPLING_PERIOD_US = 20_000
    }
}

private val Mint = Color(0xFF4ADE9F)
private val Muted = Color(0xFF97A9A1)

@Composable
private fun LiveSensorScreen(
    hasAccelerometer: Boolean,
    hasGyroscope: Boolean,
    accel: FloatArray?,
    gyro: FloatArray?,
    accelRateHz: Int?,
    buffered: Int,
    capacity: Int,
) {
    val listState = rememberScalingLazyListState()
    MaterialTheme {
        Scaffold(
            timeText = { TimeText() },
            positionIndicator = { PositionIndicator(scalingLazyListState = listState) },
        ) {
            ScalingLazyColumn(
                modifier = Modifier.fillMaxWidth(),
                state = listState,
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                item {
                    Text("FallGuard", fontWeight = FontWeight.Bold, fontSize = 18.sp, color = Mint)
                }
                item {
                    Text("Live sensors", fontSize = 12.sp, color = Muted)
                }
                item {
                    SensorBlock(
                        title = "Accelerometer · m/s²",
                        available = hasAccelerometer,
                        values = accel,
                        extra = accel?.let { "|a| ${fmt(magnitude(it))}" },
                    )
                }
                item {
                    SensorBlock(
                        title = "Gyroscope · rad/s",
                        available = hasGyroscope,
                        values = gyro,
                        extra = null,
                    )
                }
                item {
                    Text(
                        text = "Rate ${accelRateHz?.let { "$it Hz" } ?: "—"}   Window $buffered/$capacity",
                        fontSize = 12.sp,
                        color = Muted,
                        textAlign = TextAlign.Center,
                    )
                }
            }
        }
    }
}

@Composable
private fun SensorBlock(title: String, available: Boolean, values: FloatArray?, extra: String?) {
    Column(
        modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(2.dp),
    ) {
        Text(title, fontSize = 11.sp, color = Muted)
        when {
            !available -> Text("Not on this watch", fontSize = 14.sp)
            values == null -> Text("—", fontSize = 14.sp, fontFamily = FontFamily.Monospace)
            else -> Text(
                "x ${fmt(values[0])}  y ${fmt(values[1])}  z ${fmt(values[2])}",
                fontSize = 13.sp,
                fontFamily = FontFamily.Monospace,
            )
        }
        if (extra != null) Text(extra, fontSize = 12.sp, fontFamily = FontFamily.Monospace, color = Mint)
    }
}

private fun magnitude(v: FloatArray): Float = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

private fun fmt(v: Float): String = "%.2f".format(v)
