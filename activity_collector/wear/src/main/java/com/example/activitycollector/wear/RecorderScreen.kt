package com.example.activitycollector.wear

import android.os.SystemClock
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.sp
import androidx.wear.compose.foundation.lazy.ScalingLazyColumn
import androidx.wear.compose.foundation.lazy.items
import androidx.wear.compose.foundation.lazy.rememberScalingLazyListState
import androidx.wear.compose.material.Chip
import androidx.wear.compose.material.ChipDefaults
import androidx.wear.compose.material.MaterialTheme
import androidx.wear.compose.material.PositionIndicator
import androidx.wear.compose.material.Scaffold
import androidx.wear.compose.material.Text
import androidx.wear.compose.material.TimeText
import kotlinx.coroutines.delay

/**
 * What the wearer is about to do. One label per session: stop and start a new
 * session when the activity changes. The false-alarm activities come from the
 * research notes on what triggers wrist fall detectors.
 */
val RECORDING_LABELS = listOf(
    "Normal day",
    "Simulated fall onto mat",
    "Lying down quickly",
    "Sitting down hard",
    "Walking",
    "Stairs",
    "Clapping",
    "Typing",
    "Hand-washing",
    "Hitting a desk",
    "Brushing teeth",
    "Taking watch off",
)

private val Mint = Color(0xFF4ADE9F)
private val Muted = Color(0xFF97A9A1)
private val Stop = Color(0xFFF87171)

@Composable
fun RecorderScreen(
    stats: RecordingService.Stats?,
    onStart: (String) -> Unit,
    onStop: () -> Unit,
    savedSessions: Int,
) {
    val listState = rememberScalingLazyListState()
    var selected by remember { mutableStateOf<String?>(null) }

    // Elapsed time is measured from the service's start time, ticking once a second.
    var now by remember { mutableLongStateOf(SystemClock.elapsedRealtime()) }
    LaunchedEffect(stats != null) {
        while (stats != null) {
            now = SystemClock.elapsedRealtime()
            delay(1000L)
        }
    }

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
                if (stats == null) {
                    item { Text("Record data", fontWeight = FontWeight.Bold, fontSize = 16.sp, color = Mint) }
                    item {
                        Text(
                            "Pick what you're about to do",
                            fontSize = 12.sp, color = Muted, textAlign = TextAlign.Center,
                        )
                    }
                    items(RECORDING_LABELS) { label ->
                        val isSelected = label == selected
                        Chip(
                            onClick = { selected = label },
                            label = { Text(label, fontSize = 13.sp) },
                            modifier = Modifier.fillMaxWidth(),
                            colors = if (isSelected) {
                                ChipDefaults.primaryChipColors(backgroundColor = Mint, contentColor = Color.Black)
                            } else {
                                ChipDefaults.secondaryChipColors()
                            },
                        )
                    }
                    item {
                        Chip(
                            onClick = { selected?.let(onStart) },
                            enabled = selected != null,
                            label = { Text(if (selected == null) "Pick a label first" else "Start recording") },
                            colors = ChipDefaults.primaryChipColors(backgroundColor = Mint, contentColor = Color.Black),
                        )
                    }
                    item { Text("$savedSessions saved session(s)", fontSize = 11.sp, color = Muted) }
                } else {
                    val elapsedMs = (now - stats.startedAtElapsedMs).coerceAtLeast(0L)
                    val elapsedS = elapsedMs / 1000L
                    item { Text("● Recording", fontWeight = FontWeight.Bold, fontSize = 15.sp, color = Stop) }
                    item { Text(stats.label, fontSize = 14.sp, textAlign = TextAlign.Center) }
                    item {
                        Text(
                            "%d:%02d:%02d".format(elapsedS / 3600, (elapsedS / 60) % 60, elapsedS % 60),
                            fontSize = 22.sp, fontFamily = FontFamily.Monospace,
                        )
                    }
                    item {
                        val rate = if (elapsedMs >= 2000L) "%.1f Hz".format(stats.accelEvents * 1000.0 / elapsedMs) else "—"
                        Stat("Accel samples", "${stats.accelEvents}  ($rate)")
                    }
                    item { Stat("Gyro samples", "${stats.gyroEvents}") }
                    item { Stat("File size", "%.1f MB".format(stats.bytesWritten / 1_000_000.0)) }
                    item {
                        val start = stats.batteryAtStartPct
                        val nowPct = stats.batteryNowPct
                        Stat(
                            "Battery",
                            if (start != null && nowPct != null) "$start% → $nowPct%" else "—",
                        )
                    }
                    item {
                        Chip(
                            onClick = onStop,
                            label = { Text("Stop and save") },
                            colors = ChipDefaults.primaryChipColors(backgroundColor = Stop, contentColor = Color.Black),
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun Stat(name: String, value: String) {
    Text("$name: $value", fontSize = 12.sp, fontFamily = FontFamily.Monospace, textAlign = TextAlign.Center)
}
