package com.example.activitycollector

import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.UUID

/**
 * AlertEvent
 * ==========
 * Data model for a recorded fall alert incident.
 */
data class AlertEvent(
    val id: String = UUID.randomUUID().toString(),
    val timestamp: Long = System.currentTimeMillis(),
    val outcome: Outcome = Outcome.DISMISSED_SAFE,
    val summary: String = "",
    val latitude: Double? = null,
    val longitude: Double? = null,
    /** Peak acceleration of the fall in m/s², or null when no fall was measured (manual SOS). */
    val impactForce: Float? = null,
    /** Posture change after the fall in degrees, or null when not measured. */
    val tiltAngle: Float? = null,
    /** Model's fall probability at impact (0-100), or null when not measured. */
    val confidencePct: Int? = null,
    /** Last activity the model reported before the fall, or null when unknown. */
    val activityBeforeFall: String? = null
) {
    enum class Outcome {
        DISMISSED_SAFE,
        SOS_DISPATCHED,
        CANCELLED_EARLY
    }

    val formattedDateTime: String
        get() = SimpleDateFormat("MMM dd, yyyy • hh:mm a", Locale.getDefault()).format(Date(timestamp))

    val relativeTimeSpan: String
        get() {
            val diff = System.currentTimeMillis() - timestamp
            val minutes = diff / 60_000L
            val hours = minutes / 60L
            val days = hours / 24L
            return when {
                minutes < 1 -> "Just now"
                minutes < 60 -> "$minutes min ago"
                hours < 24 -> "$hours hr ago"
                days == 1L -> "Yesterday"
                else -> "$days days ago"
            }
        }

    val mapsUrl: String?
        get() = if (latitude != null && longitude != null) {
            "https://maps.google.com/?q=$latitude,$longitude"
        } else null

    fun toJson(): JSONObject {
        return JSONObject().apply {
            put("id", id)
            put("timestamp", timestamp)
            put("outcome", outcome.name)
            put("summary", summary)
            if (latitude != null) put("latitude", latitude)
            if (longitude != null) put("longitude", longitude)
            impactForce?.let { put("impactForce", it.toDouble()) }
            tiltAngle?.let { put("tiltAngle", it.toDouble()) }
            confidencePct?.let { put("confidencePct", it) }
            activityBeforeFall?.let { put("activityBeforeFall", it) }
        }
    }

    companion object {
        fun fromJson(json: JSONObject): AlertEvent {
            return AlertEvent(
                id = json.optString("id", UUID.randomUUID().toString()),
                timestamp = json.optLong("timestamp", System.currentTimeMillis()),
                outcome = try {
                    Outcome.valueOf(json.optString("outcome", Outcome.DISMISSED_SAFE.name))
                } catch (_: Exception) {
                    Outcome.DISMISSED_SAFE
                },
                summary = json.optString("summary", ""),
                latitude = if (json.has("latitude")) json.optDouble("latitude") else null,
                longitude = if (json.has("longitude")) json.optDouble("longitude") else null,
                impactForce = if (json.has("impactForce")) json.getDouble("impactForce").toFloat() else null,
                tiltAngle = if (json.has("tiltAngle")) json.getDouble("tiltAngle").toFloat() else null,
                confidencePct = if (json.has("confidencePct")) json.getInt("confidencePct") else null,
                activityBeforeFall = if (json.has("activityBeforeFall")) json.getString("activityBeforeFall") else null
            )
        }
    }
}
