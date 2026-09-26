package com.example.spennotecollector.data.model

/**
 * Summary metrics of an entire data collection session.
 */
data class SessionSummary(
    val sessionId: String,
    val sessionTimestampUtc: Long,
    val formattedDate: String,
    val deviceManufacturer: String,
    val deviceModel: String,
    val androidVersion: String,
    val screenWidthPx: Int,
    val screenHeightPx: Int,
    val xdpi: Float,
    val ydpi: Float,
    val totalPointsLogged: Int,
    val totalOnSurfaceStrokes: Int,
    val totalInAirSegments: Int,
    val totalSessionDurationMs: Long,
    val onSurfaceWritingDurationMs: Long,
    val inAirHoverDurationMs: Long,
    val inAirTimeRatio: Float, // inAirTime / totalTime
    val averageSamplingRateHz: Float,
    val meanWritingVelocityMmPerSec: Float,
    val meanStrokePressure: Float,
    val paperStyleUsed: String
)
