package com.example.spennotecollector.data.model

import kotlin.math.pow
import kotlin.math.sqrt

/**
 * Encapsulates a complete continuous handwriting stroke (from DOWN to UP)
 * or continuous in-air hovering flight segment.
 */
data class StrokeData(
    val strokeId: Int,
    val isHoverStroke: Boolean = false,
    val points: MutableList<KinematicPoint> = mutableListOf()
) {
    val startTimeMs: Long
        get() = points.firstOrNull()?.timestampMs ?: 0L

    val endTimeMs: Long
        get() = points.lastOrNull()?.timestampMs ?: 0L

    val durationMs: Long
        get() = if (points.size >= 2) endTimeMs - startTimeMs else 0L

    val pointsCount: Int
        get() = points.size

    val pathLengthMm: Float
        get() {
            if (points.size < 2) return 0f
            var length = 0f
            for (i in 1 until points.size) {
                val dx = points[i].xMm - points[i - 1].xMm
                val dy = points[i].yMm - points[i - 1].yMm
                length += sqrt(dx * dx + dy * dy)
            }
            return length
        }

    val meanPressure: Float
        get() {
            if (points.isEmpty()) return 0f
            return points.map { it.pressure }.average().toFloat()
        }

    val maxPressure: Float
        get() = points.maxOfOrNull { it.pressure } ?: 0f

    val pressureVariance: Float
        get() {
            if (points.size < 2) return 0f
            val mean = meanPressure
            return points.map { (it.pressure - mean).pow(2) }.average().toFloat()
        }

    val meanVelocityMmPerSec: Float
        get() {
            val valid = points.filter { it.velocityMmPerSec > 0f }
            if (valid.isEmpty()) return 0f
            return valid.map { it.velocityMmPerSec }.average().toFloat()
        }

    val maxVelocityMmPerSec: Float
        get() = points.maxOfOrNull { it.velocityMmPerSec } ?: 0f

    val meanJerkMmPerSec3: Float
        get() {
            val valid = points.filter { it.jerkMmPerSec3 > 0f }
            if (valid.isEmpty()) return 0f
            return valid.map { it.jerkMmPerSec3 }.average().toFloat()
        }

    /**
     * Count micro-hesitations or pauses within the stroke (velocity dropping below 5 mm/s).
     * High hesitation count within a single continuous letter/stroke is a strong dysgraphia indicator.
     */
    val hesitationCount: Int
        get() {
            if (points.size < 3) return 0
            var count = 0
            for (i in 1 until points.size - 1) {
                if (points[i].velocityMmPerSec < 5f &&
                    points[i - 1].velocityMmPerSec >= 5f
                ) {
                    count++
                }
            }
            return count
        }
}
