package com.example.spennotecollector.data.kinematics

import com.example.spennotecollector.data.model.KinematicPoint
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Computes instantaneous kinematic variables:
 * - Velocity (v in mm/s)
 * - Acceleration (a in mm/s^2)
 * - Jerk (j in mm/s^3)
 * - Dynamic azimuth change rate
 * - Pressure rate dp/dt
 */
class KinematicCalculator(
    private val xdpi: Float = 500f,
    private val ydpi: Float = 500f
) {
    private var prevPoint: KinematicPoint? = null
    private var prevPrevPoint: KinematicPoint? = null

    // Window for sampling frequency calculation
    private val recentTimestampsMs = ArrayDeque<Long>(64)

    /**
     * Converts screen pixel coordinates to physical millimeters based on display calibration.
     */
    fun pxToMmx(px: Float): Float = px * 25.4f / (if (xdpi > 0f) xdpi else 500f)
    fun pxToMmy(py: Float): Float = py * 25.4f / (if (ydpi > 0f) ydpi else 500f)

    /**
     * Resets the kinematic history (e.g. at stroke start or tool change).
     */
    fun reset() {
        prevPoint = null
        prevPrevPoint = null
        recentTimestampsMs.clear()
    }

    /**
     * Updates timestamps and computes the instantaneous sampling rate in Hz.
     */
    fun recordTimestamp(timeMs: Long): Float {
        recentTimestampsMs.addLast(timeMs)
        if (recentTimestampsMs.size > 50) {
            recentTimestampsMs.removeFirst()
        }
        if (recentTimestampsMs.size < 4) return 0f

        val deltaTTotalMs = recentTimestampsMs.last() - recentTimestampsMs.first()
        if (deltaTTotalMs <= 0L) return 0f

        val count = recentTimestampsMs.size - 1
        return (count * 1000f) / deltaTTotalMs
    }

    /**
     * Computes the derived kinematic metrics for a newly arrived raw point.
     */
    fun computeKinematics(
        xMm: Float,
        yMm: Float,
        pressure: Float,
        orientationRad: Float,
        timeMs: Long
    ): KinematicMetrics {
        val p1 = prevPoint
        if (p1 == null) {
            return KinematicMetrics(
                velocityMmPerSec = 0f,
                accelMmPerSec2 = 0f,
                jerkMmPerSec3 = 0f,
                azimuthVelocityRadPerSec = 0f,
                pressureRatePerSec = 0f
            )
        }

        val dtSec = (timeMs - p1.timestampMs) / 1000f
        if (dtSec <= 0.0001f) {
            return KinematicMetrics(
                velocityMmPerSec = p1.velocityMmPerSec,
                accelMmPerSec2 = p1.accelMmPerSec2,
                jerkMmPerSec3 = p1.jerkMmPerSec3,
                azimuthVelocityRadPerSec = p1.azimuthVelocityRadPerSec,
                pressureRatePerSec = p1.pressureRatePerSec
            )
        }

        val dxMm = xMm - p1.xMm
        val dyMm = yMm - p1.yMm
        val distMm = sqrt(dxMm * dxMm + dyMm * dyMm)
        val velocity = distMm / dtSec

        // Instantaneous acceleration
        val prevVel = p1.velocityMmPerSec
        val accel = (velocity - prevVel) / dtSec

        // Instantaneous jerk
        val prevAccel = p1.accelMmPerSec2
        val jerk = (accel - prevAccel) / dtSec

        // Dynamic orientation change rate (accounting for circular wrap-around [-PI, PI])
        val dAzimuth = normalizeAngleDifference(orientationRad, p1.orientationRad)
        val azimuthVelocity = abs(dAzimuth) / dtSec

        // Dynamic pressure change rate
        val dp = pressure - p1.pressure
        val pressureRate = dp / dtSec

        return KinematicMetrics(
            velocityMmPerSec = velocity,
            accelMmPerSec2 = accel,
            jerkMmPerSec3 = jerk,
            azimuthVelocityRadPerSec = azimuthVelocity,
            pressureRatePerSec = pressureRate
        )
    }

    fun updateState(newPoint: KinematicPoint) {
        prevPrevPoint = prevPoint
        prevPoint = newPoint
    }

    private fun normalizeAngleDifference(a: Float, b: Float): Float {
        var diff = a - b
        while (diff > Math.PI.toFloat()) diff -= (2 * Math.PI).toFloat()
        while (diff < -Math.PI.toFloat()) diff += (2 * Math.PI).toFloat()
        return diff
    }

    data class KinematicMetrics(
        val velocityMmPerSec: Float,
        val accelMmPerSec2: Float,
        val jerkMmPerSec3: Float,
        val azimuthVelocityRadPerSec: Float,
        val pressureRatePerSec: Float
    )
}
