package com.example.spennotecollector.data.model

/**
 * High-precision multidimensional sensor sample captured from S-Pen / Wacom EMR digitizer.
 * Captures all standard offline and online kinematic properties required for dysgraphia screening.
 */
data class KinematicPoint(
    val timestampMs: Long,
    val elapsedNanos: Long,
    val strokeId: Int,
    val pointIndex: Int,
    val action: String, // DOWN, MOVE, UP, HOVER_ENTER, HOVER_MOVE, HOVER_EXIT, CANCEL
    val contactState: String, // ON_SURFACE or IN_AIR
    val xPx: Float,
    val yPx: Float,
    val xMm: Float,
    val yMm: Float,
    val pressure: Float, // 0.0f to 1.0f (4096 levels on Samsung S-Pen)
    val tiltRad: Float, // AXIS_TILT in radians (0 = perpendicular, PI/2 = flat)
    val orientationRad: Float, // AXIS_ORIENTATION in radians (-PI to +PI)
    val hoverDistance: Float, // AXIS_DISTANCE (0.0 on surface, >0.0 in-air)
    val touchMajor: Float, // Contact ellipse major axis in px
    val touchMinor: Float, // Contact ellipse minor axis in px
    val toolMajor: Float, // Tool ellipse major axis in px
    val toolMinor: Float, // Tool ellipse minor axis in px
    val buttonState: Int, // S-Pen barrel button bitmask
    val velocityMmPerSec: Float = 0f, // Instantaneous velocity in mm/s
    val accelMmPerSec2: Float = 0f, // Instantaneous acceleration in mm/s^2
    val jerkMmPerSec3: Float = 0f, // Instantaneous jerk in mm/s^3 (tremor metric)
    val azimuthVelocityRadPerSec: Float = 0f, // Dynamic change rate in pen orientation
    val pressureRatePerSec: Float = 0f // Rate of pressure variation dp/dt
) {
    /**
     * Serializes this point into standard CSV format.
     */
    fun toCsvRow(): String {
        return "$timestampMs,$elapsedNanos,$strokeId,$pointIndex,$action,$contactState," +
                "%.3f,%.3f,%.4f,%.4f,%.5f,%.4f,%.4f,%.4f,".format(
                    xPx, yPx, xMm, yMm, pressure, tiltRad, orientationRad, hoverDistance
                ) +
                "%.2f,%.2f,%.2f,%.2f,$buttonState,".format(
                    touchMajor, touchMinor, toolMajor, toolMinor
                ) +
                "%.3f,%.3f,%.3f,%.4f,%.4f".format(
                    velocityMmPerSec, accelMmPerSec2, jerkMmPerSec3, azimuthVelocityRadPerSec, pressureRatePerSec
                )
    }

    companion object {
        const val CSV_HEADER = "timestamp_ms,elapsed_nanos,stroke_id,point_idx,action,contact_state," +
                "x_px,y_px,x_mm,y_mm,pressure,tilt_rad,orientation_rad,hover_dist," +
                "touch_major,touch_minor,tool_major,tool_minor,button_state," +
                "velocity_mms,accel_mms2,jerk_mms3,azimuth_vel_rads,pressure_rate_s"
    }
}
