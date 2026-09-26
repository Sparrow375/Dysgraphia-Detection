package com.example.spennotecollector.ui.canvas

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.os.Build
import android.os.SystemClock
import android.util.AttributeSet
import android.view.InputDevice
import android.view.MotionEvent
import android.view.View
import com.example.spennotecollector.data.kinematics.KinematicCalculator
import com.example.spennotecollector.data.model.CanvasTool
import com.example.spennotecollector.data.model.KinematicPoint
import com.example.spennotecollector.data.model.PaperStyle
import com.example.spennotecollector.data.model.SessionSummary
import com.example.spennotecollector.data.model.StrokeData
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.UUID
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Custom high-performance Drawing View optimized for Samsung S26 Ultra S-Pen digitizer.
 * Features:
 * - Unbuffered touch dispatch (bypasses frame batching)
 * - Historical event buffer unpacking (captures every intermediate micro-sample)
 * - In-air hover trajectory logging (ACTION_HOVER_MOVE)
 * - Strict palm rejection (ignores finger when S-Pen is active)
 * - S-Pen hardware button toggle for eraser
 * - Variable-width pressure-sensitive ink rendering
 * - Paper background simulation (Ruled, Grid, Blank)
 */
class SPenDrawingView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    interface TelemetryListener {
        fun onTelemetryUpdate(
            pressure: Float,
            tiltDeg: Float,
            orientationDeg: Float,
            samplingHz: Float,
            isHovering: Boolean,
            hoverDist: Float,
            totalPoints: Int,
            strokeCount: Int,
            velocityMmPerSec: Float
        )
    }

    var telemetryListener: TelemetryListener? = null

    // Session State
    val sessionId: String = UUID.randomUUID().toString().take(8).uppercase()
    val sessionStartTimeUtc: Long = System.currentTimeMillis()
    private val sessionStartUptimeMs: Long = SystemClock.uptimeMillis()

    // Kinematics and display metrics
    private val displayMetrics = resources.displayMetrics
    private val kinematicCalc = KinematicCalculator(displayMetrics.xdpi, displayMetrics.ydpi)

    // Storage for all recorded points and strokes in this session
    private val allPoints = mutableListOf<KinematicPoint>()
    private val completedStrokes = mutableListOf<StrokeData>()
    private var currentStroke: StrokeData? = null
    private var currentHoverStroke: StrokeData? = null

    // Active tool and styling
    var activeTool: CanvasTool = CanvasTool.PEN
    var paperStyle: PaperStyle = PaperStyle.RULED
        set(value) {
            field = value
            invalidate()
        }

    var baseStrokeWidthPx: Float = 6.0f
    var inkColor: Int = Color.rgb(20, 24, 32)
    var palmRejectionEnabled: Boolean = true

    // Drawing state
    private val strokePaths = mutableListOf<DrawableStroke>()
    private var currentPath = Path()
    private var currentStrokeSegments = mutableListOf<PathSegment>()
    private var lastX: Float = 0f
    private var lastY: Float = 0f

    // Hover state
    private var hoverX: Float = -1f
    private var hoverY: Float = -1f
    private var isHoveringNow: Boolean = false

    // Telemetry tracking
    private var strokeCounter: Int = 0
    private var hoverSegmentCounter: Int = 0
    private var pointCounter: Int = 0
    private var totalOnSurfaceTimeMs: Long = 0L
    private var totalInAirTimeMs: Long = 0L
    private var lastEventTimeMs: Long = 0L

    // Paints
    private val inkPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
    }

    private val ruledLinePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(45, 60, 110, 180) // Light subtle blue ruled lines
        strokeWidth = 1.5f
    }

    private val marginLinePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(55, 220, 60, 60) // Red notebook left margin
        strokeWidth = 2.0f
    }

    private val gridLinePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(35, 120, 140, 160)
        strokeWidth = 1.0f
    }

    private val hoverCursorPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(120, 33, 150, 243)
        style = Paint.Style.STROKE
        strokeWidth = 2.0f
    }

    private val hoverDotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(180, 33, 150, 243)
        style = Paint.Style.FILL
    }

    init {
        // Enable generic motion events for hover capture
        isFocusable = true
        isFocusableInTouchMode = true
    }

    // =========================================================================
    // TOUCH EVENTS: ON-SURFACE DRAWING & KINEMATIC BATCH EXTRACTION
    // =========================================================================

    override fun onTouchEvent(event: MotionEvent): Boolean {
        // Request unbuffered dispatch to eliminate Android VSync frame-batching delays
        if (event.actionMasked == MotionEvent.ACTION_DOWN) {
            requestUnbufferedDispatch(event)
        }

        val toolType = event.getToolType(0)
        val isStylus = (toolType == MotionEvent.TOOL_TYPE_STYLUS || toolType == MotionEvent.TOOL_TYPE_ERASER)

        // Palm rejection: filter out finger touches if palm rejection is on and stylus was used
        if (palmRejectionEnabled && !isStylus) {
            return false
        }

        // Detect S-Pen barrel button press
        val buttonState = event.buttonState
        val isButtonPressed = (buttonState and MotionEvent.BUTTON_STYLUS_PRIMARY) != 0
        val effectiveTool = if (isButtonPressed || toolType == MotionEvent.TOOL_TYPE_ERASER) {
            CanvasTool.ERASER
        } else {
            activeTool
        }

        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                isHoveringNow = false
                strokeCounter++
                kinematicCalc.reset()

                val stroke = StrokeData(strokeId = strokeCounter, isHoverStroke = false)
                currentStroke = stroke
                currentStrokeSegments.clear()
                currentPath.reset()

                lastX = event.x
                lastY = event.y
                currentPath.moveTo(lastX, lastY)

                val pt = recordSample(
                    eventTimeMs = event.eventTime,
                    elapsedNanos = SystemClock.elapsedRealtimeNanos(),
                    xPx = event.x,
                    yPx = event.y,
                    pressure = event.pressure,
                    tiltRad = event.getAxisValue(MotionEvent.AXIS_TILT),
                    orientationRad = event.getAxisValue(MotionEvent.AXIS_ORIENTATION),
                    hoverDistance = 0f,
                    touchMajor = event.getAxisValue(MotionEvent.AXIS_TOUCH_MAJOR),
                    touchMinor = event.getAxisValue(MotionEvent.AXIS_TOUCH_MINOR),
                    toolMajor = event.getAxisValue(MotionEvent.AXIS_TOOL_MAJOR),
                    toolMinor = event.getAxisValue(MotionEvent.AXIS_TOOL_MINOR),
                    action = "DOWN",
                    contactState = "ON_SURFACE",
                    strokeId = strokeCounter,
                    buttonState = buttonState
                )

                lastEventTimeMs = event.eventTime
                invalidate()
                return true
            }

            MotionEvent.ACTION_MOVE -> {
                isHoveringNow = false
                val strokeId = strokeCounter

                // CRITICAL: Unpack all intermediate historical samples generated by 240/480Hz digitizer
                val historySize = event.historySize
                for (h in 0 until historySize) {
                    val hTime = event.getHistoricalEventTime(h)
                    val hX = event.getHistoricalX(0, h)
                    val hY = event.getHistoricalY(0, h)
                    val hP = event.getHistoricalPressure(0, h)
                    val hTilt = event.getHistoricalAxisValue(MotionEvent.AXIS_TILT, 0, h)
                    val hOrient = event.getHistoricalAxisValue(MotionEvent.AXIS_ORIENTATION, 0, h)
                    val hTouchMaj = event.getHistoricalAxisValue(MotionEvent.AXIS_TOUCH_MAJOR, 0, h)
                    val hTouchMin = event.getHistoricalAxisValue(MotionEvent.AXIS_TOUCH_MINOR, 0, h)
                    val hToolMaj = event.getHistoricalAxisValue(MotionEvent.AXIS_TOOL_MAJOR, 0, h)
                    val hToolMin = event.getHistoricalAxisValue(MotionEvent.AXIS_TOOL_MINOR, 0, h)

                    recordSample(
                        eventTimeMs = hTime,
                        elapsedNanos = SystemClock.elapsedRealtimeNanos(),
                        xPx = hX,
                        yPx = hY,
                        pressure = hP,
                        tiltRad = hTilt,
                        orientationRad = hOrient,
                        hoverDistance = 0f,
                        touchMajor = hTouchMaj,
                        touchMinor = hTouchMin,
                        toolMajor = hToolMaj,
                        toolMinor = hToolMin,
                        action = "MOVE",
                        contactState = "ON_SURFACE",
                        strokeId = strokeId,
                        buttonState = buttonState
                    )

                    appendPointToPath(hX, hY, hP, effectiveTool)
                }

                // Process the current active sample in the event
                val currentPt = recordSample(
                    eventTimeMs = event.eventTime,
                    elapsedNanos = SystemClock.elapsedRealtimeNanos(),
                    xPx = event.x,
                    yPx = event.y,
                    pressure = event.pressure,
                    tiltRad = event.getAxisValue(MotionEvent.AXIS_TILT),
                    orientationRad = event.getAxisValue(MotionEvent.AXIS_ORIENTATION),
                    hoverDistance = 0f,
                    touchMajor = event.getAxisValue(MotionEvent.AXIS_TOUCH_MAJOR),
                    touchMinor = event.getAxisValue(MotionEvent.AXIS_TOUCH_MINOR),
                    toolMajor = event.getAxisValue(MotionEvent.AXIS_TOOL_MAJOR),
                    toolMinor = event.getAxisValue(MotionEvent.AXIS_TOOL_MINOR),
                    action = "MOVE",
                    contactState = "ON_SURFACE",
                    strokeId = strokeId,
                    buttonState = buttonState
                )

                appendPointToPath(event.x, event.y, event.pressure, effectiveTool)
                lastEventTimeMs = event.eventTime
                invalidate()
                return true
            }

            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                val actionName = if (event.actionMasked == MotionEvent.ACTION_UP) "UP" else "CANCEL"
                val strokeId = strokeCounter

                val upPt = recordSample(
                    eventTimeMs = event.eventTime,
                    elapsedNanos = SystemClock.elapsedRealtimeNanos(),
                    xPx = event.x,
                    yPx = event.y,
                    pressure = 0f,
                    tiltRad = event.getAxisValue(MotionEvent.AXIS_TILT),
                    orientationRad = event.getAxisValue(MotionEvent.AXIS_ORIENTATION),
                    hoverDistance = 0f,
                    touchMajor = event.getAxisValue(MotionEvent.AXIS_TOUCH_MAJOR),
                    touchMinor = event.getAxisValue(MotionEvent.AXIS_TOUCH_MINOR),
                    toolMajor = event.getAxisValue(MotionEvent.AXIS_TOOL_MAJOR),
                    toolMinor = event.getAxisValue(MotionEvent.AXIS_TOOL_MINOR),
                    action = actionName,
                    contactState = "ON_SURFACE",
                    strokeId = strokeId,
                    buttonState = buttonState
                )

                // Finalize active stroke
                currentStroke?.let { stroke ->
                    completedStrokes.add(stroke)
                    totalOnSurfaceTimeMs += stroke.durationMs
                }
                currentStroke = null

                // Commit drawing segments to persistent list
                if (currentStrokeSegments.isNotEmpty()) {
                    strokePaths.add(
                        DrawableStroke(
                            strokeId = strokeId,
                            segments = currentStrokeSegments.toList(),
                            tool = effectiveTool
                        )
                    )
                    currentStrokeSegments.clear()
                }

                lastEventTimeMs = event.eventTime
                invalidate()
                return true
            }
        }

        return super.onTouchEvent(event)
    }

    // =========================================================================
    // GENERIC MOTION: IN-AIR S-PEN HOVER TRAJECTORY TRACKING
    // =========================================================================

    override fun onGenericMotionEvent(event: MotionEvent): Boolean {
        val toolType = event.getToolType(0)
        val isStylus = (toolType == MotionEvent.TOOL_TYPE_STYLUS || toolType == MotionEvent.TOOL_TYPE_ERASER)

        if (!isStylus) return super.onGenericMotionEvent(event)

        when (event.actionMasked) {
            MotionEvent.ACTION_HOVER_ENTER -> {
                isHoveringNow = true
                hoverSegmentCounter--
                hoverX = event.x
                hoverY = event.y

                val stroke = StrokeData(strokeId = hoverSegmentCounter, isHoverStroke = true)
                currentHoverStroke = stroke

                recordHoverPoint(event, "HOVER_ENTER")
                invalidate()
                return true
            }

            MotionEvent.ACTION_HOVER_MOVE -> {
                isHoveringNow = true
                hoverX = event.x
                hoverY = event.y

                // Unpack any historical hover points
                val historySize = event.historySize
                for (h in 0 until historySize) {
                    val hTime = event.getHistoricalEventTime(h)
                    val hX = event.getHistoricalX(0, h)
                    val hY = event.getHistoricalY(0, h)
                    val hDist = event.getHistoricalAxisValue(MotionEvent.AXIS_DISTANCE, 0, h)
                    val hTilt = event.getHistoricalAxisValue(MotionEvent.AXIS_TILT, 0, h)
                    val hOrient = event.getHistoricalAxisValue(MotionEvent.AXIS_ORIENTATION, 0, h)

                    recordSample(
                        eventTimeMs = hTime,
                        elapsedNanos = SystemClock.elapsedRealtimeNanos(),
                        xPx = hX,
                        yPx = hY,
                        pressure = 0f,
                        tiltRad = hTilt,
                        orientationRad = hOrient,
                        hoverDistance = hDist,
                        touchMajor = 0f,
                        touchMinor = 0f,
                        toolMajor = 0f,
                        toolMinor = 0f,
                        action = "HOVER_MOVE",
                        contactState = "IN_AIR",
                        strokeId = hoverSegmentCounter,
                        buttonState = event.buttonState
                    )
                }

                recordHoverPoint(event, "HOVER_MOVE")
                invalidate()
                return true
            }

            MotionEvent.ACTION_HOVER_EXIT -> {
                isHoveringNow = false
                hoverX = -1f
                hoverY = -1f

                recordHoverPoint(event, "HOVER_EXIT")

                currentHoverStroke?.let { stroke ->
                    completedStrokes.add(stroke)
                    totalInAirTimeMs += stroke.durationMs
                }
                currentHoverStroke = null

                invalidate()
                return true
            }
        }

        return super.onGenericMotionEvent(event)
    }

    private fun recordHoverPoint(event: MotionEvent, action: String) {
        val hoverDist = event.getAxisValue(MotionEvent.AXIS_DISTANCE)
        val tiltRad = event.getAxisValue(MotionEvent.AXIS_TILT)
        val orientRad = event.getAxisValue(MotionEvent.AXIS_ORIENTATION)

        recordSample(
            eventTimeMs = event.eventTime,
            elapsedNanos = SystemClock.elapsedRealtimeNanos(),
            xPx = event.x,
            yPx = event.y,
            pressure = 0f,
            tiltRad = tiltRad,
            orientationRad = orientRad,
            hoverDistance = hoverDist,
            touchMajor = 0f,
            touchMinor = 0f,
            toolMajor = 0f,
            toolMinor = 0f,
            action = action,
            contactState = "IN_AIR",
            strokeId = hoverSegmentCounter,
            buttonState = event.buttonState
        )
    }

    // =========================================================================
    // POINT RECORDING & TELEMETRY DISPATCH
    // =========================================================================

    private fun recordSample(
        eventTimeMs: Long,
        elapsedNanos: Long,
        xPx: Float,
        yPx: Float,
        pressure: Float,
        tiltRad: Float,
        orientationRad: Float,
        hoverDistance: Float,
        touchMajor: Float,
        touchMinor: Float,
        toolMajor: Float,
        toolMinor: Float,
        action: String,
        contactState: String,
        strokeId: Int,
        buttonState: Int
    ): KinematicPoint {
        pointCounter++
        val xMm = kinematicCalc.pxToMmx(xPx)
        val yMm = kinematicCalc.pxToMmy(yPx)

        val kinematics = kinematicCalc.computeKinematics(
            xMm = xMm,
            yMm = yMm,
            pressure = pressure,
            orientationRad = orientationRad,
            timeMs = eventTimeMs
        )

        val point = KinematicPoint(
            timestampMs = eventTimeMs,
            elapsedNanos = elapsedNanos,
            strokeId = strokeId,
            pointIndex = pointCounter,
            action = action,
            contactState = contactState,
            xPx = xPx,
            yPx = yPx,
            xMm = xMm,
            yMm = yMm,
            pressure = pressure,
            tiltRad = tiltRad,
            orientationRad = orientationRad,
            hoverDistance = hoverDistance,
            touchMajor = touchMajor,
            touchMinor = touchMinor,
            toolMajor = toolMajor,
            toolMinor = toolMinor,
            buttonState = buttonState,
            velocityMmPerSec = kinematics.velocityMmPerSec,
            accelMmPerSec2 = kinematics.accelMmPerSec2,
            jerkMmPerSec3 = kinematics.jerkMmPerSec3,
            azimuthVelocityRadPerSec = kinematics.azimuthVelocityRadPerSec,
            pressureRatePerSec = kinematics.pressureRatePerSec
        )

        kinematicCalc.updateState(point)
        allPoints.add(point)

        if (contactState == "ON_SURFACE") {
            currentStroke?.points?.add(point)
        } else {
            currentHoverStroke?.points?.add(point)
        }

        // Sampling frequency calculation
        val currentHz = kinematicCalc.recordTimestamp(eventTimeMs)

        // Telemetry callback
        telemetryListener?.onTelemetryUpdate(
            pressure = pressure,
            tiltDeg = Math.toDegrees(tiltRad.toDouble()).toFloat(),
            orientationDeg = Math.toDegrees(orientationRad.toDouble()).toFloat(),
            samplingHz = currentHz,
            isHovering = contactState == "IN_AIR",
            hoverDist = hoverDistance,
            totalPoints = pointCounter,
            strokeCount = strokeCounter,
            velocityMmPerSec = kinematics.velocityMmPerSec
        )

        return point
    }

    private fun appendPointToPath(x: Float, y: Float, pressure: Float, tool: CanvasTool) {
        val midX = (lastX + x) / 2f
        val midY = (lastY + y) / 2f

        // Variable width based on pressure: stroke width modulates with applied force
        val width = if (tool == CanvasTool.ERASER) {
            36f // Eraser thickness
        } else {
            // Pressure modulates from 40% to 160% of base width
            baseStrokeWidthPx * (0.4f + 1.2f * max(0.05f, pressure))
        }

        currentStrokeSegments.add(
            PathSegment(
                startX = lastX,
                startY = lastY,
                endX = x,
                endY = y,
                widthPx = width,
                color = if (tool == CanvasTool.ERASER) Color.WHITE else inkColor
            )
        )

        lastX = x
        lastY = y
    }

    // =========================================================================
    // RENDERING: PAPER LINES, STROKES, AND HOVER RETICLE
    // =========================================================================

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        // 1. Draw Paper Background
        drawPaperBackground(canvas)

        // 2. Draw Committed Strokes
        for (stroke in strokePaths) {
            for (segment in stroke.segments) {
                inkPaint.strokeWidth = segment.widthPx
                inkPaint.color = segment.color
                canvas.drawLine(segment.startX, segment.startY, segment.endX, segment.endY, inkPaint)
            }
        }

        // 3. Draw Active Stroke Segments
        for (segment in currentStrokeSegments) {
            inkPaint.strokeWidth = segment.widthPx
            inkPaint.color = segment.color
            canvas.drawLine(segment.startX, segment.startY, segment.endX, segment.endY, inkPaint)
        }

        // 4. Draw In-Air S-Pen Hover Indicator
        if (isHoveringNow && hoverX >= 0 && hoverY >= 0) {
            canvas.drawCircle(hoverX, hoverY, 12f, hoverCursorPaint)
            canvas.drawCircle(hoverX, hoverY, 3f, hoverDotPaint)
        }
    }

    private fun drawPaperBackground(canvas: Canvas) {
        val width = width.toFloat()
        val height = height.toFloat()

        when (paperStyle) {
            PaperStyle.RULED -> {
                // Left margin line (red)
                val marginX = width * 0.12f
                canvas.drawLine(marginX, 0f, marginX, height, marginLinePaint)

                // Horizontal ruled lines (blue)
                val lineSpacingPx = 72f
                var y = 140f
                while (y < height) {
                    canvas.drawLine(0f, y, width, y, ruledLinePaint)
                    y += lineSpacingPx
                }
            }

            PaperStyle.GRID -> {
                val gridSizePx = 48f
                var x = 0f
                while (x < width) {
                    canvas.drawLine(x, 0f, x, height, gridLinePaint)
                    x += gridSizePx
                }
                var y = 0f
                while (y < height) {
                    canvas.drawLine(0f, y, width, y, gridLinePaint)
                    y += gridSizePx
                }
            }

            PaperStyle.BLANK -> {
                // Pure clean canvas (no background lines)
            }
        }
    }

    // =========================================================================
    // ACTIONS & CONTROLS
    // =========================================================================

    fun undoLastStroke() {
        if (strokePaths.isNotEmpty()) {
            strokePaths.removeAt(strokePaths.size - 1)
            invalidate()
        }
    }

    fun clearCanvas() {
        strokePaths.clear()
        currentStrokeSegments.clear()
        allPoints.clear()
        completedStrokes.clear()
        pointCounter = 0
        strokeCounter = 0
        hoverSegmentCounter = 0
        totalOnSurfaceTimeMs = 0L
        totalInAirTimeMs = 0L
        kinematicCalc.reset()
        invalidate()
    }

    fun renderToBitmap(whiteBackground: Boolean = true): Bitmap {
        val w = max(1, width)
        val h = max(1, height)
        val bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)

        if (whiteBackground) {
            canvas.drawColor(Color.WHITE)
        }

        // Draw strokes on bitmap
        val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeCap = Paint.Cap.ROUND
            strokeJoin = Paint.Join.ROUND
        }

        for (stroke in strokePaths) {
            for (segment in stroke.segments) {
                paint.strokeWidth = segment.widthPx
                paint.color = segment.color
                canvas.drawLine(segment.startX, segment.startY, segment.endX, segment.endY, paint)
            }
        }

        return bitmap
    }

    fun getCurrentSessionPoints(): List<KinematicPoint> = allPoints.toList()
    fun getCurrentStrokes(): List<StrokeData> = completedStrokes.toList()

    fun getSessionSummary(): SessionSummary {
        val now = System.currentTimeMillis()
        val totalDuration = max(1L, now - sessionStartTimeUtc)
        val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US)

        val onSurfacePoints = allPoints.filter { it.contactState == "ON_SURFACE" }
        val meanVel = if (onSurfacePoints.isNotEmpty()) {
            onSurfacePoints.map { it.velocityMmPerSec }.average().toFloat()
        } else 0f

        val meanPress = if (onSurfacePoints.isNotEmpty()) {
            onSurfacePoints.map { it.pressure }.average().toFloat()
        } else 0f

        val inAirRatio = if (totalDuration > 0) {
            min(1f, totalInAirTimeMs.toFloat() / totalDuration.toFloat())
        } else 0f

        val avgHz = if (totalDuration > 0) {
            (allPoints.size * 1000f) / totalDuration.toFloat()
        } else 0f

        return SessionSummary(
            sessionId = sessionId,
            sessionTimestampUtc = sessionStartTimeUtc,
            formattedDate = dateFormat.format(Date(sessionStartTimeUtc)),
            deviceManufacturer = Build.MANUFACTURER,
            deviceModel = Build.MODEL,
            androidVersion = "Android ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})",
            screenWidthPx = width,
            screenHeightPx = height,
            xdpi = displayMetrics.xdpi,
            ydpi = displayMetrics.ydpi,
            totalPointsLogged = allPoints.size,
            totalOnSurfaceStrokes = strokePaths.size,
            totalInAirSegments = abs(hoverSegmentCounter),
            totalSessionDurationMs = totalDuration,
            onSurfaceWritingDurationMs = totalOnSurfaceTimeMs,
            inAirHoverDurationMs = totalInAirTimeMs,
            inAirTimeRatio = inAirRatio,
            averageSamplingRateHz = avgHz,
            meanWritingVelocityMmPerSec = meanVel,
            meanStrokePressure = meanPress,
            paperStyleUsed = paperStyle.displayName
        )
    }

    private data class PathSegment(
        val startX: Float,
        val startY: Float,
        val endX: Float,
        val endY: Float,
        val widthPx: Float,
        val color: Int
    )

    private data class DrawableStroke(
        val strokeId: Int,
        val segments: List<PathSegment>,
        val tool: CanvasTool
    )
}
