package com.example.spennotecollector.ui.screens

import android.app.Activity
import android.widget.Toast
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CenterAlignedTopAppBar
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import com.example.spennotecollector.data.export.SessionExporter
import com.example.spennotecollector.data.model.CanvasTool
import com.example.spennotecollector.data.model.PaperStyle
import com.example.spennotecollector.ui.canvas.SPenDrawingView
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun NoteWorkspaceScreen() {
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()

    var drawingViewRef by remember { mutableStateOf<SPenDrawingView?>(null) }

    // Live S-Pen Telemetry State
    var livePressure by remember { mutableFloatStateOf(0f) }
    var liveTiltDeg by remember { mutableFloatStateOf(0f) }
    var liveOrientationDeg by remember { mutableFloatStateOf(0f) }
    var liveSamplingHz by remember { mutableFloatStateOf(0f) }
    var isHoveringState by remember { mutableStateOf(false) }
    var hoverDistance by remember { mutableFloatStateOf(0f) }
    var totalPointsCount by remember { mutableIntStateOf(0) }
    var totalStrokeCount by remember { mutableIntStateOf(0) }
    var liveVelocity by remember { mutableFloatStateOf(0f) }

    // Canvas UI State
    var activeTool by remember { mutableStateOf(CanvasTool.PEN) }
    var activePaperStyle by remember { mutableStateOf(PaperStyle.RULED) }
    var isTelemetryExpanded by remember { mutableStateOf(true) }
    var isExporting by remember { mutableStateOf(false) }

    Scaffold(
        topBar = {
            CenterAlignedTopAppBar(
                title = {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(
                            text = "S-Pen Dysgraphia Collector",
                            fontWeight = FontWeight.Bold,
                            fontSize = 17.sp
                        )
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            Box(
                                modifier = Modifier
                                    .size(7.dp)
                                    .clip(CircleShape)
                                    .background(if (livePressure > 0f) Color(0xFF4CAF50) else if (isHoveringState) Color(0xFF00BCD4) else Color.Gray)
                            )
                            Text(
                                text = "${"%.0f".format(liveSamplingHz)} Hz | $totalPointsCount pts | $totalStrokeCount strokes",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                fontSize = 11.sp,
                                fontFamily = FontFamily.Monospace
                            )
                        }
                    }
                },
                actions = {
                    // Undo Button
                    IconButton(onClick = { drawingViewRef?.undoLastStroke() }) {
                        Text("↶", fontSize = 20.sp, fontWeight = FontWeight.Bold)
                    }

                    // Clear Canvas Button
                    IconButton(onClick = {
                        drawingViewRef?.clearCanvas()
                        livePressure = 0f
                        isHoveringState = false
                        totalPointsCount = 0
                        totalStrokeCount = 0
                        Toast.makeText(context, "Canvas Cleared", Toast.LENGTH_SHORT).show()
                    }) {
                        Text("↺", fontSize = 20.sp, fontWeight = FontWeight.Bold)
                    }

                    // Export Data Button
                    Button(
                        onClick = {
                            val view = drawingViewRef ?: return@Button
                            val points = view.getCurrentSessionPoints()
                            if (points.isEmpty()) {
                                Toast.makeText(context, "Write something before exporting!", Toast.LENGTH_SHORT).show()
                                return@Button
                            }

                            isExporting = true
                            coroutineScope.launch {
                                try {
                                    val exporter = SessionExporter(context)
                                    val strokes = view.getCurrentStrokes()
                                    val summary = view.getSessionSummary()
                                    val bitmap = view.renderToBitmap(whiteBackground = true)

                                    val exportResult = withContext(Dispatchers.IO) {
                                        exporter.exportSession(points, strokes, summary, bitmap)
                                    }

                                    isExporting = false
                                    Toast.makeText(
                                        context,
                                        "Exported ${exportResult.filesCount} files (${exportResult.totalBytes / 1024} KB)",
                                        Toast.LENGTH_SHORT
                                    ).show()

                                    // Launch Android Sharesheet
                                    val shareIntent = exporter.createShareIntent(exportResult)
                                    context.startActivity(shareIntent)
                                } catch (e: Exception) {
                                    isExporting = false
                                    Toast.makeText(context, "Export error: ${e.message}", Toast.LENGTH_LONG).show()
                                }
                            }
                        },
                        enabled = !isExporting,
                        modifier = Modifier.padding(end = 8.dp),
                        colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
                    ) {
                        Text(if (isExporting) "Exporting..." else "Export ⤓", fontSize = 13.sp)
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.surfaceContainer
                )
            )
        }
    ) { innerPadding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
        ) {
            // Main Drawing Surface (Hardware-accelerated S-Pen Canvas)
            AndroidView(
                modifier = Modifier.fillMaxSize(),
                factory = { ctx ->
                    SPenDrawingView(ctx).apply {
                        telemetryListener = object : SPenDrawingView.TelemetryListener {
                            override fun onTelemetryUpdate(
                                pressure: Float,
                                tiltDeg: Float,
                                orientationDeg: Float,
                                samplingHz: Float,
                                isHovering: Boolean,
                                hoverDist: Float,
                                totalPoints: Int,
                                strokeCount: Int,
                                velocityMmPerSec: Float
                            ) {
                                livePressure = pressure
                                liveTiltDeg = tiltDeg
                                liveOrientationDeg = orientationDeg
                                liveSamplingHz = samplingHz
                                isHoveringState = isHovering
                                hoverDistance = hoverDist
                                totalPointsCount = totalPoints
                                totalStrokeCount = strokeCount
                                liveVelocity = velocityMmPerSec
                            }
                        }
                        drawingViewRef = this
                    }
                },
                update = { view ->
                    view.activeTool = activeTool
                    view.paperStyle = activePaperStyle
                }
            )

            // Top Floating Toolbar (Tool & Paper Style Switchers)
            Surface(
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .padding(top = 10.dp),
                shape = RoundedCornerShape(24.dp),
                shadowElevation = 6.dp,
                color = MaterialTheme.colorScheme.surface.copy(alpha = 0.92f)
            ) {
                Row(
                    modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    // Pen Tool Button
                    ToolChip(
                        label = "✎ Pen",
                        isSelected = activeTool == CanvasTool.PEN,
                        onClick = {
                            activeTool = CanvasTool.PEN
                            drawingViewRef?.activeTool = CanvasTool.PEN
                        }
                    )

                    // Eraser Tool Button
                    ToolChip(
                        label = "⌫ Eraser",
                        isSelected = activeTool == CanvasTool.ERASER,
                        onClick = {
                            activeTool = CanvasTool.ERASER
                            drawingViewRef?.activeTool = CanvasTool.ERASER
                        }
                    )

                    Spacer(modifier = Modifier.width(4.dp))
                    Box(
                        modifier = Modifier
                            .height(20.dp)
                            .width(1.dp)
                            .background(Color.Gray.copy(alpha = 0.4f))
                    )
                    Spacer(modifier = Modifier.width(4.dp))

                    // Paper Style Switcher
                    PaperStyleChip(
                        label = "Ruled ☰",
                        isSelected = activePaperStyle == PaperStyle.RULED,
                        onClick = { activePaperStyle = PaperStyle.RULED }
                    )

                    PaperStyleChip(
                        label = "Grid ▦",
                        isSelected = activePaperStyle == PaperStyle.GRID,
                        onClick = { activePaperStyle = PaperStyle.GRID }
                    )

                    PaperStyleChip(
                        label = "Blank ◻",
                        isSelected = activePaperStyle == PaperStyle.BLANK,
                        onClick = { activePaperStyle = PaperStyle.BLANK }
                    )
                }
            }

            // Bottom S-Pen Live Kinematics Telemetry Card
            Card(
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .fillMaxWidth()
                    .padding(horizontal = 12.dp, vertical = 8.dp),
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceContainerHigh.copy(alpha = 0.94f)
                ),
                elevation = CardDefaults.cardElevation(defaultElevation = 8.dp)
            ) {
                Column(modifier = Modifier.padding(horizontal = 14.dp, vertical = 10.dp)) {
                    // Header Row with Status & Collapse Toggle
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            val statusText = when {
                                livePressure > 0f -> "ON-SURFACE (WRITING)"
                                isHoveringState -> "IN-AIR (HOVERING)"
                                else -> "S-PEN READY"
                            }
                            val statusColor = when {
                                livePressure > 0f -> Color(0xFF4CAF50)
                                isHoveringState -> Color(0xFF00BCD4)
                                else -> Color(0xFF9E9E9E)
                            }

                            Box(
                                modifier = Modifier
                                    .size(9.dp)
                                    .clip(CircleShape)
                                    .background(statusColor)
                            )
                            Text(
                                text = statusText,
                                fontWeight = FontWeight.Bold,
                                fontSize = 12.sp,
                                color = statusColor,
                                letterSpacing = 0.5.sp
                            )
                        }

                        Text(
                            text = if (isTelemetryExpanded) "Hide ▲" else "Telemetry ▼",
                            fontSize = 12.sp,
                            color = MaterialTheme.colorScheme.primary,
                            modifier = Modifier.clickable { isTelemetryExpanded = !isTelemetryExpanded }
                        )
                    }

                    // Expanded Telemetry Metrics
                    AnimatedVisibility(visible = isTelemetryExpanded) {
                        Column(modifier = Modifier.padding(top = 8.dp)) {
                            // Pressure Bar
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.SpaceBetween
                            ) {
                                Text(
                                    text = "Pressure (4096 lvls):",
                                    fontSize = 11.sp,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                                Text(
                                    text = "${"%.3f".format(livePressure)} (${(livePressure * 100).toInt()}%)",
                                    fontSize = 11.sp,
                                    fontFamily = FontFamily.Monospace,
                                    fontWeight = FontWeight.SemiBold
                                )
                            }
                            Spacer(modifier = Modifier.height(3.dp))
                            LinearProgressIndicator(
                                progress = { livePressure.coerceIn(0f, 1f) },
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(6.dp)
                                    .clip(RoundedCornerShape(3.dp)),
                                color = if (livePressure > 0.8f) Color(0xFFE91E63) else MaterialTheme.colorScheme.primary
                            )

                            Spacer(modifier = Modifier.height(8.dp))

                            // Grid of Secondary Metrics
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween
                            ) {
                                TelemetryMetricItem(label = "Tilt Angle", value = "${"%.1f".format(liveTiltDeg)}°")
                                TelemetryMetricItem(label = "Orientation", value = "${"%.1f".format(liveOrientationDeg)}°")
                                TelemetryMetricItem(label = "Velocity", value = "${"%.1f".format(liveVelocity)} mm/s")
                                TelemetryMetricItem(
                                    label = "In-Air Dist",
                                    value = if (isHoveringState) "%.2f".format(hoverDistance) else "0.00"
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun ToolChip(
    label: String,
    isSelected: Boolean,
    onClick: () -> Unit
) {
    Surface(
        modifier = Modifier
            .clip(RoundedCornerShape(12.dp))
            .clickable(onClick = onClick),
        color = if (isSelected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.surfaceVariant,
        contentColor = if (isSelected) MaterialTheme.colorScheme.onPrimary else MaterialTheme.colorScheme.onSurfaceVariant
    ) {
        Text(
            text = label,
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
            fontSize = 12.sp,
            fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Medium
        )
    }
}

@Composable
private fun PaperStyleChip(
    label: String,
    isSelected: Boolean,
    onClick: () -> Unit
) {
    Surface(
        modifier = Modifier
            .clip(RoundedCornerShape(12.dp))
            .clickable(onClick = onClick),
        color = if (isSelected) MaterialTheme.colorScheme.secondaryContainer else Color.Transparent,
        contentColor = if (isSelected) MaterialTheme.colorScheme.onSecondaryContainer else MaterialTheme.colorScheme.onSurfaceVariant
    ) {
        Text(
            text = label,
            modifier = Modifier.padding(horizontal = 8.dp, vertical = 6.dp),
            fontSize = 11.sp,
            fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal
        )
    }
}

@Composable
private fun TelemetryMetricItem(label: String, value: String) {
    Column {
        Text(text = label, fontSize = 10.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text(
            text = value,
            fontSize = 12.sp,
            fontFamily = FontFamily.Monospace,
            fontWeight = FontWeight.SemiBold,
            color = MaterialTheme.colorScheme.onSurface
        )
    }
}
