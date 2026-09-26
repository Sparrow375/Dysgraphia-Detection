package com.example.spennotecollector.data.export

import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Build
import androidx.core.content.FileProvider
import com.example.spennotecollector.data.model.KinematicPoint
import com.example.spennotecollector.data.model.SessionSummary
import com.example.spennotecollector.data.model.StrokeData
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

/**
 * Handles serializing and packaging note handwriting sessions into:
 * - timeseries_raw.csv
 * - timeseries_raw.jsonl
 * - strokes_summary.json
 * - session_metadata.json
 * - note_render.png
 * - Bundled ZIP archive with Android Sharesheet integration
 */
class SessionExporter(private val context: Context) {

    data class ExportResult(
        val zipFile: File,
        val zipUri: Uri,
        val filesCount: Int,
        val totalBytes: Long
    )

    fun exportSession(
        points: List<KinematicPoint>,
        strokes: List<StrokeData>,
        summary: SessionSummary,
        renderedBitmap: Bitmap?
    ): ExportResult {
        val dateFormat = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US)
        val timestampStr = dateFormat.format(Date(summary.sessionTimestampUtc))
        val sessionDirName = "spen_session_${summary.sessionId}_$timestampStr"

        // Use cache/exports or files/exports directory
        val exportBaseDir = File(context.cacheDir, "exports")
        if (!exportBaseDir.exists()) {
            exportBaseDir.mkdirs()
        }

        val sessionDir = File(exportBaseDir, sessionDirName)
        if (!sessionDir.exists()) {
            sessionDir.mkdirs()
        }

        // 1. Export timeseries_raw.csv
        val csvFile = File(sessionDir, "timeseries_raw.csv")
        csvFile.bufferedWriter().use { writer ->
            writer.write(KinematicPoint.CSV_HEADER)
            writer.newLine()
            for (p in points) {
                writer.write(p.toCsvRow())
                writer.newLine()
            }
        }

        // 2. Export timeseries_raw.jsonl
        val jsonlFile = File(sessionDir, "timeseries_raw.jsonl")
        jsonlFile.bufferedWriter().use { writer ->
            for (p in points) {
                val obj = JSONObject().apply {
                    put("t_ms", p.timestampMs)
                    put("t_nanos", p.elapsedNanos)
                    put("stroke_id", p.strokeId)
                    put("pt_idx", p.pointIndex)
                    put("action", p.action)
                    put("state", p.contactState)
                    put("x_px", p.xPx.toDouble())
                    put("y_px", p.yPx.toDouble())
                    put("x_mm", p.xMm.toDouble())
                    put("y_mm", p.yMm.toDouble())
                    put("pressure", p.pressure.toDouble())
                    put("tilt_rad", p.tiltRad.toDouble())
                    put("orient_rad", p.orientationRad.toDouble())
                    put("hover_dist", p.hoverDistance.toDouble())
                    put("v_mms", p.velocityMmPerSec.toDouble())
                    put("a_mms2", p.accelMmPerSec2.toDouble())
                    put("j_mms3", p.jerkMmPerSec3.toDouble())
                    put("btn", p.buttonState)
                }
                writer.write(obj.toString())
                writer.newLine()
            }
        }

        // 3. Export strokes_summary.json
        val strokesFile = File(sessionDir, "strokes_summary.json")
        strokesFile.bufferedWriter().use { writer ->
            val root = JSONObject()
            root.put("session_id", summary.sessionId)
            root.put("total_strokes", strokes.size)

            val strokesArr = JSONArray()
            for (s in strokes) {
                val sObj = JSONObject().apply {
                    put("stroke_id", s.strokeId)
                    put("is_hover", s.isHoverStroke)
                    put("points_count", s.pointsCount)
                    put("start_time_ms", s.startTimeMs)
                    put("end_time_ms", s.endTimeMs)
                    put("duration_ms", s.durationMs)
                    put("path_length_mm", s.pathLengthMm.toDouble())
                    put("mean_pressure", s.meanPressure.toDouble())
                    put("max_pressure", s.maxPressure.toDouble())
                    put("pressure_variance", s.pressureVariance.toDouble())
                    put("mean_velocity_mms", s.meanVelocityMmPerSec.toDouble())
                    put("max_velocity_mms", s.maxVelocityMmPerSec.toDouble())
                    put("mean_jerk_mms3", s.meanJerkMmPerSec3.toDouble())
                    put("hesitation_count", s.hesitationCount)
                }
                strokesArr.put(sObj)
            }
            root.put("strokes", strokesArr)
            writer.write(root.toString(2))
        }

        // 4. Export session_metadata.json
        val metadataFile = File(sessionDir, "session_metadata.json")
        metadataFile.bufferedWriter().use { writer ->
            val meta = JSONObject().apply {
                put("session_id", summary.sessionId)
                put("timestamp_utc", summary.sessionTimestampUtc)
                put("date_formatted", summary.formattedDate)
                put("device_manufacturer", summary.deviceManufacturer)
                put("device_model", summary.deviceModel)
                put("android_release", summary.androidVersion)
                put("screen_width_px", summary.screenWidthPx)
                put("screen_height_px", summary.screenHeightPx)
                put("xdpi", summary.xdpi.toDouble())
                put("ydpi", summary.ydpi.toDouble())
                put("total_points", summary.totalPointsLogged)
                put("on_surface_strokes", summary.totalOnSurfaceStrokes)
                put("in_air_segments", summary.totalInAirSegments)
                put("total_duration_ms", summary.totalSessionDurationMs)
                put("on_surface_duration_ms", summary.onSurfaceWritingDurationMs)
                put("in_air_duration_ms", summary.inAirHoverDurationMs)
                put("in_air_ratio", summary.inAirTimeRatio.toDouble())
                put("avg_sampling_hz", summary.averageSamplingRateHz.toDouble())
                put("mean_velocity_mms", summary.meanWritingVelocityMmPerSec.toDouble())
                put("mean_pressure", summary.meanStrokePressure.toDouble())
                put("paper_style", summary.paperStyleUsed)
            }
            writer.write(meta.toString(2))
        }

        // 5. Export note_render.png
        if (renderedBitmap != null) {
            val pngFile = File(sessionDir, "note_render.png")
            FileOutputStream(pngFile).use { out ->
                renderedBitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
            }
        }

        // 6. Compress sessionDir into a standalone ZIP file
        val zipFile = File(exportBaseDir, "${sessionDirName}.zip")
        val generatedFiles = sessionDir.listFiles() ?: emptyArray()

        ZipOutputStream(BufferedOutputStream(FileOutputStream(zipFile))).use { zos ->
            for (file in generatedFiles) {
                if (file.isFile) {
                    val entry = ZipEntry(file.name)
                    zos.putNextEntry(entry)
                    file.inputStream().use { input ->
                        input.copyTo(zos)
                    }
                    zos.closeEntry()
                }
            }
        }

        // Generate FileProvider Uri for sharing
        val zipUri = FileProvider.getUriForFile(
            context,
            "${context.packageName}.fileprovider",
            zipFile
        )

        return ExportResult(
            zipFile = zipFile,
            zipUri = zipUri,
            filesCount = generatedFiles.size + 1,
            totalBytes = zipFile.length()
        )
    }

    /**
     * Builds an Android Sharesheet Intent to send or save the exported ZIP.
     */
    fun createShareIntent(exportResult: ExportResult): Intent {
        val shareIntent = Intent(Intent.ACTION_SEND).apply {
            type = "application/zip"
            putExtra(Intent.EXTRA_STREAM, exportResult.zipUri)
            putExtra(Intent.EXTRA_SUBJECT, "Dysgraphia Data Collection: ${exportResult.zipFile.name}")
            putExtra(
                Intent.EXTRA_TEXT,
                "S-Pen Kinematic Dataset Export (${exportResult.zipFile.name}, size: ${exportResult.totalBytes / 1024} KB)"
            )
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        return Intent.createChooser(shareIntent, "Export S-Pen Dysgraphia Dataset")
    }
}
