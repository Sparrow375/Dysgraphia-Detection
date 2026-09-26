package com.example.spennotecollector.data.model

/**
 * Visual background style for the note-taking canvas.
 */
enum class PaperStyle(val displayName: String) {
    RULED("Ruled Notebook"),
    GRID("Math Grid"),
    BLANK("Blank Sheet")
}

/**
 * Active tool on the canvas.
 */
enum class CanvasTool(val displayName: String) {
    PEN("Pen"),
    ERASER("Eraser")
}
