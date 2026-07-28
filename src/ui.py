"""
ui.py - OpenCV-based UI rendering helpers for the gesture recognition system
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple

from src.config import GESTURE_CLASSES, UNKNOWN_LABEL


# ─── Colour palette ──────────────────────────────────────────────────────────
PALETTE = {
    "bg_dark":    (15,  15,  20),
    "accent":     (0,   220, 130),
    "warn":       (0,   140, 255),
    "danger":     (50,  50,  220),
    "text_light": (240, 240, 240),
    "text_dim":   (140, 140, 140),
    "skeleton":   (0,   200, 100),
    "panel":      (25,  25,  35),
}

FONT        = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL  = cv2.FONT_HERSHEY_SIMPLEX


def alpha_blend(frame, overlay, alpha: float = 0.6):
    """Blend semi-transparent overlay onto frame."""
    return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)


def draw_rounded_rect(img, pt1, pt2, color, radius=12, thickness=-1):
    """Draw a rounded rectangle on img."""
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)


def draw_confidence_bar(
    frame, x, y, width, height,
    confidence: float,
    label: str,
):
    """Animated confidence progress bar."""
    # Background
    draw_rounded_rect(frame, (x, y), (x + width, y + height), (40, 40, 50), radius=6)

    # Fill
    color = PALETTE["accent"] if confidence >= 0.8 else (
            PALETTE["warn"]   if confidence >= 0.6 else PALETTE["danger"])
    fill_w = int(width * confidence)
    if fill_w > 10:
        draw_rounded_rect(frame, (x, y), (x + fill_w, y + height), color, radius=6)

    # Text
    pct_txt = f"{confidence * 100:.1f}%"
    cv2.putText(frame, pct_txt,
                (x + width + 8, y + height - 4),
                FONT_SMALL, 0.55, PALETTE["text_light"], 1, cv2.LINE_AA)


def draw_gesture_panel(
    frame,
    gesture:    str,
    confidence: float,
    mode:       str = "LIVE",
    fps:        float = 0.0,
    top_k:      Optional[List[Tuple[str, float]]] = None,
):
    """
    Draw the right-side info panel.
    """
    h, w = frame.shape[:2]
    panel_x = w - 280

    # Panel background
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x - 10, 0), (w, h), PALETTE["panel"], -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Separator
    cv2.line(frame, (panel_x - 10, 0), (panel_x - 10, h), PALETTE["accent"], 2)

    # Mode badge
    badge_color = PALETTE["accent"] if mode == "LIVE" else PALETTE["warn"]
    draw_rounded_rect(frame, (panel_x, 15), (panel_x + 100, 40), badge_color, radius=8)
    cv2.putText(frame, mode, (panel_x + 10, 34),
                FONT_SMALL, 0.6, (10, 10, 10), 2, cv2.LINE_AA)

    # FPS
    cv2.putText(frame, f"FPS {fps:.0f}", (panel_x + 115, 34),
                FONT_SMALL, 0.55, PALETTE["text_dim"], 1, cv2.LINE_AA)

    # Gesture label
    is_unknown = gesture == UNKNOWN_LABEL
    label_color = PALETTE["text_dim"] if is_unknown else PALETTE["accent"]
    cv2.putText(frame, "GESTURE", (panel_x, 80),
                FONT_SMALL, 0.5, PALETTE["text_dim"], 1, cv2.LINE_AA)

    # Large gesture text
    gtext = gesture if not is_unknown else "Unknown"
    scale = 0.85 if len(gtext) <= 8 else 0.65
    cv2.putText(frame, gtext, (panel_x, 115),
                FONT, scale, label_color, 2, cv2.LINE_AA)

    # Confidence bar
    cv2.putText(frame, "CONFIDENCE", (panel_x, 145),
                FONT_SMALL, 0.5, PALETTE["text_dim"], 1, cv2.LINE_AA)
    draw_confidence_bar(frame, panel_x, 155, 180, 22, confidence, gesture)

    # Top-K predictions
    if top_k:
        cv2.putText(frame, "TOP PREDICTIONS", (panel_x, 220),
                    FONT_SMALL, 0.5, PALETTE["text_dim"], 1, cv2.LINE_AA)
        for i, (lbl, conf) in enumerate(top_k[:3]):
            y_off = 240 + i * 38
            bar_w = int(180 * conf)
            cv2.rectangle(frame, (panel_x, y_off),
                          (panel_x + bar_w, y_off + 16), (50, 60, 70), -1)
            cv2.putText(frame, f"{lbl}  {conf*100:.0f}%",
                        (panel_x + 4, y_off + 13),
                        FONT_SMALL, 0.5, PALETTE["text_light"], 1, cv2.LINE_AA)

    return frame


def draw_recording_indicator(frame, gesture: str, count: int, target: int):
    """Overlay for data collection mode."""
    h, w = frame.shape[:2]
    # Red pulsing dot
    cv2.circle(frame, (30, 30), 12, (0, 0, 220), -1)
    cv2.circle(frame, (30, 30), 12, (255, 255, 255), 2)
    cv2.putText(frame, "REC", (50, 38),
                FONT_SMALL, 0.8, (0, 0, 220), 2, cv2.LINE_AA)

    # Gesture name
    cv2.putText(frame, f"Gesture: {gesture}",
                (10, 70), FONT_SMALL, 0.8, PALETTE["accent"], 2, cv2.LINE_AA)

    # Progress
    pct = min(count / target, 1.0)
    bar_len = w - 20
    cv2.rectangle(frame, (10, h - 30), (10 + bar_len, h - 10), (50, 50, 50), -1)
    cv2.rectangle(frame, (10, h - 30), (10 + int(bar_len * pct), h - 10),
                  PALETTE["accent"], -1)
    cv2.putText(frame, f"{count}/{target}",
                (10, h - 35), FONT_SMALL, 0.6, PALETTE["text_light"], 1, cv2.LINE_AA)

    return frame


def draw_help_overlay(frame):
    """Draw keyboard shortcut help."""
    h, w = frame.shape[:2]
    lines = [
        ("L", "Toggle live inference"),
        ("R", "Record dataset"),
        ("T", "Train model"),
        ("S", "Toggle skeleton"),
        ("M", "Cycle model mode"),
        ("Q", "Quit"),
    ]
    y_start = h - len(lines) * 28 - 10
    for i, (key, desc) in enumerate(lines):
        y = y_start + i * 26
        cv2.putText(frame, f"[{key}]", (10, y),
                    FONT_SMALL, 0.55, PALETTE["accent"], 1, cv2.LINE_AA)
        cv2.putText(frame, desc, (50, y),
                    FONT_SMALL, 0.55, PALETTE["text_dim"], 1, cv2.LINE_AA)
    return frame


def draw_status_bar(frame, message: str, ok: bool = True):
    h, w = frame.shape[:2]
    color = PALETTE["accent"] if ok else PALETTE["danger"]
    cv2.rectangle(frame, (0, h - 28), (w - 290, h), (20, 20, 25), -1)
    cv2.putText(frame, message, (10, h - 8),
                FONT_SMALL, 0.55, color, 1, cv2.LINE_AA)
    return frame
