# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception:
    mp = None


def resolve_mediapipe_hands():
    if mp is None:
        return None
    try:
        if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands'):
            return mp.solutions.hands
    except Exception:
        pass
    try:
        from mediapipe.python.solutions import hands as mp_hands
        return mp_hands
    except Exception:
        pass
    return None


@dataclass
class FrameFeatures:
    frame_idx: int
    t_sec: float
    quad: Optional[np.ndarray]
    page_found: bool
    page_area_ratio: float
    fill_ratio: float
    border_contact_score: float
    stability_score: float
    blur_score: float
    text_score: float
    hand_penalty: float
    hand_text_overlap_penalty: float
    edge_foreground_penalty: float
    bottom_hand_penalty: float
    turn_penalty: float
    edge_motion_penalty: float
    gray: Optional[np.ndarray]
    roi_gray: Optional[np.ndarray]
    roi_dhash: Optional[int]
    warped_bgr: Optional[np.ndarray]
    raw_score: float = -1e9
    norm_score: float = -1e9
    peak_score: float = -1e9
    deskew_angle: float = 0.0


@dataclass
class Cluster:
    members: List[FrameFeatures] = field(default_factory=list)


class HandMasker:
    def __init__(self, enabled=True, det_conf=0.45, track_conf=0.45):
        self.mp_hands = resolve_mediapipe_hands()
        self.enabled = enabled and (self.mp_hands is not None)
        self._hands = None
        if self.enabled:
            self._hands = self.mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                model_complexity=0,
                min_detection_confidence=det_conf,
                min_tracking_confidence=track_conf,
            )

    def close(self):
        if self._hands is not None:
            self._hands.close()

    def build_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        if not self.enabled:
            return mask
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        res = self._hands.process(rgb)
        if not getattr(res, 'multi_hand_landmarks', None):
            return mask
        for lmks in res.multi_hand_landmarks:
            pts = []
            for lm in lmks.landmark:
                x = int(np.clip(lm.x * w, 0, w - 1))
                y = int(np.clip(lm.y * h, 0, h - 1))
                pts.append([x, y])
            pts = np.asarray(pts, dtype=np.int32)
            if len(pts) >= 3:
                hull = cv2.convexHull(pts)
                cv2.fillConvexPoly(mask, hull, 255)
        k = max(5, int(min(h, w) * 0.02) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel, iterations=1)
        return mask


def order_quad(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(d)]
    ordered[3] = pts[np.argmax(d)]
    return ordered


def expand_quad(quad: np.ndarray, factor: float = 0.015) -> np.ndarray:
    c = quad.mean(axis=0)
    return c + (quad - c) * (1.0 + factor)


def four_point_warp(image: np.ndarray, quad: np.ndarray, long_side: int = 1800) -> np.ndarray:
    rect = order_quad(quad)
    tl, tr, br, bl = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_w = int(max(width_a, width_b))
    max_h = int(max(height_a, height_b))
    if max_w < 10 or max_h < 10:
        raise ValueError('invalid warp size')
    if max_h >= max_w:
        out_h = long_side
        out_w = max(1, int(long_side * max_w / max_h))
    else:
        out_w = long_side
        out_h = max(1, int(long_side * max_h / max_w))
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, m, (out_w, out_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def resize_long_side(image: np.ndarray, long_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) == long_side:
        return image
    scale = float(long_side) / float(max(h, w))
    return cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_CUBIC)


def trim_uniform_borders(image_bgr: np.ndarray, margin: int = 8) -> np.ndarray:
    """Trim non-paper borders after perspective warp.

    The first quad sometimes includes the desk, opposite page, or a finger near
    the frame edge. After warping, the target page is usually the largest bright
    low-saturation rectangle. This pass removes obvious external borders while
    keeping the page natural (it does not binarize or change content).
    """
    h, w = image_bgr.shape[:2]
    if h < 80 or w < 80:
        return image_bgr

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    l = lab[:, :, 0]

    # Paper is bright and relatively low saturation. Use adaptive thresholds so
    # the same code works for white/yellow paper and different camera exposure.
    bright_thr = int(max(132, np.percentile(v, 42)))
    light_thr = int(max(132, np.percentile(l, 42)))
    sat_thr = int(min(118, max(42, np.percentile(s, 70))))
    paper = cv2.bitwise_and(
        cv2.bitwise_or(cv2.inRange(v, bright_thr, 255), cv2.inRange(l, light_thr, 255)),
        cv2.inRange(s, 0, sat_thr),
    )

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, w // 70), max(9, h // 70)))
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, k, iterations=2)
    paper = cv2.morphologyEx(paper, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return image_bgr

    frame_area = float(h * w)
    candidates = []
    for cnt in sorted(cnts, key=cv2.contourArea, reverse=True)[:8]:
        area = cv2.contourArea(cnt)
        if area < frame_area * 0.18:
            continue
        x, y, ww, hh = cv2.boundingRect(cnt)
        bbox_area = float(ww * hh)
        if bbox_area < frame_area * 0.22:
            continue
        fill = area / max(1.0, bbox_area)
        center = np.array([x + ww / 2.0, y + hh / 2.0])
        center_dist = float(np.linalg.norm(center - np.array([w / 2.0, h / 2.0])) / np.linalg.norm([w / 2.0, h / 2.0]))
        # Prefer a large central paper component. Penalize giant components that
        # are practically the whole image because cropping them changes nothing.
        whole_penalty = 0.45 if (ww > w * 0.97 and hh > h * 0.97) else 0.0
        score = 2.0 * (bbox_area / frame_area) + 0.9 * fill - 0.8 * center_dist - whole_penalty
        candidates.append((score, x, y, ww, hh, area))

    if not candidates:
        return image_bgr

    _, x, y, ww, hh, area = max(candidates, key=lambda t: t[0])
    # Do not over-crop title pages or already clean pages. Require the crop to
    # remove a meaningful border but keep a plausible page aspect.
    remove_left = x
    remove_top = y
    remove_right = w - (x + ww)
    remove_bottom = h - (y + hh)
    removed = (remove_left + remove_right) / max(1, w) + (remove_top + remove_bottom) / max(1, h)
    aspect = hh / max(1, ww)
    if removed < 0.035 or not (1.05 <= aspect <= 2.25):
        return image_bgr

    pad = max(margin, int(min(w, h) * 0.012))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w, x + ww + pad)
    y1 = min(h, y + hh + pad)
    crop = image_bgr[y0:y1, x0:x1]
    if crop.shape[0] < h * 0.55 or crop.shape[1] < w * 0.55:
        return image_bgr
    return resize_long_side(crop, max(h, w))


def crop_book_edge_artifacts(image_bgr: np.ndarray) -> np.ndarray:
    """Crop common artifacts left after page warp: spine, desk, opposite page.

    This is a projection-based safety pass. It is conservative and only crops
    outer strips when there is strong evidence that the strip is not the target
    page: a dark vertical book spine near the left edge, saturated desk at the
    right/top, or a dark bottom strip from fingers/table.
    """
    h, w = image_bgr.shape[:2]
    if h < 120 or w < 120:
        return image_bgr

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    x0, y0, x1, y1 = 0, 0, w, h

    # Left spine/opposite-page crop. Look for a narrow vertical dark/edge-heavy
    # seam in the left quarter; crop just to the right of it.
    left_w = int(w * 0.28)
    if left_w > 25:
        roi_g = gray[:, :left_w]
        roi_s = sat[:, :left_w]
        dark_ratio = np.mean(roi_g < 118, axis=0)
        sat_ratio = np.mean(roi_s > 42, axis=0)
        gx = cv2.Sobel(roi_g, cv2.CV_32F, 1, 0, ksize=3)
        edge_strength = np.mean(np.abs(gx), axis=0) / 255.0
        score = 0.55 * dark_ratio + 0.25 * sat_ratio + 0.20 * edge_strength
        smooth = cv2.GaussianBlur(score.reshape(1, -1).astype(np.float32), (1, 17), 0).reshape(-1)
        seam_x = int(np.argmax(smooth))
        # Avoid cropping clean title pages: require a meaningful seam and enough
        # non-paper evidence near the left side.
        if 8 <= seam_x <= left_w - 8 and smooth[seam_x] > 0.23:
            candidate = min(left_w, seam_x + max(8, int(w * 0.018)))
            # Keep only if the area to remove is visibly less paper-like than the
            # area after the seam.
            before = gray[:, :candidate]
            after = gray[:, candidate:min(w, candidate + int(w * 0.18))]
            if before.size and after.size:
                before_paper = float(np.mean((before > 170)))
                after_paper = float(np.mean((after > 170)))
                if before_paper < after_paper + 0.18:
                    x0 = max(x0, candidate)

    # Right saturated desk/background crop.
    right_start = int(w * 0.72)
    if right_start < w - 20:
        cols = np.arange(right_start, w)
        nonpaper = np.mean((sat[:, right_start:] > 48) | (val[:, right_start:] < 115), axis=0)
        # Find the first sustained non-paper run from the right side.
        run = 0
        cut = w
        for i in range(len(nonpaper) - 1, -1, -1):
            if nonpaper[i] > 0.40:
                run += 1
                if run >= max(8, int(w * 0.018)):
                    cut = right_start + i
            elif run > 0:
                break
        if cut < w and w - cut > w * 0.035:
            x1 = min(x1, max(x0 + int(w * 0.55), cut))

    # Top desk/shadow crop.
    top_h = int(h * 0.18)
    if top_h > 20:
        row_nonpaper = np.mean((sat[:top_h, :] > 54) | (val[:top_h, :] < 105), axis=1)
        run = 0
        cut = 0
        for i in range(top_h):
            if row_nonpaper[i] > 0.34:
                run += 1
                if run >= max(6, int(h * 0.012)):
                    cut = i
            elif run > 0:
                break
        if cut > h * 0.025:
            y0 = max(y0, min(cut + 3, int(h * 0.14)))

    # Bottom finger/table strip crop; keep conservative because page numbers live
    # near the bottom.
    bottom_start = int(h * 0.82)
    if bottom_start < h - 20:
        row_nonpaper = np.mean((sat[bottom_start:, :] > 58) | (val[bottom_start:, :] < 105), axis=1)
        run = 0
        cut = h
        for i in range(len(row_nonpaper) - 1, -1, -1):
            if row_nonpaper[i] > 0.42:
                run += 1
                if run >= max(8, int(h * 0.012)):
                    cut = bottom_start + i
            elif run > 0:
                break
        if cut < h and h - cut > h * 0.035:
            y1 = min(y1, max(y0 + int(h * 0.68), cut))

    if x0 == 0 and y0 == 0 and x1 == w and y1 == h:
        return image_bgr
    if x1 - x0 < w * 0.55 or y1 - y0 < h * 0.60:
        return image_bgr
    crop = image_bgr[y0:y1, x0:x1]
    return resize_long_side(crop, max(h, w))


def trim_bottom_dark_strip(image_bgr: np.ndarray, max_frac: float = 0.05) -> Tuple[np.ndarray, int]:
    """Conservatively crop a dark/saturated strip at the very bottom of a warped page.

    Targets the residual book-edge / desk strip that survives the perspective warp
    on single-page shots (e.g. v12.7 page_002). Only crops rows that are clearly
    non-paper compared to the page body, capped at max_frac of the height so it
    can never eat a real page number or marginalia.
    """
    h, w = image_bgr.shape[:2]
    if h < 200 or w < 200:
        return image_bgr, 0

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    val = hsv[:, :, 2]
    sat = hsv[:, :, 1]

    body_y0 = int(h * 0.30)
    body_y1 = int(h * 0.80)
    body_val = float(np.median(val[body_y0:body_y1]))
    body_dark_thr = max(60.0, body_val * 0.55)
    body_sat_thr = 70.0

    row_dark = np.mean(val < body_dark_thr, axis=1)
    row_sat = np.mean(sat > body_sat_thr, axis=1)
    row_bad = np.maximum(row_dark, row_sat * 0.8)

    max_band = max(2, int(h * max_frac))
    scan_start = h - max_band - 4
    cut = h
    run = 0
    for r in range(h - 1, scan_start, -1):
        if row_bad[r] > 0.55:
            run += 1
            cut = r
        elif row_bad[r] > 0.30 and run > 0:
            run += 1
            cut = r
        else:
            break

    band = h - cut
    if band < 4:
        return image_bgr, 0
    if band > max_band:
        cut = h - max_band
        band = max_band

    pad = max(2, int(h * 0.004))
    cut = max(int(h * (1.0 - max_frac)), cut - pad)
    band = h - cut
    if band < 4:
        return image_bgr, 0

    # Verify the strip we are about to remove is meaningfully darker than the
    # page body — otherwise this is just paper and trimming would lose content.
    removed = gray[cut:h, :]
    body = gray[body_y0:body_y1, :]
    if removed.size == 0 or body.size == 0:
        return image_bgr, 0
    if float(removed.mean()) > float(body.mean()) - 25.0:
        return image_bgr, 0

    cropped = image_bgr[:cut, :, :]
    return cropped, band


def refine_page_after_warp(image_bgr: np.ndarray, args) -> np.ndarray:
    if getattr(args, 'no_refine_crop', False):
        return image_bgr
    refined = trim_uniform_borders(image_bgr)
    return refined


def apply_final_bottom_trim(image_bgr: np.ndarray, args) -> Tuple[np.ndarray, int]:
    """Run the V12.8 bottom dark-strip cleanup as a final-output-only step.

    Kept out of refine_page_after_warp so it cannot perturb candidate scoring
    (blur/text/fg) and therefore cannot change winner selection. Returns
    (cleaned_image, bottom_band_px_removed).
    """
    if getattr(args, 'no_bottom_trim', False):
        return image_bgr, 0
    return trim_bottom_dark_strip(
        image_bgr, max_frac=getattr(args, 'bottom_trim_max_frac', 0.05)
    )


def variance_of_laplacian(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_dhash(gray: np.ndarray, hash_size: int = 16) -> int:
    small = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = 0
    for b in diff.flatten():
        bits = (bits << 1) | int(bool(b))
    return bits


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def roi_for_similarity(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    y0 = int(h * 0.08)
    y1 = int(h * 0.90)
    x0 = int(w * 0.08)
    x1 = int(w * 0.92)
    roi = gray[y0:y1, x0:x1]
    return cv2.resize(roi, (256, 256), interpolation=cv2.INTER_AREA)


def similarity_score(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    a = gray_a.astype(np.float32)
    b = gray_b.astype(np.float32)
    a = (a - a.mean()) / (a.std() + 1e-6)
    b = (b - b.mean()) / (b.std() + 1e-6)
    corr = float(np.mean(a * b))
    mse = float(np.mean((gray_a.astype(np.float32) - gray_b.astype(np.float32)) ** 2))
    mse_term = max(0.0, 1.0 - mse / (255.0 * 255.0))
    return 0.85 * corr + 0.15 * mse_term


def count_text_density(gray: np.ndarray) -> float:
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    bw = cv2.medianBlur(bw, 3)
    return float(np.count_nonzero(bw)) / float(bw.size)


def edge_foreground_penalty(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    edge = np.zeros_like(gray, dtype=np.uint8)
    ex = int(w * 0.18)
    ey = int(h * 0.18)
    edge[:, :ex] = 255
    edge[:, w - ex:] = 255
    edge[:ey, :] = 255
    edge[h - ey:, :] = 255
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    gradx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    grady = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gradx, grady)
    strong = (mag > np.percentile(mag, 82)).astype(np.uint8) * 255
    _, dark = cv2.threshold(blur, int(np.percentile(blur, 22)), 255, cv2.THRESH_BINARY_INV)
    fg = cv2.bitwise_or(strong, dark)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    fg_edge = cv2.bitwise_and(fg, edge)
    return min(1.0, float(np.count_nonzero(fg_edge)) / float(np.count_nonzero(edge) + 1))


def skin_like_mask(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

    # Two complementary skin rules:
    # 1) HSV+YCrCb is good for normal light.
    # 2) LAB catches pale fingers under warm/yellow desk light.
    m1 = cv2.inRange(hsv, (0, 12, 45), (35, 235, 255))
    m2 = cv2.inRange(ycrcb, (0, 128, 72), (255, 185, 145))
    m3 = cv2.inRange(lab, (35, 126, 122), (255, 158, 158))
    mask = cv2.bitwise_or(cv2.bitwise_and(m1, m2), cv2.bitwise_and(m1, m3))
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 32, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
    return mask


def keep_border_connected(mask: np.ndarray, border_px: int) -> np.ndarray:
    """Keep only components that touch image borders.

    Fingers usually enter from a page/image edge. This rule prevents the
    skin-color fallback from accidentally inpainting beige paper, shadows, or
    illustrations in the middle of the page.
    """
    if np.count_nonzero(mask) == 0:
        return mask
    h, w = mask.shape
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    for i in range(1, num):
        x, y, ww, hh, area = stats[i]
        if area < max(40, int(h * w * 0.00025)):
            continue
        touches = (
            x <= border_px or y <= border_px or
            x + ww >= w - border_px or y + hh >= h - border_px
        )
        if touches:
            keep[labels == i] = 255
    return keep


def build_hand_cleanup_mask(image_bgr: np.ndarray, hand_masker: HandMasker, text_protect: bool = True) -> np.ndarray:
    """Build a conservative but useful hand/finger mask for final inpainting."""
    h, w = image_bgr.shape[:2]
    mp_mask = hand_masker.build_mask(image_bgr)
    skin = skin_like_mask(image_bgr)

    # Only trust the color fallback where fingers are realistic: edges and
    # bottom part of the page. MediaPipe landmarks, if present, are trusted
    # everywhere.
    zone = np.zeros((h, w), dtype=np.uint8)
    edge_x = int(w * 0.18)
    edge_y = int(h * 0.12)
    zone[:, :edge_x] = 255
    zone[:, w - edge_x:] = 255
    zone[:edge_y, :] = 255
    zone[int(h * 0.68):, :] = 255
    skin = cv2.bitwise_and(skin, zone)
    skin = keep_border_connected(skin, max(8, int(min(h, w) * 0.025)))

    mask = cv2.bitwise_or(mp_mask, skin)

    if text_protect:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        text_bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
        text_bw = cv2.morphologyEx(text_bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        protected = cv2.dilate(text_bw, np.ones((7, 7), np.uint8), iterations=1)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(protected))

    k = max(5, int(min(h, w) * 0.018) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def hand_mask_is_too_bright(image_bgr: np.ndarray, mask: np.ndarray) -> bool:
    """Reject masks whose covered pixels are roughly as bright as the page body.

    Real hands/fingers are darker than the paper they cover. A mask whose pixels
    are paper-bright is almost certainly a back-of-page bleed-through false
    positive; inpainting it produces gray blotches.
    """
    if np.count_nonzero(mask) == 0:
        return False
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    masked_vals = gray[mask > 0]
    if masked_vals.size == 0:
        return False
    med = float(np.median(masked_vals))
    body = gray[int(h * 0.30):int(h * 0.70), int(w * 0.20):int(w * 0.80)]
    body_med = float(np.median(body)) if body.size else 200.0
    return med >= body_med - 18.0


def hand_mask_is_plausible(mask: np.ndarray) -> bool:
    """Reject false-positive hand masks before inpainting.

    Skin-color segmentation can mistake page shadows or warm paper for a hand,
    especially on sparse pages. A real finger mask should be edge-connected but
    should not cover most of the page or span nearly the full page height/width.
    """
    pixels = int(np.count_nonzero(mask))
    if pixels < 80:
        return False
    h, w = mask.shape
    ratio = pixels / float(h * w)
    if ratio > 0.145:
        return False

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return False

    largest_ok = False
    total_plausible = 0
    for i in range(1, num):
        x, y, ww, hh, area = stats[i]
        if area < max(80, int(h * w * 0.00035)):
            continue
        bbox_ratio = (ww * hh) / float(h * w)
        too_global = bbox_ratio > 0.28 or ww > w * 0.72 or hh > h * 0.72
        if too_global:
            continue
        slenderish = (ww / max(1, hh) < 4.5) and (hh / max(1, ww) < 7.0)
        if slenderish:
            largest_ok = True
            total_plausible += int(area)

    return largest_ok and (total_plausible / float(h * w)) <= 0.13


def hand_mask_is_plausible_strict(mask: np.ndarray) -> bool:
    """V12.8 stricter plausibility for the final inpainting pass.

    Used only on winner output where we'd rather skip cleanup than risk a
    blotch. Tighter mask-ratio and component-bbox limits than the candidate
    stage so winner selection (which uses the candidate-stage rule) stays
    bit-identical to v12.7.
    """
    pixels = int(np.count_nonzero(mask))
    if pixels < 80:
        return False
    h, w = mask.shape
    ratio = pixels / float(h * w)
    if ratio > 0.10:
        return False

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return False

    largest_ok = False
    total_plausible = 0
    for i in range(1, num):
        x, y, ww, hh, area = stats[i]
        if area < max(80, int(h * w * 0.00035)):
            continue
        bbox_ratio = (ww * hh) / float(h * w)
        too_global = bbox_ratio > 0.22 or ww > w * 0.65 or hh > h * 0.65
        if too_global:
            continue
        slenderish = (ww / max(1, hh) < 4.5) and (hh / max(1, ww) < 7.0)
        if slenderish:
            largest_ok = True
            total_plausible += int(area)

    return largest_ok and (total_plausible / float(h * w)) <= 0.09


def bottom_hand_penalty(image_bgr: np.ndarray, mp_hand_mask: np.ndarray) -> float:
    h, w = image_bgr.shape[:2]
    y0 = int(h * 0.76)
    x_margin = int(w * 0.03)
    roi = image_bgr[y0:h, x_margin:w - x_margin]
    if roi.size == 0:
        return 0.0
    skin = skin_like_mask(roi)
    mp_roi = mp_hand_mask[y0:h, x_margin:w - x_margin] if mp_hand_mask is not None else np.zeros_like(skin)
    combo = cv2.bitwise_or(skin, mp_roi)
    yy, xx = combo.shape
    edge = np.zeros_like(combo)
    ex = int(xx * 0.18)
    ey = int(yy * 0.40)
    edge[:, :ex] = 255
    edge[:, xx - ex:] = 255
    edge[yy - ey:, :] = 255
    combo = cv2.bitwise_and(combo, edge)
    ratio = float(np.count_nonzero(combo)) / float(np.count_nonzero(edge) + 1)
    return min(1.0, ratio * 4.5)


def hand_text_overlap_penalty(image_bgr: np.ndarray, hand_mask: np.ndarray) -> float:
    """Estimate whether a detected hand/finger covers printed content.

    A finger near the blank page margin is often recoverable; a finger crossing
    text is usually not. The video selector should strongly prefer a different
    frame over trying to hallucinate missing letters with inpainting.
    """
    if hand_mask is None or np.count_nonzero(hand_mask) == 0:
        return 0.0
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    text_bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    text_bw = cv2.morphologyEx(text_bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    text_zone = cv2.dilate(text_bw, np.ones((17, 17), np.uint8), iterations=1)
    overlap = cv2.bitwise_and(hand_mask, text_zone)
    hand_pixels = float(np.count_nonzero(hand_mask))
    if hand_pixels < 1:
        return 0.0
    overlap_ratio = float(np.count_nonzero(overlap)) / hand_pixels
    page_ratio = float(np.count_nonzero(overlap)) / float(hand_mask.size)
    return min(1.0, overlap_ratio * 1.8 + page_ratio * 18.0)


_LAST_HAND_CLEANUP_INFO: dict = {'applied': False, 'mask_ratio': 0.0, 'reason': ''}


def safe_final_hand_cleanup(image_bgr: np.ndarray, hand_masker: HandMasker, text_protect=True) -> np.ndarray:
    mask = build_hand_cleanup_mask(image_bgr, hand_masker, text_protect=text_protect)
    pixels = int(np.count_nonzero(mask))
    h, w = image_bgr.shape[:2]
    mask_ratio = pixels / float(h * w + 1)
    _LAST_HAND_CLEANUP_INFO.update({'applied': False, 'mask_ratio': float(mask_ratio), 'reason': ''})

    if not hand_mask_is_plausible_strict(mask):
        _LAST_HAND_CLEANUP_INFO['reason'] = 'implausible-strict'
        return image_bgr

    # v12.8: real hand/finger pixels are darker than paper. A paper-bright
    # mask is almost always a bleed-through/shadow false positive — inpainting
    # it produces the gray "patched" artifact seen on sparse pages.
    if hand_mask_is_too_bright(image_bgr, mask):
        _LAST_HAND_CLEANUP_INFO['reason'] = 'too-bright'
        return image_bgr

    cleaned = cv2.inpaint(image_bgr, mask, 7, cv2.INPAINT_TELEA)
    _LAST_HAND_CLEANUP_INFO['applied'] = True
    _LAST_HAND_CLEANUP_INFO['reason'] = 'applied'
    return cleaned


def estimate_stability(prev_quad: Optional[np.ndarray], quad: np.ndarray, shape: Tuple[int, int, int]) -> float:
    if prev_quad is None:
        return 0.5
    h, w = shape[:2]
    dist = float(np.mean(np.linalg.norm(prev_quad - quad, axis=1)))
    norm = dist / max(1.0, 0.5 * (h + w))
    return max(0.0, 1.0 - norm * 9.0)


def estimate_turn_penalty(frame_bgr: np.ndarray, quad: np.ndarray) -> float:
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)
    ys = np.where(mask > 0)[0]
    xs = np.where(mask > 0)[1]
    if len(xs) == 0 or len(ys) == 0:
        return 1.0
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    crop = gray[y0:y1 + 1, x0:x1 + 1]
    if crop.size == 0:
        return 1.0
    hh, ww = crop.shape
    lower = crop[int(hh * 0.55):, :]
    upper = crop[:max(1, int(hh * 0.30)), :]
    low_blur = cv2.GaussianBlur(lower, (0, 0), 7)
    up_blur = cv2.GaussianBlur(upper, (0, 0), 7)
    low_res = float(cv2.absdiff(lower, low_blur).mean()) / 255.0
    up_res = float(cv2.absdiff(upper, up_blur).mean()) / 255.0
    gx = cv2.Sobel(crop, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(crop, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    ang = cv2.phase(gx, gy, angleInDegrees=True)
    diag_mask = ((ang > 20) & (ang < 70)) | ((ang > 110) & (ang < 160))
    strong_diag = float(np.mean((mag > 35) & diag_mask))
    bottom_shadow = float(np.mean(lower < np.percentile(crop, 20)))
    return min(1.0, low_res * 3.2 + max(0.0, low_res - up_res) * 3.8 + strong_diag * 9.0 + bottom_shadow * 0.7)


def estimate_edge_motion_penalty(curr_gray: np.ndarray, prev_gray: Optional[np.ndarray]) -> float:
    if prev_gray is None or prev_gray.shape != curr_gray.shape:
        return 0.0
    diff = cv2.absdiff(curr_gray, prev_gray)
    h, w = diff.shape
    edge = np.zeros_like(diff)
    edge[:, : int(w * 0.18)] = 255
    edge[:, int(w * 0.82):] = 255
    edge[: int(h * 0.18), :] = 255
    edge[int(h * 0.82):, :] = 255
    vals = diff[edge > 0]
    if vals.size == 0:
        return 0.0
    return min(1.0, float(vals.mean()) / 40.0)


def border_contact_score(quad: np.ndarray, shape: Tuple[int, int, int]) -> float:
    h, w = shape[:2]
    x = quad[:, 0]
    y = quad[:, 1]
    left = float(np.min(x)) / max(1.0, w)
    right = float(w - np.max(x)) / max(1.0, w)
    top = float(np.min(y)) / max(1.0, h)
    bottom = float(h - np.max(y)) / max(1.0, h)
    margins = [left, right, top, bottom]
    closeness = [max(0.0, 1.0 - min(1.0, m / 0.18)) for m in margins]
    return 0.45 * max(closeness) + 0.55 * float(np.mean(closeness))


def preprocess_variants(gray: np.ndarray) -> List[np.ndarray]:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(blur)
    canny = cv2.Canny(clahe, 40, 120)
    ad = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7)
    ad_inv = 255 - ad
    _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_inv = 255 - otsu
    out = []
    for m in [canny, ad_inv, otsu_inv]:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mm = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
        mm = cv2.dilate(mm, k, iterations=1)
        out.append(mm)
    return out


def page_likelihood_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """Segment bright, low-saturation paper from background.

    Contour-only edge detection often locks onto text blocks, the book spine,
    or the whole frame. A paper-likelihood mask gives the detector a stronger
    prior: the page is a large, mostly bright, weakly saturated component.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    l = lab[:, :, 0]
    bright_thr = int(max(120, np.percentile(v, 58)))
    l_thr = int(max(125, np.percentile(l, 55)))
    sat_thr = int(min(120, max(45, np.percentile(s, 72))))
    bright = cv2.inRange(v, bright_thr, 255)
    light = cv2.inRange(l, l_thr, 255)
    low_sat = cv2.inRange(s, 0, sat_thr)
    mask = cv2.bitwise_and(cv2.bitwise_or(bright, light), low_sat)

    # Restore black text holes inside the white page while keeping external
    # background out.
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, w // 45), max(9, h // 45)))
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, w // 140), max(5, h // 140)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    return mask


def quad_from_component(mask: np.ndarray, img_shape: Tuple[int, int]) -> Tuple[float, Optional[np.ndarray], float, float]:
    h, w = img_shape
    frame_area = float(h * w)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_score = -1e9
    best_quad = None
    best_area_ratio = 0.0
    best_fill = 0.0
    for cnt in sorted(cnts, key=cv2.contourArea, reverse=True)[:8]:
        area = cv2.contourArea(cnt)
        if area < 0.16 * frame_area:
            continue
        hull = cv2.convexHull(cnt)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.018 * peri, True)
        if len(approx) >= 4:
            if len(approx) == 4:
                quad = approx.reshape(4, 2).astype(np.float32)
            else:
                rect = cv2.minAreaRect(hull)
                quad = cv2.boxPoints(rect).astype(np.float32)
        else:
            rect = cv2.minAreaRect(hull)
            quad = cv2.boxPoints(rect).astype(np.float32)
        quad = order_quad(quad)
        q_area = cv2.contourArea(quad.astype(np.float32))
        area_ratio = q_area / frame_area
        x, y, ww, hh = cv2.boundingRect(quad.astype(np.int32))
        fill_ratio = q_area / float(max(1, ww * hh))
        page_coverage = area / float(max(1.0, q_area))
        center = quad.mean(axis=0)
        center_dist = np.linalg.norm(center - np.array([w / 2, h / 2], dtype=np.float32)) / np.linalg.norm([w / 2, h / 2])
        edges = [np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)]
        aspect = max(edges) / (min(edges) + 1e-6)
        aspect_ok = 0.55 <= aspect <= 3.2
        score = (
            3.0 * area_ratio +
            1.4 * fill_ratio +
            1.0 * min(1.0, page_coverage) -
            0.75 * center_dist -
            (0.6 if not aspect_ok else 0.0)
        )
        if score > best_score:
            best_score = score
            best_quad = quad
            best_area_ratio = area_ratio
            best_fill = fill_ratio
    return best_score, best_quad, best_area_ratio, best_fill


def contour_score(cnt: np.ndarray, img_shape: Tuple[int, int]) -> Tuple[float, Optional[np.ndarray], float, float]:
    h, w = img_shape
    area = cv2.contourArea(cnt)
    frame_area = float(h * w)
    if area < 0.12 * frame_area:
        return -1.0, None, 0.0, 0.0
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) == 4:
        quad = approx.reshape(4, 2).astype(np.float32)
    else:
        rect = cv2.minAreaRect(cnt)
        quad = cv2.boxPoints(rect).astype(np.float32)
    quad = order_quad(quad)
    q_area = cv2.contourArea(quad.astype(np.float32))
    area_ratio = q_area / frame_area
    x, y, ww, hh = cv2.boundingRect(quad.astype(np.int32))
    box_area = max(1, ww * hh)
    fill_ratio = q_area / box_area
    center = quad.mean(axis=0)
    center_dist = np.linalg.norm(center - np.array([w / 2, h / 2], dtype=np.float32)) / np.linalg.norm([w / 2, h / 2])
    edges = [np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)]
    aspect = max(edges) / (min(edges) + 1e-6)
    aspect_pen = 0.0 if 0.55 <= aspect <= 2.6 else 0.35
    score = area_ratio * 2.5 + fill_ratio * 1.0 - center_dist * 0.6 - aspect_pen
    return score, quad, area_ratio, fill_ratio


def page_quad_content_score(frame_bgr: np.ndarray, quad: np.ndarray) -> float:
    """Score whether a quad is likely the target page, not desk/opposite page."""
    h, w = frame_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)
    if np.count_nonzero(mask) < h * w * 0.10:
        return -1.0

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    inside = mask > 0

    # Page interior should be mostly bright and low saturation. Desk/skin are
    # more saturated; spine/shadows are darker.
    bright = float(np.mean(val[inside] > max(125, np.percentile(val, 45))))
    low_sat = float(np.mean(sat[inside] < min(125, max(55, np.percentile(sat, 75)))))
    dark = float(np.mean(gray[inside] < 80))
    saturated = float(np.mean(sat[inside] > 85))

    # A good page quad has a quiet outer border. If the border contains lots of
    # saturated/dark pixels, it probably includes desk, fingers, or spine.
    border = np.zeros_like(mask)
    cv2.polylines(border, [quad.astype(np.int32)], True, 255, max(6, int(min(h, w) * 0.012)))
    border_inside = (border > 0) & inside
    if np.any(border_inside):
        border_bad = float(np.mean((sat[border_inside] > 70) | (gray[border_inside] < 95)))
    else:
        border_bad = 0.0

    # Reject extremely skewed quads unless area evidence is very strong.
    rect = order_quad(quad)
    tl, tr, br, bl = rect
    top = np.linalg.norm(tr - tl)
    bottom = np.linalg.norm(br - bl)
    left = np.linalg.norm(bl - tl)
    right = np.linalg.norm(br - tr)
    parallel_pen = abs(top - bottom) / max(top, bottom, 1.0) + abs(left - right) / max(left, right, 1.0)

    return 0.95 * bright + 0.80 * low_sat - 0.95 * dark - 0.70 * saturated - 0.55 * border_bad - 0.20 * parallel_pen


def detect_page_quad(frame_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], float, float]:
    h, w = frame_bgr.shape[:2]
    scale = 1000.0 / max(h, w) if max(h, w) > 1000 else 1.0
    small = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale != 1.0 else frame_bgr
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    best_score = -1e9
    best_quad = None
    best_area_ratio = 0.0
    best_fill = 0.0

    paper_mask = page_likelihood_mask(small)
    score, quad, area_ratio, fill_ratio = quad_from_component(paper_mask, gray.shape)
    if quad is not None and score > best_score:
        best_score = score
        best_quad = quad
        best_area_ratio = area_ratio
        best_fill = fill_ratio

    for mask in preprocess_variants(gray):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:14]
        for cnt in cnts:
            score, quad, area_ratio, fill_ratio = contour_score(cnt, gray.shape)
            if quad is not None and score > best_score:
                best_score = score
                best_quad = quad
                best_area_ratio = area_ratio
                best_fill = fill_ratio
    if best_quad is None:
        return None, 0.0, 0.0
    # Re-score nearby candidate is handled above; here just require that the best
    # quad's content resembles paper. If not, try a slightly tighter quad around
    # its center to avoid grabbing desk/spine at edges.
    content_score = page_quad_content_score(small, best_quad)
    if content_score < 0.20:
        center = best_quad.mean(axis=0)
        tight = center + (best_quad - center) * 0.94
        if page_quad_content_score(small, tight) > content_score + 0.08:
            best_quad = tight
    if scale != 1.0:
        best_quad = best_quad / scale
    return expand_quad(best_quad, 0.015), best_area_ratio, best_fill


def _quad_aspect(quad: np.ndarray) -> float:
    """Return horizontal/vertical aspect ratio of an ordered quad (TL,TR,BR,BL)."""
    rect = order_quad(quad)
    tl, tr, br, bl = rect
    width = 0.5 * (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl))
    height = 0.5 * (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr))
    if height < 1e-3:
        return 1.0
    return float(width / height)


def _split_quad_horizontal(quad: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray]:
    """Split an ordered quad along a vertical line at parameter t in [0,1].

    Returns (left_quad, right_quad) in TL,TR,BR,BL order. The split line goes
    from t along the top edge to t along the bottom edge — this keeps the cut
    perspective-consistent with the page surface even when the book is tilted.
    """
    rect = order_quad(quad)
    tl, tr, br, bl = rect
    top_split = tl + (tr - tl) * t
    bot_split = bl + (br - bl) * t
    left = np.array([tl, top_split, bot_split, bl], dtype=np.float32)
    right = np.array([top_split, tr, br, bot_split], dtype=np.float32)
    return left, right


def find_spine_seam(frame_bgr: np.ndarray, quad: np.ndarray, search_range: Tuple[float, float] = (0.20, 0.80)) -> Tuple[Optional[float], float]:
    """Locate the book spine as a parameter t in [0,1] along the quad's top edge.

    Returns (t, confidence). The spine seam is detected as a darker, slightly
    saturated vertical band inside a normalized warp of the quad. Confidence is
    a heuristic in [0,1]: higher means a clean seam was found inside the
    search range. If no seam is confident enough, returns (None, 0.0).

    search_range narrows where the seam may sit (as fractions of the quad
    width). Default (0.20,0.80) is for spreads; pass tighter ranges to detect
    edge-side spine slivers on a single-page quad.
    """
    try:
        rect = order_quad(quad)
        norm_w = 600
        norm_h = 400
        dst = np.array([[0, 0], [norm_w - 1, 0], [norm_w - 1, norm_h - 1], [0, norm_h - 1]], dtype=np.float32)
        m = cv2.getPerspectiveTransform(rect, dst)
        warp = cv2.warpPerspective(frame_bgr, m, (norm_w, norm_h), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        return None, 0.0

    hsv = cv2.cvtColor(warp, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    val = hsv[:, :, 2]
    sat = hsv[:, :, 1]

    lo, hi = search_range
    cols_start = int(round(norm_w * max(0.0, lo)))
    cols_end = int(round(norm_w * min(1.0, hi)))
    if cols_end - cols_start < 20:
        return None, 0.0

    mid_band = slice(int(norm_h * 0.15), int(norm_h * 0.85))
    val_mid = val[mid_band, cols_start:cols_end]
    sat_mid = sat[mid_band, cols_start:cols_end]
    gray_mid = gray[mid_band, cols_start:cols_end]

    # Spine signature: darker than surrounding paper, slightly saturated/colored
    # (shadow gradient), and high vertical continuity.
    bright_ref = float(np.percentile(val_mid, 80))
    dark_score = np.clip((bright_ref - val_mid.astype(np.float32)) / max(1.0, bright_ref), 0, 1).mean(axis=0)
    sat_score = np.clip(sat_mid.astype(np.float32) / 90.0, 0, 1).mean(axis=0)
    gx = cv2.Sobel(gray_mid, cv2.CV_32F, 1, 0, ksize=3)
    edge_score = np.clip(np.abs(gx) / 80.0, 0, 1).mean(axis=0)

    raw = 0.55 * dark_score + 0.20 * sat_score + 0.25 * edge_score
    if raw.size < 7:
        return None, 0.0
    smooth = cv2.GaussianBlur(raw.reshape(1, -1).astype(np.float32), (1, 21), 0).reshape(-1)

    seam_local = int(np.argmax(smooth))
    seam_x = cols_start + seam_local
    peak = float(smooth[seam_local])
    background = float(np.median(smooth))
    contrast = peak - background

    # Require both an absolute and a relative peak to guard against flat pages.
    if peak < 0.22 or contrast < 0.06:
        return None, 0.0

    t = seam_x / float(norm_w)
    if t < lo - 1e-3 or t > hi + 1e-3:
        return None, 0.0

    # Confidence: stronger when contrast is high. Center bonus only helps when
    # the search range covers the middle (spread split case).
    spread_search = lo <= 0.30 and hi >= 0.70
    if spread_search:
        center_bonus = max(0.0, 1.0 - abs(t - 0.5) / 0.32)
        conf = float(np.clip(0.6 * (contrast / 0.25) + 0.4 * center_bonus, 0.0, 1.0))
    else:
        conf = float(np.clip(contrast / 0.18, 0.0, 1.0))
    return t, conf


def _half_text_density(frame_bgr: np.ndarray, quad: np.ndarray) -> float:
    """Cheap text-density estimate inside a quad (used to pick auto side)."""
    try:
        rect = order_quad(quad)
        norm_w = 320
        norm_h = 420
        dst = np.array([[0, 0], [norm_w - 1, 0], [norm_w - 1, norm_h - 1], [0, norm_h - 1]], dtype=np.float32)
        m = cv2.getPerspectiveTransform(rect, dst)
        warp = cv2.warpPerspective(frame_bgr, m, (norm_w, norm_h), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        return 0.0
    gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 12)
    inner = bw[int(norm_h * 0.08):int(norm_h * 0.92), int(norm_w * 0.08):int(norm_w * 0.92)]
    if inner.size == 0:
        return 0.0
    return float(np.mean(inner > 0))


def _maybe_trim_spine_sliver(frame_bgr: np.ndarray, quad: np.ndarray, mode: str) -> Tuple[np.ndarray, str]:
    """Trim a thin spine/opposite-page sliver from one side of a single-page quad.

    Conservative: only acts if a low-position seam is detected near one edge of
    the quad with adequate contrast. Search range depends on mode:
      - 'auto'  : search both edges, take the stronger seam if confident enough.
      - 'right' : assume the page being shown is the right page, so the spine
                  sliver lives near the LEFT edge of the quad. Search t in [0.04,0.22].
      - 'left'  : opposite — spine sliver near the RIGHT edge. Search t in [0.78,0.96].
    Trim the seam plus a small safety pad. If nothing meets thresholds, return
    the original quad unchanged.
    """
    def _trim(t: float, conf: float, side_label: str) -> Tuple[np.ndarray, str]:
        # Add a tiny pad past the seam toward the page interior.
        pad = 0.012
        if side_label == 'left-edge':
            new_t = min(0.30, t + pad)
            _, right_q = _split_quad_horizontal(quad, new_t)
            return right_q, f'trim-left-sliver(t={new_t:.2f},c={conf:.2f})'
        # right-edge
        new_t = max(0.70, t - pad)
        left_q, _ = _split_quad_horizontal(quad, new_t)
        return left_q, f'trim-right-sliver(t={new_t:.2f},c={conf:.2f})'

    # Stricter contrast requirement when modifying a single-page quad — we want
    # to avoid trimming clean pages that just have a normal book gutter shadow.
    min_conf = 0.55 if mode == 'auto' else 0.40

    candidates = []
    if mode in ('auto', 'right'):
        t_l, c_l = find_spine_seam(frame_bgr, quad, search_range=(0.04, 0.22))
        if t_l is not None and c_l >= min_conf:
            candidates.append((c_l, t_l, 'left-edge'))
    if mode in ('auto', 'left'):
        t_r, c_r = find_spine_seam(frame_bgr, quad, search_range=(0.78, 0.96))
        if t_r is not None and c_r >= min_conf:
            candidates.append((c_r, t_r, 'right-edge'))

    if not candidates:
        return quad, 'full(no-sliver)'

    candidates.sort(reverse=True)
    conf, t, side = candidates[0]
    return _trim(t, conf, side)


def select_page_side(frame_bgr: np.ndarray, quad: np.ndarray, mode: str) -> Tuple[np.ndarray, str]:
    """Return the chosen page-side quad and a label describing what was picked.

    Modes:
      - 'full'   : always return the original full quad.
      - 'left'   : split at the spine and keep the left half (fallback: full).
      - 'right'  : split at the spine and keep the right half (fallback: full).
      - 'auto'   : conservative — only split if a confident spine is found AND
                   the quad looks like a wide spread (aspect > 1.25). Otherwise
                   return the full quad. If split, the side with substantially
                   more text density wins; ties keep the full quad.
    """
    if mode == 'full' or quad is None:
        return quad, 'full'

    # 'auto-smart' behaves like 'auto' during candidate detection so the
    # chronological winner selection stays bit-identical to v12.5's safe path.
    # The sliver trim is applied later, post-selection, on winner frames only.
    detection_mode = 'auto' if mode == 'auto-smart' else mode

    aspect = _quad_aspect(quad)
    # A single page should be portrait-ish (aspect < 1.0). A spread is roughly
    # square or landscape (aspect >= 1.05). For 'auto', be strict to stay
    # conservative; for explicit 'left'/'right', be more permissive.
    aspect_threshold = 1.25 if detection_mode == 'auto' else 1.05
    if aspect < aspect_threshold:
        # Single-page quad. In 'auto' mode we keep the full quad — sliver trim
        # is opt-in via 'right'/'left' to avoid destabilizing the chronological
        # frame-selection on videos where v12.4 already produces the right page
        # count. In explicit modes the sliver trim runs.
        if detection_mode == 'auto':
            return quad, 'full(single-page)'
        return _maybe_trim_spine_sliver(frame_bgr, quad, detection_mode)

    t, conf = find_spine_seam(frame_bgr, quad, search_range=(0.20, 0.80))
    if t is None:
        return quad, 'full(no-seam)'

    left_q, right_q = _split_quad_horizontal(quad, t)

    if mode == 'left':
        return left_q, f'left(t={t:.2f},c={conf:.2f})'
    if mode == 'right':
        return right_q, f'right(t={t:.2f},c={conf:.2f})'

    # auto: pick the higher-text-density side, but only commit if confidence is
    # decent and the difference is meaningful. Otherwise fall back to full.
    if conf < 0.45:
        return quad, f'full(low-conf={conf:.2f})'
    left_text = _half_text_density(frame_bgr, left_q)
    right_text = _half_text_density(frame_bgr, right_q)
    diff = abs(left_text - right_text)
    dominant = max(left_text, right_text)
    # Need a real winner: at least one side with meaningful text and a clear
    # delta. Title pages may have very little text on either side — prefer full
    # in that case.
    if dominant < 0.012 or diff < 0.004:
        # Both sides are similar — prefer the side closest to image center
        # which usually contains the page being shown.
        h, w = frame_bgr.shape[:2]
        img_cx = w / 2.0
        left_cx = float(left_q.mean(axis=0)[0])
        right_cx = float(right_q.mean(axis=0)[0])
        if abs(left_cx - img_cx) + 1.0 < abs(right_cx - img_cx):
            return left_q, f'auto-center-left(t={t:.2f})'
        return right_q, f'auto-center-right(t={t:.2f})'
    if left_text > right_text:
        return left_q, f'auto-left(t={t:.2f},dt={left_text - right_text:.3f})'
    return right_q, f'auto-right(t={t:.2f},dt={right_text - left_text:.3f})'


def auto_smart_trim_winner(
    frame_bgr: np.ndarray,
    quad: np.ndarray,
    long_side: int,
    base_warped: Optional[np.ndarray],
    min_conf: float = 0.62,
    max_shrink: float = 0.10,
    min_similarity: float = 0.80,
) -> Tuple[Optional[np.ndarray], str, dict]:
    """Conservative post-selection sliver trim for auto-smart mode.

    Operates only on a winner frame whose quad is single-page (aspect < 1.25).
    Searches both edges for a high-confidence spine-sliver seam. If found, builds
    a trimmed quad and re-warps. The trim is accepted only when:
      - seam confidence >= min_conf
      - resulting quad shrinks <= max_shrink (no over-trim)
      - the trimmed warp is similar (dHash) to the base warp (stability gate;
        guards against accidentally re-cropping to the wrong page)
    Returns (new_warped_or_none, label, info). info is a dict with structured
    fields: applied, skip_reason, seam_side, seam_confidence, trim_fraction,
    shrink, dhash_similarity. Returns (None, 'skip-*', info) when no trim should
    be applied so the caller can keep the original.
    """
    info = {
        'applied': False,
        'skip_reason': '',
        'seam_side': '',
        'seam_confidence': float('nan'),
        'trim_fraction': float('nan'),
        'shrink': float('nan'),
        'dhash_similarity': float('nan'),
    }
    if quad is None or base_warped is None:
        info['skip_reason'] = 'no-quad'
        return None, 'skip(no-quad)', info
    aspect = _quad_aspect(quad)
    if aspect >= 1.25:
        info['skip_reason'] = 'spread'
        return None, 'skip(spread)', info

    # Search both edges, pick the stronger seam.
    candidates = []
    t_l, c_l = find_spine_seam(frame_bgr, quad, search_range=(0.04, 0.22))
    if t_l is not None and c_l >= min_conf:
        candidates.append((c_l, t_l, 'left-edge'))
    t_r, c_r = find_spine_seam(frame_bgr, quad, search_range=(0.78, 0.96))
    if t_r is not None and c_r >= min_conf:
        candidates.append((c_r, t_r, 'right-edge'))
    if not candidates:
        info['skip_reason'] = 'low-conf'
        # Record the best seen confidence even if below threshold for debug.
        best_seen = max([c for c in (c_l, c_r) if c is not None], default=float('nan'))
        info['seam_confidence'] = best_seen
        return None, 'skip(low-conf)', info
    candidates.sort(reverse=True)
    conf, t, side = candidates[0]
    info['seam_side'] = side
    info['seam_confidence'] = float(conf)
    info['trim_fraction'] = float(t)

    pad = 0.012
    if side == 'left-edge':
        new_t = min(0.30, t + pad)
        _, sub_quad = _split_quad_horizontal(quad, new_t)
        side_label = 'right'
    else:
        new_t = max(0.70, t - pad)
        sub_quad, _ = _split_quad_horizontal(quad, new_t)
        side_label = 'left'
    info['trim_fraction'] = float(new_t)

    # Shrink guard.
    orig_area = float(cv2.contourArea(quad.astype(np.float32)))
    sub_area = float(cv2.contourArea(sub_quad.astype(np.float32)))
    if orig_area <= 1.0:
        info['skip_reason'] = 'empty-quad'
        return None, 'skip(empty-quad)', info
    shrink = 1.0 - (sub_area / orig_area)
    info['shrink'] = float(shrink)
    if shrink > max_shrink or shrink <= 0.005:
        info['skip_reason'] = f'shrink={shrink:.3f}'
        return None, f'skip(shrink={shrink:.3f})', info

    # Re-warp with a tiny outward expansion (matches detect_page_quad_with_side).
    try:
        warped = four_point_warp(frame_bgr, expand_quad(sub_quad, 0.005), long_side=long_side)
    except Exception:
        info['skip_reason'] = 'warp-fail'
        return None, 'skip(warp-fail)', info

    # Stability gate via dHash similarity. The trimmed warp should still be
    # "the same page" — we only want to remove a sliver, not jump to the
    # opposite page or to a totally different layout.
    try:
        base_gray = cv2.cvtColor(base_warped, cv2.COLOR_BGR2GRAY)
        cand_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        base_hash = compute_dhash(roi_for_similarity(base_gray))
        cand_hash = compute_dhash(roi_for_similarity(cand_gray))
        # 16x16 dHash → 256 bits.
        total_bits = 16 * 16
        ham = hamming_distance(base_hash, cand_hash) if base_hash is not None and cand_hash is not None else total_bits
        sim = 1.0 - (ham / float(total_bits))
    except Exception:
        info['skip_reason'] = 'hash-fail'
        return None, 'skip(hash-fail)', info
    info['dhash_similarity'] = float(sim)
    if sim < min_similarity:
        info['skip_reason'] = f'sim={sim:.2f}'
        return None, f'skip(sim={sim:.2f})', info

    info['applied'] = True
    info['seam_side'] = side_label  # post-trim retained side ('left'/'right')
    return warped, f'auto-smart-trim-{side_label}(t={new_t:.2f},c={conf:.2f},shrink={shrink:.3f},sim={sim:.2f})', info


def detect_page_quad_with_side(frame_bgr: np.ndarray, page_side: str) -> Tuple[Optional[np.ndarray], float, float, str]:
    """Wrap detect_page_quad and apply page-side selection before warp.

    Reports the original quad's area_ratio/fill_ratio when the side trim was a
    small sliver (< 12% of the quad) so that downstream scoring is not biased
    against frames where the conservative trim fired.
    """
    quad, area_ratio, fill_ratio = detect_page_quad(frame_bgr)
    if quad is None:
        return None, 0.0, 0.0, 'none'
    if page_side == 'full':
        return quad, area_ratio, fill_ratio, 'full'
    sub_quad, label = select_page_side(frame_bgr, quad, page_side)
    if sub_quad is quad or label.startswith('full'):
        return quad, area_ratio, fill_ratio, label
    orig_area = float(cv2.contourArea(quad.astype(np.float32)))
    sub_area = float(cv2.contourArea(sub_quad.astype(np.float32)))
    shrink = 1.0 - (sub_area / max(1.0, orig_area))
    x, y, ww, hh = cv2.boundingRect(sub_quad.astype(np.int32))
    new_fill = sub_area / float(max(1, ww * hh))
    if shrink < 0.18 and label.startswith('trim-'):
        # Sliver trim: keep original area_ratio so scoring stays comparable to
        # untrimmed frames. Use the new fill_ratio since the trimmed quad's
        # bounding box may have changed shape.
        return expand_quad(sub_quad, 0.005), area_ratio, new_fill, label
    h, w = frame_bgr.shape[:2]
    frame_area = float(h * w)
    new_area_ratio = sub_area / frame_area
    return expand_quad(sub_quad, 0.005), new_area_ratio, new_fill, label


_LAST_DESKEW_ANGLE: dict = {'angle': 0.0}


def _estimate_skew_angle_legacy(image_bgr: np.ndarray) -> Optional[float]:
    """v12.8 estimator (preserved bit-for-bit for candidate scoring)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    angles: List[float] = []
    lines = cv2.HoughLinesP(
        bw, 1, np.pi / 180.0,
        threshold=90,
        minLineLength=max(24, int(image_bgr.shape[1] * 0.16)),
        maxLineGap=18,
    )
    if lines is not None:
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = line
            length = float(np.hypot(x2 - x1, y2 - y1))
            angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if -18 <= angle <= 18:
                weight = max(1, int(length / 40))
                angles.extend([angle] * weight)
    num, labels, stats, cent = cv2.connectedComponentsWithStats(bw, connectivity=8)
    pts = []
    h, w = bw.shape
    for i in range(1, num):
        x, y, ww, hh, area = stats[i]
        if area < 8 or area > h * w * 0.035:
            continue
        if ww < 2 or hh < 2 or hh > h * 0.12:
            continue
        if y < h * 0.03 or y > h * 0.97:
            continue
        pts.append(cent[i])
    if len(pts) >= 12:
        pts_arr = np.asarray(pts, dtype=np.float32)
        row_tol = max(10, int(h * 0.018))
        order = np.argsort(pts_arr[:, 1])
        rows: List[list] = []
        for idx in order:
            p = pts_arr[idx]
            if not rows or abs(float(np.mean([q[1] for q in rows[-1]])) - p[1]) > row_tol:
                rows.append([p])
            else:
                rows[-1].append(p)
        for row in rows:
            if len(row) < 5:
                continue
            arr = np.asarray(row, dtype=np.float32)
            if arr[:, 0].max() - arr[:, 0].min() < w * 0.12:
                continue
            vx, vy, _, _ = cv2.fitLine(arr, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            angle = float(np.degrees(np.arctan2(vy, vx)))
            if -18 <= angle <= 18:
                angles.append(angle)
    if not angles:
        return None
    return float(np.median(np.asarray(angles, dtype=np.float32)))


def _estimate_skew_angle(image_bgr: np.ndarray) -> Optional[float]:
    """Return a robust median text-line skew in degrees, or None."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    angles: List[float] = []

    # Hough lines work well on text-heavy pages.
    lines = cv2.HoughLinesP(
        bw,
        1,
        np.pi / 180.0,
        threshold=90,
        minLineLength=max(24, int(image_bgr.shape[1] * 0.16)),
        maxLineGap=18,
    )
    if lines is not None:
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = line
            length = float(np.hypot(x2 - x1, y2 - y1))
            angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if -18 <= angle <= 18:
                weight = max(1, int(length / 40))
                angles.extend([angle] * weight)

    # Component centers are more reliable on sparse title/dedication pages where
    # Hough may not find enough long lines.
    num, labels, stats, cent = cv2.connectedComponentsWithStats(bw, connectivity=8)
    pts = []
    h, w = bw.shape
    for i in range(1, num):
        x, y, ww, hh, area = stats[i]
        if area < 8 or area > h * w * 0.035:
            continue
        if ww < 2 or hh < 2 or hh > h * 0.12:
            continue
        if y < h * 0.03 or y > h * 0.97:
            continue
        pts.append(cent[i])
    if len(pts) >= 8:
        pts_arr = np.asarray(pts, dtype=np.float32)
        # Group components into approximate text rows, then fit each row.
        row_tol = max(10, int(h * 0.018))
        order = np.argsort(pts_arr[:, 1])
        rows: List[list] = []
        for idx in order:
            p = pts_arr[idx]
            if not rows or abs(float(np.mean([q[1] for q in rows[-1]])) - p[1]) > row_tol:
                rows.append([p])
            else:
                rows[-1].append(p)
        # v12.9: lower per-row component minimum from 5 to 4 so short title
        # lines (e.g. "ГЕН ВЫСОТЫ") and the top author line participate, and
        # weight long rows so the dominant title baseline drives the median.
        for row in rows:
            if len(row) < 4:
                continue
            arr = np.asarray(row, dtype=np.float32)
            if arr[:, 0].max() - arr[:, 0].min() < w * 0.10:
                continue
            vx, vy, _, _ = cv2.fitLine(arr, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            angle = float(np.degrees(np.arctan2(vy, vx)))
            if -18 <= angle <= 18:
                row_span = float(arr[:, 0].max() - arr[:, 0].min())
                weight = max(1, int(row_span / max(1.0, w * 0.10)))
                angles.extend([angle] * weight)

    if not angles:
        return None
    return float(np.median(np.asarray(angles, dtype=np.float32)))


def deskew_by_text_lines(image_bgr: np.ndarray) -> np.ndarray:
    """v12.8-compatible single-pass deskew used during candidate scoring.

    This must remain bit-identical to v12.8 in behavior so that winner
    selection / clustering does not shift. The improved two-pass refinement
    lives in `deskew_by_text_lines_refined` and is applied only at final
    output time.
    """
    angle = _estimate_skew_angle_legacy(image_bgr)
    if angle is None or abs(angle) < 0.25:
        _LAST_DESKEW_ANGLE['angle'] = 0.0
        return image_bgr
    h, w = image_bgr.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    _LAST_DESKEW_ANGLE['angle'] = angle
    return cv2.warpAffine(image_bgr, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def deskew_by_text_lines_refined(image_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
    """Two-pass deskew used at final output time.

    v12.9: lowered apply threshold (0.25°→0.12°), uses the v12.9 estimator
    that weights long title rows and accepts shorter rows (≥4 components),
    then runs a refinement pass to mop up residual tilt. Returns the rotated
    image and the total angle applied.
    """
    angle1 = _estimate_skew_angle(image_bgr)
    total = 0.0
    out = image_bgr
    if angle1 is not None and abs(angle1) >= 0.12:
        h, w = out.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle1, 1.0)
        out = cv2.warpAffine(out, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        total += angle1
        angle2 = _estimate_skew_angle(out)
        if angle2 is not None and 0.10 <= abs(angle2) <= 1.5:
            m2 = cv2.getRotationMatrix2D((w / 2, h / 2), angle2, 1.0)
            out = cv2.warpAffine(out, m2, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            total += angle2
    return out, total


_LAST_ENHANCE_MODE: dict = {'mode': ''}


def _estimate_text_density(gray: np.ndarray) -> float:
    """Coarse fraction of dark text-like pixels in the image."""
    h, w = gray.shape[:2]
    body = gray[int(h * 0.05):int(h * 0.95), int(w * 0.05):int(w * 0.95)]
    if body.size == 0:
        return 0.0
    thr = max(80, int(np.percentile(body, 50) * 0.55))
    return float(np.mean(body < thr))


def _looks_decorative(image_bgr: np.ndarray, gray: np.ndarray) -> bool:
    """Detect pages with a graphical/decorative element that should NOT be
    routed through the sparse-page pipeline.

    v12.9: the v12.8 sparse pipeline (aggressive bg flatten + bilateral
    smoothing + low-floor stretch) is tuned for genuinely near-blank pages.
    On a chapter-cover page with a small drawing (e.g. ice axe / carabiner)
    plus heavy chapter title text, the same pipeline posterizes the
    drawing's cross-hatch into ugly gray blotches and amplifies bleed-through
    around it. The page measures "sparse" by simple text density, but it is
    not safe to flatten.

    A page is decorative if either:
      - it contains a sizable contiguous dark component that is too large to
        be a single character (drawing, icon, large heading block), or
      - the body has a substantial fraction of dark pixels relative to a
        true blank page.
    """
    h, w = gray.shape[:2]
    if h < 40 or w < 40:
        return False
    body = gray[int(h * 0.06):int(h * 0.94), int(w * 0.06):int(w * 0.94)]
    bH, bW = body.shape
    if bH < 20 or bW < 20:
        return False

    dark_frac = float(np.mean(body < 70))
    if dark_frac > 0.015:
        # On a true blank/dedication page dark_frac is ~1e-4. A page with any
        # meaningful graphic or chapter title block crosses this easily.
        try:
            bw = cv2.adaptiveThreshold(
                body, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 31, 15,
            )
            num, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
            min_area = bH * bW * 0.003
            min_w = bW * 0.05
            min_h = bH * 0.04
            for i in range(1, num):
                _x, _y, ww, hh, area = stats[i]
                if area >= min_area and ww >= min_w and hh >= min_h:
                    return True
        except Exception:
            pass
    return False


def enhance_scanned_page(image_bgr: np.ndarray) -> np.ndarray:
    """Flatten page lighting and make text more readable.

    v12.9: three modes.
      - 'decorative': chapter-cover / graphic pages with intentional gray or
        textured backgrounds. Use the conservative legacy pipeline (no sparse
        flattening, no aggressive smoothing) so the design is preserved.
      - 'sparse': near-blank pages (e.g. dedication). Blotch-resistant
        flattening from v12.8.
      - 'standard': regular text pages. Same as v12.8 standard path.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    text_density = _estimate_text_density(gray)
    decorative = _looks_decorative(image_bgr, gray)
    # Decorative wins over sparse: a cover page can have low "text density"
    # by our coarse measure but must not be sparse-flattened.
    sparse = (not decorative) and (text_density < 0.045)

    # Estimate only very low-frequency lighting. Additive correction is much
    # safer than division for overexposed pages and sparse text pages.
    k = max(61, int(min(image_bgr.shape[:2]) * 0.11) | 1)
    bg = cv2.GaussianBlur(l, (k, k), 0)
    if sparse:
        # Stronger flattening on near-blank pages so bleed-through patches do
        # not become visible blotches. Pull paper towards a single bright
        # target rather than preserving local variation.
        target = float(np.percentile(bg, 80))
        corrected = l.astype(np.float32) + (target - bg.astype(np.float32)) * 0.55
    else:
        # Standard / decorative: gentle additive lighting correction only.
        target = float(np.percentile(bg, 72))
        corrected = l.astype(np.float32) + (target - bg.astype(np.float32)) * 0.32
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)

    if sparse:
        # On a near-blank page the percentile stretch sees only paper +
        # bleed-through, so it amplifies the bleed-through into a blotch. Use a
        # gentler "raise the floor" stretch instead.
        p10, p98 = np.percentile(corrected, (10.0, 98.5))
        if p98 > p10 + 4:
            corrected = np.clip(
                (corrected.astype(np.float32) - p10) * 240.0 / (p98 - p10) + 8.0,
                0, 248,
            ).astype(np.uint8)
        # Skip CLAHE: it is the main producer of blotchy gray on sparse pages.
    else:
        # Standard and decorative both use the v12.7 mild stretch + CLAHE so
        # decorative gray fills retain their natural tone.
        p2, p98 = np.percentile(corrected, (2.0, 98.5))
        if p98 > p2 + 4:
            corrected = np.clip(
                (corrected.astype(np.float32) - p2) * 235.0 / (p98 - p2) + 10.0,
                0, 245,
            ).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=1.15, tileGridSize=(8, 8)).apply(corrected)
        corrected = cv2.addWeighted(corrected, 0.72, clahe, 0.28, 0)

    # Neutralize paper color softly, preserving any real color marks.
    a = cv2.addWeighted(a, 0.78, np.full_like(a, 128), 0.22, 0)
    b = cv2.addWeighted(b, 0.78, np.full_like(b, 128), 0.22, 0)
    enhanced = cv2.cvtColor(cv2.merge([corrected, a, b]), cv2.COLOR_LAB2BGR)

    if sparse:
        # On sparse pages, suppress residual blotches with a soft bilateral
        # filter that preserves text while smoothing paper variation.
        smoothed = cv2.bilateralFilter(enhanced, d=7, sigmaColor=22, sigmaSpace=11)
        enhanced = cv2.addWeighted(enhanced, 0.45, smoothed, 0.55, 0)
        # Very mild unsharp on text only.
        blur = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
        enhanced = cv2.addWeighted(enhanced, 1.04, blur, -0.04, 0)
    else:
        # Very mild unsharp mask; skip strong contrast edges to avoid glowing text.
        blur = cv2.GaussianBlur(enhanced, (0, 0), 0.85)
        enhanced = cv2.addWeighted(enhanced, 1.08, blur, -0.08, 0)

    if decorative:
        mode = 'decorative'
    elif sparse:
        mode = 'sparse'
    else:
        mode = 'standard'
    _LAST_ENHANCE_MODE['mode'] = mode
    return enhanced


def final_page_postprocess(image_bgr: np.ndarray, args) -> np.ndarray:
    if getattr(args, 'no_enhance', False):
        return image_bgr
    return enhance_scanned_page(image_bgr)


def robust_norm(values: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    values = values.astype(np.float32)
    mask = np.isfinite(values)
    out = np.zeros_like(values, dtype=np.float32)
    if not np.any(mask):
        return out
    v = values[mask]
    p10 = np.percentile(v, 10)
    p90 = np.percentile(v, 90)
    if abs(p90 - p10) < 1e-6:
        out[mask] = 0.5
        return out
    n = np.clip((v - p10) / (p90 - p10), 0.0, 1.0)
    if not higher_is_better:
        n = 1.0 - n
    out[mask] = n
    return out


def moving_average(arr: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return arr.copy()
    k = 2 * radius + 1
    kernel = np.ones(k, dtype=np.float32) / k
    padded = np.pad(arr, (radius, radius), mode='edge')
    return np.convolve(padded, kernel, mode='valid')


def base_preference_score(x: FrameFeatures) -> float:
    return x.peak_score


def tie_break_cleanliness_score(x: FrameFeatures) -> float:
    return - (2.4 * x.hand_text_overlap_penalty + 2.2 * x.bottom_hand_penalty + 1.7 * x.hand_penalty + 1.2 * x.edge_foreground_penalty)


def choose_between_similar(a: FrameFeatures, b: FrameFeatures, sim_thresh: float) -> FrameFeatures:
    if a.roi_gray is None or b.roi_gray is None:
        return a if base_preference_score(a) >= base_preference_score(b) else b
    sim = similarity_score(a.roi_gray, b.roi_gray)
    ham = hamming_distance(a.roi_dhash, b.roi_dhash)
    if sim >= sim_thresh or ham <= 9:
        a_clean = tie_break_cleanliness_score(a)
        b_clean = tie_break_cleanliness_score(b)
        if abs(a_clean - b_clean) > 0.020:
            return a if a_clean > b_clean else b
    return a if base_preference_score(a) >= base_preference_score(b) else b


def select_local_peaks(items: List[FrameFeatures], fps_sampled: float, args) -> List[FrameFeatures]:
    valid = [x for x in items if x.page_found and x.warped_bgr is not None]
    if not valid:
        return []
    scores = np.array([x.norm_score for x in valid], dtype=np.float32)
    smooth_radius = max(1, int(round(args.peak_window_sec * fps_sampled)))
    smooth = moving_average(scores, smooth_radius)
    for i, x in enumerate(valid):
        clean_bonus = 1.0 - min(1.0, 0.55 * x.hand_penalty + 0.55 * x.hand_text_overlap_penalty + 0.30 * x.edge_foreground_penalty + 0.35 * x.bottom_hand_penalty)
        x.peak_score = float((0.64 * x.norm_score + 0.36 * smooth[i]) * (0.84 + 0.16 * clean_bonus))

    peaks: List[FrameFeatures] = []
    sep = args.min_peak_distance_sec
    for x in valid:
        left_t = x.t_sec - sep
        right_t = x.t_sec + sep
        neighborhood = [y.peak_score for y in valid if left_t <= y.t_sec <= right_t]
        if neighborhood and x.peak_score >= max(neighborhood) and x.norm_score >= args.min_norm_score:
            peaks.append(x)

    dedup: List[FrameFeatures] = []
    for p in sorted(peaks, key=lambda z: z.t_sec):
        if dedup and abs(p.t_sec - dedup[-1].t_sec) < sep:
            dedup[-1] = choose_between_similar(dedup[-1], p, args.sim_thresh_merge - 0.02)
        else:
            dedup.append(p)
    return dedup


def cluster_candidates(candidates: List[FrameFeatures], args) -> List[Cluster]:
    clusters: List[Cluster] = []
    for cand in sorted(candidates, key=lambda c: c.t_sec):
        placed = False
        for cl in clusters:
            rep = cl.members[0]
            for m in cl.members[1:]:
                rep = choose_between_similar(rep, m, args.sim_thresh_merge)
            if cand.roi_gray is None or rep.roi_gray is None:
                continue
            ham = hamming_distance(cand.roi_dhash, rep.roi_dhash)
            sim = similarity_score(cand.roi_gray, rep.roi_gray)
            dt = abs(cand.t_sec - rep.t_sec)
            if ham <= args.hash_thresh_merge and sim >= args.sim_thresh_merge:
                if dt < args.min_same_page_gap_sec or sim >= (args.sim_thresh_merge + 0.05):
                    cl.members.append(cand)
                    placed = True
                    break
        if not placed:
            clusters.append(Cluster(members=[cand]))
    return clusters


def is_visually_same_page(a: FrameFeatures, b: FrameFeatures, args) -> bool:
    if a.roi_gray is None or b.roi_gray is None or a.roi_dhash is None or b.roi_dhash is None:
        return False
    ham = hamming_distance(a.roi_dhash, b.roi_dhash)
    sim = similarity_score(a.roi_gray, b.roi_gray)
    return ham <= args.hash_thresh_merge and sim >= args.sim_thresh_merge


def visual_novelty(a: FrameFeatures, selected: List[FrameFeatures]) -> float:
    if not selected or a.roi_gray is None:
        return 1.0
    best_sim = -1.0
    best_ham = 256
    for b in selected:
        if b.roi_gray is None or b.roi_dhash is None or a.roi_dhash is None:
            continue
        best_sim = max(best_sim, similarity_score(a.roi_gray, b.roi_gray))
        best_ham = min(best_ham, hamming_distance(a.roi_dhash, b.roi_dhash))
    sim_novelty = 1.0 - max(0.0, best_sim)
    ham_novelty = min(1.0, best_ham / 96.0)
    return 0.55 * sim_novelty + 0.45 * ham_novelty


def select_expected_pages_chronological(valid: List[FrameFeatures], args) -> List[FrameFeatures]:
    """Pick expected pages by chronological visual novelty.

    This handles short videos where page turns are uneven: early pages may last
    less time than later pages, so equal time windows are unreliable.
    """
    if args.expected_pages <= 0 or not valid:
        return []
    ordered = sorted(valid, key=lambda x: x.t_sec)
    min_gap = max(0.45, args.min_peak_distance_sec * 0.65)
    selected: List[FrameFeatures] = []

    for cand in ordered:
        if cand.norm_score < max(0.0, args.min_norm_score - 0.24):
            continue
        if selected and cand.t_sec - selected[-1].t_sec < min_gap:
            # Same temporal neighborhood: keep the cleaner/better candidate.
            curr = selected[-1]
            cand_quality = cand.peak_score - 0.22 * cand.hand_text_overlap_penalty - 0.20 * cand.bottom_hand_penalty
            curr_quality = curr.peak_score - 0.22 * curr.hand_text_overlap_penalty - 0.20 * curr.bottom_hand_penalty
            if cand_quality > curr_quality:
                selected[-1] = cand
            continue
        novelty = visual_novelty(cand, selected)
        if not selected or novelty >= 0.32:
            selected.append(cand)
        elif selected:
            # If it looks similar but is much cleaner than the last accepted
            # candidate, update that candidate rather than creating a duplicate.
            last = selected[-1]
            cand_quality = cand.peak_score - 0.25 * cand.hand_text_overlap_penalty - 0.20 * cand.bottom_hand_penalty
            last_quality = last.peak_score - 0.25 * last.hand_text_overlap_penalty - 0.20 * last.bottom_hand_penalty
            if cand_quality > last_quality and cand.t_sec - last.t_sec < args.min_same_page_gap_sec * 2.2:
                selected[-1] = cand
        if len(selected) >= args.expected_pages:
            break

    if len(selected) < args.expected_pages:
        leftovers = [x for x in ordered if all(abs(x.t_sec - s.t_sec) >= min_gap for s in selected)]
        leftovers = sorted(leftovers, key=lambda x: (
            visual_novelty(x, selected) * 1.35
            + x.peak_score * 0.65
            - 0.25 * x.hand_text_overlap_penalty
            - 0.20 * x.bottom_hand_penalty
        ), reverse=True)
        for cand in leftovers:
            if len(selected) >= args.expected_pages:
                break
            selected.append(cand)

    selected = sorted(selected, key=lambda x: x.t_sec)
    if len(selected) > args.expected_pages:
        selected = selected[:args.expected_pages]
    return selected


def repair_close_duplicate_gaps(selected: List[FrameFeatures], valid: List[FrameFeatures], args) -> List[FrameFeatures]:
    """Replace likely temporal duplicates with candidates from missed gaps.

    The perceptual hashes are intentionally conservative because page photos can
    change a lot with perspective and hand shadows. A second useful signal is
    time: if two selected pages are much closer to each other than the typical
    page interval, and there is a large empty interval elsewhere, we likely kept
    a duplicate and missed a page.
    """
    if args.expected_pages <= 0 or len(selected) < 2 or not valid:
        return selected

    selected = sorted(selected, key=lambda x: x.t_sec)
    t0 = min(x.t_sec for x in valid)
    t1 = max(x.t_sec for x in valid)
    typical_gap = max(0.8, (t1 - t0) / max(1, args.expected_pages))
    close_thresh = max(args.min_peak_distance_sec * 1.35, typical_gap * 0.58)

    def quality(x: FrameFeatures) -> float:
        return (
            x.peak_score
            + 0.20 * x.norm_score
            - 0.28 * x.hand_text_overlap_penalty
            - 0.22 * x.bottom_hand_penalty
            - 0.16 * x.hand_penalty
        )

    # Repeat because replacing one duplicate can reveal another.
    for _ in range(args.expected_pages):
        selected = sorted(selected, key=lambda x: x.t_sec)
        close_pairs = [(i, selected[i + 1].t_sec - selected[i].t_sec) for i in range(len(selected) - 1)]
        close_pairs = [(i, g) for i, g in close_pairs if g < close_thresh]
        if not close_pairs:
            break

        # Remove the weaker page from the closest pair.
        pair_i, _ = min(close_pairs, key=lambda z: z[1])
        a, b = selected[pair_i], selected[pair_i + 1]
        # Protect the first early page. In page-turn videos the first page often
        # lasts briefly and scores poorly because a hand is already turning it,
        # but it is still a real unique page.
        early_protected = pair_i == 0 and (a.t_sec - t0) <= typical_gap * 0.75
        if early_protected:
            remove_idx = pair_i + 1
        else:
            remove_idx = pair_i if quality(a) < quality(b) else pair_i + 1
        trial = selected[:remove_idx] + selected[remove_idx + 1:]

        # Find the largest uncovered temporal gap, including beginning/end.
        anchors = [t0] + [x.t_sec for x in trial] + [t1]
        gaps = []
        for i in range(len(anchors) - 1):
            gaps.append((anchors[i + 1] - anchors[i], anchors[i], anchors[i + 1]))
        _, ga, gb = max(gaps, key=lambda z: z[0])
        margin = min(0.90, max(0.18, (gb - ga) * 0.18))
        pool = [
            x for x in valid
            if ga + margin <= x.t_sec <= gb - margin
            and all(abs(x.t_sec - y.t_sec) >= args.min_peak_distance_sec * 0.70 for y in trial)
            and x.norm_score >= max(0.0, args.min_norm_score - 0.24)
        ]
        if not pool:
            break
        novel_pool = [x for x in pool if visual_novelty(x, trial) >= 0.30]
        if novel_pool:
            pool = novel_pool
        gap_center = 0.5 * (ga + gb)
        gap_half = max(1e-6, 0.5 * (gb - ga))
        replacement = max(pool, key=lambda x: (
            quality(x)
            + 1.25 * visual_novelty(x, trial)
            + 1.70 * max(0.0, 1.0 - abs(x.t_sec - gap_center) / gap_half)
        ))
        selected = sorted(trial + [replacement], key=lambda x: x.t_sec)

    return selected


def prefer_cleaner_equivalent_winners(selected: List[FrameFeatures], valid: List[FrameFeatures], args) -> List[FrameFeatures]:
    """For each chosen page, replace it with a cleaner equivalent nearby.

    Example: the first title page may have one sharp frame with a hand and one
    slightly softer frame without a hand. For final JPEG output, the clean frame
    is better.
    """
    out: List[FrameFeatures] = []

    def final_quality(x: FrameFeatures) -> float:
        return (
            0.42 * x.peak_score
            + 0.25 * x.norm_score
            - 0.95 * x.hand_penalty
            - 0.85 * x.hand_text_overlap_penalty
            - 0.70 * x.bottom_hand_penalty
            - 0.25 * x.edge_foreground_penalty
        )

    for win in selected:
        pool = []
        for cand in valid:
            if abs(cand.t_sec - win.t_sec) > max(0.55, args.min_peak_distance_sec * 0.70):
                continue
            if cand.roi_gray is None or win.roi_gray is None:
                continue
            sim = similarity_score(cand.roi_gray, win.roi_gray)
            ham = hamming_distance(cand.roi_dhash, win.roi_dhash)
            # Use looser similarity because hand occlusion/perspective changes can
            # alter hashes even for the same physical page.
            if sim >= 0.32 or ham <= 62:
                pool.append(cand)
        out.append(max(pool, key=final_quality) if pool else win)

    # Special case for the very first page: the cleanest title/cover frame is
    # often at the start before the hand enters, but perspective changes during
    # the first page turn can make image hashes look different. If the first
    # selected winner is still early, search only before it and prefer a cleaner
    # frame with comparable text density.
    if out and valid:
        first = out[0]
        t0 = min(x.t_sec for x in valid)
        if first.t_sec - t0 <= max(2.0, args.min_same_page_gap_sec * 1.8):
            density_ref = first.text_score
            early_pool = [
                x for x in valid
                if t0 <= x.t_sec <= first.t_sec
                and abs(x.text_score - density_ref) <= max(0.018, density_ref * 0.65)
                and x.page_area_ratio >= first.page_area_ratio * 0.72
            ]
            if early_pool:
                out[0] = min(early_pool, key=lambda x: x.t_sec)

    # Keep chronological order and prevent accidental duplicates after swaps.
    cleaned: List[FrameFeatures] = []
    for cand in sorted(out, key=lambda x: x.t_sec):
        if cleaned and abs(cand.t_sec - cleaned[-1].t_sec) < args.min_peak_distance_sec * 0.55:
            cleaned[-1] = choose_between_similar(cleaned[-1], cand, 0.42)
        else:
            cleaned.append(cand)
    return cleaned


def fill_expected_pages_by_time(winners: List[FrameFeatures], valid: List[FrameFeatures], args) -> List[FrameFeatures]:
    """Recover under-represented pages when the expected count is known.

    In phone videos, a low-text page may score worse than a sharp duplicate of a
    neighboring page. When the user provides --expected-pages, use the timeline
    as an additional cue: divide the valid part of the video into chronological
    slots and pick the best clean candidate for slots that do not yet have a
    winner. This remains automatic but prevents missing sparse/low-contrast
    pages.
    """
    if args.expected_pages <= 0 or not valid:
        return winners

    selected = list(winners)
    t0 = min(x.t_sec for x in valid)
    t1 = max(x.t_sec for x in valid)
    if t1 <= t0:
        return selected

    # First, one best candidate per expected temporal slot.
    slot_candidates: List[FrameFeatures] = []
    for i in range(args.expected_pages):
        a = t0 + (t1 - t0) * i / args.expected_pages
        b = t0 + (t1 - t0) * (i + 1) / args.expected_pages
        in_slot = [x for x in valid if a <= x.t_sec <= b and x.norm_score >= max(0.0, args.min_norm_score - 0.20)]
        if not in_slot:
            continue
        best = max(in_slot, key=lambda x: (
            x.peak_score
            - 0.24 * x.hand_text_overlap_penalty
            - 0.18 * x.bottom_hand_penalty
            - 0.14 * x.hand_penalty
        ))
        slot_candidates.append(best)

    # Add missing slots if they are not visual duplicates of an already selected
    # page or if they occupy a large temporal gap.
    for cand in slot_candidates:
        if len(selected) >= args.expected_pages:
            break
        same = any(is_visually_same_page(cand, s, args) for s in selected)
        close_time = any(abs(cand.t_sec - s.t_sec) < args.min_peak_distance_sec for s in selected)
        if not same and not close_time:
            selected.append(cand)

    # If there are too many, prefer timeline coverage first and quality second.
    selected = sorted(selected, key=lambda x: x.t_sec)
    while len(selected) > args.expected_pages:
        best_remove_idx = None
        best_remove_cost = 1e9
        for i, x in enumerate(selected):
            left_gap = x.t_sec - selected[i - 1].t_sec if i > 0 else args.min_peak_distance_sec
            right_gap = selected[i + 1].t_sec - x.t_sec if i + 1 < len(selected) else args.min_peak_distance_sec
            temporal_value = min(left_gap, right_gap)
            quality = x.peak_score - 0.28 * x.hand_text_overlap_penalty - 0.20 * x.bottom_hand_penalty
            remove_cost = 0.65 * temporal_value + 0.35 * quality
            if remove_cost < best_remove_cost:
                best_remove_cost = remove_cost
                best_remove_idx = i
        if best_remove_idx is None:
            break
        del selected[best_remove_idx]

    selected = repair_close_duplicate_gaps(selected, valid, args)
    selected = prefer_cleaner_equivalent_winners(selected, valid, args)
    return sorted(selected, key=lambda x: x.t_sec)


def force_reduce(clusters: List[Cluster], expected_pages: int) -> List[Cluster]:
    if expected_pages <= 0 or len(clusters) <= expected_pages:
        return clusters
    while len(clusters) > expected_pages:
        best_pair = None
        best_score = -1e9
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                a = clusters[i].members[0]
                for m in clusters[i].members[1:]:
                    a = choose_between_similar(a, m, 0.90)
                b = clusters[j].members[0]
                for m in clusters[j].members[1:]:
                    b = choose_between_similar(b, m, 0.90)
                if a.roi_gray is None or b.roi_gray is None:
                    continue
                ham = hamming_distance(a.roi_dhash, b.roi_dhash)
                sim = similarity_score(a.roi_gray, b.roi_gray)
                score = sim - 0.03 * ham
                if score > best_score:
                    best_score = score
                    best_pair = (i, j)
        if best_pair is None:
            break
        i, j = best_pair
        clusters[i].members.extend(clusters[j].members)
        del clusters[j]
    return clusters


def process_video(args):
    video_path = Path(args.video)
    if args.output_dir:
        out_dir = Path(args.output_dir)
        dbg_dir = out_dir.with_name(out_dir.name + '_debug')
    else:
        out_dir = video_path.with_name(video_path.stem + '_pages_v12_9')
        dbg_dir = video_path.with_name(video_path.stem + '_debug_v12_9')
    if args.clean_output and out_dir.exists():
        shutil.rmtree(out_dir)
    if args.clean_output and dbg_dir.exists():
        shutil.rmtree(dbg_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    if args.debug:
        dbg_dir.mkdir(exist_ok=True, parents=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError('Could not open video')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / args.sample_fps))) if args.sample_fps > 0 else 1
    sampled_fps = fps / step

    hand_masker = HandMasker(enabled=not args.no_hands, det_conf=args.hand_det_conf, track_conf=args.hand_track_conf)
    features: List[FrameFeatures] = []
    prev_quad = None
    prev_gray_small = None
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % step != 0:
                frame_idx += 1
                continue

            quad, page_area_ratio, fill_ratio, side_label = detect_page_quad_with_side(frame, getattr(args, 'page_side', 'auto'))
            if quad is None:
                features.append(FrameFeatures(frame_idx, frame_idx / fps, None, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, None, None, None, None))
                frame_idx += 1
                continue

            stability_score = estimate_stability(prev_quad, quad, frame.shape)
            prev_quad = quad.copy()
            contact_score = border_contact_score(quad, frame.shape)
            turn_penalty = estimate_turn_penalty(frame, quad)
            curr_gray_small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (320, 320), interpolation=cv2.INTER_AREA)
            edge_motion_penalty = estimate_edge_motion_penalty(curr_gray_small, prev_gray_small)
            prev_gray_small = curr_gray_small

            warped_bgr = None
            gray = None
            roi_gray = None
            roi_dhash = None
            blur_score = 0.0
            text_score = 0.0
            hand_penalty = 0.0
            hand_text_penalty = 0.0
            fg_penalty = 0.0
            btm_hand = 0.0

            cand_deskew = 0.0
            try:
                warped = four_point_warp(frame, quad, long_side=args.long_side)
                warped = deskew_by_text_lines(warped)
                cand_deskew = float(_LAST_DESKEW_ANGLE.get('angle', 0.0))
                warped = refine_page_after_warp(warped, args)
                hand_mask = build_hand_cleanup_mask(warped, hand_masker, text_protect=False)
                hand_penalty = float(np.count_nonzero(hand_mask)) / float(hand_mask.size)
                hand_text_penalty = hand_text_overlap_penalty(warped, hand_mask)
                btm_hand = bottom_hand_penalty(warped, hand_mask)
                cleaned = cv2.inpaint(warped, hand_mask, 7, cv2.INPAINT_TELEA) if (
                    not args.no_inpaint and hand_mask_is_plausible(hand_mask)
                ) else warped
                gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
                roi_gray = roi_for_similarity(gray)
                roi_dhash = compute_dhash(roi_gray)
                blur_score = variance_of_laplacian(gray)
                text_score = count_text_density(gray)
                fg_penalty = edge_foreground_penalty(cleaned)
                warped_bgr = cleaned
            except Exception:
                pass

            features.append(FrameFeatures(
                frame_idx=frame_idx,
                t_sec=frame_idx / fps,
                quad=quad,
                page_found=True,
                page_area_ratio=page_area_ratio,
                fill_ratio=fill_ratio,
                border_contact_score=contact_score,
                stability_score=stability_score,
                blur_score=blur_score,
                text_score=text_score,
                hand_penalty=hand_penalty,
                hand_text_overlap_penalty=hand_text_penalty,
                edge_foreground_penalty=fg_penalty,
                bottom_hand_penalty=btm_hand,
                turn_penalty=turn_penalty,
                edge_motion_penalty=edge_motion_penalty,
                gray=gray,
                roi_gray=roi_gray,
                roi_dhash=roi_dhash,
                warped_bgr=warped_bgr,
                deskew_angle=cand_deskew,
            ))
            frame_idx += 1
    finally:
        cap.release()

    valid = [x for x in features if x.page_found and x.warped_bgr is not None]
    if not valid:
        hand_masker.close()
        print('No valid warped page candidates found.')
        return

    area_n = robust_norm(np.array([x.page_area_ratio for x in valid]), True)
    fill_n = robust_norm(np.array([x.fill_ratio for x in valid]), True)
    contact_n = robust_norm(np.array([x.border_contact_score for x in valid]), True)
    stab_n = robust_norm(np.array([x.stability_score for x in valid]), True)
    blur_n = robust_norm(np.array([x.blur_score for x in valid]), True)
    text_n = robust_norm(np.array([x.text_score for x in valid]), True)
    hand_n = robust_norm(np.array([x.hand_penalty for x in valid]), False)
    hand_text_n = robust_norm(np.array([x.hand_text_overlap_penalty for x in valid]), False)
    fg_n = robust_norm(np.array([x.edge_foreground_penalty for x in valid]), False)
    btm_n = robust_norm(np.array([x.bottom_hand_penalty for x in valid]), False)
    turn_n = robust_norm(np.array([x.turn_penalty for x in valid]), False)
    motion_n = robust_norm(np.array([x.edge_motion_penalty for x in valid]), False)

    for i, x in enumerate(valid):
        x.raw_score = (
            1.45 * area_n[i] +
            1.00 * fill_n[i] +
            1.00 * contact_n[i] +
            1.25 * stab_n[i] +
            2.10 * blur_n[i] +
            1.10 * text_n[i] +
            1.35 * hand_n[i] +
            1.55 * hand_text_n[i] +
            1.25 * fg_n[i] +
            0.65 * btm_n[i] +
            1.20 * turn_n[i] +
            1.10 * motion_n[i]
        ) - (
            2.15 * x.hand_penalty +
            2.40 * x.hand_text_overlap_penalty +
            1.55 * x.edge_foreground_penalty +
            0.75 * x.bottom_hand_penalty +
            0.70 * x.turn_penalty
        )

    raw_all = np.array([x.raw_score for x in valid], dtype=np.float32)
    norm_all = robust_norm(raw_all, higher_is_better=True)
    for i, x in enumerate(valid):
        x.norm_score = float(norm_all[i])

    winners_pre = select_local_peaks(features, sampled_fps, args)
    if not winners_pre:
        winners_pre = sorted(valid, key=base_preference_score, reverse=True)[:max(1, args.expected_pages or 5)]

    clusters = cluster_candidates(winners_pre, args)
    clusters = force_reduce(clusters, args.expected_pages)

    winners = []
    for cl in sorted(clusters, key=lambda c: min(m.t_sec for m in c.members)):
        best = cl.members[0]
        for m in cl.members[1:]:
            best = choose_between_similar(best, m, args.sim_thresh_merge)
        winners.append(best)

    winners = fill_expected_pages_by_time(winners, valid, args)

    if args.expected_pages > 0 and len(winners) > args.expected_pages:
        winners = sorted(winners, key=base_preference_score, reverse=True)[:args.expected_pages]
        winners = sorted(winners, key=lambda c: c.t_sec)

    # auto-smart post-selection sliver trim. Re-open the video and re-read the
    # winner frames only — winner selection already happened, so this cannot
    # change page count or ordering.
    # smart_trim_log entries: (frame_idx, label, info_dict, orig_h, orig_w)
    smart_trim_log: List[Tuple[int, str, dict, int, int]] = []
    if getattr(args, 'page_side', 'auto-smart') == 'auto-smart' and winners:
        cap2 = cv2.VideoCapture(str(video_path))
        if cap2.isOpened():
            try:
                for cand in winners:
                    if cand.quad is None or cand.warped_bgr is None:
                        continue
                    orig_h, orig_w = cand.warped_bgr.shape[:2]
                    cap2.set(cv2.CAP_PROP_POS_FRAMES, cand.frame_idx)
                    ok, frame = cap2.read()
                    if not ok or frame is None:
                        continue
                    new_warp, label, info = auto_smart_trim_winner(
                        frame,
                        cand.quad,
                        long_side=args.long_side,
                        base_warped=cand.warped_bgr,
                        min_conf=args.auto_trim_confidence,
                        max_shrink=args.auto_trim_max_shrink,
                        min_similarity=args.auto_trim_min_similarity,
                    )
                    if new_warp is not None:
                        # Re-run deskew + post-warp refine on the new crop so
                        # the output matches the rest of the pipeline.
                        try:
                            new_warp = deskew_by_text_lines(new_warp)
                            cand.deskew_angle = float(_LAST_DESKEW_ANGLE.get('angle', 0.0))
                            new_warp = refine_page_after_warp(new_warp, args)
                        except Exception:
                            pass
                        cand.warped_bgr = new_warp
                    smart_trim_log.append((cand.frame_idx, label, info, orig_h, orig_w))
            finally:
                cap2.release()

    final_dims_by_frame: dict = {}
    finalize_diag_by_frame: dict = {}
    for idx, cand in enumerate(winners, start=1):
        # v12.9: apply the refined two-pass deskew on the final winner image
        # only. Candidate scoring already used the v12.8 single-pass deskew so
        # winner selection is unchanged; this pass mops up residual tilt on
        # title/cover pages without affecting which frames win.
        try:
            refined, refined_total = deskew_by_text_lines_refined(cand.warped_bgr)
            cand.warped_bgr = refined
            cand.deskew_angle = float(cand.deskew_angle) + float(refined_total)
        except Exception:
            pass
        # V12.8: bottom dark-strip cleanup is final-output only so it does not
        # perturb candidate scoring or winner selection.
        pre_bottom_trim, bottom_band = apply_final_bottom_trim(cand.warped_bgr, args)
        final_img = safe_final_hand_cleanup(pre_bottom_trim, hand_masker, text_protect=not args.allow_text_touch)
        cleanup_info = dict(_LAST_HAND_CLEANUP_INFO)
        final_img = final_page_postprocess(final_img, args)
        enhance_mode = _LAST_ENHANCE_MODE.get('mode', '')
        deskew_angle = float(getattr(cand, 'deskew_angle', 0.0))
        fh, fw = final_img.shape[:2]
        final_dims_by_frame[cand.frame_idx] = (fh, fw)
        finalize_diag_by_frame[cand.frame_idx] = {
            'bottom_trim_px': int(bottom_band),
            'cleanup_applied': bool(cleanup_info.get('applied', False)),
            'cleanup_mask_ratio': float(cleanup_info.get('mask_ratio', 0.0)),
            'cleanup_reason': cleanup_info.get('reason', ''),
            'enhance_mode': enhance_mode,
            'deskew_angle_final': deskew_angle,
        }
        cv2.imwrite(str(out_dir / f'page_{idx:03d}.jpg'), final_img, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])

    if args.debug:
        with open(dbg_dir / 'scores.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['frame_idx', 't_sec', 'page_found', 'area', 'fill', 'contact', 'stability', 'blur', 'text', 'hand', 'hand_text_overlap', 'edge_fg', 'bottom_hand', 'turn', 'edge_motion', 'raw_score', 'norm_score', 'peak_score'])
            for x in features:
                w.writerow([
                    x.frame_idx, f'{x.t_sec:.3f}', int(x.page_found),
                    f'{x.page_area_ratio:.4f}', f'{x.fill_ratio:.4f}', f'{x.border_contact_score:.4f}',
                    f'{x.stability_score:.4f}', f'{x.blur_score:.2f}', f'{x.text_score:.5f}',
                    f'{x.hand_penalty:.5f}', f'{x.hand_text_overlap_penalty:.5f}', f'{x.edge_foreground_penalty:.5f}', f'{x.bottom_hand_penalty:.5f}',
                    f'{x.turn_penalty:.5f}', f'{x.edge_motion_penalty:.5f}',
                    f'{x.raw_score:.5f}', f'{x.norm_score:.5f}', f'{x.peak_score:.5f}'
                ])
        # smart trim entries indexed by frame_idx → (label, info, orig_h, orig_w)
        smart_by_frame = {fi: (lab, info, oh, ow) for (fi, lab, info, oh, ow) in smart_trim_log}
        page_side_arg = getattr(args, 'page_side', 'auto-smart')
        with open(dbg_dir / 'winners.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            # Structured columns + retained human-readable smart_trim string at
            # the end so existing tooling that parses smart_trim still works.
            w.writerow([
                'page', 'frame_idx', 't_sec',
                'raw_score', 'norm_score', 'peak_score',
                'hand', 'hand_text_overlap', 'bottom_hand', 'blur', 'text',
                'page_side_mode',
                'trim_applied', 'trim_skip_reason', 'seam_side',
                'seam_confidence', 'trim_fraction', 'shrink', 'dhash_similarity',
                'orig_w', 'orig_h', 'final_w', 'final_h',
                'bottom_trim_px', 'cleanup_applied', 'cleanup_mask_ratio', 'cleanup_reason', 'enhance_mode',
                'deskew_angle_final',
                'smart_trim',
            ])
            for idx, x in enumerate(winners, start=1):
                entry = smart_by_frame.get(x.frame_idx)
                if entry is not None:
                    label, info, orig_h, orig_w = entry
                    applied = '1' if info.get('applied') else '0'
                    skip = info.get('skip_reason', '') or ''
                    seam_side = info.get('seam_side', '') or ''
                    sc = info.get('seam_confidence', float('nan'))
                    tf = info.get('trim_fraction', float('nan'))
                    sh = info.get('shrink', float('nan'))
                    sim = info.get('dhash_similarity', float('nan'))
                    sc_s = '' if sc != sc else f'{sc:.3f}'  # NaN check
                    tf_s = '' if tf != tf else f'{tf:.3f}'
                    sh_s = '' if sh != sh else f'{sh:.4f}'
                    sim_s = '' if sim != sim else f'{sim:.3f}'
                    smart_label = label
                else:
                    # No smart-trim attempt was made (mode != auto-smart, or
                    # winner had no quad/warped). Record the mode so consumers
                    # can disambiguate "skipped" vs "not attempted".
                    applied = '0'
                    skip = 'not-attempted'
                    seam_side = ''
                    sc_s = tf_s = sh_s = sim_s = ''
                    orig_h, orig_w = 0, 0
                    if x.warped_bgr is not None:
                        orig_h, orig_w = x.warped_bgr.shape[:2]
                    smart_label = ''
                fh, fw = final_dims_by_frame.get(x.frame_idx, (0, 0))
                fdiag = finalize_diag_by_frame.get(x.frame_idx, {})
                bt_px = int(fdiag.get('bottom_trim_px', 0))
                cu_app = '1' if fdiag.get('cleanup_applied', False) else '0'
                cu_mr = f"{float(fdiag.get('cleanup_mask_ratio', 0.0)):.4f}"
                cu_rs = fdiag.get('cleanup_reason', '') or ''
                en_md = fdiag.get('enhance_mode', '') or ''
                dsk = f"{float(fdiag.get('deskew_angle_final', 0.0)):.3f}"
                w.writerow([
                    idx, x.frame_idx, f'{x.t_sec:.3f}',
                    f'{x.raw_score:.5f}', f'{x.norm_score:.5f}', f'{x.peak_score:.5f}',
                    f'{x.hand_penalty:.5f}', f'{x.hand_text_overlap_penalty:.5f}', f'{x.bottom_hand_penalty:.5f}',
                    f'{x.blur_score:.2f}', f'{x.text_score:.5f}',
                    page_side_arg,
                    applied, skip, seam_side,
                    sc_s, tf_s, sh_s, sim_s,
                    orig_w, orig_h, fw, fh,
                    bt_px, cu_app, cu_mr, cu_rs, en_md,
                    dsk,
                    smart_label,
                ])
        print(f'Debug files: {dbg_dir}')

    hand_masker.close()
    print(f'Saved {len(winners)} unique pages to: {out_dir}')
    print(f'Valid warped candidates: {len(valid)}, peak winners before clustering: {len(winners_pre)}, clusters: {len(clusters)}')


def iter_image_paths(path: Path) -> List[Path]:
    exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'}
    if path.is_file() and path.suffix.lower() in exts:
        return [path]
    if path.is_dir():
        return sorted([p for p in path.iterdir() if p.suffix.lower() in exts])
    return []


def process_images(args):
    """Process still frames/images with the same page cleanup pipeline.

    This is primarily useful for debugging a video run: pass the saved candidate
    JPEGs or frame exports and inspect how page detection and hand cleanup work
    without re-reading the whole video.
    """
    src = Path(args.images)
    paths = iter_image_paths(src)
    if not paths:
        raise RuntimeError(f'No images found: {src}')
    out_dir = Path(args.output_dir) if args.output_dir else src.with_name(src.stem + '_rectified_v12_2')
    if out_dir.exists() and args.clean_output:
        shutil.rmtree(out_dir)
    out_dir.mkdir(exist_ok=True)

    hand_masker = HandMasker(enabled=not args.no_hands, det_conf=args.hand_det_conf, track_conf=args.hand_track_conf)
    saved = 0
    try:
        for pth in paths:
            frame = cv2.imread(str(pth))
            if frame is None:
                print(f'Skip unreadable image: {pth}')
                continue
            quad, area_ratio, fill_ratio, side_label = detect_page_quad_with_side(frame, getattr(args, 'page_side', 'auto'))
            if quad is None:
                print(f'Skip no page: {pth.name}')
                continue
            warped = four_point_warp(frame, quad, long_side=args.long_side)
            warped = deskew_by_text_lines(warped)
            warped = refine_page_after_warp(warped, args)
            warped, _ = apply_final_bottom_trim(warped, args)
            final_img = safe_final_hand_cleanup(warped, hand_masker, text_protect=not args.allow_text_touch)
            final_img = final_page_postprocess(final_img, args)
            saved += 1
            cv2.imwrite(str(out_dir / f'page_{saved:03d}.jpg'), final_img, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
            if args.debug:
                dbg = frame.copy()
                cv2.polylines(dbg, [quad.astype(np.int32)], True, (0, 0, 255), 4)
                cv2.imwrite(str(out_dir / f'debug_{saved:03d}_{pth.stem}.jpg'), dbg, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                print(f'{pth.name}: area={area_ratio:.3f}, fill={fill_ratio:.3f}, side={side_label}')
    finally:
        hand_masker.close()
    print(f'Saved {saved} rectified pages to: {out_dir}')


def build_parser():
    p = argparse.ArgumentParser(
        description=(
            'Extract unique book pages from a video '
            '(V12.8: V12.7 pipeline plus a conservative bottom dark-strip cleanup, '
            'a stricter hand-mask plausibility/brightness gate to avoid inpainting '
            'gray blotches on sparse pages, and a sparse-page enhancement mode that '
            'skips CLAHE to suppress back-of-page bleed-through artifacts). '
            'Use --page-side auto for the previous v12.5/v12.6 default behavior, '
            'or --page-side full to disable all page-side trimming.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Page-side modes:\n'
            '  auto-smart (default, V12.7)  Conservative spread split during detection,\n'
            '                               then a stability-gated post-selection sliver\n'
            '                               trim on each winner. Cleans inner-spine slivers\n'
            '                               on single-page shots without changing winner\n'
            '                               selection. Recommended for most book videos.\n'
            '  auto                         The v12.5/v12.6 default. Same conservative\n'
            '                               spread split, but no sliver trim. Use this if\n'
            '                               you want bit-identical legacy behavior or if\n'
            '                               the smart trim is misbehaving on your input.\n'
            '  right / left                 Force the right or left half of every detected\n'
            '                               two-page spread. Use when the camera framing\n'
            '                               is stable and you only photograph one side.\n'
            '  full                         Keep the entire detected quad without any\n'
            '                               page-side trimming. Use to debug detection or\n'
            '                               when your video is already cropped to one page.\n'
        ),
    )
    p.add_argument('video', nargs='?', help='Input video file. You can drag-and-drop MOV/MP4 here.')
    p.add_argument('--images', help='Debug mode: process a still image or a folder of images instead of a video.')
    p.add_argument('--output-dir', help='Optional output folder.')
    p.add_argument('--clean-output', action='store_true', help='Delete output folder before writing new files.')
    p.add_argument('--sample-fps', type=float, default=2.0)
    p.add_argument('--expected-pages', type=int, default=0)
    p.add_argument('--long-side', type=int, default=1800)
    p.add_argument('--jpeg-quality', type=int, default=95)
    p.add_argument('--peak-window-sec', type=float, default=0.8)
    p.add_argument('--min-peak-distance-sec', type=float, default=0.9)
    p.add_argument('--min-norm-score', type=float, default=0.28)
    p.add_argument('--hash-thresh-merge', type=int, default=11)
    p.add_argument('--sim-thresh-merge', type=float, default=0.89)
    p.add_argument('--min-same-page-gap-sec', type=float, default=1.3)
    p.add_argument('--hand-det-conf', type=float, default=0.45)
    p.add_argument('--hand-track-conf', type=float, default=0.45)
    p.add_argument('--no-hands', action='store_true')
    p.add_argument('--no-inpaint', action='store_true')
    p.add_argument('--no-enhance', action='store_true', help='Disable final scan-like lighting/contrast enhancement.')
    p.add_argument('--no-refine-crop', action='store_true', help='Disable second-pass crop refinement after perspective warp.')
    p.add_argument('--no-bottom-trim', action='store_true', help='Disable conservative bottom dark-strip cleanup (V12.8).')
    p.add_argument('--bottom-trim-max-frac', type=float, default=0.05,
                   help='Cap on the fraction of page height removable by the V12.8 bottom dark-strip cleanup. Default 0.05.')
    p.add_argument('--page-side', choices=['auto', 'auto-smart', 'right', 'left', 'full'], default='auto-smart',
                   help=('Pre-warp page selection. Default is auto-smart (V12.7): conservative '
                         'spread split during detection plus a safe post-selection sliver trim on '
                         'winners that pass seam-confidence, shrink, and dHash similarity gates. '
                         'Pass --page-side auto for the previous v12.5/v12.6 default (no sliver '
                         'trim, bit-identical legacy behavior). Use right/left to force a side or '
                         'full to keep the whole detected quad. See the epilog for guidance.'))
    p.add_argument('--auto-trim-confidence', type=float, default=0.62,
                   help='Minimum spine-seam confidence required for auto-smart sliver trim on a winner frame. Only used when --page-side is auto-smart.')
    p.add_argument('--auto-trim-max-shrink', type=float, default=0.10,
                   help='Maximum fractional area shrink allowed when applying auto-smart sliver trim (guards against eating real page content). Only used when --page-side is auto-smart.')
    p.add_argument('--auto-trim-min-similarity', type=float, default=0.65,
                   help='Required dHash-based similarity between original and trimmed warps for auto-smart trim to be accepted (stability gate). A clean sliver trim typically scores 0.7-0.85; jumping to a different page scores well under 0.5. Only used when --page-side is auto-smart.')
    p.add_argument('--allow-text-touch', action='store_true', help='Allow cleanup mask to affect text regions (off by default).')
    p.add_argument('--debug', action='store_true')
    return p


def main():
    args = build_parser().parse_args()
    try:
        if args.images:
            process_images(args)
        elif args.video:
            process_video(args)
        else:
            raise RuntimeError('Pass a video path, or use --images IMAGE_OR_FOLDER for still-image debugging.')
    except KeyboardInterrupt:
        print('Interrupted.')
        sys.exit(130)
    except Exception as e:
        print(f'Error: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
