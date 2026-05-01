# IMG_4899 v15.11 Run Report

**Video:** `/home/user/workspace/Book_Scanner/IMG_4899.MOV` — 13.44 s, 403 frames @ 29.99 fps
**Script:** `/home/user/workspace/extract_book_pages_v15_11.py`
**Truth:** book pages 187, 188, 189, 190, 191 (5 distinct pages)

## Summary

v15.11 implements two narrow, additive patches on top of v15.10 (both default-on, opt-out via flags):

* **Patch A** — `_v155_adjacent_same_page` Path D ("blur-asymmetry rescue").
* **Patch B** — local-peak picker blur-aware tiebreak.

Both were targeted at the IMG_4899 v15.10 review diagnostic.

## Result on IMG_4899 (default mode)

5 pages saved, no duplicate, page p190 now uses the sharp f315 capture instead of motion-blurred f330.

| Output    | Frame | t (s)  | Book page | blur   | notes                                             |
|-----------|------:|-------:|-----------|-------:|---------------------------------------------------|
| page_001  |     0 |  0.000 | p187      | 619.21 |                                                   |
| page_002  |   120 |  4.001 | p188      | 467.94 | absorbed f150 via Path D blur_asym (kept sharper) |
| page_003  |   225 |  7.503 | p189      | 638.55 |                                                   |
| page_004  |   315 | 10.504 | p190      | 529.97 | promoted by blur tiebreak over f330 (blur=60.77)  |
| page_005  |   375 | 12.505 | p191      | 601.97 |                                                   |

Run-log evidence:

```
[v15.5] adjacent-dedup: removed winner frames [150] (checked 3 pair(s), merged 1)
[v15.5]   keep frame 120 over frame 150: blur_asym(dt=1.00s,blur_ratio=0.14,sim=0.23,warp=0.69,prof=0.20) (dt=1.00s, delta_q=+0.396)
```

The blur-aware peak picker promoted f315 directly at the local-max stage; winners.csv shows `peak_score=0.90735, blur=529.97` for page_004.

## Patch detail

### Patch A — Path D in `_v155_adjacent_same_page`

Adds a fourth same-page acceptance path that fires when geometric/text gates fail because of motion-blur asymmetry. The text-density early-return is bypassed only when the path is eligible (close in time, neither winner showing a turn-in-progress, blur ratio min/max ≤ 0.25). Final acceptance still requires a non-trivial agreement signal (warp ≥ 0.40 OR profile ≥ 0.05 OR sim ≥ 0.03), and the v15.6 footer guard still blocks distinct folios.

New flags (all opt-out):

* `--v155-adj-dedup-blur-asym-dt-sec` (default `min_peak_distance_sec * 1.5`)
* `--v155-adj-dedup-blur-asym-max-ratio` (default `0.25`; set 0 to disable)
* `--v155-adj-dedup-blur-asym-turn-max` (default `0.65`)
* `--v155-adj-dedup-blur-asym-warp-floor` (default `0.40`)
* `--v155-adj-dedup-blur-asym-profile-floor` (default `0.05`)
* `--v155-adj-dedup-blur-asym-sim-floor` (default `0.03`)

### Patch B — peak picker blur tiebreak

In the local-peak selection loop (around old line 2492), the strict `x.peak_score >= max(neighborhood)` rule is replaced with an effective-local-peak test: a candidate that is barely beaten by a neighbour (Δpeak ≤ `peak_tie_eps`) but is dramatically sharper (own blur ≥ `peak_tie_blur_min` AND ≥ `peak_tie_blur_factor` × neighbour's blur) is still accepted; conversely, a candidate that barely beats a sharper neighbour yields. This both demotes the blurry would-be winner and promotes the sharp neighbour in one pass.

New flags (all opt-out):

* `--peak-tie-eps` (default `0.01`; set 0 to disable)
* `--peak-tie-blur-min` (default `200.0`)
* `--peak-tie-blur-factor` (default `2.0`)

## Regression results (default mode, no extra flags)

| Video    | Expected pages | v15.11 pages | Status |
|----------|---------------:|-------------:|--------|
| IMG_4899 | 5              | 5            | ✅ fixed (was 6 with blurred duplicate) |
| IMG_4892 | 6              | 6            | ✅ unchanged |
| IMG_4890 | 5              | 5            | ✅ unchanged |
| IMG_4886 | 7              | 7            | ✅ unchanged |

Path D fired only on IMG_4899 (1 merge: f120↔f150). The blur-aware peak tiebreak fired only on IMG_4899 (f330 yielded to f315). Other corpora's local maxima are not within 1% of a much sharper neighbour, so the tiebreak is dormant.

## Files

* Source: `extract_book_pages_v15_11.py` (placed at workspace root and inside the result archive under `Book_Scanner/`).
* Pages: `v15_11_IMG_4899/page_001.jpg` … `page_005.jpg`.
* Debug: `v15_11_IMG_4899_debug/winners.csv`, `scores.csv`, `prefilter.csv`, `timings.json`, `calibration.json`.
* Run log: `v15_11_IMG_4899_run.log`.
