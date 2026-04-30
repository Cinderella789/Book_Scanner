# IMG_4892 v15.9 — Tail-Page Recovery Report

## TL;DR
v15.9 adds a narrow head/tail coverage rescue inside the v15.1 fast prefilter
that fires only when (a) the global top-K leaves a contiguous unselected gap
touching position 0 or n-1, AND (b) the gap contains at least one frame whose
prefilter-blur is materially sharper than the nearest selected frame. With this
fix, IMG_4892 default mode now recovers the 6th colophon/barcode page that
v15.8 silently dropped, without regressing IMG_4890 or IMG_4886.

## Default-mode results (v15.9 vs v15.8)

| video      | v15.8 pages | v15.9 pages | tail t_sec | last winner v15.8 | last winner v15.9 |
|------------|-------------|-------------|-----------|-------------------|-------------------|
| IMG_4892   | 5           | **6**       | 18.5 s    | frame 480 (16.0s) | frame 555 (18.5s) |
| IMG_4890   | 5           | 5           | 11.0 s    | frame 330 (11.0s) | frame 330 (11.0s) |
| IMG_4886   | 7           | 7           | 19.5 s    | frame 585 (19.5s) | frame 585 (19.5s) |

## IMG_4892 winners (v15.9)
| page | frame | t_sec |
|------|-------|------:|
| 1    | 0     | 0.000 |
| 2    | 150   | 5.000 |
| 3    | 270   | 9.000 |
| 4    | 345   | 11.500 |
| 5    | 480   | 16.000 |
| **6**| **555** | **18.500** |

Page 6 = colophon page with ISBN barcode and "16+" mark, captured at frame 555.
Frame 540 (sharpest in the diagnostic, blur=4083) is also in the candidate pool
via neighbourhood expansion but the cluster ranking selected frame 555
(blur=4087, slightly sharper still).

## What changed in `_v150_run_prefilter`

A single new block, gated on `expected_pages == 0` (so `--expected-pages` runs
are unaffected — they already use v15.3 per-slot retention):

1. After global top-K selection, find a **head gap** [0, first_selected) and
   **tail gap** (last_selected, n-1] of unselected sampled positions. Both
   must be at least `gap_thr = max(2, round(n / (top_k * 2)))` positions long.
2. Reject the gap unless its best prefilter blur is `>= 2.5x` the blur of the
   nearest already-selected frame. This is the *distinctness* test that
   separates "true missed page" from "more of the same content".
3. If the gap qualifies, add the position with the highest smoothed composite,
   tied by distance from the gap's far edge (so we sample as deep into the
   under-covered region as quality allows).

The change is intentionally one-sided: only edge gaps, only with a blur-ratio
distinctness gate. Mid-video gaps are never rescued because those represent
the expected behaviour of global top-K picking the strongest cluster.

## Validation diagnostics

`prefilter_diagnostics` in `timings.json` now includes
`v159_default_coverage_added` and `v159_default_coverage_positions`:

| video    | v159_default_coverage_added | positions added |
|----------|----------------------------:|-----------------|
| IMG_4892 | 1                           | tail position (frame 555) |
| IMG_4890 | 0                           | (blur-ratio gate held; tail not distinctive enough) |
| IMG_4886 | 0                           | (no qualifying edge gap)  |

The fix fired exactly where the diagnostic said it needed to fire and
nowhere else.

## Runtime
v15.9 default-mode runtime is essentially unchanged from v15.8: the rescue
adds at most one extra prefilter position, expanded to ~3 extras after
neighbourhood expansion. Wall-clock impact is dominated by the existing
parallel full-process pass (15-20 s).

## Files
- `extract_book_pages_v15_9.py` — patched script
- `v15_9_IMG_4892/page_001..006.jpg` — 6 extracted pages
- `v15_9_IMG_4892_debug/{prefilter,scores,winners}.csv`,
  `v15_9_IMG_4892_debug/timings.json`,
  `v15_9_IMG_4892_debug/calibration.json` — debug artefacts
- `v15_9_IMG_4892_run.log` — full run log
