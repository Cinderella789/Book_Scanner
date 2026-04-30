# v15.8 — within-region finger-relief refinement

## Goal

User review of v15.7 IMG_4890 default reported that `page_003.jpg`
(frame 165) shows a finger occluding the bottom of the warped page.
The user noted that a cleaner same-page candidate exists and asked
for a *generic* (no hardcoded video / frame / page count) within-
peak-region refinement that can replace such a winner with a cleaner
same-page candidate, while:

* preserving the v15.7 steal_zone guard (no cross-region replacement);
* preserving page count and coverage on IMG_4890 (5 pages);
* not regressing IMG_4886 default (7 pages);
* not adding a broad full-video pass.

## What v15.8 adds (additive on v15.7)

A new bounded pass, `finger_relief_pass`, that runs after
`quality_refinement_pass` and before `v155_adjacent_winner_dedup`.
Code-wise it is one new function plus a single call site; argparse
gains seven `--finger-relief-*` flags (all defaulted ON / sensible).
No file structure changes.

The pass acts only on winners with high `finger_penalty`
(v13.5 skin + bottom-skin metric on the warped page; default floor
`--finger-relief-finger-floor 0.30`) OR whose
`_v154_winner_is_suspicious` reason string contains a `cvs=` /
`finger=` token. Other suspicion modes (hand_text overlap, large
deskew, turn) are intentionally NOT eligible — those do not imply
finger occlusion of the warped page and shouldn't trigger a
finger-clean replacement search.

For each finger-suspicious winner `w`:

1. **Pool**: temporal window of `--quality-refine-window-sec` (2.5s by
   default) around `w`. The v15.7 **steal_zone guard is preserved**:
   any candidate sitting inside another existing winner's
   `min_peak_distance_sec` zone AND closer to that other winner than
   to `w` is excluded. This is what makes the pass strictly within-
   region.
2. **Same-page identity**: same gate v15.4 uses
   (`_v154_same_page_gate` — combines v13.4 relaxed test and v13.2
   alt-search test, with a temporal-buddy fallback for hand-occluded
   ROIs).
3. **Geometric / motion gates** (kept): `|deskew| <=
   quality-refine-deskew-max`, `turn_penalty <= 0.85`,
   `stability >= --quality-refine-stability-min`,
   `edge_motion <= --quality-refine-edge-motion-max`.
4. **Loosened blur / raw_score gates**: blur floor drops from 0.55
   (v15.4 default) to `--finger-relief-blur-frac` (default 0.30) of
   the winner's blur, with an absolute floor of 100; raw_score drop
   ceiling rises from 1.5 to `--finger-relief-raw-score-max-drop`
   (default 2.5). These let a slightly motion-blurred but visibly
   cleaner same-page candidate compete.
5. **Finger-specific gain test**: candidate must reduce
   `finger_penalty` by at least
   `--finger-relief-min-finger-improve` (default 0.20). The v15.4
   global `_v154_refine_score` is used only as a **regression check**
   (candidate may not regress overall by more than
   `--finger-relief-max-overall-regress`, default −0.10). This is
   what unlocks finger-driven replacements that v15.4 rejects with
   `insufficient_gain`.
6. **Twin-of-other-winner guard**: even after passing all the above,
   the candidate must NOT be near-identical to any other existing
   winner. Three signals are checked: SSIM/Hamming, the
   `_v154_same_page_gate` itself, and the v13.4 warp-thumbnail
   correlation. This anticipates the v15.5 adjacent-dedup merge
   logic and structurally prevents finger-relief from collapsing
   two regions into one (which would lower the page count).

## How v15.8 keeps page count safe

The page-count-preserving property is a structural consequence of
two things:

* **Steal_zone guard** (v15.7): a candidate closer to a *different*
  existing winner than to `w` cannot enter the pool.
* **Twin-of-other-winner guard** (v15.8): a candidate that looks
  like another existing winner under the same merge logic that
  v15.5 adjacent-dedup uses cannot be picked. Even if v15.7's
  steal_zone is not triggered (the candidate is between two
  winners and equidistant), v15.5 dedup would have collapsed the
  pair after the swap, so the swap is rejected up-front.

If both guards reject every candidate, the original winner is
retained (the pass is non-destructive). The diagnostic in
`winners.csv` records the rejection counts so the operator can see
why no swap occurred.

## Diagnostic surface (additive to v15.7)

`winners.csv` gains the following columns:

* `v158_finger_relief_applied` (0/1)
* `v158_finger_relief_reason` (e.g. `not-finger-suspicious`,
  `no-finger-improvement(checked=N,min_finger_gain=...,
  max_regress=...,rej[insufficient_finger_gain=...,
  twin_of_other_winner=...,...])`,
  `replaced(finger A->B,d_finger=+...,d_overall=+...,checked=N)`)
* `v158_finger_relief_orig_frame`, `v158_finger_relief_new_frame`
* `v158_finger_relief_orig_finger`, `v158_finger_relief_new_finger`
* `v158_finger_relief_orig_cvs`, `v158_finger_relief_new_cvs`
* `v158_finger_relief_finger_delta`,
  `v158_finger_relief_overall_delta`
* `v158_finger_relief_candidates_checked`,
  `v158_finger_relief_candidates_examined`

`timings.json` gains a `finger_relief_summary` block with the
list of replacements (original/replacement frame, finger penalties,
overall delta, reason).

## Results

### IMG_4890 (default, no `--expected-pages`)

```
Saved 5 unique pages to: v15_8_IMG_4890
Valid warped candidates: 19, peak winners before clustering: 5, clusters: 5
Winners (frame_idx, t_sec): 30 (1.0), 105 (3.5), 165 (5.5), 240 (8.0), 330 (11.0)
[v15.8] finger-relief: 1 finger-suspicious winner(s), 5 candidates examined,
        3 strict same-page, 0 replacement(s)
[v15.5] adjacent-dedup: no merges (checked 4 pair(s))
```

Page count: **5** (unchanged from v15.7). All five region winners are
preserved.

### IMG_4890 — diagnosis of page_003 (frame 165)

`finger_penalty(165) = 0.5747`. Finger-relief examined the 5 valid
candidates within the steal_zone-filtered window:

| frame |  blur | stability | finger_penalty | same-page? | rejection |
|-------|-------|-----------|----------------|------------|-----------|
|  135  |  339  |  0.665    |  0.06          | yes (buddy)| `twin_of_other_winner` (warp 0.83 vs frame 105) |
|  150  |  870  |  0.855    |  0.61          | yes        | `insufficient_finger_gain` (delta -0.07) |
|  180  |  226  |  0.088    | 1.00           | yes        | `low_stability` (0.088 < 0.55 floor) |
|  195  |   11  |  0.167    | 1.00           | (skipped — turning, blur=11, low_stability) |
|  210  |  256  |  0.304    | 0.33           | (skipped — different content / low_stability) |

The only candidate with materially lower `finger_penalty` is frame
135, but frame 135 is visually near-identical to the page_002
winner (frame 105) under the v15.5 dedup gate
(warp_thumb_match = 0.83, prof = 0.50, text_rel = 0.05). Replacing
frame 165 with 135 would have caused v15.5 to merge frames 105 and
135 — losing region C and dropping the page count to 4. The v15.8
twin-of-other-winner guard correctly anticipates this and rejects
frame 135. Within-region (region C) only, no candidate provides a
material `finger_penalty` improvement, so the winner is retained.
This is the correct, safe result — exactly the "if safe" condition
the user described.

### IMG_4886 (default) — regression check

```
Saved 7 unique pages to: v15_8_IMG_4886
[v15.4] quality refinement: 9 suspicious winner(s), 36 candidates examined,
        30 same-page, 1 replacement(s)
[v15.4]   frame 30 -> 15: replaced(...)
[v15.8] finger-relief: 1 finger-suspicious winner(s), 2 candidates examined,
        2 strict same-page, 0 replacement(s)
[v15.5] adjacent-dedup: removed winner frames [420, 480]
```

Page count: **7** (unchanged from v15.7).

### Runtime

```
v158_finger_relief         0.110s (IMG_4890), 0.015s (IMG_4886)
total                     23.96s (IMG_4890), 50.98s (IMG_4886)
```

The new pass adds <0.15s per video (well under 1% of total runtime).
No additional video pass. Only existing decoded `warped_bgr` data
and the bounded valid-candidate pool are used.

## Files in zip

* `extract_book_pages_v15_8.py` — patched script.
* `v15_8_IMG_4890/page_001..005.jpg` — 5 saved pages (page_003 still
  contains a finger; no safe within-region replacement available).
* `v15_8_IMG_4890_debug/{prefilter.csv, scores.csv, winners.csv,
  calibration.json, timings.json}` — debug artefacts (`--debug
  --profile`); `winners.csv` includes the v15.8 finger-relief
  diagnostic columns.
* `v15_8_IMG_4890_run.log` — full stdout from the IMG_4890 run.
* `v15_8_IMG_4886/page_001..007.jpg` — IMG_4886 regression output
  (7 pages, no regression).
* `v15_8_IMG_4886_debug/...` — debug artefacts for IMG_4886.
* `v15_8_IMG_4886_run.log` — IMG_4886 stdout.
* `v15_8_report.md` — this report.
