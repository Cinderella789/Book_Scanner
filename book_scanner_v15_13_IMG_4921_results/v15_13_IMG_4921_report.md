# Book Scanner v15.13 — IMG_4921 Run Report

**Script:** `extract_book_pages_v15_13.py`
**Predecessor:** `extract_book_pages_v15_12.py`
**Diagnostic source:** v15.12 IMG_4921 artifacts + user's visual review (animal-illustration section is two distinct pages, not one)
**Run date:** 2026-05-02

## Diagnosis (using v15.12 artifacts and direct frame inspection)

I rendered IMG_4921 frames at the suspect time positions (0, 30, 60, 75, 90, 105, 150, 195, 210, 240, 375, 405, 420, 435, 450, 465, 495, 510, 525, 555, 570) and matched them against `v15_12_IMG_4921_debug/{prefilter,scores,winners}.csv`. The actual physical-page sequence in the video is:

| seq | t_sec     | content                                          | v15.12 outcome              |
|-----|-----------|--------------------------------------------------|-----------------------------|
| 1   | 0.0       | **"Sporting Seasons"** — illustrated calendar    | **MISSED**                  |
| 2   | 1.0–3.5   | "A Father's Advice" — illustrated poem           | kept (frame 60)             |
| 3   | 4.5–6.0   | Blank verso                                      | kept by v15.12 blank rescue |
| 4   | 6.5–11.0  | Half-title "Birds, Boots & Barrels" + handwritten Russian inscription | kept (frame 240) |
| 5   | 12.0–13.0 | Full title page                                  | kept (frame 375)            |
| 6   | 14.5–17.0 | Copyright / dedication                           | kept (frame 495)            |
| 7   | 17.5–19.0 | Contents list                                    | kept (frame 570)            |

So the v15.12 output was 6 pages but should be **7**. The user's "two animal-illustration pages" comment maps to: f0 (Sporting Seasons) + f60 (Father's Advice). They are visually unrelated (dHash hamming = 127 of 128 bits, roi-similarity = 0.16) but v15.12 only kept the second.

The user also mentioned "another near-blank/low-contrast page after the title around t≈14–14.5s." On direct frame inspection that span is the title→copyright/contents flip itself; the prefilter rows at f420/f435 already show the copyright page settling on the right side. No physically distinct post-title flyleaf was found in the captured footage, so v15.13 does not synthesise one — adding a rescue path for this would need fabricating a page that is not in the video, which is out of scope.

### Why was Sporting Seasons (frame 0) missed?

`v15_12_IMG_4921_debug/scores.csv` row 0:
- `page_found=1, area=0.78, fill=0.997, blur=101, text=0.046, hand=0.152` — a real settled page detection, **not** a hand/turn frame.
- BUT `stability=0.500` (initialisation default — there is no prior frame to stabilise against), `hand_text_overlap=1.000`, `bottom_hand=1.000` — these last two are degenerate first-frame outputs of the existing penalty estimators when there is no temporal context.
- `peak_score = (0.64·norm + 0.36·smooth) · (0.84 + 0.16·clean_bonus) = 0.159` because `clean_bonus` collapses with `hand_text_overlap=1` and `bottom_hand=1`.
- The neighbour at frame 30 has `peak_score=0.945` (Father's Advice, sharp). They are 1.0 s apart, well inside `min_peak_distance_sec=2.0`, so `select_local_peaks` picks frame 30 (and later 60) and discards frame 0.
- Nothing downstream looks for a real distinct page that was suppressed at the peak picker; the v15.12 blank-rescue covers only inter-winner gaps with the strict "settled blank paper" signature (text~0, edges~0), which a 0.046-text illustrated page does not match.

So the failure mode is generic: **a real page that is the very first sampled frame can be peak-suppressed by degenerate first-frame penalty defaults, and there is no recovery path.**

## v15.13 patch (additive on v15.12; no hardcoded frames or page counts)

### `_v1513_leading_edge_distinct_page_rescue` (default-on, opt-out)

After the v15.12 blank-rescue stage and before smart_trim, inspect the first kept winner. If `t_first >= --v1513-leading-edge-min-gap-sec` (default 1.5 s), scan the full processed candidate list (`features`) for a frame strictly before the first winner that satisfies all of:

| gate                         | default   | rationale                                     |
|------------------------------|-----------|-----------------------------------------------|
| `t_first - t >= 1.5 s`       | 1.5       | excludes anything inside `min_peak_distance`  |
| `page_found=1`               | —         | must already pass the contour scorer          |
| `area >= 0.55, fill >= 0.85` | 0.55/0.85 | settled page contour                          |
| `edge_motion_penalty <= 0.5` | 0.5       | not a page-turn frame                         |
| `turn_penalty <= 0.5`        | 0.5       | not a page-turn frame                         |
| `blur_score >= 50`           | 50        | not a smear                                   |
| `text_score >= 0.005`        | 0.005     | not blank — v15.12 already handles blanks     |
| **distinctness vs first winner** | ham≥22 OR sim≤0.55 | provably a different physical page |

The candidate set is sorted by sharpness (then lower edge_motion, lower turn, earlier t_sec) and the top is **prepended** as a winner. Winners are then re-sorted by `t_sec` so downstream stages (smart_trim, page-numbering) see the correct order. The candidate already has `warped_bgr` from the parallel full-process stage, so no re-decode is needed.

The distinctness gate is the critical safety net: the candidate must be *visibly unrelated* to the existing first winner (dHash hamming ≥ 22 of 128 OR ROI structural similarity ≤ 0.55). On IMG_4921 frame 0 vs frame 60 we measure **ham=127 of 128 and sim=0.16** — unambiguously different physical pages.

Skipped entirely when `--expected-pages` is set (the user has authored the count). Opt out with `--no-v1513-leading-edge-rescue`.

## IMG_4921 result

```
[v15.12] blank-front-matter-rescue: inserted 1 blank page(s) in inter-winner gap(s)
[v15.12]   frame 150 @t=5.00s: blur=2,text=0.0019,gap_dt=6.00s
[v15.13] leading-edge-rescue: prepended frame 0 @t=0.00s (blur=101,text=0.0461,ham=127,sim=0.16)
Saved 7 unique pages to: v15_13_IMG_4921
```

| page | frame | t_sec | content                                |
|------|-------|-------|----------------------------------------|
| 001  | 0     | 0.00  | **Sporting Seasons (NEW — animal page A)** |
| 002  | 60    | 2.00  | A Father's Advice (animal page B)      |
| 003  | 150   | 5.00  | Blank verso                            |
| 004  | 240   | 8.00  | Half-title + handwritten inscription   |
| 005  | 375   | 12.50 | Title page                             |
| 006  | 495   | 16.50 | Copyright / dedication                 |
| 007  | 570   | 19.00 | Contents                               |

## Regression results (default mode, no `--expected-pages`)

| video      | expected | v15.12 | v15.13 | leading-edge fired? | status |
|------------|----------|--------|--------|---------------------|--------|
| IMG_4921   | 7 (was 6)| 6      | **7**  | yes (frame 0)       | FIXED  |
| IMG_4899   | 5        | 5      | 5      | no                  | OK     |
| IMG_4892   | 6        | 6      | 6      | no                  | OK     |
| IMG_4890   | 5        | 5      | 5      | no                  | OK     |
| IMG_4886   | 7        | 7      | 7      | no                  | OK     |

The leading-edge rescue correctly does not fire on the four regression videos because their first kept winners are already at small `t_sec` and/or there is no qualifying distinct candidate before them.

## Note on the 8th-page question

The user's count of "8 pages including both animal pages and both blank/low-text pages" implied a second blank/low-text page near t≈14–14.5 s. Direct frame inspection at f420/f435/f450 shows those frames are mid-flip (title → copyright) and the page that is settled at f450/495 is the printed copyright/dedication — there is no physically separate post-title flyleaf in this video. v15.13 therefore reaches **7** pages on IMG_4921, which is the count the actual capture supports. If the user's source-of-truth book has an 8th page that simply was not captured in this video, it cannot be recovered without fabricating content.

The scaffolding for adding a second rescue path (extending the v15.12 blank rescue to also bracket the *front* of the video, not just inter-winner gaps) is a one-line change — opt-in flag — but I have not enabled it because the available footage has no qualifying frame for it (f0 at text=0.046 already gets the leading-edge rescue treatment, not blank rescue).

## Opt-out flags

| Patch | Disable                              | Tunable defaults                                                                                       |
|-------|--------------------------------------|--------------------------------------------------------------------------------------------------------|
| v15.13 | `--no-v1513-leading-edge-rescue`    | `--v1513-leading-edge-{min-gap-sec,ham-min,sim-max,area-min,fill-min,edge-motion-max,turn-max,blur-min,text-min}` |
| v15.12 A | `--v1512-blur-asym-severe-ratio 0` | severe-ratio/dt/turn-max, moderate-*, footer-bypass-*                                                  |
| v15.12 B | `--v1512-sharp-tiebreak-bonus 0`   | sharp-tiebreak-{ratio,skew-max,turn-max,bonus}                                                         |
| v15.12 C | `--no-v1512-blank-rescue`          | blank-rescue-{paper,motion,blur,bottom-dark,edge,skin,bright,post-text}-*                              |

All inherited v15.12 defaults are unchanged.
