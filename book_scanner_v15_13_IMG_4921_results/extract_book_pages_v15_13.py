# -*- coding: utf-8 -*-
# extract_book_pages_v15_13.py
# v15.13 — leading-edge distinct-page rescue (additive on v15.12).
#
# Motivation (IMG_4921 v15.12 review by user):
#   The video begins with two visually different illustrated pages
#   ("Sporting Seasons" at t=0, then "A Father's Advice" by t~1-2s). v15.12
#   only kept the second one because the first sampled frame (t=0) is
#   evaluated with degenerate context (stability=0.5 init default,
#   hand_text_overlap=1.0, bottom_hand=1.0); its peak_score collapses to
#   ~0.16 and it loses the local-peak contest to the neighbour that
#   follows it within min_peak_distance_sec. No v15.x path surfaces a
#   real distinct page that was suppressed at the peak picker for that
#   reason.
#
# v15.13 patch (additive, opt-out, no hardcoded frames or page counts):
#   _v1513_leading_edge_distinct_page_rescue: after v15.12 blank rescue,
#   inspect the first kept winner. If a sampled candidate with
#   page_found=1, stable contour (area>=0.55, fill>=0.85), low
#   edge_motion (<=0.5) and low turn_penalty (<=0.5) exists strictly
#   before it (gap >= 1.5s) AND is visually distinct from the first
#   winner (dHash hamming >= 22 OR roi-similarity <= 0.55), recover the
#   sharpest such candidate (full warp + smart_trim) and prepend it.
#   The distinctness gate is strict so the rescue will never duplicate
#   the existing first winner; the gap rule means nothing within
#   min_peak_distance_sec is eligible. Skipped when --expected-pages is
#   set. Opt-out via --no-v1513-leading-edge-rescue.
#
# v15.12 inherited — Path D severity-aware turn ceiling, sharpness/skew
# dedup tiebreaker, and conservative blank front-matter rescue.
#
# Motivation (IMG_4921 v15.11 review diagnostic):
#   The Father's Advice page was duplicated in v15.11 output (frames 30 and
#   60 — page_001 and page_002). v15.11 Path D (blur-asymmetry rescue) was
#   blocked because BOTH sides had to pass turn_penalty <= 0.65: f60 (the
#   sharp side) sat at 0.845 due to operator reframe noise, not an actual
#   page flip. Even if Path D fired, the unified `_v155_quality_score`
#   would have kept f30 (blur=86) over f60 (blur=727) because
#   hand_text_overlap dominated the keeper score on this low-text page.
#   Separately, a real blank front-matter verso at t≈4.5-6.0s was filtered
#   out by the prefilter on principle (text=0, edge_density<0.10).
#
# v15.12 patches (all additive, opt-out, no hardcoded frames/page counts):
#   A. _v155_adjacent_same_page Path D severity-aware turn ceiling: replace
#      the AND-of-both turn gate with a single max-turn ceiling that floats
#      with blur asymmetry severity:
#        blur_ratio<=0.15, dt<=1.0s -> max_turn ceiling 0.90
#        blur_ratio<=0.25, dt<=1.5s -> max_turn ceiling 0.75
#        otherwise                  -> existing 0.65
#      Physically, when one capture is razor-sharp and the other heavily
#      blurred within a 1 s window, the operator cannot have flipped a
#      page; turn_penalty inflation under those conditions is reframe
#      noise. Opt-out via --v1512-blur-asym-severe-ratio 0.
#   B. v155_adjacent_winner_dedup explicit sharpness/skew tiebreaker:
#      after computing q_a / q_b, when one side is >=4x sharper than the
#      other AND its |deskew_angle| <= 1.5 deg AND its turn_penalty <=
#      0.90, add an explicit +0.25 sharpness bonus to the sharp side. On
#      low-text pages the unified score's hand_text_overlap term can
#      dominate over blur (hand-occluded empty margins, not text), so an
#      explicit sharpness signal is needed when one duplicate is sharp/
#      level and the other is blurry/skewed. Opt-out via
#      --v1512-sharp-tiebreak-bonus 0.
#   C. Conservative blank front-matter rescue: after dedup, scan every
#      inter-winner temporal gap >= 1.5s. If the gap contains a sampled
#      prefilter frame with the strict "settled blank paper" signature
#      (paper_ratio>=0.75, motion<=0.04, blur>=80, bottom_dark<=0.50,
#      edge_density<=0.10, skin<=0.05, bright_mean>=120), decode the
#      sharpest such frame, validate text_score remains <=0.012 after
#      warp/deskew, and add it as a winner. At most one rescue per gap.
#      Front and back of video are NOT rescued. Default-on to fulfil the
#      user requirement that real pages should be found by default; opt-
#      out via --no-v1512-blank-rescue.
#
# Inherited from v15.11 — motion-blur-aware adjacent-dedup Path D + blur-
# tiebreak peak picker (additive on v15.10).
#
# Motivation (IMG_4899 v15.10 review diagnostic):
#   v15.10's local-peak retention surfaced the right candidate set but two
#   specific failure modes remained, both rooted in motion blur:
#     1. f120 (sharp p188) and f150 (motion-blurred p188) survived as two
#        winners because every same-page test (sim, dHash, text density)
#        collapses on the blurred side. The v15.5 adjacent-dedup short-
#        circuited on the text-density floor before any geometric path ran.
#     2. f330 (blur=61) beat f315 (blur=530) for page p190 by a peak_score
#        margin of only 0.004 — the local-max rule promoted the blurry frame.
#
#   v15.11 adds two narrow, additive, opt-out patches:
#     A. _v155_adjacent_same_page Path D ("blur-asymmetry rescue"): when two
#        adjacent winners are within min_peak_distance*1.5 seconds, neither
#        shows turn-in-progress, and one is >=4x sharper than the other,
#        accept as same physical page even when sim/dHash/text-rel fail.
#        Footer-distinctness guard still applies. Path D fires only after
#        Paths A/B/C; quality keeper picks the sharper frame. Opt-out via
#        --v155-adj-dedup-blur-asym-max-ratio 0 (or related flags).
#     B. Local-peak picker blur tiebreak: when a neighbour is within
#        --peak-tie-eps (1%) of x.peak_score, has blur_score >= 200, and is
#        >=2x sharper than x, defer to the sharper neighbour. Opt-out via
#        --peak-tie-eps 0.
#
# v15.10 — enable local-peak retention in default mode (additive on v15.9).
#
# Motivation (IMG_4899 default-mode diagnostic):
#   v15.9's prefilter selects via global top-K plus neighbourhood
#   expansion, head/tail edge-gap rescue, and the v15.9 Van der Corput
#   tiebreaker. The slot-retention and local-peak retention guardrails
#   stayed gated behind `expected_pages > 0`. On IMG_4899, the smoothed
#   composite saturated at 1.85 across most of the video, so the
#   global top-K consumed its budget on the leading and trailing
#   plateau positions and the two genuine mid-video pages near frames
#   90 and 285 were dropped at the prefilter (never decoded).
#
# v15.10 fix: ungate local-peak retention in default mode. The
# strict-local-max gate (peak must be strictly above at least one
# neighbour) is still in place, so saturated plateau frames cannot
# inflate the candidate set — only true peaks of the smoothed
# composite are added. New flag `--prefilter-default-local-peaks`
# defaults to ON; `--no-prefilter-default-local-peaks` restores v15.9
# behaviour for any regression case. Behaviour in `--expected-pages`
# mode is unchanged (slot retention already enabled the same path).
#
# Original v15.9 notes follow:
# v15.9 — prefilter temporal-coverage tiebreaker (additive on top of v15.8).
#
# Motivation (IMG_4892 default-mode diagnostic):
#   The v15.1 fast prefilter selects candidates by ranking the smoothed
#   composite score and keeping the global top-K (plus neighbourhood and,
#   in expected-pages mode, per-slot/local-peak retention). When many
#   sampled frames share the saturated composite ceiling (1.85), the
#   smoothed-score plateau leaves the tiebreak entirely up to numpy's
#   stable argsort order. On IMG_4892, four tail frames at 510/525/540/
#   555 share composite=1.8500 with kept frames immediately before them
#   but lose the plain argsort tiebreak, never reach full processing,
#   and the 6th (colophon/barcode) page is silently dropped. Without
#   --expected-pages, v15.3's per-slot and local-peak guardrails are
#   disabled, so global top-K is the only filter and starves whole
#   temporal regions on plateaus.
#
# v15.9 fix: in `_v150_run_prefilter`, add a deterministic low-amplitude
# Van der Corput (base-2 reversed-binary) tiebreaker (< 1e-3) to the
# smoothed composite before argsort. On true plateaus this picks
# positions in low-discrepancy interleaved order, guaranteeing that
# head, tail and middle of the video all get representation in the
# global top-K pool. The bonus is at least an order of magnitude below
# any meaningful composite difference, so on non-plateau frames the
# selection is unchanged. No new flags, no hardcoded frames or page
# counts, no behaviour change in expected-pages mode (per-slot retention
# already protected those runs).
#
# Inherited from v15.8 — within-region finger-relief refinement.
#
# Additive on top of v15.7. v15.7 introduced a steal_zone guard inside
# quality_refinement_pass that prevents a suspicious winner from being
# replaced with a candidate sitting closer to a *different* existing
# winner (cross-region replacement that would collapse a peak/page
# region under v15.5 adjacent-dedup). The guard is preserved.
#
# Motivation (user review of IMG_4890 v15.7 default):
#   page_003 (frame 165) shows a finger occluding the bottom-left of the
#   warped page. Within the same peak/page region a cleaner same-page
#   candidate exists, but v15.7's quality_refinement_pass rejects it
#   because the global-quality `_v154_refine_score` improvement is below
#   `--quality-refine-min-improvement` (default 0.12), or because the
#   blur floor (0.55 * winner.blur) excludes a slightly motion-blurred
#   but visibly cleaner same-page frame.
#
# v15.8 adds, inside `quality_refinement_pass`, a *bounded* relaxed
# branch that fires ONLY when the winner is suspicious specifically due
# to a high finger / bottom-skin signal (cvs/finger reason in
# `_v154_winner_is_suspicious`, OR finger_penalty >= a configurable
# floor). The relaxed branch:
#
#   * Keeps the v15.7 steal_zone guard (within-region only — no cross-
#     region replacement).
#   * Requires the strict same-page identity gate (v134 relaxed OR alt
#     same-page accept), NOT the temporal-buddy fallback.
#   * Keeps geometric / motion gates: deskew_max, turn_penalty,
#     stability, edge_motion (turn frames must NOT win).
#   * Loosens the blur floor (0.55 -> --finger-relief-blur-frac, default
#     0.30) and the absolute raw_score drop (1.5 -> 2.5) — but only when
#     finger_penalty improvement on the candidate is large.
#   * Replaces the global `min_improve` gate with a finger-specific
#     gain test: candidate.finger_penalty <= winner.finger_penalty
#     - --finger-relief-min-finger-improve (default 0.20), AND the
#     overall `_v154_refine_score` does not regress beyond
#     --finger-relief-max-overall-regress (default -0.10).
#   * After picking the best candidate, checks that the candidate is
#     not visually near-identical to any OTHER existing winner (so the
#     subsequent v15.5 adjacent-dedup cannot collapse two regions onto
#     one frame and reduce the page count).
#
# Properties of the fix (commercial-safety / generic-by-design):
#   * No hardcoded video name, frame index, page number, page count, or
#     timestamp. All thresholds are CLI-tunable.
#   * Does not require --expected-pages.
#   * No OCR; no per-frame model invocations; no extra full-video pass.
#     Uses the already-decoded `warped_bgr` and the existing valid pool.
#     Worst case adds O(K * S) same-page tests where K is the number of
#     finger-suspicious winners (typically 0-2) and S is bounded by
#     `--quality-refine-top-k` (default 6).
#   * Cross-region collapse is structurally impossible: the steal_zone
#     guard and the post-pick "near-identical to another winner" check
#     both forbid replacements that would lose a region. Page count and
#     coverage are preserved.
#
# CLI additions:
#   --finger-relief / --no-finger-relief                   (default ON)
#   --finger-relief-finger-floor FLOAT                      (default 0.30)
#   --finger-relief-min-finger-improve FLOAT                (default 0.20)
#   --finger-relief-max-overall-regress FLOAT               (default -0.10)
#   --finger-relief-blur-frac FLOAT                         (default 0.30)
#   --finger-relief-raw-score-max-drop FLOAT                (default 2.5)
#   --finger-relief-other-winner-sim FLOAT                  (default 0.85)
#   --finger-relief-other-winner-ham INT                    (default 8)
#
# Diagnostics in winners.csv (additive):
#   * v158_finger_relief_applied   (0/1)
#   * v158_finger_relief_reason    string
#   * v158_finger_relief_orig_finger / v158_finger_relief_new_finger
#   * v158_finger_relief_orig_cvs / v158_finger_relief_new_cvs
#
# v15.7 — narrower peak smoothing default to recover missed pages.
#
# Minimal change relative to v15.6: lower the default of --peak-window-sec
# from 0.8 to 0.5 (single-line change to the argparse default). All other
# v15.6 behaviour (footer/folio-region distinctness guard, v15.5 adjacent-
# dedup, v15.4 quality refinement, etc.) is preserved unchanged.
#
# Motivation (see IMG_4890_v15_6_missed_pages_diagnostic.md):
#   IMG_4890 default v15.6 saved 3 pages (frames 120, 240, 330) but the
#   video contains 5 stable page regions. Frames 30/45 (region A) and
#   150/165 (region C) survived prefilter and were valid warped candidates
#   with page_found=1, but were lost in select_local_peaks because the
#   smoothing radius (peak_window_sec * fps_sampled = 0.8 * 2 ≈ 2 samples)
#   pulled down the peak near t=5.5s through low-quality handover frames at
#   t=6.0–7.0s. With peak_window_sec=0.5 the smoothing radius drops to 1
#   sample, so region C's local maximum survives and the global peak picker
#   selects 5 regions instead of 4. No hardcoded video name / frame index /
#   page count anywhere.
#
# v15.6 — bottom/folio-region distinctness guard for adjacent-winner dedup
# (additive on top of v15.5).
#
# User review of v15.5 IMG_4886 default output (6 unique pages: p13, p15, p17,
# p21, p23, p25) reported one specific defect: a single page (p19) was missed
# between output pages 003 and 004. The bottom page numbering on the printed
# book makes the omission obvious. IMG_4883 v15.5 default and exp5 outputs
# remained excellent and must NOT regress.
#
# Root cause analysis (v15.5 IMG_4886 fast log):
#   [v15.5] adjacent-dedup: removed winner frames [420, 480]
#     keep frame 390 over frame 420: rescan(dt=1.00s,sim=0.25,prof=0.16,
#       warp=0.68,text_rel=0.24)
#     keep frame 510 over frame 480: rescan(dt=1.00s,sim=0.08,prof=0.44,
#       warp=0.55,text_rel=0.15)
#
# The temporal-rescan path inside _v155_adjacent_same_page accepts a same-
# page merge purely on the basis of "close in time + similar text density +
# non-trivial warp/profile/SSIM agreement". For two ADJACENT physical pages
# this is dangerous: ~1s is enough for the operator to flip one page in this
# scanning rig, the warp thumbnail explicitly trims the bottom 6 rows to
# discount hand occlusion (which also discards the page-number band), and
# the row profile uses a max_shift tolerance that can absorb a one-line
# shift between consecutive pages of a paragraph. The outcome: a true
# different-page pair was merged.
#
# v15.6 adds a footer/folio distinctness GUARD that runs on every non-strict
# same-page verdict (i.e. anything that relies on temporal-rescan or
# primary+corroboration instead of strict identity). The guard:
#
#   1. Extracts a small bottom-center ROI from each winner's perspective-
#      rectified `warped_bgr` (the page-number / folio band — by default
#      the bottom 9% of page height, horizontally centered over 60% of
#      width, with the outer 8% trimmed to avoid edge artefacts and gutter).
#   2. Computes lightweight, OCR-free signatures of that ROI:
#        * adaptive-binarized ink mask (cross-illumination invariant);
#        * column ink projection (1-D profile, normalized);
#        * row    ink projection (1-D profile, normalized);
#        * total ink ratio in the band;
#        * 64-bit dHash on a 9x8 normalized thumbnail of the band.
#   3. Combines those into a "distinct" verdict that requires at least two
#      independent signals to disagree confidently AND the bands to
#      actually contain ink (so a blank-bottom pair, e.g. page-end
#      whitespace, never falsely separates).
#   4. If the bottom bands are confidently DIFFERENT, the same-page merge
#      is REJECTED even if temporal-rescan / whole-page similarity said
#      "same". A diagnostic reason `footer_distinct(...)` is recorded.
#   5. Strict-identity merges (very high SSIM AND very low dHash hamming
#      over the whole page) BYPASS the guard since true near-duplicates of
#      the same physical page must have matching footers. This preserves
#      the v15.4/v15.5 behaviour of removing true duplicates of skewed /
#      hand-occluded pages.
#
# Properties of the fix (commercial-safety / generic-by-design):
#   * No hardcoded video name, frame index, page number, page count, or
#     timestamp. Footer ROI is geometric (fractions of warp height/width).
#   * Does not require --expected-pages.
#   * No OCR; no per-frame model invocations; no extra video pass. Uses the
#     already-decoded `warped_bgr` per winner. O(K) winner pairs.
#   * Fails OPEN on missing data (warped_bgr is None, ROI extraction
#     fails) so it cannot regress runs that lack the inputs to evaluate
#     the guard. The pre-existing dedup decision is used in that case.
#   * Honours an explicit "blank footer" verdict: if either page's footer
#     band has near-zero ink, the guard cannot confidently separate them
#     and falls back to the v15.5 decision (so blank-bottom pages still
#     dedup correctly).
#
# Diagnostics in winners.csv:
#   * v156_footer_guard_applied         (0/1) — guard ran on this row
#   * v156_footer_guard_blocked         (0/1) — guard blocked a merge
#   * v156_footer_distinct_reason       — e.g. "col_corr=0.41,row_corr=0.55,
#                                              ink_delta=0.18,ham=12"
#   * v156_footer_ink_a / v156_footer_ink_b   — ink ratios
#
# Performance: the guard adds <50ms total on a 6-winner case. Total runtime
# stays close to v15.5.
#
# CLI additions:
#   --v156-footer-guard / --no-v156-footer-guard         (default ON)
#   --v156-footer-band-frac FLOAT                         (default 0.09)
#   --v156-footer-center-frac FLOAT                       (default 0.60)
#   --v156-footer-side-trim-frac FLOAT                    (default 0.08)
#   --v156-footer-col-corr-max FLOAT                      (default 0.70)
#   --v156-footer-row-corr-max FLOAT                      (default 0.70)
#   --v156-footer-ink-delta-min FLOAT                     (default 0.12)
#   --v156-footer-ham-min INT                             (default 14)
#   --v156-footer-min-ink FLOAT                           (default 0.012)
#
# ---------------------------------------------------------------------------
# v15.5 — adjacent-winner quality dedup with replacement (additive on top of v15.4).
#
# v15.4 introduced per-winner quality refinement (replace one winner with a
# cleaner same-page candidate from the broader candidate pool). User review of
# v15.4 IMG_4886 default output reported three remaining defects on the same
# video (IMG_4883 was excellent and must NOT regress):
#   * page_005 still showed a hand;
#   * page_006 was visibly skewed/curved;
#   * page_007 was a *good* duplicate of page_006.
#
# Root cause: v15.4 replaces a single winner with a cleaner candidate, but it
# never compares two ADJACENT winners against each other. When two adjacent
# winners are actually the same physical page, the worse one stays in the
# output even though the cleaner one is right next to it. The v14.2a auto-
# dedup pass that runs earlier had thresholds tuned to avoid false merges and
# did not consider clean_visual_score / finger_penalty / abs(deskew) when
# choosing the keeper, so a skewed-but-high-raw-score winner could outrank a
# cleaner near-duplicate.
#
# v15.5 adds a single, bounded post-refinement pass:
#   1. After quality_refinement_pass produces the final candidate set, walk
#      the winners in temporal order and test each adjacent pair (and
#      neighbour-2 for tail cleanup) for same-physical-page identity using
#      the v13.4 relaxed test PLUS warp-thumb correlation, profile
#      correlation, temporal proximity, and similar text density. Treat the
#      pair as the same page only when "primary AND corroborated" (same
#      strict-evidence pattern as v14.2a auto-dedup).
#   2. When two adjacent winners are the same page, score both with a
#      unified visual quality score that is the same combined metric used
#      to *rank* refinement candidates inside v15.4 (clean_visual_score,
#      finger_penalty, hand_text_overlap, bottom_hand_penalty,
#      hand_penalty, abs(deskew), turn_penalty, edge_motion, blur). Drop the
#      lower-scoring winner. The clean_visual_score / finger_penalty /
#      |deskew| terms ensure a skewed-but-high-raw winner LOSES to a clean
#      adjacent duplicate.
#   3. Diagnostics: each removed winner records v155_adj_dedup_removed,
#      v155_adj_dedup_keeper, v155_adj_dedup_reason in winners.csv so the
#      operator can audit each merge.
#
# Performance discipline:
#   * O(K) where K = #winners. Each pair test reuses already-computed warp
#     thumbnails, profiles, dHashes — no new decode, no full-video pass.
#   * Runs after auto-dedup, alt-search, and quality_refinement; if those
#     already settled the winners, this pass is a no-op.
#   * Default ON; --no-v155-adjacent-dedup disables.
#   * Honours --expected-pages: when set, the user wants exactly that many
#     pages, so we skip dedup (don't shrink the count below expected_pages).
#
# CLI additions:
#   --v155-adjacent-dedup / --no-v155-adjacent-dedup       (default ON)
#   --v155-adj-dedup-window-sec FLOAT                       (default 3.0)
#   --v155-adj-dedup-sim-min FLOAT                          (default 0.62)
#   --v155-adj-dedup-warp-min FLOAT                         (default 0.78)
#   --v155-adj-dedup-profile-min FLOAT                      (default 0.65)
#   --v155-adj-dedup-text-rel-max FLOAT                     (default 0.30)
#
# v15.4 — bounded same-page quality refinement for suspicious winners
# (additive on top of v15.3).
#
# v15.3 produced excellent IMG_4883 results and good IMG_4886 results, but
# user review of IMG_4886 highlighted two specific defects:
#   * page_005 picked a frame containing a hand;
#   * page_006 was tilted/skewed.
# Everything else on both videos was excellent. The fix here is generic
# (NOT hardcoded to IMG_4886/pages/frames), does not regress IMG_4883 and
# keeps performance close to v15.3.
#
# v15.4 introduces a bounded post-selection quality-refinement pass that:
#   1. flags individual winners as "suspicious" using generic, threshold-
#      based criteria (high hand_penalty / hand_text_overlap / bottom_hand,
#      large abs(deskew_angle), high turn or edge_motion penalty, low
#      clean_visual_score combined with high finger_penalty).
#   2. for each suspicious winner only, searches a small temporal window
#      (default +/- 2.5s) for already-decoded candidates from the same
#      physical page that are visibly cleaner. Same-page identity reuses
#      _v134_relaxed_same_page (warp-thumb / profile / SSIM / dHash) so we
#      never replace a winner with a different page.
#   3. ranks candidates with a combined refinement score that rewards
#      clean_visual_score, blur, raw_score and penalises hand metrics,
#      bottom_hand, |deskew|, turn_penalty, edge_motion. Replaces only when
#      improvement exceeds --quality-refine-min-improvement (default 0.12).
#   4. emits diagnostics into winners.csv: quality_refinement_applied,
#      quality_refinement_reason, qr_original_frame, qr_replacement_frame,
#      qr_score_delta, qr_candidates_checked.
#
# Performance discipline:
#   * The refinement pass only operates on suspicious winners (typically
#     0-2 per video on these inputs), uses the existing valid-candidate
#     pool (no new decode), bounded by --quality-refine-top-k (default 6)
#     candidates per suspicious winner.
#   * Runs AFTER alt-search/auto-dedup/expected-pages-fill so it never
#     fires on already-stable winners.
#   * No contact sheets unless --debug-contact-sheets is also set.
#
# CLI additions:
#   --quality-refinement / --no-quality-refinement              (default ON)
#   --quality-refine-window-sec FLOAT                            (default 2.5)
#   --quality-refine-top-k N                                     (default 6)
#   --quality-refine-min-improvement FLOAT                       (default 0.12)
#   --quality-refine-hand-thresh FLOAT                           (default 0.40)
#   --quality-refine-skew-thresh FLOAT                           (default 4.0 deg)
#
# v15.3 — quality-safe fast prefilter on top of v15.2.
#
# v15.2 cut runtime via parallel candidate processing on top of v15.1's
# fast prefilter. On IMG_4883 --expected-pages 5 the v15.2 fast pipeline
# however dropped clean, temporally-isolated pages (e.g. dedication frame 60
# and chapter title frame 390 on IMG_4883) because the prefilter retains
# only the global top-K composite scores, which let a saturated cluster of
# high-blur frames mid-video crowd out otherwise good frames elsewhere.
# v15.3 keeps the v15.2 parallel candidate path and adds three production-
# safe quality guardrails to the fast prefilter:
#   1. Per-temporal-slot retention. When --expected-pages > 0, the video is
#      split into N = expected_pages * slot_factor temporal slots and the
#      prefilter retains the top-K candidates from each slot (in addition to
#      the global top-K). This guarantees each expected page slot has at
#      least one full-processed candidate to compete for selection.
#   2. Local-peak retention. Frames whose smoothed prefilter composite is a
#      strict local maximum (within +/- prefilter-peak-radius) are retained
#      even if they fall below the global top-K cut-off.
#   3. Fast-mode quality fallback. After the existing reselection pipeline,
#      if winners count < expected_pages or visual duplicates remain among
#      winners, decode a small bounded set of extra sampled frames from the
#      uncovered temporal intervals only and re-run reselection. Bounded by
#      --fast-fallback-max-extra (default 24).
# v15.2 parallelism, prefilter-driven calibration, and clean_visual_score
# logic are preserved verbatim — the v15.3 changes are purely additive on
# the candidate-retention / fallback side.
#
# CLI additions:
#   --prefilter-slot-retention / --no-prefilter-slot-retention (default ON)
#   --prefilter-slot-factor N (default 2)
#   --prefilter-per-slot-top-k K (default 2)
#   --prefilter-peak-radius R (default 2)
#   --fast-quality-fallback / --no-fast-quality-fallback (default ON)
#   --fast-fallback-max-extra N (default 24)
#
# v15.2 — parallel candidate processing built on top of v15.1.
#
# v15.1 cut runtime by adding a fast prefilter and a sequential grab/retrieve
# decode path. The dominant remaining stage is full_process_candidates
# (detect/warp/deskew/inpaint/dhash). v15.2 keeps decode strictly sequential
# (cv2.VideoCapture is not thread-safe) and parallelizes only the per-frame
# heavy CPU work using a ThreadPoolExecutor. Each worker owns its own
# HandMasker (MediaPipe Hands.process is not safe across threads). Results
# are sorted by frame_idx and a small sequential pass computes
# stability_score / edge_motion_penalty so the output is bit-identical to
# v15.1 when --candidate-workers 1 is used and visually equivalent at
# higher worker counts.
#
# CLI:
#   --parallel-candidates / --no-parallel-candidates  (default ON)
#   --candidate-workers N                             (default auto)
#
# v14.1b — adds expected-pages count-fill repair on top of v14.1a sanity gate.
#
# v14.1a fixed false-duplicate merges (sanity gate refuses to merge two pages
# whose raw visual evidence is too weak even when the relaxed same-page test
# fires via profile/warp-thumb). On IMG_4886 --expected-pages 7 that prevented
# the merge of pages 13/17 but the resulting selection was still 6 pages —
# one expected slot was left unfilled because the original peak/cluster pass
# never produced a winner there.
#
# v14.1b adds a generic expected-count repair pass that runs AFTER all v13.4
# / v13.5 / v14.1a reselection logic:
#   * If len(winners) < expected_pages, identify the largest temporal gap(s)
#     (including head/tail of the video). For each gap consider the top-K
#     valid candidates ranked by quality + clean_visual_score + raw score.
#   * Apply strict distinctness checks against existing winners: relaxed
#     same-page test must NOT fire, raw SSIM/hash distance must exceed a
#     conservative threshold, and the new candidate must occupy a distinct
#     time cluster (>=min_peak_distance_sec from any winner).
#   * Add the best surviving candidate, re-evaluate gaps, repeat until
#     count == expected_pages or no safe candidate exists.
#   * Per-fill diagnostics are recorded in reselection_diag and surfaced in
#     winners.csv via four new columns: expected_fill_applied,
#     expected_fill_reason, fill_source_gap, fill_distinctness_score.
#
# Behaviour is otherwise identical to v14.1a:
#   * Sanity gate from v14.1a is preserved verbatim.
#   * No hardcoded video/frame logic — the fill is purely temporal/quality.
#   * Production performance: bounded top-K per gap (default 6), bounded
#     iterations (= expected_pages), no contact-sheet generation.
#
# v14.0 — production / commercial-app oriented refactor of v13.5.
#
# Architectural goals over v13.5:
#   1. Clear mode separation:
#        * production / fast (default):
#            - no contact sheets, no candidate audit, no broad alt-search;
#            - bounded reselection (--reselection-top-k, --max-alternatives-per-winner);
#            - minimal logging, no debug artifacts.
#        * --debug:
#            - writes scores.csv, winners.csv, calibration.json (as in v13.x);
#            - writes timings.json with stage timings.
#        * --audit-candidates (or --debug-contact-sheets):
#            - opt-in expensive candidate audit / contact sheet generation;
#            - never default; only available when explicitly requested.
#   2. Timing instrumentation:
#        * --profile / --timing-report prints per-stage timings to stdout.
#        * In debug mode timings.json is written to <video>_debug_v14_0/.
#   3. Performance guardrails:
#        * --reselection-top-k limits the same-page pool size considered by
#          prefer_cleaner_equivalent_winners (default 6).
#        * --max-alternatives-per-winner limits find_alternative_winner's
#          temporal-window candidate count (default 8). Without alt-search
#          enabled this is a no-op.
#        * Contact-sheet generation only runs when explicitly requested.
#        * Adaptive calibration sample is bounded by --calibration-max-frames.
#   4. Quality features from v13.5 are preserved generically (no IMG_4883-
#      specific hardcode). Specifically:
#        * clean_visual_score-driven prefer_cleaner_equivalent_winners,
#        * rescue_early_first_page,
#        * repair_visual_duplicate_winners,
#        * destructive cleanup remains opt-in (--experimental-hand-cleanup).
#
# v13.5 (IMG_4883 focused iteration) extends v13.4 with a CLEAN-VISUAL
# RESELECTION layer. Page identities discovered by v13.4 stay correct, but
# within each identity we now actively pick the cleanest equivalent frame
# instead of the one with the highest peak/norm score.
#
# Symptoms on IMG_4883 with --expected-pages 5 that v13.4 still has:
#   * page_002 (dedication, frame 105): page borders mostly clean but the
#     bottom-left has a finger blob and the global background tone is grayer
#     than other dedication candidates (frame 60/90 are visibly whiter).
#   * page_003 (Пролог, frame 255): a clean equivalent exists at frame 195
#     where no fingers are visible at all and the bottom page number "7" is
#     legible. v13.4 picked 255 because the script-level bottom_hand_penalty
#     is a false positive on f195's lower margin.
#
# v13.5 additions:
#   1. compute_clean_visual_score(warped_bgr): an HSV-based score that is
#      independent of the existing (and sometimes false-positive) hand
#      penalties. It rewards bright, even paper background, low skin/finger
#      pixels on the page, low bottom-band skin, low background blotchiness
#      and stable text-edge readability.
#   2. clean_visual_select_within_window: for each winner, examine candidates
#      within a wider same-page window than v13.4. Same-page identity uses
#      _v134_relaxed_same_page (already proven on the 195/255 case in reverse).
#      The cleanest equivalent (max clean_visual_score) replaces the winner if
#      it materially beats the original; tie-breaks fall back on peak_score.
#   3. The dirty-winner branch in prefer_cleaner_equivalent_winners is
#      generalized: clean_visual_score now drives the swap on EVERY winner,
#      not just ones flagged as dirty. The dirty flag still widens the window.
#   4. Extra diagnostics in winners.csv: clean_visual_score, bg_gray_penalty,
#      finger_penalty, candidate_search_window, original_frame,
#      replacement_frame, reselection_reason. The existing v13.4 columns are
#      reused; clean_visual_* are appended.
#   5. Cleanup remains DISABLED by default (this is a selection change, not
#      an inpainting change).
#
# Previous v13.4 layer is unchanged:
#   1. repair_visual_duplicate_winners: detects adjacent winners that are the
#      same physical page using a RELAXED dHash/SSIM threshold and replaces
#      the worse one with the best clean novel candidate from the largest
#      empty time gap. This catches the 195/255 "Пролог" duplicate.
#   2. rescue_early_first_page: when --expected-pages > 0 and the first
#      selected winner leaves a substantial gap from the start of the video,
#      and a distinct (different-content) clean candidate exists earlier,
#      inject it. This rescues frame 0 (title page) on IMG_4883.
#   3. prefer_cleaner_equivalent_winners: now driven by clean_visual_score
#      (see v13.5 additions above). The widened window for "dirty" winners
#      is preserved.
#   4. Diagnostics: reselection_reason and duplicate_repaired columns added to
#      winners.csv. The existing original_frame / replacement_frame columns
#      are reused when a v13.4/v13.5 reselection fires.
#
# Default behaviour for non-expected-pages mode is unchanged. Cleanup is still
# disabled by default. v13.3 prologue retained:
# extract_book_pages_v13_3.py
# v13.3 is a CONSERVATIVE REGRESSION FIX over v13.2.
#
# v13.1 introduced cluster_select_score reranking + adaptive high-hand mode +
# secondary cluster merging, and v13.2 added an alternative-winner replacement
# search and a conservative_bottom_hand_cleanup post-pass. These improved
# IMG_4885-style high-hand videos but caused regressions on well-behaved
# inputs (IMG_4883: lost page + duplicate pages 2/3) and damaged final JPEGs
# on IMG_4885 (page 5 half invisible).
#
# v13.3 strategy:
#  * Restore v13.0 / v12.9 stable winner selection by default
#    (high_hand_mode=false): cluster reranking, secondary/tertiary merge
#    heuristics, deskew penalties in peak/quality/final selection, and the
#    high-hand-biased force_reduce all become NO-OPS unless high_hand_mode is
#    on.
#  * Keep v13.1 high-hand selection improvements gated to high_hand_mode=true
#    (auto-detected from calibration skin stats, or forced via
#    --force-high-hand). IMG_4885 still gets the deskew-cleaner winners.
#  * Disable v13.2 alternative-winner replacement by default. It can be opted
#    into with --enable-alt-search; even then it only applies when
#    high_hand_mode=true so well-behaved videos are never perturbed.
#  * Disable v13.2 conservative_bottom_hand_cleanup by default. It can be
#    opted into via --experimental-hand-cleanup. Default JPEGs are never
#    modified by this destructive cleanup.
#  * Diagnostic columns are kept for inspection (cluster_select_score etc.)
#    so debug output stays informative.
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception:
    mp = None


# ---------------------------------------------------------------------------
# v14.0: lightweight timing instrumentation. Stage timings accumulate into
# _STAGE_TIMINGS (seconds) and are surfaced via --profile / --timing-report
# and (in debug mode) timings.json. Always-on, low overhead, never blocks.
# ---------------------------------------------------------------------------
_STAGE_TIMINGS: Dict[str, float] = {}
_STAGE_ORDER: List[str] = []


def _reset_timings() -> None:
    _STAGE_TIMINGS.clear()
    _STAGE_ORDER.clear()


@contextmanager
def stage_timer(name: str):
    """Context manager that records elapsed seconds under `name`.

    Repeated names accumulate (e.g. nested per-winner work)."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        if name not in _STAGE_TIMINGS:
            _STAGE_ORDER.append(name)
        _STAGE_TIMINGS[name] = _STAGE_TIMINGS.get(name, 0.0) + dt


def _format_timings_report() -> str:
    if not _STAGE_ORDER:
        return '[v14.0] timings: (none recorded)'
    lines = ['[v14.0] stage timings (seconds):']
    width = max(len(n) for n in _STAGE_ORDER)
    total = _STAGE_TIMINGS.get('total', None)
    for name in _STAGE_ORDER:
        dt = _STAGE_TIMINGS[name]
        pct = ''
        if total and total > 0 and name != 'total':
            pct = f' ({dt / total * 100:5.1f}%)'
        lines.append(f'  {name.ljust(width)}  {dt:7.3f}s{pct}')
    return '\n'.join(lines)


def _timings_dict() -> Dict[str, float]:
    return {name: float(_STAGE_TIMINGS.get(name, 0.0)) for name in _STAGE_ORDER}


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
    # v13.1: per-merge reason strings, parallel to members. members[0] is the
    # cluster seed so its reason is 'seed'; subsequent reasons describe whether
    # the candidate was merged via the strict primary path (dHash+ssim) or the
    # secondary heuristic (temporal + text-density + relaxed similarity).
    merge_reasons: List[str] = field(default_factory=list)


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
    # v13.5: gate sparse mode on the cleanliness of the actual page
    # background. The sparse pipeline was tuned for pages with visible
    # bleed-through where its strong flattening + bilateral helps. On a
    # genuinely clean dedication page (e.g. IMG_4883 frame 60) the same
    # pipeline amplifies low-amplitude paper noise into the blotchy gray
    # texture the user complained about. Bypass sparse mode when the paper
    # is bright AND uniform — we then fall through to the gentler
    # 'standard' path.
    try:
        _hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        _Hch, _Sch, _Vch = cv2.split(_hsv)
        _paper_mask = (_Vch > 170) & (_Sch < 60)
        _paper_ratio = float(_paper_mask.mean())
        if _paper_ratio > 0.10:
            _Vsm = cv2.GaussianBlur(_Vch, (51, 51), 0)
            _bg_blotch = float(_Vsm[_paper_mask].std())
            _bg_v_mean = float(_Vch[_paper_mask].mean())
        else:
            _bg_blotch = 999.0
            _bg_v_mean = 0.0
        clean_paper = (_paper_ratio >= 0.78
                       and _bg_v_mean >= 248.0
                       and _bg_blotch <= 8.5)
    except Exception:
        clean_paper = False
    # Decorative wins over sparse: a cover page can have low "text density"
    # by our coarse measure but must not be sparse-flattened.
    sparse = (not decorative) and (text_density < 0.045) and (not clean_paper)

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


def deskew_soft_penalty(angle_deg: float, soft_thresh: float, alpha: float) -> float:
    """Smooth penalty for residual deskew that exceeds `soft_thresh` (deg).

    Returns 0.0 for moderately straight pages; grows linearly with the absolute
    excess past the threshold. Multiplied by `alpha` so callers can tune
    aggressiveness from a single weight constant.
    """
    excess = max(0.0, abs(float(angle_deg)) - float(soft_thresh))
    return float(alpha) * excess


def cluster_select_score(x: FrameFeatures, args, high_hand_mode: bool = False) -> float:
    """Composite score used to pick the cleanest member inside a cluster.

    The intent is to break ties between near-duplicate frames of the same page
    so the final winner is the straightest, least hand-occluded one rather than
    merely the candidate with the highest raw_score. v13.0 used max(raw_score)
    inside clusters which on IMG_4885 produced winners with -22 deg deskew and
    bottom-hand=1.0 even though cleaner siblings existed.

    cluster_select_score = base - alpha*deskew_excess - beta*hand_text_overlap
                                  - gamma*bottom_hand - delta*hand_penalty
                                  - epsilon*cleanup_mask_ratio
    where alpha/beta/gamma/delta/epsilon are conservative (so well-behaved
    videos like IMG_4883 are not perturbed) but escalate in high_hand_mode.
    """
    soft = float(getattr(args, 'deskew_soft_threshold', 12.0))
    alpha = float(getattr(args, 'cluster_deskew_weight', 0.030))
    beta = float(getattr(args, 'cluster_hand_text_weight', 0.55))
    gamma = float(getattr(args, 'cluster_bottom_hand_weight', 0.40))
    delta = float(getattr(args, 'cluster_hand_weight', 0.40))
    if high_hand_mode:
        alpha *= 1.7
        beta *= 1.6
        gamma *= 1.7
        delta *= 1.5
    # base is primarily peak_score (already normalised 0..1) plus a small
    # contribution from norm_score (also 0..1). raw_score is intentionally
    # NOT used here because it has unbounded magnitude (~5..7 on this video)
    # and would let high blur scores dominate over deskew/hand penalties.
    base = 0.55 * float(getattr(x, 'peak_score', 0.0)) + 0.45 * float(getattr(x, 'norm_score', 0.0))
    deskew_pen = deskew_soft_penalty(getattr(x, 'deskew_angle', 0.0), soft, alpha)
    hand_pen = (
        beta * float(getattr(x, 'hand_text_overlap_penalty', 0.0))
        + gamma * float(getattr(x, 'bottom_hand_penalty', 0.0))
        + delta * float(getattr(x, 'hand_penalty', 0.0))
    )
    return base - deskew_pen - hand_pen


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
    # v15.7: allow peak-window-sec to disable smoothing entirely (radius 0)
    # so a strong but isolated page region is not pulled down by adjacent
    # low-quality handover frames. The previous v15.6 floor of max(1, ...)
    # forced at least one neighbour into the moving average, which on
    # IMG_4890 wiped out region C (frame 165) because frame 180's norm
    # score was 0. The default --peak-window-sec is now 0.2; users can
    # raise it for very noisy footage if needed.
    smooth_radius = max(0, int(round(args.peak_window_sec * fps_sampled)))
    smooth = moving_average(scores, smooth_radius)
    high_hand_mode = bool(getattr(args, '_high_hand_mode', False))
    # v13.3: in default (non-high-hand) mode use v13.0 / v12.9 weights exactly
    # so well-behaved videos like IMG_4883 produce identical peak scores to
    # the stable baseline. v13.1 high-hand weights remain available when
    # high_hand_mode is on (e.g. IMG_4885).
    if high_hand_mode:
        h_w, ht_w, fg_w, bh_w = 0.75, 0.85, 0.30, 0.55
        bonus_min, bonus_max = 0.78, 0.22
        soft = float(getattr(args, 'deskew_soft_threshold', 12.0))
        deskew_pen_w = float(getattr(args, 'peak_deskew_weight', 0.020)) * 1.8
        for i, x in enumerate(valid):
            clean_bonus = 1.0 - min(1.0, h_w * x.hand_penalty + ht_w * x.hand_text_overlap_penalty + fg_w * x.edge_foreground_penalty + bh_w * x.bottom_hand_penalty)
            deskew_pen = deskew_soft_penalty(getattr(x, 'deskew_angle', 0.0), soft, deskew_pen_w)
            x.peak_score = float((0.64 * x.norm_score + 0.36 * smooth[i]) * (bonus_min + bonus_max * clean_bonus) - deskew_pen)
    else:
        # v13.0 / v12.9 stable formula: no deskew penalty, original weights.
        for i, x in enumerate(valid):
            clean_bonus = 1.0 - min(1.0, 0.55 * x.hand_penalty + 0.55 * x.hand_text_overlap_penalty + 0.30 * x.edge_foreground_penalty + 0.35 * x.bottom_hand_penalty)
            x.peak_score = float((0.64 * x.norm_score + 0.36 * smooth[i]) * (0.84 + 0.16 * clean_bonus))

    peaks: List[FrameFeatures] = []
    sep = args.min_peak_distance_sec
    peak_tie_eps = float(getattr(args, 'peak_tie_eps', 0.01))
    peak_tie_blur_min = float(getattr(args, 'peak_tie_blur_min', 200.0))
    peak_tie_blur_factor = float(getattr(args, 'peak_tie_blur_factor', 2.0))

    # v15.11 blur-aware tiebreak: a candidate x is a "local peak" if no
    # neighbour beats it on peak_score AND no neighbour is BOTH within
    # peak_tie_eps of x's peak_score AND substantially sharper. This both
    # demotes a blurry frame that barely won a tied local max AND promotes
    # the sharper neighbour to local-max status (since the only frame that
    # had been beating it is now disqualified). Opt out via peak-tie-eps=0.
    def _is_local_peak(x: FrameFeatures, group: List[FrameFeatures]) -> bool:
        if x.norm_score < args.min_norm_score:
            return False
        x_peak = float(x.peak_score)
        x_blur = float(getattr(x, 'blur_score', 0.0))
        for y in group:
            if y is x:
                continue
            y_peak = float(y.peak_score)
            y_blur = float(getattr(y, 'blur_score', 0.0))
            if y_peak > x_peak:
                if peak_tie_eps > 0.0 and (y_peak - x_peak) <= peak_tie_eps \
                        and x_blur >= peak_tie_blur_min \
                        and x_blur > y_blur * peak_tie_blur_factor:
                    # y nominally beats x but x is dramatically sharper and
                    # essentially tied on peak_score: x stays in contention.
                    continue
                return False
            if y_peak == x_peak:
                continue
            # y_peak < x_peak: x beats y on peak_score. Check if y is within
            # tie band and dramatically sharper -> defer to y.
            if peak_tie_eps > 0.0 and (x_peak - y_peak) <= peak_tie_eps \
                    and y_blur >= peak_tie_blur_min \
                    and y_blur > x_blur * peak_tie_blur_factor:
                return False
        return True

    for x in valid:
        left_t = x.t_sec - sep
        right_t = x.t_sec + sep
        group = [y for y in valid if left_t <= y.t_sec <= right_t]
        if not group:
            continue
        if _is_local_peak(x, group):
            peaks.append(x)

    dedup: List[FrameFeatures] = []
    for p in sorted(peaks, key=lambda z: z.t_sec):
        if dedup and abs(p.t_sec - dedup[-1].t_sec) < sep:
            dedup[-1] = choose_between_similar(dedup[-1], p, args.sim_thresh_merge - 0.02)
        else:
            dedup.append(p)
    return dedup


def cluster_candidates(candidates: List[FrameFeatures], args) -> List[Cluster]:
    """Group near-duplicate winner candidates into clusters.

    v13.1 adds a *secondary* merge path on top of the v13.0 strict
    dHash+structural-similarity test. Perspective-warped duplicates of the same
    page on IMG_4885 produced sim ~ 0.58 which is below sim_thresh_merge=0.89
    so v13.0 left them as separate winners. v13.1 fuses them when:
      * temporal proximity (dt <= 1.5 * min_same_page_gap_sec) AND
      * text density is similar (relative diff <= 0.30) AND
      * structural similarity is at least sim_secondary_min (default 0.50) OR
        hamming distance <= hash_thresh_merge + 8.
    The merge reason for each cluster is stored on `cl.merge_reason` so the
    debug winners.csv can show *why* candidates collapsed.

    For default no expected-pages this remains conservative: the secondary path
    only fires when temporal proximity + text density already strongly suggest
    the same page.
    """
    clusters: List[Cluster] = []
    sim_secondary_min = float(getattr(args, 'sim_secondary_min', 0.50))
    hash_secondary_extra = int(getattr(args, 'hash_secondary_extra', 8))
    text_rel_tol = float(getattr(args, 'cluster_text_rel_tol', 0.30))
    time_factor = float(getattr(args, 'cluster_time_factor', 1.5))
    high_hand_mode = bool(getattr(args, '_high_hand_mode', False))
    if high_hand_mode:
        sim_secondary_min = max(0.42, sim_secondary_min - 0.05)
        hash_secondary_extra += 2
        text_rel_tol += 0.05
        time_factor = max(time_factor, 1.7)

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
            merged = False
            reason = ''
            if ham <= args.hash_thresh_merge and sim >= args.sim_thresh_merge:
                if dt < args.min_same_page_gap_sec or sim >= (args.sim_thresh_merge + 0.05):
                    merged = True
                    reason = 'primary(sim+ham)'
            # v13.3: secondary/tertiary merge heuristics are gated on
            # high_hand_mode so default behaviour matches v13.0 / v12.9 exactly
            # for normal videos (e.g. IMG_4883). Without this gate the
            # secondary path collapsed pages 2 and 3 into duplicates on
            # IMG_4883.
            if not merged and high_hand_mode:
                # Secondary heuristic: perspective-warped duplicates.
                gap = float(getattr(args, 'min_same_page_gap_sec', 1.3))
                if dt <= time_factor * gap:
                    text_a = max(1e-6, float(rep.text_score))
                    text_b = max(1e-6, float(cand.text_score))
                    text_rel = abs(text_a - text_b) / max(text_a, text_b)
                    sim_ok = sim >= sim_secondary_min
                    ham_ok = ham <= args.hash_thresh_merge + hash_secondary_extra
                    if text_rel <= text_rel_tol and (sim_ok or ham_ok):
                        merged = True
                        reason = (
                            f'secondary(dt={dt:.2f}s,sim={sim:.2f},ham={ham},'
                            f'text_rel={text_rel:.2f})'
                        )
                # Tertiary heuristic for high-hand mode: very tight temporal
                # proximity + very tight text density even when dHash hamming is
                # large because heavy hand occlusion + skew destroy both ssim
                # and dHash. Only fires in high_hand_mode and only when dt is
                # below the min same-page gap (i.e. clearly within the same
                # page interval) AND text density is essentially identical.
                # Cluster-span guard: also require that the resulting cluster
                # stays within a tight temporal window so we do not
                # accidentally chain page A -> A' -> B when A' has matching
                # text density to both neighbours.
                if not merged and high_hand_mode:
                    text_tight = float(getattr(args, 'cluster_text_rel_tight', 0.20))
                    time_tight = float(getattr(args, 'cluster_time_tight_factor', 1.25))
                    span_factor = float(getattr(args, 'cluster_span_max_factor', 1.6))
                    if dt <= time_tight * gap:
                        text_a = max(1e-6, float(rep.text_score))
                        text_b = max(1e-6, float(cand.text_score))
                        text_rel = abs(text_a - text_b) / max(text_a, text_b)
                        # Prevent transitive chaining: cluster bounding-box
                        # in time must remain <= span_factor * gap.
                        cluster_t_min = min(m.t_sec for m in cl.members)
                        cluster_t_max = max(m.t_sec for m in cl.members)
                        new_min = min(cluster_t_min, cand.t_sec)
                        new_max = max(cluster_t_max, cand.t_sec)
                        span_ok = (new_max - new_min) <= span_factor * gap
                        if text_rel <= text_tight and span_ok:
                            merged = True
                            reason = (
                                f'tertiary_high_hand(dt={dt:.2f}s,'
                                f'text_rel={text_rel:.2f},sim={sim:.2f},ham={ham})'
                            )
            if merged:
                cl.members.append(cand)
                cl.merge_reasons.append(reason)
                placed = True
                break
        if not placed:
            clusters.append(Cluster(members=[cand], merge_reasons=['seed']))
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

    # v13.3: deskew penalty in quality() is only used in high_hand_mode so
    # default behaviour matches v13.0 / v12.9 exactly for normal videos.
    soft_q = float(getattr(args, 'deskew_soft_threshold', 12.0))
    deskew_w_q = float(getattr(args, 'cluster_deskew_weight', 0.030))
    high_hand_q = bool(getattr(args, '_high_hand_mode', False))
    if high_hand_q:
        deskew_w_q *= 1.7
    else:
        deskew_w_q = 0.0  # disable deskew penalty entirely on default

    def quality(x: FrameFeatures) -> float:
        return (
            x.peak_score
            + 0.20 * x.norm_score
            - 0.28 * x.hand_text_overlap_penalty
            - 0.22 * x.bottom_hand_penalty
            - 0.16 * x.hand_penalty
            - deskew_soft_penalty(getattr(x, 'deskew_angle', 0.0), soft_q, deskew_w_q)
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


# ----------------------------------------------------------------------------
# v13.4 (IMG_4883) helpers ---------------------------------------------------
# ----------------------------------------------------------------------------

def _v134_text_row_profile(gray: np.ndarray) -> np.ndarray:
    """Coarse text-row profile (row-wise dark-pixel ratio after thresholding).

    Robust against perspective warp because we only compare 1D profiles, and
    against hand occlusion because the hand affects only a few rows in the
    bottom band; the upper text rows still match.
    """
    if gray is None:
        return np.zeros(64, dtype=np.float32)
    g = cv2.resize(gray, (192, 192), interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 12)
    rows = bw.mean(axis=1).astype(np.float32) / 255.0
    # Crop top/bottom 8% to reduce header/hand influence.
    n = rows.size
    rows = rows[int(n * 0.08):int(n * 0.92)]
    if rows.size == 0:
        return np.zeros(64, dtype=np.float32)
    rows = (rows - rows.mean()) / (rows.std() + 1e-6)
    return rows


def _v134_shift_corr(pa: np.ndarray, pb: np.ndarray, max_shift: int = 12) -> float:
    """Best correlation between two normalized 1D profiles allowing small shifts.

    Same physical page rendered with slightly different perspective produces a
    text-row profile that is shifted vertically. A direct dot-product can come
    out near zero if rows are off by even 4-6 pixels. Try shifts in
    [-max_shift, max_shift] and take the maximum.
    """
    if pa.size == 0 or pb.size == 0:
        return 0.0
    n = min(pa.size, pb.size)
    pa = pa[:n]
    pb = pb[:n]
    best = -1.0
    for s in range(-max_shift, max_shift + 1):
        if s >= 0:
            la = pa[s:]
            lb = pb[:n - s] if s > 0 else pb
        else:
            la = pa[:n + s]
            lb = pb[-s:]
        m = min(la.size, lb.size)
        if m < 8:
            continue
        c = float(np.mean(la[:m] * lb[:m]))
        if c > best:
            best = c
    return best


def _v134_warp_phash(warped_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Coarse 32x32 binarized thumbnail of the warped page.

    Used for a robust same-page test: the warped page is already
    perspective-rectified, so two takes of the same physical page produce
    nearly identical thumbnails apart from hand occlusion in the bottom band.
    """
    if warped_bgr is None:
        return None
    g = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    med = float(np.median(g))
    return (g > med).astype(np.uint8)


def _v134_warp_thumb_match(a: FrameFeatures, b: FrameFeatures) -> Tuple[float, int]:
    """Returns (match_ratio, hamming) over 32x32 warped-thumbnail bits.

    Ignores the bottom 6 rows of the thumbnail to discount hand occlusion at
    the page bottom.
    """
    pa = _v134_warp_phash(a.warped_bgr)
    pb = _v134_warp_phash(b.warped_bgr)
    if pa is None or pb is None or pa.shape != pb.shape:
        return 0.0, 32 * 32
    pa = pa[:26, :]
    pb = pb[:26, :]
    eq = (pa == pb).astype(np.uint8)
    ham = int(np.count_nonzero(pa != pb))
    ratio = float(np.count_nonzero(eq)) / float(eq.size)
    return ratio, ham


def _v134_profile_corr(a: FrameFeatures, b: FrameFeatures) -> float:
    if a.roi_gray is None or b.roi_gray is None:
        return 0.0
    pa = _v134_text_row_profile(a.roi_gray)
    pb = _v134_text_row_profile(b.roi_gray)
    if pa.size == 0 or pb.size == 0 or pa.size != pb.size:
        return 0.0
    return _v134_shift_corr(pa, pb, max_shift=14)


# ----------------------------------------------------------------------------
# v13.5 (IMG_4883) clean-visual scoring -------------------------------------
# ----------------------------------------------------------------------------

def _v135_compute_clean_visual_metrics(warped_bgr: np.ndarray) -> Dict[str, float]:
    """HSV-based descriptors of how clean a warped page looks.

    These are independent of the script's MediaPipe-based hand_penalty /
    bottom_hand_penalty, which are noisy on plain page margins. The metrics
    are intentionally simple so they remain robust across pages.
    """
    if warped_bgr is None or warped_bgr.size == 0:
        return dict(paper_ratio=0.0, v_mean=0.0, v_std=0.0,
                    skin_ratio=1.0, bot_skin=1.0, bot_v_mean=0.0,
                    blotch=255.0, text_edge=0.0)
    h, w = warped_bgr.shape[:2]
    hsv = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    paper = (V > 170) & (S < 60)
    paper_ratio = float(paper.mean())
    if paper.sum() > 100:
        v_mean = float(V[paper].mean())
        v_std = float(V[paper].std())
    else:
        v_mean = float(V.mean()); v_std = float(V.std())
    skin = ((H >= 0) & (H <= 25) & (S >= 30) & (S <= 170) & (V >= 60)).astype(np.uint8) * 255
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    skin_ratio = float(skin.mean()) / 255.0
    y0 = int(h * 0.72)
    bot_skin = float(skin[y0:].mean()) / 255.0 if y0 < h else 0.0
    bot_v_mean = float(V[y0:].mean()) if y0 < h else float(V.mean())
    Vsm = cv2.GaussianBlur(V, (51, 51), 0)
    if paper.sum() > 100:
        blotch = float(Vsm[paper].std())
    else:
        blotch = float(Vsm.std())
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    hp = cv2.absdiff(gray, cv2.GaussianBlur(gray, (15, 15), 0))
    text_edge = float(hp.std())
    return dict(
        paper_ratio=paper_ratio,
        v_mean=v_mean,
        v_std=v_std,
        skin_ratio=skin_ratio,
        bot_skin=bot_skin,
        bot_v_mean=bot_v_mean,
        blotch=blotch,
        text_edge=text_edge,
    )


def _v135_clean_visual_score(metrics: Dict[str, float]) -> float:
    """Combine the per-page metrics into a single 'cleaner is higher' score.

    Calibrated empirically against IMG_4883 dedication (frames 30-150) and
    Прlog (frames 180-285) candidates. Range typically [-1.5, +1.5].
    """
    paper = metrics.get('paper_ratio', 0.0)
    v_mean = metrics.get('v_mean', 0.0)
    bot_v = metrics.get('bot_v_mean', 0.0)
    skin = metrics.get('skin_ratio', 1.0)
    bot_skin = metrics.get('bot_skin', 1.0)
    blotch = metrics.get('blotch', 255.0)
    text_edge = metrics.get('text_edge', 0.0)
    # Brighter paper => higher.
    paper_term = 1.20 * paper
    v_term = 0.014 * (v_mean - 240.0)
    bot_v_term = 0.005 * (bot_v - 200.0)
    # Skin / fingers => penalty.
    skin_term = -1.40 * skin
    bot_skin_term = -1.10 * bot_skin
    # Background unevenness (gray patches, shadows) => penalty.
    blotch_term = -0.045 * max(0.0, blotch - 6.0)
    # Reward visible text edges, capped.
    edge_term = 0.012 * min(text_edge, 30.0)
    return float(paper_term + v_term + bot_v_term + skin_term
                 + bot_skin_term + blotch_term + edge_term)


def _v135_bg_gray_penalty(metrics: Dict[str, float]) -> float:
    """Diagnostic: how 'gray' the background is (0=white, ~1=quite gray)."""
    v_mean = metrics.get('v_mean', 255.0)
    blotch = metrics.get('blotch', 0.0)
    paper = metrics.get('paper_ratio', 0.0)
    a = max(0.0, (245.0 - v_mean) / 60.0)
    b = max(0.0, (blotch - 6.0) / 12.0)
    c = max(0.0, (0.92 - paper) / 0.30)
    return float(min(1.0, 0.40 * a + 0.40 * b + 0.20 * c))


def _v135_finger_penalty(metrics: Dict[str, float]) -> float:
    """Diagnostic: how much finger/skin is on the warped page (0..1)."""
    skin = metrics.get('skin_ratio', 0.0)
    bot_skin = metrics.get('bot_skin', 0.0)
    return float(min(1.0, 1.6 * bot_skin + 0.6 * skin))


def _v135_visual_metrics_cached(x: 'FrameFeatures') -> Dict[str, float]:
    """Compute and memoize clean-visual metrics on the FrameFeatures object."""
    cached = getattr(x, '_v135_visual_metrics', None)
    if cached is not None:
        return cached
    m = _v135_compute_clean_visual_metrics(x.warped_bgr) if x.warped_bgr is not None else {}
    try:
        setattr(x, '_v135_visual_metrics', m)
    except Exception:
        pass
    return m


def _v135_visual_score_cached(x: 'FrameFeatures') -> float:
    metrics = _v135_visual_metrics_cached(x)
    return _v135_clean_visual_score(metrics) if metrics else -1e9


def _v134_relaxed_same_page(a: FrameFeatures, b: FrameFeatures, args) -> Tuple[bool, float, int]:
    """Relaxed visual same-page test for adjacent winners.

    `is_visually_same_page` uses the strict (sim>=sim_thresh_merge AND
    ham<=hash_thresh_merge) gate which misses perspective-warped duplicates
    of the same physical page. The relaxed test triggers when ANY of three
    conditions holds together with similar text density:
      * SSIM-like similarity is moderately high; OR
      * dHash distance is small; OR
      * The horizontal-row text profile correlates strongly (most robust to
        perspective and hand occlusion).
    """
    if a.roi_gray is None or b.roi_gray is None or a.roi_dhash is None or b.roi_dhash is None:
        return False, 0.0, 64
    sim = float(similarity_score(a.roi_gray, b.roi_gray))
    ham = int(hamming_distance(a.roi_dhash, b.roi_dhash))
    text_a = max(1e-6, float(a.text_score))
    text_b = max(1e-6, float(b.text_score))
    text_rel = abs(text_a - text_b) / max(text_a, text_b)
    sim_relaxed = float(getattr(args, 'v134_dup_sim_min', 0.62))
    ham_relaxed = int(getattr(args, 'v134_dup_ham_max', 22))
    text_tol = float(getattr(args, 'v134_dup_text_rel', 0.35))
    profile_min = float(getattr(args, 'v134_dup_profile_min', 0.45))
    warp_ratio_min = float(getattr(args, 'v134_dup_warp_ratio', 0.78))
    if sim >= sim_relaxed and text_rel <= text_tol:
        return True, sim, ham
    if ham <= ham_relaxed and text_rel <= text_tol:
        return True, sim, ham
    # Profile-based test for perspective-warped duplicates of the same page
    # whose ROI ssim/hash is destroyed by warp/hand. Require similar text
    # density to avoid matching unrelated low-text pages.
    if text_rel <= text_tol:
        prof = _v134_profile_corr(a, b)
        if prof >= profile_min:
            return True, sim, ham
    # Warped-thumbnail match: the warped image is already perspective-
    # rectified, so two takes of the same physical page should produce
    # near-identical 32x32 binarized thumbnails apart from a hand band at
    # the bottom (which we ignore by clipping the bottom rows).
    if text_rel <= text_tol + 0.10:
        ratio, _ham_warp = _v134_warp_thumb_match(a, b)
        if ratio >= warp_ratio_min:
            return True, sim, ham
    # Strict path remains a positive too.
    if sim >= float(getattr(args, 'sim_thresh_merge', 0.89)) and ham <= int(getattr(args, 'hash_thresh_merge', 11)):
        return True, sim, ham
    return False, sim, ham


def repair_visual_duplicate_winners(
    selected: List[FrameFeatures],
    valid: List[FrameFeatures],
    args,
    diag: Dict[int, Dict[str, Any]],
) -> List[FrameFeatures]:
    """Replace visual duplicates among adjacent winners with novel candidates.

    Complements `repair_close_duplicate_gaps` which only triggers when
    selected pages are temporally close. On IMG_4883 the duplicate "Пролог"
    pair (frames 195, 255) sits 2.0s apart — wider than the close-pair
    threshold — so the temporal repair never sees it. This helper adds a
    visual check for adjacent winners regardless of temporal closeness.
    """
    if args.expected_pages <= 0 or len(selected) < 2 or not valid:
        return selected

    selected = sorted(selected, key=lambda x: x.t_sec)
    t0 = min(x.t_sec for x in valid)
    t1 = max(x.t_sec for x in valid)

    def quality(x: FrameFeatures) -> float:
        return (
            x.peak_score
            + 0.20 * x.norm_score
            - 0.30 * x.hand_text_overlap_penalty
            - 0.24 * x.bottom_hand_penalty
            - 0.16 * x.hand_penalty
        )

    max_iter = max(1, args.expected_pages)
    debug_print = bool(getattr(args, 'debug', False)) and bool(getattr(args, 'v134_verbose', False))
    for _iter in range(max_iter):
        selected = sorted(selected, key=lambda x: x.t_sec)
        dup_idx = None
        dup_sim = 0.0
        dup_ham = 64
        if debug_print:
            for i in range(len(selected) - 1):
                a, b = selected[i], selected[i + 1]
                ok, sim, ham = _v134_relaxed_same_page(a, b, args)
                prof = _v134_profile_corr(a, b)
                ratio, _h = _v134_warp_thumb_match(a, b)
                print(f'[v13.4 dup-check iter={_iter}] frame {a.frame_idx}<->{b.frame_idx}'
                      f' sim={sim:.3f} ham={ham} prof={prof:.3f} warp_ratio={ratio:.3f} same={ok}')
        # v14.1a strict sanity gate: even if the relaxed same-page test
        # fires (e.g. via profile-correlation or warped-thumbnail match),
        # refuse to treat the pair as duplicates when both raw visual
        # signals are weak. On IMG_4886 with --expected-pages 7 the v14.0
        # relaxed test fired at sim=0.33/ham=93 and merged two genuinely
        # different pages, losing pages 13/17.
        ham_strict_max = max(int(getattr(args, 'hash_thresh_merge', 11)) * 4, 40)
        sim_strict_min = float(getattr(args, 'v141a_dup_sim_floor', 0.50))
        for i in range(len(selected) - 1):
            a, b = selected[i], selected[i + 1]
            ok, sim, ham = _v134_relaxed_same_page(a, b, args)
            if ok and sim < sim_strict_min and ham > ham_strict_max:
                # Record skipped merge for diagnostics on the kept winner.
                skip_diag = diag.setdefault(int(a.frame_idx), {})
                prev_reason = skip_diag.get('reselection_reason', '')
                skip_diag['reselection_reason'] = (
                    (prev_reason + '|' if prev_reason else '')
                    + f'duplicate_merge_rejected(other={int(b.frame_idx)},'
                    f'sim={sim:.2f},ham={ham},reason=weak_visual_evidence)'
                )
                ok = False
            if ok:
                dup_idx, dup_sim, dup_ham = i, sim, ham
                break
        if dup_idx is None:
            break

        a, b = selected[dup_idx], selected[dup_idx + 1]
        # Drop the worse-quality member of the duplicate pair.
        # Protect a clear early-first-page situation: if the first slot is
        # uniquely covered only by `a`, keep `a` and drop `b`.
        early_protected = (dup_idx == 0 and (a.t_sec - t0) <= max(0.5, args.min_peak_distance_sec))
        if early_protected:
            remove_idx = dup_idx + 1
        else:
            remove_idx = dup_idx if quality(a) < quality(b) else dup_idx + 1
        removed = selected[remove_idx]
        trial = selected[:remove_idx] + selected[remove_idx + 1:]

        # Find the largest uncovered temporal gap (including head/tail).
        anchors = [t0] + [x.t_sec for x in trial] + [t1]
        gaps = [(anchors[i + 1] - anchors[i], anchors[i], anchors[i + 1]) for i in range(len(anchors) - 1)]
        _, ga, gb = max(gaps, key=lambda z: z[0])
        margin = min(0.85, max(0.15, (gb - ga) * 0.15))
        pool = [
            x for x in valid
            if ga + margin <= x.t_sec <= gb - margin
            and all(abs(x.t_sec - y.t_sec) >= max(0.5, args.min_peak_distance_sec * 0.65) for y in trial)
            and x.norm_score >= max(0.0, args.min_norm_score - 0.22)
        ]
        # Reject candidates that look like the kept duplicate or any other
        # already-selected winner.
        pool = [
            x for x in pool
            if all(not _v134_relaxed_same_page(x, y, args)[0] for y in trial)
        ]
        if not pool:
            break
        gap_center = 0.5 * (ga + gb)
        gap_half = max(1e-6, 0.5 * (gb - ga))
        replacement = max(pool, key=lambda x: (
            quality(x)
            + 1.10 * visual_novelty(x, trial)
            + 1.40 * max(0.0, 1.0 - abs(x.t_sec - gap_center) / gap_half)
        ))
        selected = sorted(trial + [replacement], key=lambda x: x.t_sec)
        rep_diag = diag.setdefault(int(replacement.frame_idx), {})
        rep_diag['reselection_reason'] = (
            f'duplicate_repaired(removed={int(removed.frame_idx)},'
            f'kept={int((b if remove_idx == dup_idx else a).frame_idx)},'
            f'sim={dup_sim:.2f},ham={dup_ham})'
        )
        rep_diag['duplicate_repaired'] = 1
        rep_diag['original_frame'] = int(removed.frame_idx)
        rep_diag['replacement_frame'] = int(replacement.frame_idx)

    return selected


def rescue_early_first_page(
    selected: List[FrameFeatures],
    valid: List[FrameFeatures],
    args,
    diag: Dict[int, Dict[str, Any]],
) -> List[FrameFeatures]:
    """Inject a clean first-page candidate when a substantial early gap exists.

    On IMG_4883 the title page is at frame 0 with norm_score=0.085 (passes
    the relaxed --expected-pages floor) but is never selected because no peak
    survives that low. When the first chosen winner sits at 4.5s but valid
    candidates exist at 0..2s with DIFFERENT visual content, we should add
    one as the new first page and drop the worst surplus winner.
    """
    if args.expected_pages <= 0 or not selected or not valid:
        return selected
    selected = sorted(selected, key=lambda x: x.t_sec)
    t0 = min(x.t_sec for x in valid)
    t1 = max(x.t_sec for x in valid)
    if t1 <= t0:
        return selected
    typical_gap = max(0.8, (t1 - t0) / max(1, args.expected_pages))
    first = selected[0]
    early_gap = first.t_sec - t0
    # Only fire if the gap before the first winner is non-trivial.
    if early_gap < max(1.5, typical_gap * 0.55):
        return selected

    floor = max(0.0, args.min_norm_score - 0.22)
    upper = first.t_sec - max(0.30, args.min_peak_distance_sec * 0.50)
    early_pool = [x for x in valid if t0 <= x.t_sec <= upper and x.norm_score >= floor]
    if not early_pool:
        return selected

    # Reject candidates that already match any selected winner visually.
    early_pool = [
        x for x in early_pool
        if all(not _v134_relaxed_same_page(x, y, args)[0] for y in selected)
    ]
    if not early_pool:
        return selected

    def early_quality(x: FrameFeatures) -> float:
        return (
            0.55 * x.peak_score
            + 0.35 * x.norm_score
            - 0.45 * x.hand_text_overlap_penalty
            - 0.40 * x.bottom_hand_penalty
            - 0.18 * x.hand_penalty
            + 0.08 * x.page_area_ratio
        )

    best = max(early_pool, key=early_quality)
    # Require some minimal cleanliness — otherwise we'd inject a hand-heavy
    # frame for the sake of slot coverage.
    if best.hand_text_overlap_penalty > 0.85 and best.bottom_hand_penalty > 0.85:
        return selected

    new_selected = list(selected)
    new_selected.append(best)
    new_selected = sorted(new_selected, key=lambda x: x.t_sec)
    if len(new_selected) > args.expected_pages:
        # Drop the worst surplus winner. Prefer dropping a winner that is a
        # visual duplicate of any other selected winner; otherwise drop the
        # lowest-quality late winner.
        def drop_score(x: FrameFeatures) -> float:
            return (
                x.peak_score
                - 0.30 * x.hand_text_overlap_penalty
                - 0.24 * x.bottom_hand_penalty
                - 0.16 * x.hand_penalty
            )
        # Never drop the new early winner.
        droppable = [x for x in new_selected if x.frame_idx != best.frame_idx]
        # Look for a duplicate among droppables.
        dup_target = None
        for i, x in enumerate(droppable):
            for y in droppable:
                if y is x:
                    continue
                ok, _, _ = _v134_relaxed_same_page(x, y, args)
                if ok and drop_score(x) < drop_score(y):
                    dup_target = x
                    break
            if dup_target is not None:
                break
        if dup_target is not None:
            new_selected = [x for x in new_selected if x.frame_idx != dup_target.frame_idx]
            removed_frame = int(dup_target.frame_idx)
        else:
            worst = min(droppable, key=drop_score)
            new_selected = [x for x in new_selected if x.frame_idx != worst.frame_idx]
            removed_frame = int(worst.frame_idx)
        rep_diag = diag.setdefault(int(best.frame_idx), {})
        rep_diag['reselection_reason'] = (
            f'first_page_rescue(t={best.t_sec:.2f}s,gap={early_gap:.2f}s,'
            f'replaced={removed_frame})'
        )
        rep_diag['original_frame'] = removed_frame
        rep_diag['replacement_frame'] = int(best.frame_idx)
    else:
        rep_diag = diag.setdefault(int(best.frame_idx), {})
        rep_diag['reselection_reason'] = (
            f'first_page_rescue(t={best.t_sec:.2f}s,gap={early_gap:.2f}s,added)'
        )
        rep_diag['original_frame'] = int(best.frame_idx)
        rep_diag['replacement_frame'] = int(best.frame_idx)
    return sorted(new_selected, key=lambda x: x.t_sec)


def prefer_cleaner_equivalent_winners(
    selected: List[FrameFeatures],
    valid: List[FrameFeatures],
    args,
    reselection_diag: Optional[Dict[int, Dict[str, Any]]] = None,
) -> List[FrameFeatures]:
    """For each chosen page, replace it with a cleaner equivalent nearby.

    Example: the first title page may have one sharp frame with a hand and one
    slightly softer frame without a hand. For final JPEG output, the clean frame
    is better.
    """
    out: List[FrameFeatures] = []

    # v13.3: deskew penalty is gated to high_hand_mode (default disabled).
    soft = float(getattr(args, 'deskew_soft_threshold', 12.0))
    deskew_w = float(getattr(args, 'cluster_deskew_weight', 0.030))
    high_hand_mode = bool(getattr(args, '_high_hand_mode', False))
    if high_hand_mode:
        deskew_w *= 1.7
    else:
        deskew_w = 0.0

    def final_quality(x: FrameFeatures) -> float:
        return (
            0.42 * x.peak_score
            + 0.25 * x.norm_score
            - 0.95 * x.hand_penalty
            - 0.85 * x.hand_text_overlap_penalty
            - 0.70 * x.bottom_hand_penalty
            - 0.25 * x.edge_foreground_penalty
            - deskew_soft_penalty(getattr(x, 'deskew_angle', 0.0), soft, deskew_w)
        )

    # v13.5: dynamic combination of final_quality and clean_visual_score.
    # final_quality uses script-internal hand penalties (sometimes false-
    # positive) while clean_visual_score is independent and HSV-based. Mixing
    # both makes us robust to either signal alone failing.
    cv_weight = float(getattr(args, 'v135_clean_visual_weight', 0.55))
    cv_min_gain = float(getattr(args, 'v135_clean_visual_min_gain', 0.06))
    cv_dominant_delta = float(getattr(args, 'v135_clean_visual_dominant_delta', 0.18))

    for win in selected:
        # v13.4: widen the same-page search window when the winner has a
        # visible hand penalty, because the cleaner equivalent (no hand at
        # all on the page) may be 1-2 seconds earlier or later than the peak
        # frame the scorer chose. v13.5: also widen the window unconditionally
        # to a moderate size so cleaner same-page equivalents (judged by
        # clean_visual_score, not by the hand penalty alone) can be found.
        dirty_winner = (
            float(getattr(win, 'hand_text_overlap_penalty', 0.0)) > 0.55
            or float(getattr(win, 'bottom_hand_penalty', 0.0)) > 0.65
            or float(getattr(win, 'hand_penalty', 0.0)) > 0.85
        )
        # v13.5: also treat winners as dirty when clean_visual diagnostics
        # signal real bottom-band skin or grayish background. This catches the
        # IMG_4883 page_003 (frame 255) case where hand_penalty is moderate
        # but the clean_visual finger_penalty is high.
        win_metrics = _v135_visual_metrics_cached(win)
        win_finger = _v135_finger_penalty(win_metrics) if win_metrics else 0.0
        win_bg_gray = _v135_bg_gray_penalty(win_metrics) if win_metrics else 0.0
        clean_dirty = (win_finger > 0.18) or (win_bg_gray > 0.32)
        effective_dirty = dirty_winner or clean_dirty
        if effective_dirty:
            window_sec = max(3.0, args.min_peak_distance_sec * 3.0)
        else:
            # v13.5: widen the default window so we can find the canonical
            # clean equivalent (e.g. f60 dedication 2 seconds before f105) even
            # when the winner does not look obviously dirty by hand penalty.
            window_sec = max(1.8, args.min_peak_distance_sec * 1.8)
        pool = []
        # v13.5: same-physical-page gate. We require ALL of:
        #   (a) similarity_score (SSIM-like on roi_gray) above a moderate floor
        #       OR a tight dHash hamming, OR a high warp-thumbnail ratio;
        #   (b) text-row profile correlation above a strict floor.
        # (b) is what stops the dedication win=135 from pooling Prologue f195
        # (which has prof=0.242 between f195 and f135, well below the 0.45
        # floor). Same-page Prologue f195<->255 also has prof=0.243 in this
        # video, but f255 vs f255 obviously passes; the value f195 vs f255 is
        # only used by the duplicate-repair pass, not here.
        prof_min = float(getattr(args, 'v135_pool_profile_min', 0.20))
        warp_min_strong = float(getattr(args, 'v135_pool_warp_strong', 0.72))
        # v14.0: bound how many candidates we visually score against the
        # winner. Without --audit-candidates we keep at most reselection_top_k
        # entries (chosen by visual proximity to the winner) so high sample-FPS
        # videos do not balloon this loop into O(N) per winner.
        top_k = int(getattr(args, 'reselection_top_k', 6) or 0)
        audit_mode = bool(getattr(args, 'audit_candidates', False))
        scored_pool: List[Tuple[float, 'FrameFeatures']] = []
        for cand in valid:
            if abs(cand.t_sec - win.t_sec) > window_sec:
                continue
            if cand.roi_gray is None or win.roi_gray is None:
                continue
            if cand.frame_idx == win.frame_idx:
                pool.append(cand)
                continue
            sim = similarity_score(cand.roi_gray, win.roi_gray)
            ham = hamming_distance(cand.roi_dhash, win.roi_dhash)
            try:
                wr, _ = _v134_warp_thumb_match(cand, win)
            except Exception:
                wr = 0.0
            try:
                prof = _v134_profile_corr(cand, win)
            except Exception:
                prof = 0.0
            # v13.5: a candidate is in the same-physical-page pool when EITHER:
            #  * warp_thumb match is strong (>= warp_min_strong, robust to
            #    hand occlusion since the bottom rows are excluded), OR
            #  * any of {sim>=0.32, ham<=62} AND profile correlation>=prof_min.
            # The second branch keeps cross-page rejects (which usually have
            # both low sim/ham AND low profile) out of the pool.
            strong_warp = wr >= warp_min_strong
            visual_match = sim >= 0.32 or ham <= 62
            if strong_warp or (visual_match and prof >= prof_min):
                # closeness key: the more similar the candidate, the bigger.
                key = max(sim, wr) + 0.05 * max(0.0, prof) - 0.005 * ham
                scored_pool.append((key, cand))
        if scored_pool:
            if not audit_mode and top_k > 0 and len(scored_pool) > top_k:
                scored_pool.sort(key=lambda kv: kv[0], reverse=True)
                scored_pool = scored_pool[:top_k]
            pool.extend(c for _, c in scored_pool)
        # v13.5: rank candidates by a blend of final_quality and
        # clean_visual_score (cv_weight). Require a minimum gain over the
        # current winner so we never swap on noise. When clean_visual_score
        # of a candidate dominates the winner by `cv_dominant_delta` AND its
        # finger_penalty is much lower, override the blended ranking — this
        # rescues frames whose script-internal hand_penalty is a false
        # positive on plain page margins (e.g. IMG_4883 dedication f60).
        def blended(x: 'FrameFeatures') -> float:
            return (1.0 - cv_weight) * final_quality(x) + cv_weight * _v135_visual_score_cached(x)

        if pool:
            # Primary ranking: blended score.
            chosen = max(pool, key=blended)
            # Override branch: a dramatically cleaner candidate (cvs much
            # higher AND finger_penalty much lower) can win even when its
            # blended score is slightly lower because of false-positive hand
            # penalties.
            win_cvs = _v135_visual_score_cached(win)
            verbose = bool(getattr(args, 'v134_verbose', False))
            for cand in pool:
                cm = _v135_visual_metrics_cached(cand)
                cf = _v135_finger_penalty(cm)
                cb = _v135_bg_gray_penalty(cm)
                cs = _v135_clean_visual_score(cm) if cm else -1e9
                cs_gain = cs - win_cvs
                fg_ok = cf <= max(0.10, win_finger - 0.10)
                bg_ok = cb <= win_bg_gray + 0.05
                if verbose and cand.frame_idx != win.frame_idx:
                    print(f'  [v13.5 cand] {cand.frame_idx} cvs={cs:.3f} (gain={cs_gain:.3f}) '
                          f'finger={cf:.2f} bg={cb:.2f} fg_ok={fg_ok} bg_ok={bg_ok} '
                          f'win_finger={win_finger:.2f} win_bg={win_bg_gray:.2f}')
                if cs_gain >= cv_dominant_delta and fg_ok and bg_ok:
                    if _v135_clean_visual_score(_v135_visual_metrics_cached(chosen)) < cs:
                        chosen = cand
            if chosen.frame_idx != win.frame_idx:
                # For dominant-clean swaps, clean_visual gain alone is enough.
                cm = _v135_visual_metrics_cached(chosen)
                cs = _v135_clean_visual_score(cm) if cm else -1e9
                cvs_gain = cs - win_cvs
                bl_gain = blended(chosen) - blended(win)
                if bl_gain < cv_min_gain and cvs_gain < cv_dominant_delta:
                    chosen = win
        else:
            chosen = win
        chosen_metrics = _v135_visual_metrics_cached(chosen)
        chosen_cvs = _v135_visual_score_cached(chosen)
        chosen_bg = _v135_bg_gray_penalty(chosen_metrics) if chosen_metrics else 0.0
        chosen_finger = _v135_finger_penalty(chosen_metrics) if chosen_metrics else 0.0
        if bool(getattr(args, 'v134_verbose', False)):
            print(f'[v13.5 prefer] win={win.frame_idx} t={win.t_sec:.2f} '
                  f'dirty={int(dirty_winner)}+cv_dirty={int(clean_dirty)} '
                  f'window={window_sec:.2f}s pool=[{",".join(str(c.frame_idx) for c in pool)}] '
                  f'chosen={chosen.frame_idx} cvs={chosen_cvs:.3f} '
                  f'bg_gray={chosen_bg:.2f} finger={chosen_finger:.2f}')
        out.append(chosen)
        if reselection_diag is not None and chosen.frame_idx != win.frame_idx:
            # Preserve diagnostics from the previous reselection step (e.g.
            # rescue_early_first_page set diag for `win` and we are now
            # replacing `win` with a cleaner same-page neighbour). Otherwise
            # record a fresh "cleaner_equivalent" entry.
            prev = reselection_diag.pop(win.frame_idx, None)
            entry = reselection_diag.setdefault(int(chosen.frame_idx), {})
            if prev:
                # Chain reasons: keep the original cause (e.g. first_page_rescue)
                # but note the cleaner-equivalent swap.
                entry['reselection_reason'] = (
                    f"{prev.get('reselection_reason', '')}|"
                    f"cleaner_equivalent_v135(orig={int(win.frame_idx)},new={int(chosen.frame_idx)},dirty={int(effective_dirty)})"
                )
                entry['original_frame'] = prev.get('original_frame', int(win.frame_idx))
                entry['replacement_frame'] = int(chosen.frame_idx)
                entry['duplicate_repaired'] = prev.get('duplicate_repaired', 0)
            else:
                entry['reselection_reason'] = (
                    f"cleaner_equivalent_v135(orig={int(win.frame_idx)},new={int(chosen.frame_idx)},dirty={int(effective_dirty)})"
                )
                entry['original_frame'] = int(win.frame_idx)
                entry['replacement_frame'] = int(chosen.frame_idx)
        # v13.5: stash diagnostics for winners.csv on every winner (not only
        # swapped ones). This is consumed by the writer at the bottom.
        if reselection_diag is not None:
            entry = reselection_diag.setdefault(int(chosen.frame_idx), {})
            entry['clean_visual_score'] = float(chosen_cvs)
            entry['bg_gray_penalty'] = float(chosen_bg)
            entry['finger_penalty'] = float(chosen_finger)
            entry['candidate_search_window'] = float(window_sec)

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
                new_first = min(early_pool, key=lambda x: x.t_sec)
                if new_first.frame_idx != first.frame_idx:
                    if reselection_diag is not None:
                        prev = reselection_diag.pop(first.frame_idx, None)
                        entry = reselection_diag.setdefault(int(new_first.frame_idx), {})
                        if prev:
                            entry['reselection_reason'] = (
                                f"{prev.get('reselection_reason', '')}|"
                                f"first_page_earliest(orig={int(first.frame_idx)},new={int(new_first.frame_idx)})"
                            )
                            entry['original_frame'] = prev.get('original_frame', int(first.frame_idx))
                            entry['replacement_frame'] = int(new_first.frame_idx)
                            entry['duplicate_repaired'] = prev.get('duplicate_repaired', 0)
                        else:
                            entry['reselection_reason'] = (
                                f"first_page_earliest(orig={int(first.frame_idx)},new={int(new_first.frame_idx)})"
                            )
                            entry['original_frame'] = int(first.frame_idx)
                            entry['replacement_frame'] = int(new_first.frame_idx)
                out[0] = new_first

    # Keep chronological order and prevent accidental duplicates after swaps.
    cleaned: List[FrameFeatures] = []
    for cand in sorted(out, key=lambda x: x.t_sec):
        if cleaned and abs(cand.t_sec - cleaned[-1].t_sec) < args.min_peak_distance_sec * 0.55:
            cleaned[-1] = choose_between_similar(cleaned[-1], cand, 0.42)
        else:
            cleaned.append(cand)
    return cleaned


def auto_dedup_default_winners(
    winners: List[FrameFeatures],
    args,
    diag: Optional[Dict[int, Dict[str, Any]]] = None,
) -> List[FrameFeatures]:
    """V14.2a: production-safe late dedup for default mode (no --expected-pages).

    Compares each winner only to its next ``auto_dedup_neighbors`` neighbors
    (default 2) and removes one of the pair only when there is *strong*
    evidence that they depict the same physical page. Strong evidence =
    relaxed-same-page test passes AND a corroborating signal (warp-thumb
    ratio at a tighter floor or central-row profile correlation at a
    tighter floor). False merge is worse than a duplicate, so when in
    doubt the function keeps both winners.

    The kept winner is the one with the better quality score (lower hand
    penalty / higher peak_score / higher clean_visual_score when present).
    Diagnostics are recorded against the *removed* frame_idx via
    ``auto_dedup_removed`` / ``auto_dedup_reason`` /
    ``auto_dedup_pair_frame``.
    """
    if not winners or len(winners) < 2:
        return winners
    if not bool(getattr(args, 'auto_dedup_default', True)):
        return winners
    if int(getattr(args, 'expected_pages', 0) or 0) > 0:
        return winners
    neighbor_window = max(1, int(getattr(args, 'auto_dedup_neighbors', 2)))

    # Tighter thresholds than the relaxed test alone to avoid false merges.
    sim_strong = 0.78
    ham_strong = 14
    profile_strong = 0.80
    warp_strong = 0.88
    text_tol_strong = 0.22

    def quality_key(x: FrameFeatures) -> float:
        # Higher is better.
        peak = float(getattr(x, 'peak_score', -1e9))
        if peak <= -1e8:
            peak = float(getattr(x, 'norm_score', 0.0))
        cv = float(getattr(x, 'clean_visual_score', 0.0) or 0.0)
        hand = float(getattr(x, 'hand_text_overlap_penalty', 0.0) or 0.0)
        bot = float(getattr(x, 'bottom_hand_penalty', 0.0) or 0.0)
        hp = float(getattr(x, 'hand_penalty', 0.0) or 0.0)
        return peak + 0.20 * cv - 0.45 * hand - 0.35 * bot - 0.20 * hp

    kept: List[FrameFeatures] = []
    removed_ids: set = set()
    for i, cand in enumerate(winners):
        if cand.frame_idx in removed_ids:
            continue
        merged_into_prev = False
        for back in range(1, neighbor_window + 1):
            if not kept:
                break
            j = len(kept) - back
            if j < 0:
                break
            prev = kept[j]
            try:
                ok, sim, ham = _v134_relaxed_same_page(prev, cand, args)
            except Exception:
                ok, sim, ham = False, 0.0, 64
            if not ok:
                continue
            text_a = max(1e-6, float(getattr(prev, 'text_score', 0.0)))
            text_b = max(1e-6, float(getattr(cand, 'text_score', 0.0)))
            text_rel = abs(text_a - text_b) / max(text_a, text_b)
            if text_rel > text_tol_strong:
                continue
            primary = (sim >= sim_strong) or (ham <= ham_strong)
            corroborated = False
            reason_parts: List[str] = []
            if primary:
                reason_parts.append(f'primary(sim={sim:.2f},ham={ham})')
            try:
                prof = float(_v134_profile_corr(prev, cand))
            except Exception:
                prof = 0.0
            if prof >= profile_strong:
                corroborated = True
                reason_parts.append(f'profile={prof:.2f}')
            try:
                ratio, _hw = _v134_warp_thumb_match(prev, cand)
                ratio = float(ratio)
            except Exception:
                ratio = 0.0
            if ratio >= warp_strong:
                corroborated = True
                reason_parts.append(f'warp={ratio:.2f}')
            if not (primary and corroborated):
                continue
            # Strong evidence: keep the better-quality winner.
            if quality_key(cand) > quality_key(prev):
                # Replace prev with cand.
                loser = prev
                kept[j] = cand
            else:
                loser = cand
            reason = 'auto_dedup:' + '+'.join(reason_parts) + f',text_rel={text_rel:.2f}'
            removed_ids.add(loser.frame_idx)
            if diag is not None:
                diag.setdefault(loser.frame_idx, {}).update({
                    'auto_dedup_removed': 1,
                    'auto_dedup_reason': reason,
                    'auto_dedup_pair_frame': int(kept[j].frame_idx),
                })
            merged_into_prev = True
            break
        if not merged_into_prev:
            kept.append(cand)
    return kept


def fill_expected_pages_by_time(
    winners: List[FrameFeatures],
    valid: List[FrameFeatures],
    args,
    reselection_diag: Optional[Dict[int, Dict[str, Any]]] = None,
) -> List[FrameFeatures]:
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
    if reselection_diag is None:
        reselection_diag = {}

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
    # v13.4: catch visual duplicates that fall outside the close-pair time
    # threshold (e.g. IMG_4883 frames 195/255).
    selected = repair_visual_duplicate_winners(selected, valid, args, reselection_diag)
    # v13.4: rescue an early clean first page (e.g. title page at frame 0 on
    # IMG_4883) when a substantial gap precedes the first selected winner.
    selected = rescue_early_first_page(selected, valid, args, reselection_diag)
    selected = prefer_cleaner_equivalent_winners(selected, valid, args, reselection_diag)
    # v14.1b: count repair. After all v13.4/v13.5/v14.1a reselection logic,
    # if the sanity gate rejected a false merge or the original cluster pass
    # never produced a winner in some temporal slot, len(selected) may still
    # be < expected_pages. Try to fill the count from the largest uncovered
    # temporal gaps with strict anti-duplicate checks.
    selected = repair_expected_count_fill(selected, valid, args, reselection_diag)
    # v13.5: the post-cleaner-equivalent repair pass was firing false positives
    # on IMG_4883 (e.g. classifying Prologue f195 and chapter f315 as the same
    # page because their warp_thumb ratio is 0.80). The cleaner_equivalent
    # pass already enforces same-physical-page identity via its own pool
    # gate, so a second relaxed-threshold sweep is no longer needed here.
    # If a real new duplicate appears, it will still be visible in the
    # winners.csv reselection diagnostics for manual review.
    return sorted(selected, key=lambda x: x.t_sec)


def repair_expected_count_fill(
    selected: List[FrameFeatures],
    valid: List[FrameFeatures],
    args,
    diag: Dict[int, Dict[str, Any]],
) -> List[FrameFeatures]:
    """v14.1b: fill missing expected-pages slots from largest temporal gaps.

    Activates only when --expected-pages > 0 and len(selected) < expected.
    For each iteration:
      * compute uncovered temporal gaps (head, tail, between winners),
      * pick the largest gap and its top-K candidates by quality,
      * accept the first candidate that passes a strict distinctness check
        against every existing winner (relaxed same-page must NOT fire,
        SSIM and hash margins must exceed conservative thresholds, and time
        clustering must respect min_peak_distance_sec).
    Diagnostics: per-added winner reselection_reason, expected_fill_*
    columns are populated through `diag`.
    """
    if args.expected_pages <= 0 or not valid:
        return selected
    if len(selected) >= args.expected_pages:
        return selected

    selected = sorted(selected, key=lambda x: x.t_sec)
    t0 = min(x.t_sec for x in valid)
    t1 = max(x.t_sec for x in valid)
    if t1 <= t0:
        return selected

    typical_gap = (t1 - t0) / max(1, args.expected_pages)
    floor = max(0.0, args.min_norm_score - 0.22)
    top_k = int(getattr(args, 'v141b_fill_top_k', 6) or 6)
    sim_distinct_max = float(getattr(args, 'v141b_fill_sim_max', 0.70))
    ham_distinct_min = int(getattr(args, 'v141b_fill_ham_min', 18))
    novelty_min = float(getattr(args, 'v141b_fill_novelty_min', 0.18))
    min_gap_factor = float(getattr(args, 'v141b_fill_min_gap_factor', 0.60))
    min_time_sep = max(0.6, args.min_peak_distance_sec * 0.85)
    debug_print = bool(getattr(args, 'debug', False))

    def quality(x: FrameFeatures) -> float:
        return (
            x.peak_score
            + 0.20 * x.norm_score
            - 0.30 * x.hand_text_overlap_penalty
            - 0.24 * x.bottom_hand_penalty
            - 0.16 * x.hand_penalty
        )

    def is_distinct(cand: FrameFeatures, winners: List[FrameFeatures]) -> Tuple[bool, str, float]:
        # Reject if cand is the same frame as any winner.
        for w in winners:
            if w.frame_idx == cand.frame_idx:
                return False, f'same_frame_as_{int(w.frame_idx)}', 0.0
            # Reject if too close in time.
            if abs(w.t_sec - cand.t_sec) < min_time_sep:
                return False, f'too_close_t={w.t_sec:.2f}_vs_{cand.t_sec:.2f}', 0.0
            # Reject if relaxed same-page test fires (uses v14.1a sanity gate-
            # compatible thresholds in _v134_relaxed_same_page).
            ok, sim_pair, ham_pair = _v134_relaxed_same_page(cand, w, args)
            if ok:
                return False, f'relaxed_dup_with_{int(w.frame_idx)}(sim={sim_pair:.2f},ham={ham_pair})', 0.0
            # Strict raw distinctness margins: refuse if SSIM is high OR
            # hash distance is small. These are conservative — must clear
            # both gates together.
            if cand.roi_gray is not None and w.roi_gray is not None:
                sim = float(similarity_score(cand.roi_gray, w.roi_gray))
                ham = int(hamming_distance(cand.roi_dhash, w.roi_dhash))
                if sim >= sim_distinct_max and ham <= ham_distinct_min:
                    return False, f'visual_close_{int(w.frame_idx)}(sim={sim:.2f},ham={ham})', 0.0
        nov = visual_novelty(cand, winners)
        if nov < novelty_min:
            return False, f'low_novelty({nov:.2f})', nov
        return True, 'ok', nov

    max_iter = max(1, args.expected_pages - len(selected) + 2)
    for _it in range(max_iter):
        if len(selected) >= args.expected_pages:
            break
        selected = sorted(selected, key=lambda x: x.t_sec)
        anchors = [t0] + [x.t_sec for x in selected] + [t1]
        gaps = [
            (anchors[i + 1] - anchors[i], anchors[i], anchors[i + 1], i)
            for i in range(len(anchors) - 1)
        ]
        # Sort gaps by size, largest first.
        gaps.sort(key=lambda z: z[0], reverse=True)
        added_this_iter = False
        for gap_size, ga, gb, gap_idx in gaps:
            # Skip tiny gaps that cannot host an additional winner.
            if gap_size < typical_gap * min_gap_factor:
                continue
            margin = min(0.85, max(0.15, gap_size * 0.10))
            ga_in = ga + margin
            gb_in = gb - margin
            if gb_in <= ga_in:
                continue
            pool = [
                x for x in valid
                if ga_in <= x.t_sec <= gb_in
                and x.norm_score >= floor
                and all(x.frame_idx != s.frame_idx for s in selected)
            ]
            if not pool:
                continue
            # Rank top-K by quality and take the K best as initial pool.
            pool.sort(key=lambda x: (quality(x), x.peak_score), reverse=True)
            pool = pool[: max(1, top_k)]
            chosen = None
            chosen_reason = ''
            chosen_distinctness = 0.0
            for cand in pool:
                ok, why, nov = is_distinct(cand, selected)
                if ok:
                    chosen = cand
                    chosen_reason = why
                    chosen_distinctness = nov
                    break
                if debug_print:
                    print(f'[v14.1b]   gap[{ga:.2f}..{gb:.2f}] reject f{int(cand.frame_idx)}: {why}')
            if chosen is None:
                continue
            selected.append(chosen)
            entry = diag.setdefault(int(chosen.frame_idx), {})
            existing_reason = entry.get('reselection_reason', '') or ''
            fill_reason = (
                f'expected_fill(t={chosen.t_sec:.2f}s,gap=[{ga:.2f},{gb:.2f}],'
                f'novelty={chosen_distinctness:.2f})'
            )
            entry['reselection_reason'] = (
                (existing_reason + '|' if existing_reason else '') + fill_reason
            )
            entry['expected_fill_applied'] = 1
            entry['expected_fill_reason'] = chosen_reason
            entry['fill_source_gap'] = f'{ga:.2f}-{gb:.2f}'
            entry['fill_distinctness_score'] = float(chosen_distinctness)
            entry['fill_candidate_frame'] = int(chosen.frame_idx)
            if debug_print:
                print(
                    f'[v14.1b] expected_fill: added frame {int(chosen.frame_idx)} '
                    f'(t={chosen.t_sec:.2f}s) into gap [{ga:.2f},{gb:.2f}] '
                    f'novelty={chosen_distinctness:.2f}'
                )
            added_this_iter = True
            break
        if not added_this_iter:
            if debug_print:
                print(
                    f'[v14.1b] expected_fill: no safe candidate in any gap; '
                    f'leaving {len(selected)}/{args.expected_pages} pages.'
                )
            break

    return selected


def force_reduce(clusters: List[Cluster], expected_pages: int, args=None) -> List[Cluster]:
    if expected_pages <= 0 or len(clusters) <= expected_pages:
        return clusters
    high_hand = bool(getattr(args, '_high_hand_mode', False)) if args is not None else False
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
                if high_hand:
                    # In high-hand mode dHash and ssim are unreliable so the
                    # default similarity-based pair pick may end up fusing two
                    # clusters that are *not* the same physical page. Prefer
                    # to fuse clusters that are adjacent in time -- excess
                    # clusters above expected_pages are most likely
                    # consecutive duplicates of the same page turn rather than
                    # arbitrary lookalikes from different parts of the book.
                    dt = abs(a.t_sec - b.t_sec)
                    text_a = max(1e-6, float(a.text_score))
                    text_b = max(1e-6, float(b.text_score))
                    text_rel = abs(text_a - text_b) / max(text_a, text_b)
                    # Strong proximity bonus, weak text bonus.
                    score += 1.20 * max(0.0, 1.0 - min(dt, 4.0) / 4.0)
                    score += 0.20 * max(0.0, 1.0 - min(text_rel, 0.6) / 0.6)
                if score > best_score:
                    best_score = score
                    best_pair = (i, j)
        if best_pair is None:
            break
        i, j = best_pair
        clusters[i].members.extend(clusters[j].members)
        # Track that this merge happened during the force_reduce stage; this is
        # only triggered when --expected-pages forces a smaller cluster count.
        clusters[i].merge_reasons.extend([f'force_reduce(sim={best_score:.2f})' for _ in clusters[j].members])
        del clusters[j]
    return clusters


# ---------------------------------------------------------------------------
# V13.0: Per-video unsupervised self-calibration.
#
# The idea: before the main detection pass we sweep cheap, robust features
# across a small budget of evenly spaced frames (no warp, no MediaPipe), build
# robust per-video distributions (median, MAD, p10/p50/p75/p90), and then
# adapt a tightly clamped subset of thresholds the main pipeline uses. This is
# unsupervised — no labels, no learning of weights, no NN training. We only
# learn what "this particular video" looks like and nudge a couple of
# thresholds inside narrow safety bands so v12.9-quality videos see no change
# while harder videos benefit from more accurate normalization ranges.
# ---------------------------------------------------------------------------


def _percentiles(arr: np.ndarray, qs=(10, 25, 50, 75, 90)) -> Dict[str, float]:
    out: Dict[str, float] = {}
    a = np.asarray(arr, dtype=np.float32)
    a = a[np.isfinite(a)]
    if a.size == 0:
        for q in qs:
            out[f'p{q}'] = float('nan')
        out['median'] = float('nan')
        out['mad'] = float('nan')
        return out
    for q in qs:
        out[f'p{q}'] = float(np.percentile(a, q))
    med = float(np.median(a))
    out['median'] = med
    out['mad'] = float(np.median(np.abs(a - med)))
    return out


def _cheap_paper_ratio(small_bgr: np.ndarray) -> Tuple[float, float, float]:
    """Cheap paper-like area / brightness / saturation estimate on a small image."""
    hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    bright = (v > 110).astype(np.uint8)
    low_sat = (s < 70).astype(np.uint8)
    paper = cv2.bitwise_and(bright, low_sat) * 255
    paper = cv2.morphologyEx(paper, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    ratio = float(np.count_nonzero(paper)) / float(paper.size)
    bright_mean = float(np.mean(v))
    sat_mean = float(np.mean(s))
    return ratio, bright_mean, sat_mean


def _cheap_skin_ratio(small_bgr: np.ndarray) -> float:
    try:
        m = skin_like_mask(small_bgr)
        return float(np.count_nonzero(m)) / float(m.size)
    except Exception:
        return 0.0


def _v151_calibration_from_prefilter(video_path: Path, args, fps: float,
                                      prefilter_metrics: Dict[int, Dict[str, float]],
                                      sampled_indices: List[int]) -> Dict[str, Any]:
    """v15.1: build adaptive calibration from prefilter low-res metrics so
    we don't decode the video twice. Reuses the same override logic as
    run_adaptive_calibration but skips text_density-based hash adjustments
    (text_density is not part of the prefilter metric set; legacy default
    is preserved).
    """
    cal: Dict[str, Any] = {
        'enabled': True,
        'used': False,
        'reason': '',
        'video': str(video_path),
        'sample_fps': float(getattr(args, 'calibration_sample_fps', 1.0)),
        'max_frames': int(getattr(args, 'calibration_max_frames', 60)),
        'samples': 0,
        'stats': {},
        'overrides': {},
        'source': 'prefilter',
        'guardrails': {
            'min_norm_score_min': 0.20,
            'min_norm_score_max': 0.36,
            'sim_thresh_merge_min': 0.86,
            'sim_thresh_merge_max': 0.92,
            'hash_thresh_merge_min': 9,
            'hash_thresh_merge_max': 13,
        },
    }
    if not prefilter_metrics or not sampled_indices:
        cal['reason'] = 'no_prefilter_metrics'
        return cal
    paper_ratios: List[float] = []
    brights: List[float] = []
    sats: List[float] = []
    blurs: List[float] = []
    skin_ratios: List[float] = []
    for fi in sampled_indices:
        m = prefilter_metrics.get(fi)
        if not m:
            continue
        paper_ratios.append(float(m.get('paper_ratio', 0.0)))
        brights.append(float(m.get('bright_mean', 0.0)))
        sats.append(float(m.get('sat_mean', 0.0)))
        blurs.append(float(m.get('blur', 0.0)))
        skin_ratios.append(float(m.get('skin', 0.0)))
    cal['samples'] = len(paper_ratios)
    if cal['samples'] < 6:
        cal['reason'] = 'too_few_samples'
        return cal
    n_frames = 0
    try:
        capx = cv2.VideoCapture(str(video_path))
        if capx.isOpened():
            n_frames = int(capx.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        capx.release()
    except Exception:
        pass
    cal['stats'] = {
        'paper_ratio': _percentiles(np.array(paper_ratios)),
        'brightness': _percentiles(np.array(brights)),
        'saturation': _percentiles(np.array(sats)),
        'blur_lap_var': _percentiles(np.array(blurs)),
        'text_density': _percentiles(np.array([])),  # not available from prefilter
        'skin_ratio': _percentiles(np.array(skin_ratios)),
        'fps': float(fps),
        'frame_count': int(n_frames),
        'duration_sec': float(n_frames) / float(fps if fps > 0 else 30.0),
    }
    overrides: Dict[str, Any] = {}
    guards = cal['guardrails']
    # NOTE: blur is computed at the prefilter long-side (default 512) which
    # is not the same scale as the legacy 480px calibration sample. To avoid
    # producing different min_norm_score overrides than legacy calibration,
    # we deliberately skip the blur-spread adjustment here. Post-pass
    # adaptive_post_calibration still nudges min_norm_score using the
    # final norm_score distribution, so the operating point is preserved.
    skin_p75 = cal['stats']['skin_ratio']['p75']
    skin_p50 = cal['stats']['skin_ratio'].get('median', float('nan'))
    if np.isfinite(skin_p75) and skin_p75 > 0.30:
        base = float(getattr(args, 'sim_thresh_merge', 0.89))
        v = float(np.clip(base - 0.01, guards['sim_thresh_merge_min'], guards['sim_thresh_merge_max']))
        if abs(v - base) >= 0.005:
            overrides['sim_thresh_merge'] = v
    median_skin = skin_p50
    high_hand = (
        (np.isfinite(median_skin) and median_skin > 0.45)
        or (np.isfinite(skin_p75) and skin_p75 > 0.55)
    )
    if high_hand:
        cal['high_hand_mode'] = True
        base_sim = float(getattr(args, 'sim_thresh_merge', 0.89))
        v = float(np.clip(base_sim - 0.02, guards['sim_thresh_merge_min'], guards['sim_thresh_merge_max']))
        if abs(v - base_sim) >= 0.005:
            overrides['sim_thresh_merge'] = v
        base_min = float(getattr(args, 'min_norm_score', 0.28))
        v = float(np.clip(base_min - 0.02, guards['min_norm_score_min'], guards['min_norm_score_max']))
        if abs(v - base_min) >= 0.005:
            overrides['min_norm_score'] = v
    else:
        cal['high_hand_mode'] = False
    cal['overrides'] = overrides
    cal['used'] = True
    cal['reason'] = 'ok_from_prefilter'
    return cal


def run_adaptive_calibration(video_path: Path, args) -> Dict[str, Any]:
    """Pre-pass: sample frames, build robust feature distributions, and emit
    safe per-video adaptive overrides.

    Returns a dict with two top-level keys:
        'stats': raw feature percentiles for diagnostics
        'overrides': sparse dict of suggested CLI-level threshold tweaks
                     (clamped to safe bands; may be empty)
    """
    cal: Dict[str, Any] = {
        'enabled': True,
        'used': False,
        'reason': '',
        'video': str(video_path),
        'sample_fps': float(getattr(args, 'calibration_sample_fps', 1.0)),
        'max_frames': int(getattr(args, 'calibration_max_frames', 60)),
        'samples': 0,
        'stats': {},
        'overrides': {},
        'guardrails': {
            'min_norm_score_min': 0.20,
            'min_norm_score_max': 0.36,
            'sim_thresh_merge_min': 0.86,
            'sim_thresh_merge_max': 0.92,
            'hash_thresh_merge_min': 9,
            'hash_thresh_merge_max': 13,
        },
    }
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cal['reason'] = 'cap_open_failed'
        return cal
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if n <= 0:
            cal['reason'] = 'no_frames'
            return cal
        target_fps = float(getattr(args, 'calibration_sample_fps', 1.0))
        if target_fps <= 0:
            target_fps = 1.0
        approx_step = max(1, int(round(fps / target_fps)))
        max_frames = int(getattr(args, 'calibration_max_frames', 60))
        # Distribute samples evenly across the video, capped by max_frames.
        ideal = max(1, n // approx_step)
        budget = max(8, min(max_frames, ideal))
        idxs = np.linspace(0, max(0, n - 1), num=budget, dtype=np.int64)

        paper_ratios: List[float] = []
        brights: List[float] = []
        sats: List[float] = []
        blurs: List[float] = []
        text_dens: List[float] = []
        skin_ratios: List[float] = []

        for fi in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            h, w = frame.shape[:2]
            scale = 480.0 / max(h, w) if max(h, w) > 480 else 1.0
            small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale != 1.0 else frame
            try:
                pr, bm, sm = _cheap_paper_ratio(small)
            except Exception:
                pr, bm, sm = 0.0, 0.0, 0.0
            try:
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                td = count_text_density(gray)
            except Exception:
                blur = 0.0
                td = 0.0
            try:
                skn = _cheap_skin_ratio(small)
            except Exception:
                skn = 0.0
            paper_ratios.append(pr)
            brights.append(bm)
            sats.append(sm)
            blurs.append(blur)
            text_dens.append(td)
            skin_ratios.append(skn)

        cal['samples'] = len(paper_ratios)
        if cal['samples'] < 6:
            cal['reason'] = 'too_few_samples'
            return cal

        cal['stats'] = {
            'paper_ratio': _percentiles(np.array(paper_ratios)),
            'brightness': _percentiles(np.array(brights)),
            'saturation': _percentiles(np.array(sats)),
            'blur_lap_var': _percentiles(np.array(blurs)),
            'text_density': _percentiles(np.array(text_dens)),
            'skin_ratio': _percentiles(np.array(skin_ratios)),
            'fps': float(fps),
            'frame_count': int(n),
            'duration_sec': float(n) / float(fps if fps > 0 else 30.0),
        }

        # ----------------- Adaptive overrides (narrow, clamped) ----------------
        # Default: keep v12.9 thresholds. Only nudge if the per-video
        # distribution clearly suggests a different operating point. The
        # clamps are intentionally tight so a "well-behaved" video like the
        # IMG_4883 test sees no override and the v12.9 winners are preserved.
        overrides: Dict[str, Any] = {}
        guards = cal['guardrails']

        # 1) min_norm_score: only nudge on *extreme* blur distributions. The
        # band is intentionally narrow so well-behaved videos (like the
        # IMG_4883 test) are not perturbed.
        blur_p10 = cal['stats']['blur_lap_var']['p10']
        blur_p90 = cal['stats']['blur_lap_var']['p90']
        if np.isfinite(blur_p10) and np.isfinite(blur_p90) and blur_p90 > 1e-3:
            spread = (blur_p90 - blur_p10) / max(blur_p90, 1.0)
            base = float(getattr(args, 'min_norm_score', 0.28))
            v = base
            if spread < 0.20:
                v = base - 0.02
            elif spread > 0.92:
                v = base + 0.02
            v = float(np.clip(v, guards['min_norm_score_min'], guards['min_norm_score_max']))
            if abs(v - base) >= 0.005:
                overrides['min_norm_score'] = v

        # 2) sim_thresh_merge: only nudge if skin is very high. Tighter
        # threshold than before so casual hand presence does not trigger.
        skin_p75 = cal['stats']['skin_ratio']['p75']
        skin_p50 = cal['stats']['skin_ratio'].get('median', float('nan'))
        if np.isfinite(skin_p75) and skin_p75 > 0.30:
            base = float(getattr(args, 'sim_thresh_merge', 0.89))
            v = float(np.clip(base - 0.01, guards['sim_thresh_merge_min'], guards['sim_thresh_merge_max']))
            if abs(v - base) >= 0.005:
                overrides['sim_thresh_merge'] = v

        # v13.1: high-hand adaptive mode. When the per-video skin distribution
        # is unusually high (median > 0.45 OR p75 > 0.55) we expect bottom hand
        # / fingers in most frames (e.g. IMG_4885). Increase secondary-merge
        # tolerance, push sim_thresh_merge to its lower guard, and flip the
        # _high_hand_mode flag so cluster_select_score gives stronger
        # hand/deskew penalties.
        median_skin = skin_p50
        high_hand = (
            (np.isfinite(median_skin) and median_skin > 0.45)
            or (np.isfinite(skin_p75) and skin_p75 > 0.55)
        )
        if high_hand:
            cal['high_hand_mode'] = True
            base_sim = float(getattr(args, 'sim_thresh_merge', 0.89))
            v = float(np.clip(base_sim - 0.02, guards['sim_thresh_merge_min'], guards['sim_thresh_merge_max']))
            if abs(v - base_sim) >= 0.005:
                overrides['sim_thresh_merge'] = v
            # Lower min_norm_score very mildly to keep enough candidates so the
            # cluster re-ranker has alternatives to choose from.
            base_min = float(getattr(args, 'min_norm_score', 0.28))
            v = float(np.clip(base_min - 0.02, guards['min_norm_score_min'], guards['min_norm_score_max']))
            if abs(v - base_min) >= 0.005:
                overrides['min_norm_score'] = v
        else:
            cal['high_hand_mode'] = False

        # 3) hash_thresh_merge: only widen for *very* low text density videos.
        text_p50 = cal['stats']['text_density']['median']
        if np.isfinite(text_p50) and text_p50 < 0.015:
            base = int(getattr(args, 'hash_thresh_merge', 11))
            v = int(np.clip(base + 1, guards['hash_thresh_merge_min'], guards['hash_thresh_merge_max']))
            if v != base:
                overrides['hash_thresh_merge'] = v

        cal['overrides'] = overrides
        cal['used'] = True
        cal['reason'] = 'ok'
        return cal
    finally:
        cap.release()


def adaptive_post_calibration(valid: List[FrameFeatures], cal: Dict[str, Any], args) -> Dict[str, Any]:
    """Refine adaptive thresholds using the *post-feature-extraction* score
    distribution. This runs after the main detection pass and only adjusts
    `min_norm_score` based on the empirical norm_score quantiles among valid
    candidates. It will never push `min_norm_score` outside the guarded band,
    and will never raise it above the v12.9 default by more than +0.04.
    """
    info: Dict[str, Any] = {'applied': False, 'before': None, 'after': None, 'reason': ''}
    if not cal.get('enabled', False):
        info['reason'] = 'disabled'
        return info
    if not valid:
        info['reason'] = 'no_valid_candidates'
        return info
    scores = np.array([x.norm_score for x in valid], dtype=np.float32)
    if scores.size < 8:
        info['reason'] = 'too_few_scores'
        return info
    p25 = float(np.percentile(scores, 25))
    p50 = float(np.percentile(scores, 50))
    p75 = float(np.percentile(scores, 75))
    base_min = float(getattr(args, 'min_norm_score', 0.28))
    # Conservative: only nudge if the empirical p25 is *far* from the default.
    # Otherwise leave it alone. Clamp deviations to ±0.02 so winner selection
    # on typical (well-distributed) videos is preserved bit-identically.
    suggested = base_min
    if p25 < base_min - 0.10:
        # Most candidates are well below default; lower slightly so we don't
        # starve cluster selection.
        suggested = base_min - 0.02
    elif p25 > base_min + 0.20 and p50 > base_min + 0.25:
        # The whole distribution is much higher than default; tighten slightly.
        suggested = base_min + 0.02
    lo = max(0.22, base_min - 0.02)
    hi = min(0.34, base_min + 0.02)
    suggested = float(np.clip(suggested, lo, hi))
    info['before'] = base_min
    info['after'] = suggested
    info['p25'] = p25
    info['p50'] = p50
    info['p75'] = p75
    if abs(suggested - base_min) >= 0.005:
        info['applied'] = True
    return info


def apply_calibration_overrides(args, cal: Dict[str, Any]) -> Dict[str, Any]:
    """Apply the calibration-suggested overrides onto args in place. Returns a
    dict of {arg_name: (before, after)} for diagnostics.
    """
    applied: Dict[str, Any] = {}
    overrides = (cal or {}).get('overrides') or {}
    for k, v in overrides.items():
        if hasattr(args, k):
            before = getattr(args, k)
            setattr(args, k, v)
            applied[k] = {'before': before, 'after': v}
    return applied


# ---------------------------------------------------------------------------
# v13.2: same-page alternative-candidate search
# ---------------------------------------------------------------------------
def _alt_hand_score(x: 'FrameFeatures') -> float:
    """Composite hand-occlusion score (lower is cleaner).

    Combines hand_text_overlap (most important), bottom_hand, and the raw hand
    mask ratio. Used both to detect a candidate as 'high hand' and to compare
    two candidates' relative cleanliness.
    """
    return (
        0.55 * float(getattr(x, 'hand_text_overlap_penalty', 0.0))
        + 0.30 * float(getattr(x, 'bottom_hand_penalty', 0.0))
        + 0.15 * float(getattr(x, 'hand_penalty', 0.0))
    )


def _alt_winner_is_dirty(w: 'FrameFeatures', args) -> bool:
    """Return True if the winner's hand metrics warrant searching for a cleaner
    same-page alternative. Conservative — clean winners are skipped."""
    ht = float(getattr(w, 'hand_text_overlap_penalty', 0.0))
    bh = float(getattr(w, 'bottom_hand_penalty', 0.0))
    h = float(getattr(w, 'hand_penalty', 0.0))
    # Either heavy text/hand overlap, full bottom-hand band, or any combination
    # of moderate signals.
    if ht >= float(getattr(args, 'alt_dirty_hand_text', 0.55)):
        return True
    if bh >= float(getattr(args, 'alt_dirty_bottom_hand', 0.85)):
        return True
    if h >= float(getattr(args, 'alt_dirty_hand', 0.20)) and ht >= 0.45:
        return True
    return False


def _alt_same_page(rep: 'FrameFeatures', cand: 'FrameFeatures', args) -> Tuple[bool, float, int, float]:
    """Decide whether `cand` likely depicts the same page as `rep`.

    Returns (is_same_page, similarity, hamming, text_rel_diff).
    Combines dHash hamming + ROI structural similarity + text-density similarity
    so we tolerate the heavy perspective and skew variation seen on IMG_4885
    where SSIM of the same page can drop to ~0.2 due to occlusion.
    """
    if rep.roi_gray is None or cand.roi_gray is None or rep.roi_dhash is None or cand.roi_dhash is None:
        return False, 0.0, 999, 1.0
    ham = hamming_distance(rep.roi_dhash, cand.roi_dhash)
    sim = similarity_score(rep.roi_gray, cand.roi_gray)
    text_a = max(1e-6, float(rep.text_score))
    text_b = max(1e-6, float(cand.text_score))
    text_rel = abs(text_a - text_b) / max(text_a, text_b)

    sim_strict = float(getattr(args, 'alt_sim_min', 0.55))
    ham_strict = int(getattr(args, 'alt_hash_max', 18))
    text_tol = float(getattr(args, 'alt_text_rel_tol', 0.25))
    sim_relaxed = float(getattr(args, 'alt_sim_relaxed', 0.35))
    ham_relaxed = int(getattr(args, 'alt_hash_relaxed', 28))
    text_tol_strict = float(getattr(args, 'alt_text_rel_tight', 0.15))

    if sim >= sim_strict and ham <= ham_strict:
        return True, sim, ham, text_rel
    if (sim >= sim_relaxed or ham <= ham_relaxed) and text_rel <= text_tol:
        return True, sim, ham, text_rel
    if text_rel <= text_tol_strict and ham <= ham_relaxed + 6:
        return True, sim, ham, text_rel
    return False, sim, ham, text_rel


def find_alternative_winner(
    winner: 'FrameFeatures',
    valid: List['FrameFeatures'],
    used_frame_idx: set,
    args,
) -> Dict[str, Any]:
    """Search for a cleaner same-page alternative to `winner` in `valid`.

    Returns a diagnostics dict with at minimum:
      checked: int                 # candidates evaluated as same-page
      examined: int                # candidates within temporal window
      replacement: Optional[FrameFeatures]
      reason: str                  # human-readable result
      hand_improvement: float      # _alt_hand_score(winner) - _alt_hand_score(replacement)
      similarity: float            # similarity to original winner
      original_frame: int
      replacement_frame: Optional[int]
    """
    diag: Dict[str, Any] = {
        'checked': 0,
        'examined': 0,
        'replacement': None,
        'reason': 'not-attempted',
        'hand_improvement': 0.0,
        'similarity': 0.0,
        'original_frame': int(winner.frame_idx),
        'replacement_frame': None,
        'orig_hand_score': _alt_hand_score(winner),
    }

    if not _alt_winner_is_dirty(winner, args):
        diag['reason'] = 'winner-clean'
        return diag

    if winner.roi_gray is None or winner.roi_dhash is None:
        diag['reason'] = 'winner-no-features'
        return diag

    # Temporal neighborhood: look within +/- alt_window_sec around the winner.
    window = float(getattr(args, 'alt_window_sec', 4.0))
    same_page_gap = float(getattr(args, 'min_same_page_gap_sec', 1.3))
    # Don't widen past the half-distance to other winners — handled implicitly
    # by used_frame_idx and the visual sameness check below.

    orig_hand = _alt_hand_score(winner)
    orig_blur = float(winner.blur_score)
    orig_deskew = abs(float(getattr(winner, 'deskew_angle', 0.0)))

    deskew_max = float(getattr(args, 'alt_deskew_max', 5.0))
    blur_floor_frac = float(getattr(args, 'alt_blur_floor_frac', 0.55))
    min_improve = float(getattr(args, 'alt_min_hand_improvement', 0.18))
    raw_score_floor_drop = float(getattr(args, 'alt_raw_score_max_drop', 1.5))

    candidates_in_window = []
    for v in valid:
        if v.frame_idx == winner.frame_idx:
            continue
        if v.frame_idx in used_frame_idx:
            continue
        dt = abs(v.t_sec - winner.t_sec)
        if dt > window:
            continue
        candidates_in_window.append(v)

    # v14.0: bound the number of candidates we evaluate per winner. Without
    # --audit-candidates we keep at most max_alternatives_per_winner of them,
    # picking the temporally-closest first (which are most likely the same
    # page). 0 disables the cap.
    max_per_winner = int(getattr(args, 'max_alternatives_per_winner', 8) or 0)
    audit_mode = bool(getattr(args, 'audit_candidates', False))
    if not audit_mode and max_per_winner > 0 and len(candidates_in_window) > max_per_winner:
        candidates_in_window.sort(key=lambda c: abs(c.t_sec - winner.t_sec))
        candidates_in_window = candidates_in_window[:max_per_winner]

    diag['examined'] = len(candidates_in_window)

    same_page_candidates: List[Tuple['FrameFeatures', float, int, float]] = []
    for c in candidates_in_window:
        ok, sim, ham, text_rel = _alt_same_page(winner, c, args)
        if not ok:
            continue
        same_page_candidates.append((c, sim, ham, text_rel))
    diag['checked'] = len(same_page_candidates)

    if not same_page_candidates:
        diag['reason'] = f'no-same-page-alts(window={window:.1f}s,examined={diag["examined"]})'
        return diag

    # Score by hand-cleanliness with guards. We rank by improvement first,
    # tie-broken by similarity and blur.
    best = None
    best_key = None
    best_meta: Optional[Dict[str, Any]] = None
    rejections: Dict[str, int] = {}

    def _add_reject(key: str):
        rejections[key] = rejections.get(key, 0) + 1

    for c, sim, ham, text_rel in same_page_candidates:
        cand_hand = _alt_hand_score(c)
        improvement = orig_hand - cand_hand
        if improvement < min_improve:
            _add_reject('insufficient_improvement')
            continue
        if abs(float(getattr(c, 'deskew_angle', 0.0))) > deskew_max:
            _add_reject('deskew_too_large')
            continue
        if float(c.blur_score) < blur_floor_frac * max(orig_blur, 100.0):
            _add_reject('too_blurry')
            continue
        if (float(winner.raw_score) - float(c.raw_score)) > raw_score_floor_drop:
            _add_reject('raw_score_drop')
            continue
        # Avoid swapping to a frame that itself looks like an obvious turn page
        if float(getattr(c, 'turn_penalty', 0.0)) > 0.85:
            _add_reject('turn_penalty')
            continue
        # Sort key: prefer biggest hand improvement, then highest sim, then sharper.
        key = (improvement, sim, c.blur_score, -abs(float(getattr(c, 'deskew_angle', 0.0))))
        if best_key is None or key > best_key:
            best_key = key
            best = c
            best_meta = {'sim': sim, 'ham': ham, 'text_rel': text_rel,
                         'improvement': improvement, 'cand_hand': cand_hand}

    if best is None:
        rej_summary = ','.join(f'{k}={v}' for k, v in sorted(rejections.items()))
        diag['reason'] = f'no-acceptable-alt(checked={diag["checked"]},rej[{rej_summary}])'
        return diag

    diag['replacement'] = best
    diag['replacement_frame'] = int(best.frame_idx)
    diag['similarity'] = float(best_meta['sim'])
    diag['hand_improvement'] = float(best_meta['improvement'])
    diag['reason'] = (
        f'replaced(sim={best_meta["sim"]:.2f},ham={best_meta["ham"]},'
        f'text_rel={best_meta["text_rel"]:.2f},'
        f'd_hand={best_meta["improvement"]:.2f},'
        f'orig={orig_hand:.2f}->cand={best_meta["cand_hand"]:.2f})'
    )
    return diag


def search_alternatives_for_winners(
    winners: List['FrameFeatures'],
    valid: List['FrameFeatures'],
    args,
) -> Tuple[List['FrameFeatures'], Dict[int, Dict[str, Any]]]:
    """Run alternative search per winner and return (new_winners, diag_by_orig_frame).

    Diagnostics are keyed by the *original* winner frame_idx so the winners.csv
    code can look up what happened even after the winner is swapped out.
    """
    diag_by_frame: Dict[int, Dict[str, Any]] = {}
    used = {w.frame_idx for w in winners}
    new_winners: List['FrameFeatures'] = []
    for w in winners:
        d = find_alternative_winner(w, valid, used - {w.frame_idx}, args)
        diag_by_frame[w.frame_idx] = d
        if d.get('replacement') is not None:
            rep = d['replacement']
            used.discard(w.frame_idx)
            used.add(rep.frame_idx)
            new_winners.append(rep)
        else:
            new_winners.append(w)
    return new_winners, diag_by_frame


# ---------------------------------------------------------------------------
# v15.4: bounded same-page quality refinement for suspicious winners.
# ---------------------------------------------------------------------------
def _v154_winner_is_suspicious(w: 'FrameFeatures', args) -> Tuple[bool, str]:
    """Return (is_suspicious, reason_tags) for a winner using generic
    threshold-based criteria. Conservative: only flag winners with at least
    one materially elevated quality signal, so well-behaved winners are
    untouched and the refinement pass is cheap.
    """
    reasons: List[str] = []
    hand = float(getattr(w, 'hand_penalty', 0.0))
    hto = float(getattr(w, 'hand_text_overlap_penalty', 0.0))
    bh = float(getattr(w, 'bottom_hand_penalty', 0.0))
    skew_abs = abs(float(getattr(w, 'deskew_angle', 0.0)))
    turn = float(getattr(w, 'turn_penalty', 0.0))
    edge_mot = float(getattr(w, 'edge_foreground_penalty', 0.0))
    em_motion = float(getattr(w, 'edge_motion_penalty', 0.0))

    hand_thresh = float(getattr(args, 'quality_refine_hand_thresh', 0.40))
    skew_thresh = float(getattr(args, 'quality_refine_skew_thresh', 4.0))
    bottom_thresh = float(getattr(args, 'quality_refine_bottom_hand_thresh', 0.55))
    hto_thresh = float(getattr(args, 'quality_refine_hand_text_thresh', 0.35))
    turn_thresh = float(getattr(args, 'quality_refine_turn_thresh', 0.55))

    if hand >= hand_thresh:
        reasons.append(f'hand={hand:.2f}')
    if hto >= hto_thresh:
        reasons.append(f'hand_text={hto:.2f}')
    if bh >= bottom_thresh:
        reasons.append(f'bottom_hand={bh:.2f}')
    if skew_abs >= skew_thresh:
        reasons.append(f'skew={skew_abs:.2f}')
    if turn >= turn_thresh:
        reasons.append(f'turn={turn:.2f}')
    if em_motion >= 0.85:
        reasons.append(f'edge_motion={em_motion:.2f}')

    # Low clean_visual_score + finger penalty signals partially-occluded page.
    try:
        if w.warped_bgr is not None:
            metrics = _v135_visual_metrics_cached(w)
            cvs = _v135_clean_visual_score(metrics)
            fg = _v135_finger_penalty(metrics)
            if cvs < float(getattr(args, 'quality_refine_cvs_low', -0.10)) and fg >= 0.30:
                reasons.append(f'cvs={cvs:.2f}/finger={fg:.2f}')
    except Exception:
        pass

    return (len(reasons) > 0, ','.join(reasons))


def _v154_refine_score(x: 'FrameFeatures') -> float:
    """Combined refinement score (higher = better).

    Rewards clean_visual_score (the discriminating same-page cleanliness
    signal) and blur; penalises finger_penalty, hand metrics, bottom_hand,
    abs(deskew), turn_penalty, edge_motion. Weights are tuned so a swap
    only fires when the candidate is materially cleaner on the visual
    discriminators — raw_score is intentionally lightly weighted because
    on hand-occluded videos the script-level raw_score is noisy.
    """
    cvs = 0.0
    fg = 0.0
    try:
        if x.warped_bgr is not None:
            metrics = _v135_visual_metrics_cached(x)
            cvs = float(_v135_clean_visual_score(metrics))
            fg = float(_v135_finger_penalty(metrics))
    except Exception:
        cvs = 0.0
        fg = 0.0
    hand = float(getattr(x, 'hand_penalty', 0.0))
    hto = float(getattr(x, 'hand_text_overlap_penalty', 0.0))
    bh = float(getattr(x, 'bottom_hand_penalty', 0.0))
    skew_abs = abs(float(getattr(x, 'deskew_angle', 0.0)))
    turn = float(getattr(x, 'turn_penalty', 0.0))
    em = float(getattr(x, 'edge_motion_penalty', 0.0))
    blur = float(getattr(x, 'blur_score', 0.0))
    return (
        0.65 * cvs
        - 0.80 * fg
        + 0.10 * min(blur / 400.0, 1.0)
        - 0.20 * hto
        - 0.20 * bh
        - 0.20 * hand
        - 0.04 * skew_abs
        - 0.15 * turn
        - 0.10 * em
    )


def _v154_finger_penalty(x: 'FrameFeatures') -> float:
    """Compute the v13.5 finger_penalty for a candidate (cached)."""
    try:
        if x.warped_bgr is None:
            return 0.0
        return float(_v135_finger_penalty(_v135_visual_metrics_cached(x)))
    except Exception:
        return 0.0


def _v154_same_page_gate(winner: 'FrameFeatures', cand: 'FrameFeatures', args) -> Tuple[bool, float, int]:
    """Conservative same-physical-page gate combining the v13.4 relaxed test,
    the v13.2 alt-search test, and a temporal-proximity backup.

    Strategy:
      * If either v134_relaxed or alt_same_page accepts the candidate AND
        the text-density relative gap is reasonable, accept.
      * As a backup, accept if candidates are very close in time
        (< quality_refine_temporal_buddy_sec, default 1.2s) and have
        similar text density. Hand occlusion can destroy ROI-similarity but
        two adjacent decoded frames of the same physical page nearly always
        match by time + text density when no page turn has occurred.
    Returns (ok, similarity, hamming).
    """
    if winner.roi_gray is None or cand.roi_gray is None:
        return False, 0.0, 64
    ok_a, sim_a, ham_a = _v134_relaxed_same_page(winner, cand, args)
    ok_b, sim_b, ham_b, trel = _alt_same_page(winner, cand, args)
    sim = float(max(sim_a, sim_b))
    ham = int(min(ham_a, ham_b))
    if ok_a or ok_b:
        return True, sim, ham
    # Temporal-buddy backup. Adjacent decoded frames are essentially the same
    # physical page when text density agrees and no turn_penalty signal fires.
    buddy_sec = float(getattr(args, 'quality_refine_temporal_buddy_sec', 1.2))
    text_a = max(1e-6, float(winner.text_score))
    text_b = max(1e-6, float(cand.text_score))
    text_rel = abs(text_a - text_b) / max(text_a, text_b)
    if (
        abs(cand.t_sec - winner.t_sec) <= buddy_sec
        and text_rel <= 0.30
        and float(getattr(cand, 'turn_penalty', 0.0)) < 0.7
        and float(getattr(winner, 'turn_penalty', 0.0)) < 0.7
    ):
        return True, sim, ham
    return False, sim, ham


def quality_refinement_pass(
    winners: List['FrameFeatures'],
    valid: List['FrameFeatures'],
    args,
) -> Tuple[List['FrameFeatures'], Dict[int, Dict[str, Any]]]:
    """v15.4 bounded quality refinement.

    Returns (new_winners, diag_by_orig_frame) where diag_by_orig_frame is
    keyed by the *original* winner frame_idx so the winners.csv writer can
    surface the result regardless of whether a swap occurred.

    Performance: at most O(K * S) same-page tests where K is the number of
    suspicious winners (typically 0-2) and S is bounded by
    --quality-refine-top-k (default 6).
    """
    diag_by_frame: Dict[int, Dict[str, Any]] = {}
    if not winners or not valid:
        return winners, diag_by_frame

    window = float(getattr(args, 'quality_refine_window_sec', 2.5))
    top_k = int(getattr(args, 'quality_refine_top_k', 6) or 0)
    min_improve = float(getattr(args, 'quality_refine_min_improvement', 0.12))
    # Use a more permissive absolute skew ceiling for refinement than alt-
    # search; we *want* to be able to swap a skewed winner for a slightly
    # less skewed candidate. The relative skew check below also enforces
    # progress.
    deskew_max = float(getattr(args, 'quality_refine_deskew_max', 7.5))
    blur_floor_frac = float(getattr(args, 'alt_blur_floor_frac', 0.55))

    used = {w.frame_idx for w in winners}
    new_winners: List['FrameFeatures'] = []

    for w in winners:
        d: Dict[str, Any] = {
            'applied': False,
            'reason': 'not-suspicious',
            'original_frame': int(w.frame_idx),
            'replacement_frame': None,
            'score_delta': 0.0,
            'candidates_checked': 0,
            'candidates_examined': 0,
            'suspicious_reasons': '',
        }
        suspicious, why = _v154_winner_is_suspicious(w, args)
        if not suspicious:
            new_winners.append(w)
            diag_by_frame[w.frame_idx] = d
            continue

        d['suspicious_reasons'] = why

        # Pool candidates within the small temporal window. Skip already-used
        # winner frames so we never collapse two winners onto the same frame.
        # v15.7: also exclude candidates that sit closer to a *different*
        # existing winner than to `w` and are inside that other winner's
        # min_peak_distance_sec zone. Without this guard, when a suspicious
        # winner near a region boundary searches for a same-page replacement
        # the visual same-page gate can match a frame from an adjacent
        # region, and the subsequent adjacent-dedup will collapse the two —
        # losing the entire region. The guard is purely additive: any
        # candidate that was already within ±min_peak_distance_sec of `w`
        # is preserved, since for those `w` is the closest winner.
        sep_keepout = float(getattr(args, 'min_peak_distance_sec', 0.9))
        other_winners = [u for u in winners if u.frame_idx != w.frame_idx]
        pool: List['FrameFeatures'] = []
        for v in valid:
            if v.frame_idx == w.frame_idx:
                continue
            if v.frame_idx in used and v.frame_idx != w.frame_idx:
                continue
            if abs(v.t_sec - w.t_sec) > window:
                continue
            steal_zone = False
            for u in other_winners:
                if abs(v.t_sec - u.t_sec) < sep_keepout and abs(v.t_sec - u.t_sec) < abs(v.t_sec - w.t_sec):
                    steal_zone = True
                    break
            if steal_zone:
                continue
            pool.append(v)
        # Order by temporal proximity and keep top-K to bound cost.
        pool.sort(key=lambda c: abs(c.t_sec - w.t_sec))
        if top_k > 0 and len(pool) > top_k:
            pool = pool[:top_k]
        d['candidates_examined'] = len(pool)

        if not pool:
            d['reason'] = f'no-candidates(window={window:.1f}s)'
            new_winners.append(w)
            diag_by_frame[w.frame_idx] = d
            continue

        same_page: List['FrameFeatures'] = []
        for c in pool:
            ok, _sim, _ham = _v154_same_page_gate(w, c, args)
            if ok:
                same_page.append(c)
        d['candidates_checked'] = len(same_page)

        if not same_page:
            d['reason'] = f'no-same-page(examined={d["candidates_examined"]})'
            new_winners.append(w)
            diag_by_frame[w.frame_idx] = d
            continue

        base_score = _v154_refine_score(w)
        orig_blur = float(w.blur_score)
        orig_finger = _v154_finger_penalty(w)
        # Floor on how much worse a candidate's finger penalty may be vs
        # the original winner. Default keeps refinement from trading a
        # clean winner for one with a visibly larger finger.
        finger_regress_max = float(
            getattr(args, 'quality_refine_finger_regress_max', 0.08)
        )
        best = None
        best_delta = 0.0
        rejections: Dict[str, int] = {}

        def _add_reject(k: str):
            rejections[k] = rejections.get(k, 0) + 1

        # Hard guards on motion/stability: reject candidates that are clearly
        # transition / turning frames even if same-page identity matches.
        stability_floor = float(getattr(args, 'quality_refine_stability_min', 0.55))
        edge_motion_max = float(getattr(args, 'quality_refine_edge_motion_max', 0.55))
        raw_score_max_drop = float(getattr(args, 'quality_refine_raw_score_max_drop', 1.5))
        for c in same_page:
            # Hard guards: don't accept obviously worse skew or blur.
            if abs(float(getattr(c, 'deskew_angle', 0.0))) > deskew_max:
                _add_reject('deskew_too_large')
                continue
            if float(c.blur_score) < blur_floor_frac * max(orig_blur, 100.0):
                _add_reject('too_blurry')
                continue
            if float(getattr(c, 'turn_penalty', 0.0)) > 0.85:
                _add_reject('turn_penalty')
                continue
            # Reject transition / turning frames (low stability or high edge
            # motion). These can pass the relaxed same-page gate via text
            # density alone but produce visibly worse pages than the
            # original winner.
            if float(getattr(c, 'stability_score', 1.0)) < stability_floor:
                _add_reject('low_stability')
                continue
            if float(getattr(c, 'edge_motion_penalty', 0.0)) > edge_motion_max:
                _add_reject('edge_motion_high')
                continue
            # Don't accept a candidate whose raw_score is much lower than
            # the original winner — the script's raw_score is a strong
            # global quality indicator (page_found, fill, blur etc.).
            if (float(w.raw_score) - float(c.raw_score)) > raw_score_max_drop:
                _add_reject('raw_score_drop')
                continue
            # Don't replace a clean winner with one whose finger penalty
            # is materially higher — protects clean winners on videos
            # where every frame has high MediaPipe-derived hand metrics.
            cand_finger = _v154_finger_penalty(c)
            if cand_finger > orig_finger + finger_regress_max:
                _add_reject('finger_regression')
                continue
            cand_score = _v154_refine_score(c)
            delta = cand_score - base_score
            if delta < min_improve:
                _add_reject('insufficient_gain')
                continue
            if best is None or delta > best_delta:
                best = c
                best_delta = delta

        if best is None:
            rej_summary = ','.join(f'{k}={v}' for k, v in sorted(rejections.items()))
            d['reason'] = (
                f'no-improvement(checked={d["candidates_checked"]},'
                f'min_gain={min_improve:.2f},rej[{rej_summary}])'
            )
            new_winners.append(w)
            diag_by_frame[w.frame_idx] = d
            continue

        d['applied'] = True
        d['replacement_frame'] = int(best.frame_idx)
        d['score_delta'] = float(best_delta)
        d['reason'] = (
            f'replaced(reasons={why};delta={best_delta:+.3f};'
            f'checked={d["candidates_checked"]})'
        )
        used.discard(w.frame_idx)
        used.add(best.frame_idx)
        new_winners.append(best)
        diag_by_frame[w.frame_idx] = d

    return new_winners, diag_by_frame


# ---------------------------------------------------------------------------
# v15.8 — within-region finger-relief refinement pass.
#
# Runs AFTER quality_refinement_pass. Targets winners that retained a high
# finger / bottom-skin signal even after v15.4 / v15.7 refinement. Uses a
# loosened blur floor and replaces the global `_v154_refine_score` gain
# gate with a finger-specific gain test, while preserving:
#   * the v15.7 steal_zone guard (within-region only),
#   * the strict same-page identity gate (no temporal-buddy fallback),
#   * geometric / motion gates (skew, turn_penalty, stability,
#     edge_motion),
#   * a "near-identical to another winner" check that prevents the
#     subsequent v15.5 adjacent-dedup from collapsing two regions onto
#     the same frame.
# ---------------------------------------------------------------------------
def _v158_strict_same_page_gate(
    winner: 'FrameFeatures',
    cand: 'FrameFeatures',
    args,
) -> Tuple[bool, float, int]:
    """Same-physical-page gate for finger-relief refinement.

    Uses the v15.4 `_v154_same_page_gate` which combines the v13.4
    relaxed test (SSIM / dHash / row-profile / warp-thumb), the v13.2
    alt-search test, and a temporal-buddy fallback (within
    --quality-refine-temporal-buddy-sec, text-density agreement, and
    no turn signal). The temporal-buddy path is necessary because a
    finger-occluded winner has its ROI similarity destroyed by the
    hand mask, so SSIM/dHash alone often fail even between two
    confidently-same-page frames. Geometric / motion gates downstream
    in `finger_relief_pass` (turn_penalty, stability, edge_motion)
    independently reject transition frames that this gate may
    accept on temporal proximity alone.
    """
    return _v154_same_page_gate(winner, cand, args)


def _v158_winner_has_finger_pressure(
    w: 'FrameFeatures', why: str, finger_floor: float,
) -> Tuple[bool, float]:
    """True if the winner has visible finger / bottom-skin pressure.

    Uses the v13.5 finger_penalty (skin + bottom-skin in the warped page)
    as the primary signal. The `cvs=...,finger=...` reason emitted by
    `_v154_winner_is_suspicious` is a strong corroborating signal, but
    other "suspicious" reasons (hand_text overlap, large skew, turn) do
    NOT imply finger occlusion of the warped page and are excluded so we
    do not pursue clean replacements for unrelated issues.
    Returns (flag, finger_penalty).
    """
    fg = _v154_finger_penalty(w)
    flag = bool(
        fg >= finger_floor
        or 'cvs=' in (why or '')
        or 'finger=' in (why or '')
    )
    return flag, fg


def _v158_near_identical_to_other_winner(
    cand: 'FrameFeatures',
    other_winners: List['FrameFeatures'],
    sim_thresh: float,
    ham_thresh: int,
    args=None,
) -> Optional['FrameFeatures']:
    """Return the first other winner that would be merged with `cand` by
    the subsequent v15.5 adjacent-dedup pass.

    The v15.5 dedup merges adjacent winners that depict the same physical
    page using either strict similarity OR warp-thumbnail correlation OR
    a temporal-rescan path (warp + profile + text density at small dt).
    To anticipate this, we apply the SAME `_v154_same_page_gate` that
    finger-relief uses for its own same-page test against every other
    existing winner; if any other winner matches, replacing `w` with
    `cand` would collapse two regions in v15.5 dedup. As an extra
    precaution we also accept the simpler SSIM/Hamming criterion the
    caller passed (so a tightened sim/ham still triggers rejection).
    """
    if cand.roi_gray is None or cand.roi_dhash is None:
        return None
    for u in other_winners:
        if u.roi_gray is None or u.roi_dhash is None:
            continue
        try:
            sim = float(similarity_score(cand.roi_gray, u.roi_gray))
            ham = int(hamming_distance(cand.roi_dhash, u.roi_dhash))
        except Exception:
            sim, ham = 0.0, 64
        if sim >= sim_thresh and ham <= ham_thresh:
            return u
        # Also catch warp-thumb / temporal-buddy matches that the v15.5
        # adjacent-dedup would treat as same-page.
        if args is not None:
            try:
                ok, _s, _h = _v154_same_page_gate(cand, u, args)
            except Exception:
                ok = False
            if ok:
                return u
            # And the warp-thumb match alone (v155 uses it as a
            # corroborating signal in the temporal-rescan path).
            try:
                ratio, _hw = _v134_warp_thumb_match(cand, u)
            except Exception:
                ratio = 0.0
            if ratio >= float(getattr(args, 'finger_relief_other_winner_warp', 0.78)):
                return u
    return None


def finger_relief_pass(
    winners: List['FrameFeatures'],
    valid: List['FrameFeatures'],
    args,
    quality_refine_replacement_orig: Optional[Dict[int, int]] = None,
) -> Tuple[List['FrameFeatures'], Dict[int, Dict[str, Any]], Dict[int, int]]:
    """v15.8 within-region finger-relief refinement.

    Returns (new_winners, diag_by_orig_frame, replacement_orig_map).
    diag_by_orig_frame is keyed by the *original* winner frame_idx (i.e.
    the frame_idx of the winner entering this pass). replacement_orig_map
    maps new_frame_idx -> entering frame_idx for swapped winners.
    """
    diag_by_frame: Dict[int, Dict[str, Any]] = {}
    rep_map: Dict[int, int] = {}
    if not winners or not valid:
        return winners, diag_by_frame, rep_map
    if not bool(getattr(args, 'finger_relief', True)):
        return winners, diag_by_frame, rep_map

    window = float(getattr(args, 'quality_refine_window_sec', 2.5))
    top_k = int(getattr(args, 'quality_refine_top_k', 6) or 0)
    deskew_max = float(getattr(args, 'quality_refine_deskew_max', 7.5))
    stability_floor = float(getattr(args, 'quality_refine_stability_min', 0.55))
    edge_motion_max = float(getattr(args, 'quality_refine_edge_motion_max', 0.55))
    sep_keepout = float(getattr(args, 'min_peak_distance_sec', 0.9))

    finger_floor = float(getattr(args, 'finger_relief_finger_floor', 0.30))
    min_finger_improve = float(
        getattr(args, 'finger_relief_min_finger_improve', 0.20)
    )
    max_overall_regress = float(
        getattr(args, 'finger_relief_max_overall_regress', -0.10)
    )
    blur_frac = float(getattr(args, 'finger_relief_blur_frac', 0.30))
    raw_score_max_drop = float(
        getattr(args, 'finger_relief_raw_score_max_drop', 2.5)
    )
    other_sim_thresh = float(
        getattr(args, 'finger_relief_other_winner_sim', 0.85)
    )
    other_ham_thresh = int(
        getattr(args, 'finger_relief_other_winner_ham', 8)
    )

    used = {w.frame_idx for w in winners}
    new_winners: List['FrameFeatures'] = []

    for w in winners:
        d: Dict[str, Any] = {
            'applied': False,
            'reason': 'not-finger-suspicious',
            'original_frame': int(w.frame_idx),
            'replacement_frame': None,
            'orig_finger': float(_v154_finger_penalty(w)),
            'new_finger': None,
            'orig_cvs': float(_v135_visual_score_cached(w))
                if w.warped_bgr is not None else 0.0,
            'new_cvs': None,
            'finger_delta': 0.0,
            'overall_delta': 0.0,
            'candidates_checked': 0,
            'candidates_examined': 0,
        }

        # Re-evaluate suspiciousness for finger / bottom-skin specifically.
        susp, why = _v154_winner_is_suspicious(w, args)
        flag, orig_finger = _v158_winner_has_finger_pressure(
            w, why if susp else '', finger_floor,
        )
        if not flag:
            new_winners.append(w)
            diag_by_frame[w.frame_idx] = d
            continue

        d['orig_finger'] = float(orig_finger)
        d['reason'] = 'no-pool'

        other_winners = [u for u in winners if u.frame_idx != w.frame_idx]

        # Pool with the v15.7 steal_zone guard preserved.
        pool: List['FrameFeatures'] = []
        for v in valid:
            if v.frame_idx == w.frame_idx:
                continue
            if v.frame_idx in used and v.frame_idx != w.frame_idx:
                continue
            if abs(v.t_sec - w.t_sec) > window:
                continue
            steal_zone = False
            for u in other_winners:
                if (
                    abs(v.t_sec - u.t_sec) < sep_keepout
                    and abs(v.t_sec - u.t_sec) < abs(v.t_sec - w.t_sec)
                ):
                    steal_zone = True
                    break
            if steal_zone:
                continue
            pool.append(v)
        pool.sort(key=lambda c: abs(c.t_sec - w.t_sec))
        if top_k > 0 and len(pool) > top_k:
            pool = pool[:top_k]
        d['candidates_examined'] = len(pool)

        if not pool:
            new_winners.append(w)
            diag_by_frame[w.frame_idx] = d
            continue

        # Strict same-page identity (no temporal-buddy fallback).
        same_page: List[Tuple['FrameFeatures', float, int]] = []
        for c in pool:
            ok, sim, ham = _v158_strict_same_page_gate(w, c, args)
            if ok:
                same_page.append((c, sim, ham))
        d['candidates_checked'] = len(same_page)

        if not same_page:
            d['reason'] = (
                f'no-strict-same-page(examined={d["candidates_examined"]})'
            )
            new_winners.append(w)
            diag_by_frame[w.frame_idx] = d
            continue

        base_overall = _v154_refine_score(w)
        orig_blur = float(w.blur_score)

        best = None
        best_finger_delta = 0.0
        best_overall_delta = 0.0
        rejections: Dict[str, int] = {}

        def _add_reject(k: str):
            rejections[k] = rejections.get(k, 0) + 1

        for (c, _sim, _ham) in same_page:
            # Geometric / motion gates (kept).
            if abs(float(getattr(c, 'deskew_angle', 0.0))) > deskew_max:
                _add_reject('deskew_too_large')
                continue
            if float(getattr(c, 'turn_penalty', 0.0)) > 0.85:
                _add_reject('turn_penalty')
                continue
            if float(getattr(c, 'stability_score', 1.0)) < stability_floor:
                _add_reject('low_stability')
                continue
            if float(getattr(c, 'edge_motion_penalty', 0.0)) > edge_motion_max:
                _add_reject('edge_motion_high')
                continue
            # Loosened blur floor; absolute floor of 100 always applies.
            if float(c.blur_score) < blur_frac * max(orig_blur, 100.0):
                _add_reject('too_blurry')
                continue
            if float(c.blur_score) < 100.0:
                _add_reject('blur_below_abs_min')
                continue
            # Loosened raw_score drop.
            if (float(w.raw_score) - float(c.raw_score)) > raw_score_max_drop:
                _add_reject('raw_score_drop')
                continue
            # Finger-specific gain test.
            cand_finger = _v154_finger_penalty(c)
            finger_delta = float(orig_finger) - float(cand_finger)
            if finger_delta < min_finger_improve:
                _add_reject('insufficient_finger_gain')
                continue
            # Overall-score must not regress beyond the bound.
            cand_overall = _v154_refine_score(c)
            overall_delta = cand_overall - base_overall
            if overall_delta < max_overall_regress:
                _add_reject('overall_regress')
                continue
            # Don't pick a candidate visually identical to another winner —
            # would lose a region in the subsequent v15.5 adjacent-dedup.
            twin = _v158_near_identical_to_other_winner(
                c, other_winners, other_sim_thresh, other_ham_thresh,
                args=args,
            )
            if twin is not None:
                _add_reject('twin_of_other_winner')
                continue
            # Rank by finger improvement primarily; break ties on overall.
            if (
                best is None
                or finger_delta > best_finger_delta
                or (
                    abs(finger_delta - best_finger_delta) < 1e-6
                    and overall_delta > best_overall_delta
                )
            ):
                best = c
                best_finger_delta = finger_delta
                best_overall_delta = overall_delta

        if best is None:
            rej_summary = ','.join(
                f'{k}={vv}' for k, vv in sorted(rejections.items())
            )
            d['reason'] = (
                f'no-finger-improvement(checked={d["candidates_checked"]},'
                f'min_finger_gain={min_finger_improve:.2f},'
                f'max_regress={max_overall_regress:.2f},'
                f'rej[{rej_summary}])'
            )
            new_winners.append(w)
            diag_by_frame[w.frame_idx] = d
            continue

        # Apply the swap.
        d['applied'] = True
        d['replacement_frame'] = int(best.frame_idx)
        d['new_finger'] = float(_v154_finger_penalty(best))
        d['new_cvs'] = float(_v135_visual_score_cached(best))
        d['finger_delta'] = float(best_finger_delta)
        d['overall_delta'] = float(best_overall_delta)
        d['reason'] = (
            f'replaced(finger {orig_finger:.2f}->{d["new_finger"]:.2f},'
            f'd_finger={best_finger_delta:+.2f},'
            f'd_overall={best_overall_delta:+.2f},'
            f'checked={d["candidates_checked"]})'
        )
        used.discard(w.frame_idx)
        used.add(best.frame_idx)
        rep_map[best.frame_idx] = int(w.frame_idx)
        new_winners.append(best)
        diag_by_frame[w.frame_idx] = d

    return new_winners, diag_by_frame, rep_map


# ---------------------------------------------------------------------------
# v15.6: footer/folio distinctness guard for adjacent-winner dedup.
#
# When the adjacent-dedup decision is non-strict (temporal-rescan or
# primary+corroboration), compare the bottom-center band (folio / page-
# number area) of the two perspective-rectified pages. If the bands are
# confidently different we reject the merge. Strict-identity merges bypass
# the guard. No OCR, no hardcoded geometry beyond the operator-tunable
# fractions of warp height/width.
# ---------------------------------------------------------------------------
def _v156_footer_extract_band(
    warped_bgr: Optional[np.ndarray],
    band_frac: float = 0.09,
    center_frac: float = 0.60,
    side_trim_frac: float = 0.08,
) -> Optional[np.ndarray]:
    """Returns the bottom-center folio band as a grayscale ROI, or None.

    Geometry is in fractions of the warped page so it is invariant to the
    actual rectified resolution. The default footprint covers the bottom
    9% of page height across the central 60% of width, with a small
    horizontal margin on each side to avoid gutter / outer-edge artefacts.
    """
    if warped_bgr is None:
        return None
    try:
        h, w = warped_bgr.shape[:2]
    except Exception:
        return None
    if h < 40 or w < 40:
        return None
    band_h = max(8, int(round(h * float(band_frac))))
    y1 = max(0, h - band_h)
    y2 = h
    cw = max(8, int(round(w * float(center_frac))))
    cx = w // 2
    x1 = max(0, cx - cw // 2)
    x2 = min(w, cx + cw // 2)
    side_trim = int(round((x2 - x1) * float(side_trim_frac)))
    x1 += side_trim
    x2 -= side_trim
    if x2 - x1 < 16 or y2 - y1 < 6:
        return None
    band = warped_bgr[y1:y2, x1:x2]
    if band.size == 0:
        return None
    if band.ndim == 3:
        band_gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    else:
        band_gray = band
    return band_gray


def _v156_footer_signature(band_gray: np.ndarray) -> Optional[Dict[str, Any]]:
    """Compute lightweight ink-mask signatures for a footer band.

    Returns a dict with:
      mask:     adaptive-binarized ink mask (uint8, 0/1)
      col_prof: column-sum profile (float, normalized to [0,1])
      row_prof: row-sum profile (float, normalized to [0,1])
      ink:      total ink ratio in band (float in [0,1])
      thumb:    9x8 normalized grayscale thumbnail (uint8)
      dhash:    64-bit dHash int over the thumbnail
    or None on failure.
    """
    if band_gray is None or band_gray.size == 0:
        return None
    try:
        g = band_gray.astype(np.uint8)
        if g.shape[0] < 6 or g.shape[1] < 16:
            return None
        # Light blur to suppress paper texture.
        gb = cv2.GaussianBlur(g, (3, 3), 0)
        # Adaptive threshold for cross-illumination invariance. The block
        # size is odd; 25 covers most page-number glyph widths at the
        # rectified resolutions we see (~1000-1200 px wide pages).
        bs = 25 if min(gb.shape[:2]) >= 25 else (max(11, min(gb.shape[:2]) | 1))
        try:
            th = cv2.adaptiveThreshold(
                gb, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                bs, 10,
            )
        except Exception:
            # Fallback: global Otsu on inverted intensity.
            _t, th = cv2.threshold(gb, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mask = (th > 0).astype(np.uint8)
        ink = float(np.count_nonzero(mask)) / float(mask.size)
        # Column / row profiles, normalized.
        col = mask.sum(axis=0).astype(np.float32)
        row = mask.sum(axis=1).astype(np.float32)
        if col.max() > 0:
            col = col / float(col.max())
        if row.max() > 0:
            row = row / float(row.max())
        # 9x8 normalized thumbnail for dHash (8-bit dHash on 9x8 -> 64 bits).
        thumb = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
        # dHash: compare adjacent columns in each row.
        bits = (thumb[:, 1:] > thumb[:, :-1]).astype(np.uint8).flatten()
        dhash = 0
        for b in bits:
            dhash = (dhash << 1) | int(b)
        return {
            'mask': mask,
            'col_prof': col,
            'row_prof': row,
            'ink': ink,
            'thumb': thumb,
            'dhash': dhash,
        }
    except Exception:
        return None


def _v156_corr_with_shift(a: np.ndarray, b: np.ndarray, max_shift: int) -> float:
    """Maximum Pearson correlation between a and b over integer shifts in
    [-max_shift, +max_shift]. Robust to small horizontal/vertical
    misalignment of the page-number band caused by trim or warp wobble.
    """
    try:
        n = min(a.size, b.size)
        if n < 4:
            return 0.0
        a = a[:n].astype(np.float32)
        b = b[:n].astype(np.float32)
        if float(np.std(a)) < 1e-6 or float(np.std(b)) < 1e-6:
            return 0.0
        max_shift = max(0, min(int(max_shift), n - 2))
        best = -1.0
        for s in range(-max_shift, max_shift + 1):
            if s >= 0:
                aa = a[s:]
                bb = b[: len(aa)]
            else:
                bb = b[-s:]
                aa = a[: len(bb)]
            if aa.size < 4:
                continue
            sa = float(aa.std())
            sb = float(bb.std())
            if sa < 1e-6 or sb < 1e-6:
                continue
            c = float(np.corrcoef(aa, bb)[0, 1])
            if not np.isfinite(c):
                continue
            if c > best:
                best = c
        return float(best if best > -1.0 else 0.0)
    except Exception:
        return 0.0


def _v156_dhash64_hamming(a: int, b: int) -> int:
    try:
        return int(bin(int(a) ^ int(b)).count('1'))
    except Exception:
        return 64


def _v156_footer_distinctness(
    a: 'FrameFeatures',
    b: 'FrameFeatures',
    args,
) -> Tuple[bool, str, Dict[str, float]]:
    """Returns (is_distinct, reason, metrics).

    is_distinct=True means the bottom-center bands of the two pages are
    confidently different, which should BLOCK a non-strict same-page merge.
    is_distinct=False means the guard is inconclusive and the upstream
    decision should stand.

    Inconclusive cases (return False with reason):
      * either warped_bgr is None / band extraction fails (no-data),
      * either band has near-zero ink (blank-bottom — guard cannot
        separate; let upstream decide),
      * signals broadly agree (col_corr & row_corr high, low ham, small
        ink_delta) — which is the "looks like same page" finding.
    """
    metrics: Dict[str, float] = {
        'col_corr': 1.0, 'row_corr': 1.0, 'ink_a': 0.0, 'ink_b': 0.0,
        'ink_delta': 0.0, 'ham': 0.0, 'applied': 0.0,
    }
    if a is None or b is None:
        return False, 'null', metrics
    band_frac = float(getattr(args, 'v156_footer_band_frac', 0.09))
    center_frac = float(getattr(args, 'v156_footer_center_frac', 0.60))
    side_trim_frac = float(getattr(args, 'v156_footer_side_trim_frac', 0.08))
    col_corr_max = float(getattr(args, 'v156_footer_col_corr_max', 0.70))
    row_corr_max = float(getattr(args, 'v156_footer_row_corr_max', 0.70))
    ink_delta_min = float(getattr(args, 'v156_footer_ink_delta_min', 0.12))
    ham_min = int(getattr(args, 'v156_footer_ham_min', 14))
    min_ink = float(getattr(args, 'v156_footer_min_ink', 0.012))

    band_a = _v156_footer_extract_band(a.warped_bgr, band_frac, center_frac, side_trim_frac)
    band_b = _v156_footer_extract_band(b.warped_bgr, band_frac, center_frac, side_trim_frac)
    if band_a is None or band_b is None:
        return False, 'no-band', metrics
    sig_a = _v156_footer_signature(band_a)
    sig_b = _v156_footer_signature(band_b)
    if sig_a is None or sig_b is None:
        return False, 'no-sig', metrics
    metrics['applied'] = 1.0
    metrics['ink_a'] = float(sig_a['ink'])
    metrics['ink_b'] = float(sig_b['ink'])
    metrics['ink_delta'] = float(abs(sig_a['ink'] - sig_b['ink']))

    # Blank-footer fallback: if either band has too little ink to identify
    # a glyph, refuse to claim distinctness. Also fall back when the ink
    # ratios are very lopsided (e.g. one band caught the page number cleanly
    # and the other caught mostly white space below the page number — same
    # physical page, different vertical band alignment). In that case the
    # whole signature comparison is unreliable so we let upstream dedup
    # decide.
    if min(sig_a['ink'], sig_b['ink']) < min_ink:
        return False, f'blank-footer(ink_a={sig_a["ink"]:.4f},ink_b={sig_b["ink"]:.4f})', metrics
    # Lopsided fallback: ratio of the smaller to larger ink ratio is
    # required to be at least 0.35. Below that, at least one band missed
    # the glyph and we cannot claim "distinct" reliably.
    ink_lo = min(sig_a['ink'], sig_b['ink'])
    ink_hi = max(sig_a['ink'], sig_b['ink'])
    if ink_hi > 0 and (ink_lo / ink_hi) < float(getattr(args, 'v156_footer_ink_ratio_min', 0.35)):
        return False, (
            f'lopsided-ink(ink_a={sig_a["ink"]:.4f},ink_b={sig_b["ink"]:.4f},'
            f'ratio={ink_lo / max(ink_hi, 1e-6):.2f})'
        ), metrics

    # Resize column/row profiles to a common length for shifted correlation.
    def _resize_1d(p: np.ndarray, target: int) -> np.ndarray:
        if p.size == target:
            return p
        if p.size < 2 or target < 2:
            return p
        # cv2.resize on a 1xN row, then squeeze.
        r = cv2.resize(p.reshape(1, -1).astype(np.float32), (target, 1),
                       interpolation=cv2.INTER_LINEAR)
        return r.flatten()

    col_len = max(8, min(sig_a['col_prof'].size, sig_b['col_prof'].size))
    row_len = max(4, min(sig_a['row_prof'].size, sig_b['row_prof'].size))
    col_a = _resize_1d(sig_a['col_prof'], col_len)
    col_b = _resize_1d(sig_b['col_prof'], col_len)
    row_a = _resize_1d(sig_a['row_prof'], row_len)
    row_b = _resize_1d(sig_b['row_prof'], row_len)

    # Shift tolerance: ~12% of profile length, capped at 6, so a small
    # horizontal/vertical band drift between two captures of the SAME page
    # number cannot fool the guard into "distinct".
    col_shift = max(2, min(6, int(round(col_len * 0.12))))
    row_shift = max(1, min(3, int(round(row_len * 0.20))))
    col_corr = _v156_corr_with_shift(col_a, col_b, col_shift)
    row_corr = _v156_corr_with_shift(row_a, row_b, row_shift)
    metrics['col_corr'] = float(col_corr)
    metrics['row_corr'] = float(row_corr)

    ham = _v156_dhash64_hamming(int(sig_a['dhash']), int(sig_b['dhash']))
    metrics['ham'] = float(ham)

    # Independent disagreement signals.
    sig_col = col_corr < col_corr_max
    sig_row = row_corr < row_corr_max
    sig_ink = metrics['ink_delta'] >= ink_delta_min
    sig_ham = ham >= ham_min
    disagree = int(sig_col) + int(sig_row) + int(sig_ink) + int(sig_ham)

    # Require strong evidence to claim "distinct":
    #   - (column-profile correlation must be confidently low AND
    #      dHash hamming must be confidently high), OR
    #   - column-profile correlation is *very* low (< 0.40) AND at least
    #     one corroborating signal disagrees.
    # Either condition implies the printed glyph occupying the page-number
    # band has shifted enough that two captures of the SAME physical page
    # cannot plausibly produce these signatures.
    col_corr_strict = float(getattr(args, 'v156_footer_col_corr_strict', 0.40))
    ham_strict = int(getattr(args, 'v156_footer_ham_strict', 22))
    cond_a = sig_col and (ham >= ham_strict)
    cond_b = (col_corr < col_corr_strict) and (disagree >= 2)
    confidently_distinct = cond_a or cond_b

    reason = (
        f'col_corr={col_corr:.2f},row_corr={row_corr:.2f},'
        f'ink_delta={metrics["ink_delta"]:.3f},ham={ham},'
        f'ink_a={metrics["ink_a"]:.3f},ink_b={metrics["ink_b"]:.3f},'
        f'disagree={disagree}'
    )
    return bool(confidently_distinct), reason, metrics


# ---------------------------------------------------------------------------
# v15.5: adjacent-winner quality dedup with replacement.
# After v15.4 quality_refinement_pass, walk adjacent winners and merge pairs
# that depict the same physical page; keep the one with the better unified
# visual quality score. Generic, threshold-based — no hardcoded frames.
# ---------------------------------------------------------------------------
def _v155_adjacent_same_page(
    a: 'FrameFeatures',
    b: 'FrameFeatures',
    args,
) -> Tuple[bool, str, Dict[str, float]]:
    """Conservative same-physical-page test for two adjacent winners.

    Returns (is_same, reason, metrics). Three positive paths, all gated by
    a hard text-density-agreement floor:

      A. Strict path: SSIM >= sim_thresh_merge AND ham <= hash_thresh_merge.
      B. Strong-corroborated path: a "primary" signal (high SSIM or low
         dHash) AND a "corroborating" signal (high warp-thumb ratio or
         high text-profile correlation). Mirrors v14.2a auto-dedup.
      C. Temporal-rescan path: when two winners are very close in time
         (dt <= --v155-adj-dedup-rescan-dt-sec, default 1.5s) the operator
         physically cannot have turned the page between them. In this
         regime ROI/dHash similarity is destroyed by hand or skew but the
         pair must be the same physical page provided text density agrees
         and warp/profile/sim is at least *moderate* (no obvious different-
         page evidence). Conservative floors prevent false merges with a
         decoded-but-occluded *different* page on a fast pan.

    Path C is the v15.5 addition that handles IMG_4886 page_005/006/007:
    the user re-frames the same page within a second producing two winners
    whose strict similarity is poor but whose temporal proximity makes
    them the same physical page.
    """
    metrics: Dict[str, float] = {
        'sim': 0.0, 'ham': 64.0, 'prof': 0.0, 'warp': 0.0, 'text_rel': 1.0, 'dt': 0.0,
    }
    if a is None or b is None:
        return False, 'null', metrics
    if a.roi_gray is None or b.roi_gray is None:
        return False, 'no-roi', metrics
    text_a = max(1e-6, float(a.text_score))
    text_b = max(1e-6, float(b.text_score))
    text_rel = abs(text_a - text_b) / max(text_a, text_b)
    metrics['text_rel'] = float(text_rel)
    metrics['dt'] = float(abs(b.t_sec - a.t_sec))
    text_rel_max = float(getattr(args, 'v155_adj_dedup_text_rel_max', 0.30))
    # v15.11 Path D pre-check: motion-blur asymmetry can destroy text-density
    # agreement on the blurred side, so the global text_rel gate must yield
    # when both sides are close in time, neither shows turn-in-progress, and
    # one side is severely sharper than the other. Final acceptance is
    # decided below after geometric/structural signals are also computed.
    blur_a = float(getattr(a, 'blur_score', 0.0))
    blur_b = float(getattr(b, 'blur_score', 0.0))
    blur_max_v = max(blur_a, blur_b, 1.0)
    blur_min_v = min(blur_a, blur_b)
    blur_ratio = blur_min_v / blur_max_v
    metrics['blur_ratio'] = float(blur_ratio)
    metrics['blur_min'] = float(blur_min_v)
    metrics['blur_max'] = float(blur_max_v)
    blur_asym_dt_arg = getattr(args, 'v155_adj_dedup_blur_asym_dt_sec', None)
    if blur_asym_dt_arg is None:
        blur_asym_dt = float(getattr(args, 'min_peak_distance_sec', 0.9)) * 1.5
    else:
        blur_asym_dt = float(blur_asym_dt_arg)
    blur_asym_max_ratio = float(getattr(args, 'v155_adj_dedup_blur_asym_max_ratio', 0.25))
    blur_asym_turn_max = float(getattr(args, 'v155_adj_dedup_blur_asym_turn_max', 0.65))
    # v15.12 Patch A: when blur asymmetry is severe and dt is short, the
    # operator cannot have flipped a page in 1 s while one capture is razor-
    # sharp; turn_penalty inflation on either side under those conditions is
    # reframe noise, not a flip. Use a single max-turn ceiling that floats
    # with severity instead of an AND-of-both gate.
    severe_blur_dt = float(getattr(args, 'v1512_blur_asym_severe_dt_sec', 1.0))
    severe_blur_ratio = float(getattr(args, 'v1512_blur_asym_severe_ratio', 0.15))
    severe_blur_turn_max = float(getattr(args, 'v1512_blur_asym_severe_turn_max', 0.90))
    moderate_blur_dt = float(getattr(args, 'v1512_blur_asym_moderate_dt_sec', 1.5))
    moderate_blur_ratio = float(getattr(args, 'v1512_blur_asym_moderate_ratio', 0.25))
    moderate_blur_turn_max = float(getattr(args, 'v1512_blur_asym_moderate_turn_max', 0.75))
    turn_a_v = float(getattr(a, 'turn_penalty', 0.0))
    turn_b_v = float(getattr(b, 'turn_penalty', 0.0))
    max_turn_v = max(turn_a_v, turn_b_v)
    if blur_ratio <= severe_blur_ratio and metrics['dt'] <= severe_blur_dt:
        max_turn_ceiling = severe_blur_turn_max
        max_turn_tier = 'severe'
    elif blur_ratio <= moderate_blur_ratio and metrics['dt'] <= moderate_blur_dt:
        max_turn_ceiling = moderate_blur_turn_max
        max_turn_tier = 'moderate'
    else:
        max_turn_ceiling = blur_asym_turn_max
        max_turn_tier = 'strict'
    metrics['blur_asym_tier'] = max_turn_tier
    metrics['blur_asym_max_turn'] = float(max_turn_v)
    metrics['blur_asym_max_turn_ceiling'] = float(max_turn_ceiling)
    blur_asym_eligible = (
        blur_asym_max_ratio > 0.0
        and metrics['dt'] <= blur_asym_dt
        and blur_ratio <= blur_asym_max_ratio
        and max_turn_v <= max_turn_ceiling
    )
    if text_rel > text_rel_max and not blur_asym_eligible:
        return False, f'text_rel={text_rel:.2f}>{text_rel_max:.2f}', metrics
    try:
        sim = float(similarity_score(a.roi_gray, b.roi_gray))
    except Exception:
        sim = 0.0
    try:
        ham = int(hamming_distance(a.roi_dhash, b.roi_dhash)) if (a.roi_dhash is not None and b.roi_dhash is not None) else 64
    except Exception:
        ham = 64
    try:
        prof = float(_v134_profile_corr(a, b))
    except Exception:
        prof = 0.0
    try:
        warp_ratio, _hw = _v134_warp_thumb_match(a, b)
        warp_ratio = float(warp_ratio)
    except Exception:
        warp_ratio = 0.0
    metrics['sim'] = sim
    metrics['ham'] = float(ham)
    metrics['prof'] = prof
    metrics['warp'] = warp_ratio

    sim_min = float(getattr(args, 'v155_adj_dedup_sim_min', 0.62))
    ham_max = int(getattr(args, 'v155_adj_dedup_ham_max', 22))
    prof_min = float(getattr(args, 'v155_adj_dedup_profile_min', 0.65))
    warp_min = float(getattr(args, 'v155_adj_dedup_warp_min', 0.78))

    # ---- Path A: strict identity ----------------------------------------
    # Strict identity bypasses the v15.6 footer guard: when whole-page SSIM
    # is very high AND dHash hamming is very low, two captures must depict
    # the same physical page and their footers must match. Removing such
    # near-duplicates is the original v15.5 behaviour and must not regress.
    sim_strict = float(getattr(args, 'sim_thresh_merge', 0.89))
    ham_strict = int(getattr(args, 'hash_thresh_merge', 11))
    if sim >= sim_strict and ham <= ham_strict:
        return True, f'strict(sim={sim:.2f},ham={ham})', metrics

    # v15.6: footer / folio distinctness guard — applies to all non-strict
    # paths below. Computed once; if confidently distinct, we refuse the
    # merge regardless of which non-strict path would have accepted it.
    footer_enabled = bool(getattr(args, 'v156_footer_guard', True))
    footer_distinct = False
    footer_reason = ''
    footer_metrics: Dict[str, float] = {}
    if footer_enabled:
        footer_distinct, footer_reason, footer_metrics = _v156_footer_distinctness(a, b, args)
        for k, v in footer_metrics.items():
            metrics[f'footer_{k}'] = float(v)

    # ---- Path B: strong primary + corroboration -------------------------
    primary_parts: List[str] = []
    if sim >= sim_min:
        primary_parts.append(f'sim={sim:.2f}')
    if ham <= ham_max:
        primary_parts.append(f'ham={ham}')
    primary = bool(primary_parts)

    corroborating_parts: List[str] = []
    if prof >= prof_min:
        corroborating_parts.append(f'prof={prof:.2f}')
    if warp_ratio >= warp_min:
        corroborating_parts.append(f'warp={warp_ratio:.2f}')
    corroborated = bool(corroborating_parts)

    if primary and corroborated:
        if footer_distinct:
            return False, (
                f'footer_distinct[B]({footer_reason});'
                f'pre={",".join(primary_parts + corroborating_parts)}'
            ), metrics
        return True, ','.join(primary_parts + corroborating_parts), metrics

    # ---- Path C: temporal-rescan (very close in time) -------------------
    # When two winners are within a sub-page-turn time gap, the same physical
    # page must be in front of the camera. This rescues pairs whose ROI sim
    # is destroyed by a hand band or skew/curve (e.g. IMG_4886 frames 480/510
    # which depict the same printed page seconds apart but with very
    # different geometric framing).
    #
    # Safety: refuse if ANY of the geometric / structural signals hints at a
    # different page (negligible warp + negligible profile correlation +
    # negligible SSIM) or if either winner's turn_penalty is high (page
    # turn was actually in progress).
    rescan_dt = float(getattr(args, 'v155_adj_dedup_rescan_dt_sec', 1.5))
    rescan_text_rel = float(getattr(args, 'v155_adj_dedup_rescan_text_rel_max', 0.25))
    rescan_warp_floor = float(getattr(args, 'v155_adj_dedup_rescan_warp_floor', 0.50))
    rescan_prof_floor = float(getattr(args, 'v155_adj_dedup_rescan_profile_floor', 0.10))
    rescan_sim_floor = float(getattr(args, 'v155_adj_dedup_rescan_sim_floor', 0.05))
    turn_max = float(getattr(args, 'v155_adj_dedup_rescan_turn_max', 0.65))

    if metrics['dt'] <= rescan_dt and text_rel <= rescan_text_rel:
        if (
            float(getattr(a, 'turn_penalty', 0.0)) <= turn_max
            and float(getattr(b, 'turn_penalty', 0.0)) <= turn_max
        ):
            # Require non-trivial agreement on at least one of the
            # geometric/structural signals so a fleeting different-page
            # frame caught at the moment of a fast pan does not merge.
            if (
                warp_ratio >= rescan_warp_floor
                and prof >= rescan_prof_floor
                and sim >= rescan_sim_floor
            ):
                if footer_distinct:
                    return False, (
                        f'footer_distinct[C]({footer_reason});'
                        f'pre=rescan(dt={metrics["dt"]:.2f}s,sim={sim:.2f},'
                        f'prof={prof:.2f},warp={warp_ratio:.2f},'
                        f'text_rel={text_rel:.2f})'
                    ), metrics
                return True, (
                    f'rescan(dt={metrics["dt"]:.2f}s,sim={sim:.2f},'
                    f'prof={prof:.2f},warp={warp_ratio:.2f},'
                    f'text_rel={text_rel:.2f})'
                ), metrics

    # ---- Path D: motion-blur-asymmetry rescue (v15.11) ------------------
    # When the geometric/text gates above fail because ONE side is severely
    # motion-blurred, the two winners must still depict the same physical
    # page when they are closely spaced in time and neither shows a page-turn.
    # Blur asymmetry is the discriminating signal — a sharp p_N and a sharp
    # p_{N+1} cannot coexist within ~min_peak_distance*1.5s on a hand-held
    # capture, but a sharp p_N + a motion-blurred p_N can (the operator's
    # hand was repositioning).
    blur_asym_warp_floor = float(getattr(args, 'v155_adj_dedup_blur_asym_warp_floor', 0.40))
    blur_asym_prof_floor = float(getattr(args, 'v155_adj_dedup_blur_asym_profile_floor', 0.05))
    blur_asym_sim_floor = float(getattr(args, 'v155_adj_dedup_blur_asym_sim_floor', 0.03))
    if blur_asym_eligible and (
        warp_ratio >= blur_asym_warp_floor
        or prof >= blur_asym_prof_floor
        or sim >= blur_asym_sim_floor
    ):
        # v15.12 Patch A (footer-guard bypass): when blur asymmetry is
        # severe, the blurred side's footer column/row correlation is
        # itself a casualty of the motion smear — the same blank or
        # near-blank footer band correlates poorly between a sharp and a
        # heavily-smeared rendering. If footer ink density agrees within
        # a tight tolerance AND the dHash hamming over the footer band is
        # small, treat the col/row correlation gate as unreliable on this
        # pair and accept the merge.
        bypass_footer = False
        bypass_reason = ''
        if footer_distinct and blur_ratio <= severe_blur_ratio:
            ink_a_v = float(metrics.get('footer_ink_a', 0.0))
            ink_b_v = float(metrics.get('footer_ink_b', 0.0))
            ink_delta = abs(ink_a_v - ink_b_v)
            footer_ham = float(metrics.get('footer_ham', 64.0))
            ink_delta_max = float(getattr(args, 'v1512_blur_asym_footer_ink_delta_max', 0.05))
            footer_ham_max = float(getattr(args, 'v1512_blur_asym_footer_ham_max', 16.0))
            if ink_delta <= ink_delta_max and footer_ham <= footer_ham_max:
                bypass_footer = True
                bypass_reason = (
                    f'severe_blur_footer_bypass(ink_delta={ink_delta:.3f},'
                    f'ham={footer_ham:.0f})'
                )
        if footer_distinct and not bypass_footer:
            return False, (
                f'footer_distinct[D]({footer_reason});'
                f'pre=blur_asym(dt={metrics["dt"]:.2f}s,'
                f'blur_ratio={blur_ratio:.2f})'
            ), metrics
        if bypass_footer:
            metrics['v1512_path_d_footer_bypass'] = 1.0
            return True, (
                f'blur_asym(dt={metrics["dt"]:.2f}s,'
                f'blur_ratio={blur_ratio:.2f},'
                f'sim={sim:.2f},warp={warp_ratio:.2f},prof={prof:.2f},'
                f'{bypass_reason})'
            ), metrics
        return True, (
            f'blur_asym(dt={metrics["dt"]:.2f}s,'
            f'blur_ratio={blur_ratio:.2f},'
            f'sim={sim:.2f},warp={warp_ratio:.2f},prof={prof:.2f})'
        ), metrics

    return False, ('none' if not primary_parts else 'no-corroboration'), metrics


def _v155_quality_score(x: 'FrameFeatures') -> float:
    """Unified visual quality score for adjacent-winner dedup.

    Higher is better. Combines:
      * raw selection scores (peak_score / norm_score) — established
        global signal,
      * clean_visual_score / finger_penalty — paper cleanliness,
      * |deskew_angle| — strong negative weight because the user-visible
        defect on IMG_4886 page_006 was geometric (large skew/curve, not
        background tone). cvs alone does not capture page curvature, so
        skew is weighted heavily here,
      * stability / blur / turn / edge_motion — secondary frame-quality
        signals,
      * hand-related penalties — already-penalised in raw_score but
        re-applied here so a refined-out hand winner cannot reappear via
        adjacent dedup.
    """
    cvs = 0.0
    fg = 0.0
    try:
        if x.warped_bgr is not None:
            metrics = _v135_visual_metrics_cached(x)
            cvs = float(_v135_clean_visual_score(metrics))
            fg = float(_v135_finger_penalty(metrics))
    except Exception:
        cvs = 0.0
        fg = 0.0
    peak = float(getattr(x, 'peak_score', 0.0))
    if peak <= -1e8:
        peak = float(getattr(x, 'norm_score', 0.0))
    norm = float(getattr(x, 'norm_score', 0.0))
    if norm <= -1e8:
        norm = 0.0
    hand = float(getattr(x, 'hand_penalty', 0.0))
    hto = float(getattr(x, 'hand_text_overlap_penalty', 0.0))
    bh = float(getattr(x, 'bottom_hand_penalty', 0.0))
    skew_abs = abs(float(getattr(x, 'deskew_angle', 0.0)))
    turn = float(getattr(x, 'turn_penalty', 0.0))
    em = float(getattr(x, 'edge_motion_penalty', 0.0))
    blur = float(getattr(x, 'blur_score', 0.0))
    stability = float(getattr(x, 'stability_score', 0.0))
    # Skew penalty: linear up to 3 degrees (mostly OK) then steeply
    # progressive past that, since the visual artefact (waving text rows,
    # curved baselines, perceived rotation) compounds with angle.
    if skew_abs <= 3.0:
        skew_pen = 0.05 * skew_abs
    else:
        skew_pen = 0.05 * 3.0 + 0.30 * (skew_abs - 3.0)
    return (
        0.50 * peak
        + 0.20 * norm
        + 0.30 * cvs            # de-weighted vs v15.4 since cvs misses curvature
        - 0.85 * fg
        + 0.10 * stability
        + 0.08 * min(blur / 400.0, 1.0)
        - 0.22 * hto
        - 0.20 * bh
        - 0.18 * hand
        - skew_pen              # see comment above
        - 0.20 * turn
        - 0.10 * em
    )


def v155_adjacent_winner_dedup(
    winners: List['FrameFeatures'],
    args,
) -> Tuple[List['FrameFeatures'], Dict[int, Dict[str, Any]], Dict[str, Any]]:
    """v15.5 adjacent-winner quality dedup.

    Walks winners in temporal order and merges any adjacent pair that
    looks like the same physical page; keeps the one with the better
    unified visual quality score. Returns (new_winners, diag_by_frame,
    summary).

    Generic, threshold-based, bounded by len(winners) pair tests; no
    hardcoded video/page logic. No-op when fewer than 2 winners exist or
    when --expected-pages is set (don't shrink below user's expected
    count).
    """
    summary: Dict[str, Any] = {
        'pairs_checked': 0,
        'pairs_merged': 0,
        'merges': [],
        'enabled': True,
    }
    diag: Dict[int, Dict[str, Any]] = {}
    if not bool(getattr(args, 'v155_adjacent_dedup', True)):
        summary['enabled'] = False
        return winners, diag, summary
    if int(getattr(args, 'expected_pages', 0) or 0) > 0:
        summary['skipped_reason'] = 'expected_pages_set'
        return winners, diag, summary
    if not winners or len(winners) < 2:
        return winners, diag, summary

    window_sec = float(getattr(args, 'v155_adj_dedup_window_sec', 3.0))
    sorted_winners = sorted(winners, key=lambda x: x.t_sec)

    # Iteratively scan adjacent pairs. Restart after each merge so that
    # transitive duplicates (a~b, b~c) all collapse to the cleanest of
    # the three.
    merged_any = True
    safety = max(8, len(sorted_winners) * 3)
    iters = 0
    while merged_any and iters < safety:
        merged_any = False
        iters += 1
        new_list: List['FrameFeatures'] = []
        i = 0
        while i < len(sorted_winners):
            cand = sorted_winners[i]
            if i + 1 < len(sorted_winners):
                nxt = sorted_winners[i + 1]
                if abs(nxt.t_sec - cand.t_sec) <= window_sec:
                    summary['pairs_checked'] += 1
                    is_same, reason, metrics = _v155_adjacent_same_page(cand, nxt, args)
                    # v15.6: surface footer-guard outcome on every pair. If
                    # the guard ran (footer_applied=1) record the metrics on
                    # both winners so winners.csv can be audited even when
                    # the merge was *not* blocked (i.e. footers also
                    # corroborated same-page).
                    footer_applied = float(metrics.get('footer_applied', 0.0)) > 0.5
                    footer_blocked = isinstance(reason, str) and reason.startswith('footer_distinct')
                    if footer_applied:
                        for fr in (cand.frame_idx, nxt.frame_idx):
                            d = diag.setdefault(fr, {})
                            d.setdefault('v156_footer_guard_applied', 1)
                            # Don't overwrite a prior block reason if any.
                            d.setdefault('v156_footer_ink_a', float(metrics.get('footer_ink_a', 0.0)))
                            d.setdefault('v156_footer_ink_b', float(metrics.get('footer_ink_b', 0.0)))
                            d.setdefault('v156_footer_col_corr', float(metrics.get('footer_col_corr', 0.0)))
                            d.setdefault('v156_footer_row_corr', float(metrics.get('footer_row_corr', 0.0)))
                            d.setdefault('v156_footer_ham', float(metrics.get('footer_ham', 0.0)))
                    if footer_blocked:
                        summary['footer_guard_blocks'] = int(summary.get('footer_guard_blocks', 0)) + 1
                        for fr in (cand.frame_idx, nxt.frame_idx):
                            d = diag.setdefault(fr, {})
                            d['v156_footer_guard_blocked'] = 1
                            d['v156_footer_distinct_reason'] = reason
                        summary.setdefault('blocks', []).append({
                            'a': int(cand.frame_idx),
                            'b': int(nxt.frame_idx),
                            'reason': reason,
                            'dt': float(metrics.get('dt', 0.0)),
                        })
                    if is_same:
                        q_a = _v155_quality_score(cand)
                        q_b = _v155_quality_score(nxt)
                        # v15.12 Patch B: when one candidate is severely
                        # sharper than the other (and not heavily skewed),
                        # award an explicit sharpness bonus so the sharp
                        # stable side wins over the blurry side, even when
                        # the blurry side has lower hand_text_overlap on a
                        # low-text page (the f30/f60 IMG_4921 failure mode).
                        ba = float(getattr(cand, 'blur_score', 0.0))
                        bb = float(getattr(nxt, 'blur_score', 0.0))
                        sa = abs(float(getattr(cand, 'deskew_angle', 0.0)))
                        sb = abs(float(getattr(nxt, 'deskew_angle', 0.0)))
                        ta = float(getattr(cand, 'turn_penalty', 0.0))
                        tb = float(getattr(nxt, 'turn_penalty', 0.0))
                        sharp_ratio = float(getattr(args, 'v1512_sharp_tiebreak_ratio', 4.0))
                        skew_max = float(getattr(args, 'v1512_sharp_tiebreak_skew_max', 1.5))
                        turn_cap = float(getattr(args, 'v1512_sharp_tiebreak_turn_max', 0.90))
                        bonus = float(getattr(args, 'v1512_sharp_tiebreak_bonus', 0.25))
                        sharp_applied = ''
                        if sharp_ratio > 0.0 and bonus > 0.0:
                            denom = max(min(ba, bb), 1.0)
                            ratio = max(ba, bb) / denom
                            if ratio >= sharp_ratio:
                                # Skew/turn gates apply to the sharper side
                                # (the keeper-favourite). The skew gate is
                                # against an *additional* problem on the sharp
                                # side that should disqualify it; if the blur
                                # asymmetry is extreme (>=2x sharp_ratio), the
                                # skew gate is dropped because blur alone is
                                # decisive evidence the sharp frame is the
                                # better physical capture.
                                drop_skew = ratio >= 2.0 * sharp_ratio
                                if ba > bb:
                                    skew_ok = drop_skew or (sa <= skew_max)
                                    if skew_ok and ta <= turn_cap:
                                        q_a += bonus
                                        sharp_applied = (
                                            f'sharp+{bonus:.2f}@a(b={ba:.0f},'
                                            f's={sa:.2f},t={ta:.2f},r={ratio:.1f},'
                                            f'drop_skew={int(drop_skew)})'
                                        )
                                elif bb > ba:
                                    skew_ok = drop_skew or (sb <= skew_max)
                                    if skew_ok and tb <= turn_cap:
                                        q_b += bonus
                                        sharp_applied = (
                                            f'sharp+{bonus:.2f}@b(b={bb:.0f},'
                                            f's={sb:.2f},t={tb:.2f},r={ratio:.1f},'
                                            f'drop_skew={int(drop_skew)})'
                                        )
                        if q_b >= q_a:
                            keeper, loser = nxt, cand
                            q_keeper, q_loser = q_b, q_a
                        else:
                            keeper, loser = cand, nxt
                            q_keeper, q_loser = q_a, q_b
                        metrics['v1512_sharp_tiebreak_inputs'] = (
                            f'ba={ba:.0f},bb={bb:.0f},sa={sa:.2f},sb={sb:.2f},'
                            f'ta={ta:.2f},tb={tb:.2f}'
                        )
                        if sharp_applied:
                            metrics['sharp_tiebreak'] = sharp_applied
                            reason = f'{reason};{sharp_applied}'
                        merge_info = {
                            'keeper': int(keeper.frame_idx),
                            'loser': int(loser.frame_idx),
                            'reason': reason,
                            'metrics': metrics,
                            'q_keeper': float(q_keeper),
                            'q_loser': float(q_loser),
                            'delta': float(q_keeper - q_loser),
                        }
                        summary['merges'].append(merge_info)
                        summary['pairs_merged'] += 1
                        diag.setdefault(loser.frame_idx, {}).update({
                            'v155_adj_dedup_removed': 1,
                            'v155_adj_dedup_keeper': int(keeper.frame_idx),
                            'v155_adj_dedup_reason': (
                                f'{reason};dt={metrics["dt"]:.2f}s;'
                                f'q_keep={q_keeper:.3f};q_lose={q_loser:.3f};'
                                f'delta={q_keeper - q_loser:+.3f}'
                            ),
                        })
                        diag.setdefault(keeper.frame_idx, {}).update({
                            'v155_adj_dedup_kept_over': int(loser.frame_idx),
                        })
                        new_list.append(keeper)
                        i += 2  # Skip the loser
                        merged_any = True
                        continue
            new_list.append(cand)
            i += 1
        sorted_winners = new_list

    return sorted_winners, diag, summary


# ---------------------------------------------------------------------------
# v15.12 Patch C: blank front-matter coverage rescue.
#
# When the prefilter correctly classifies an inter-winner gap as containing no
# text/no edges (a blank verso/flyleaf or printer's blank), the page is dropped
# even though it is a real physical page in the operator's pass. Universal,
# default-on, but with very strict guards so it almost never fires on
# non-front-matter content:
#
#   * gap between two consecutive kept winners must be >= min gap (default 1.5s)
#   * NO existing winner sits inside the gap
#   * at least one prefilter-sampled frame in the gap has the "settled blank
#     paper" signature: high paper_ratio, very low motion, sufficient blur
#     (in focus), low edge_density (no text), low bottom_dark (held flat,
#     not edge-of-book), and low skin (no hand)
#   * pick the single sharpest qualifying frame (max blur) per gap
#   * at most one rescue per gap; rescued frame's t_sec must not collide with
#     an existing winner (already enforced by gap-strict bounds)
#
# Produces a list of (frame_idx, t_sec) to attempt to add as winners. The
# caller decodes, warps, and inserts in temporal order.
# ---------------------------------------------------------------------------
def _v1512_blank_front_matter_rescue_candidates(
    prefilter_metrics: Dict[int, Dict[str, float]],
    sampled_indices_pf: List[int],
    winners: List['FrameFeatures'],
    args,
) -> List[Tuple[int, float, Dict[str, float]]]:
    """Return a list of (frame_idx, t_sec, metrics_dict) — one per qualifying
    inter-winner gap — to rescue as blank front-matter pages.
    """
    if not bool(getattr(args, 'v1512_blank_rescue', True)):
        return []
    if int(getattr(args, 'expected_pages', 0) or 0) > 0:
        # Honour explicit user expected count; don't add blanks behind their back.
        return []
    if not winners or len(winners) < 1 or not prefilter_metrics:
        return []

    min_gap = float(getattr(args, 'v1512_blank_rescue_min_gap_sec', 1.5))
    paper_min = float(getattr(args, 'v1512_blank_rescue_paper_min', 0.75))
    motion_max = float(getattr(args, 'v1512_blank_rescue_motion_max', 0.04))
    blur_min = float(getattr(args, 'v1512_blank_rescue_blur_min', 80.0))
    bottom_dark_max = float(getattr(args, 'v1512_blank_rescue_bottom_dark_max', 0.50))
    edge_max = float(getattr(args, 'v1512_blank_rescue_edge_max', 0.10))
    skin_max = float(getattr(args, 'v1512_blank_rescue_skin_max', 0.05))
    text_max = float(getattr(args, 'v1512_blank_rescue_text_max', 1e9))  # no text feature in prefilter; reserved
    bright_min = float(getattr(args, 'v1512_blank_rescue_bright_min', 120.0))

    sorted_winners = sorted(winners, key=lambda w: w.t_sec)
    out: List[Tuple[int, float, Dict[str, float]]] = []
    # Build temporal bracketing intervals: front of video to first winner is
    # the most likely place a blank verso lives, but we limit to gaps between
    # two kept winners only — front/back edges are NOT rescued (avoids
    # over-eager pickup of pre-flip dwell frames).
    for w0, w1 in zip(sorted_winners, sorted_winners[1:]):
        dt = w1.t_sec - w0.t_sec
        if dt < min_gap:
            continue
        gap_lo = w0.t_sec + 0.05  # exclude the keepers themselves
        gap_hi = w1.t_sec - 0.05
        candidates: List[Tuple[int, Dict[str, float]]] = []
        for fi in sampled_indices_pf:
            m = prefilter_metrics.get(fi)
            if m is None:
                continue
            t_sec = float(m.get('t_sec', fi / 30.0))
            if t_sec <= gap_lo or t_sec >= gap_hi:
                continue
            paper = float(m.get('paper_ratio', 0.0))
            motion = float(m.get('motion', 1.0))
            blur = float(m.get('blur', 0.0))
            bottom_dark = float(m.get('bottom_dark', 1.0))
            edge_density = float(m.get('edge_density', 1.0))
            skin = float(m.get('skin', 1.0))
            bright = float(m.get('bright_mean', 0.0))
            if (paper >= paper_min
                and motion <= motion_max
                and blur >= blur_min
                and bottom_dark <= bottom_dark_max
                and edge_density <= edge_max
                and skin <= skin_max
                and bright >= bright_min):
                candidates.append((fi, m))
        if not candidates:
            continue
        # Sharpest blank in the gap (max blur), break ties on lowest motion,
        # then by frame_idx for determinism.
        best_fi, best_m = max(
            candidates,
            key=lambda x: (float(x[1].get('blur', 0.0)),
                           -float(x[1].get('motion', 1.0)),
                           -int(x[0])),
        )
        out.append((int(best_fi), float(best_m.get('t_sec', best_fi / 30.0)),
                    {
                        'paper_ratio': float(best_m.get('paper_ratio', 0.0)),
                        'motion': float(best_m.get('motion', 0.0)),
                        'blur': float(best_m.get('blur', 0.0)),
                        'bottom_dark': float(best_m.get('bottom_dark', 0.0)),
                        'edge_density': float(best_m.get('edge_density', 0.0)),
                        'skin': float(best_m.get('skin', 0.0)),
                        'bright': float(best_m.get('bright_mean', 0.0)),
                        'gap_dt': float(dt),
                    }))
    return out


def _v1512_decode_and_warp_blank(video_path: Path, frame_idx: int, args
                                 ) -> Optional[np.ndarray]:
    """Decode `frame_idx` from `video_path` and produce a clean BGR image
    suitable for output as a rescued blank front-matter page. Tries the
    standard quad-detect+warp; if no quad is found (often the case for a
    blank held in front of a dark background), falls back to a generous
    center-crop of the frame so the blank still appears in the output
    sequence.

    Skips deskew (no text lines available; the routine returns garbage on
    blanks) and any aggressive refinement. The downstream page-write loop
    must also skip the CLAHE/contrast enhance for rescued blanks.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        try:
            quad, _area, _fill = detect_page_quad(frame)
        except Exception:
            quad = None
        warped: Optional[np.ndarray] = None
        if quad is not None:
            try:
                warped = four_point_warp(frame, quad, long_side=args.long_side)
            except Exception:
                warped = None
        if warped is None:
            # Center-crop fallback — assume the page is roughly centred and
            # occupies the inner 70% of frame area.
            h, w = frame.shape[:2]
            cx0 = int(w * 0.15)
            cx1 = int(w * 0.85)
            cy0 = int(h * 0.10)
            cy1 = int(h * 0.90)
            warped = frame[cy0:cy1, cx0:cx1].copy()
            warped = resize_long_side(warped, args.long_side)
        return warped
    finally:
        cap.release()


def _v1513_leading_edge_distinct_page_rescue(
    features: List['FrameFeatures'],
    winners: List['FrameFeatures'],
    args,
) -> Optional['FrameFeatures']:
    """Return a single FrameFeatures to PREPEND as a recovered first page,
    or None if no qualifying candidate exists.

    A real distinct page may have been suppressed at select_local_peaks
    when it was the very first sampled frame — degenerate context
    (stability=0.5 init default, hand_text_overlap=1.0, bottom_hand=1.0)
    collapses its peak_score so the immediate neighbour wins. We look at
    sampled features that are page_found, settled (low edge_motion / low
    turn / good area+fill), reasonably sharp, with non-blank text density,
    and visually DISTINCT from the existing first winner (high dHash
    hamming OR low roi structural similarity). Only fires when
    --expected-pages is unset.
    """
    if not bool(getattr(args, 'v1513_leading_edge_rescue', True)):
        return None
    if int(getattr(args, 'expected_pages', 0) or 0) > 0:
        return None
    if not winners or not features:
        return None

    sorted_winners = sorted(winners, key=lambda w: w.t_sec)
    first = sorted_winners[0]
    min_gap = float(getattr(args, 'v1513_leading_edge_min_gap_sec', 1.5))
    if first.t_sec < min_gap:
        return None

    ham_min = int(getattr(args, 'v1513_leading_edge_ham_min', 22))
    sim_max = float(getattr(args, 'v1513_leading_edge_sim_max', 0.55))
    area_min = float(getattr(args, 'v1513_leading_edge_area_min', 0.55))
    fill_min = float(getattr(args, 'v1513_leading_edge_fill_min', 0.85))
    em_max = float(getattr(args, 'v1513_leading_edge_edge_motion_max', 0.5))
    turn_max = float(getattr(args, 'v1513_leading_edge_turn_max', 0.5))
    blur_min = float(getattr(args, 'v1513_leading_edge_blur_min', 50.0))
    text_min = float(getattr(args, 'v1513_leading_edge_text_min', 0.005))

    winner_frames = {int(w.frame_idx) for w in sorted_winners}
    candidates: List['FrameFeatures'] = []
    for f in features:
        if f is first:
            continue
        if int(getattr(f, 'frame_idx', -1)) in winner_frames:
            continue
        if not bool(getattr(f, 'page_found', False)):
            continue
        if f.warped_bgr is None or f.roi_gray is None:
            continue
        if float(getattr(f, 't_sec', 1e9)) >= first.t_sec - 0.05:
            continue
        if (first.t_sec - float(getattr(f, 't_sec', 0.0))) < min_gap:
            continue
        if float(getattr(f, 'page_area_ratio', 0.0)) < area_min:
            continue
        if float(getattr(f, 'fill_ratio', 0.0)) < fill_min:
            continue
        if float(getattr(f, 'edge_motion_penalty', 1.0)) > em_max:
            continue
        if float(getattr(f, 'turn_penalty', 1.0)) > turn_max:
            continue
        if float(getattr(f, 'blur_score', 0.0)) < blur_min:
            continue
        if float(getattr(f, 'text_score', 0.0)) < text_min:
            continue
        # Distinctness vs current first winner.
        try:
            ham = hamming_distance(f.roi_dhash, first.roi_dhash)
        except Exception:
            ham = 0
        try:
            sim = similarity_score(f.roi_gray, first.roi_gray)
        except Exception:
            sim = 1.0
        if ham < ham_min and sim > sim_max:
            continue
        # Annotate for diagnostics.
        setattr(f, '_v1513_leading_edge_ham', int(ham))
        setattr(f, '_v1513_leading_edge_sim', float(sim))
        candidates.append(f)

    if not candidates:
        return None

    # Sharpest first; tie-break by lower edge_motion then lower turn_penalty
    # then earlier t_sec (most likely to be the actual first page held).
    candidates.sort(
        key=lambda c: (
            float(getattr(c, 'blur_score', 0.0)),
            -float(getattr(c, 'edge_motion_penalty', 1.0)),
            -float(getattr(c, 'turn_penalty', 1.0)),
            -float(getattr(c, 't_sec', 0.0)),
        ),
        reverse=True,
    )
    return candidates[0]


# ---------------------------------------------------------------------------
# v13.2: conservative bottom-band hand cleanup (used only when alt search left
# the winner with high-hand metrics). Focuses on skin-like components touching
# the bottom edge and protects text. Does NOT inpaint paper-bright masks.
# ---------------------------------------------------------------------------
def conservative_bottom_hand_cleanup(image_bgr: np.ndarray, args) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Remove skin-like blobs that touch the bottom (or side) borders without
    creating obvious artifacts. Returns (image, info_dict).

    Steps:
      1. Build a skin-like mask (HSV+YCrCb).
      2. Restrict to the bottom band (default: 30% of height).
      3. Keep only components that touch the bottom or side border.
      4. Reject if the resulting mask covers any significant portion of the
         text region (use a Sauvola-like text mask as a guard).
      5. Inpaint with TELEA at small radius.
    """
    info: Dict[str, Any] = {
        'applied': False,
        'mask_ratio': 0.0,
        'reason': 'not-attempted',
    }
    if image_bgr is None or image_bgr.size == 0:
        info['reason'] = 'empty'
        return image_bgr, info

    h, w = image_bgr.shape[:2]
    band_frac = float(getattr(args, 'alt_cleanup_band_frac', 0.30))
    band_top = int(round(h * (1.0 - band_frac)))
    if band_top >= h - 4:
        info['reason'] = 'band-too-small'
        return image_bgr, info

    skin = skin_like_mask(image_bgr)
    band_mask = np.zeros_like(skin)
    band_mask[band_top:, :] = 255
    skin_band = cv2.bitwise_and(skin, band_mask)

    # Keep only components that touch the bottom or left/right borders inside
    # the band (true hands typically extend off-page).
    border_px = max(6, int(0.02 * min(h, w)))
    bordered = keep_border_connected(skin_band, border_px)
    if int(np.count_nonzero(bordered)) == 0:
        info['reason'] = 'no-border-skin'
        return image_bgr, info

    # Smooth + dilate slightly so inpainting reaches into the contour edges.
    k = max(5, int(min(h, w) * 0.012) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(bordered, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Hard cap: refuse to clean if the mask exceeds max_frac of the page area.
    mask_ratio = float(np.count_nonzero(mask)) / float(h * w + 1)
    info['mask_ratio'] = mask_ratio
    max_frac = float(getattr(args, 'alt_cleanup_max_mask_frac', 0.18))
    if mask_ratio < 0.001:
        info['reason'] = 'too-small'
        return image_bgr, info
    if mask_ratio > max_frac:
        info['reason'] = f'mask-too-large({mask_ratio:.3f}>{max_frac:.3f})'
        return image_bgr, info

    # Text-overlap guard: do not clean if the mask intersects the text region
    # excessively in the upper part of the band (i.e. text we care about).
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    text_bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 25, 12)
    # Restrict the text region to the upper 60% of the band (i.e. the area
    # above the bottom strip where the hand normally sits), so we don't get
    # tricked by paper noise at the very bottom edge.
    text_protect = np.zeros_like(text_bin)
    upper_band_top = band_top
    upper_band_bot = band_top + int((h - band_top) * 0.55)
    text_protect[upper_band_top:upper_band_bot, :] = text_bin[upper_band_top:upper_band_bot, :]
    intersect = cv2.bitwise_and(mask, text_protect)
    text_overlap = float(np.count_nonzero(intersect)) / float(np.count_nonzero(text_protect) + 1)
    if text_overlap > float(getattr(args, 'alt_cleanup_max_text_overlap', 0.18)):
        info['reason'] = f'text-overlap({text_overlap:.2f})'
        return image_bgr, info

    cleaned = cv2.inpaint(image_bgr, mask, 5, cv2.INPAINT_TELEA)
    info['applied'] = True
    info['reason'] = (
        f'applied(mask_ratio={mask_ratio:.3f},text_overlap={text_overlap:.2f})'
    )
    return cleaned, info


def _v140_write_winners_contact_sheet(winners: List['FrameFeatures'], out_path: Path,
                                      thumb_long_side: int = 360) -> None:
    """v14.0: small JPEG grid of winner thumbnails, written only on opt-in."""
    thumbs: List[np.ndarray] = []
    for w in winners:
        if w.warped_bgr is None:
            continue
        h, ww = w.warped_bgr.shape[:2]
        scale = thumb_long_side / max(h, ww)
        nh, nw = max(1, int(h * scale)), max(1, int(ww * scale))
        thumbs.append(cv2.resize(w.warped_bgr, (nw, nh), interpolation=cv2.INTER_AREA))
    if not thumbs:
        return
    cols = min(len(thumbs), 5)
    rows = (len(thumbs) + cols - 1) // cols
    cell_h = max(t.shape[0] for t in thumbs)
    cell_w = max(t.shape[1] for t in thumbs)
    sheet = np.full((rows * cell_h, cols * cell_w, 3), 240, dtype=np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        y0, x0 = r * cell_h, c * cell_w
        sheet[y0:y0 + t.shape[0], x0:x0 + t.shape[1]] = t
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def _v150_prefilter_score_frame(small_bgr: np.ndarray, prev_gray_small: Optional[np.ndarray]
                                 ) -> Tuple[Dict[str, float], np.ndarray]:
    """Cheap per-frame prefilter metrics computed on a downscaled BGR image.

    Returns (metrics, small_gray) where small_gray feeds the next call as
    prev_gray_small for temporal-difference computation.
    """
    gray = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)
    paper_ratio, bright_mean, sat_mean = _cheap_paper_ratio(small_bgr)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sob = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sob_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = cv2.magnitude(sob, sob_y)
    edge_density = float(np.mean(edge_mag > 25.0))
    if prev_gray_small is not None and prev_gray_small.shape == gray.shape:
        diff = cv2.absdiff(gray, prev_gray_small)
        motion = float(np.mean(diff)) / 255.0
    else:
        motion = 0.0
    skin = _cheap_skin_ratio(small_bgr)
    h, w = gray.shape[:2]
    bottom_band = gray[int(h * 0.7):, :]
    bottom_dark_ratio = float(np.mean(bottom_band < 80))
    metrics = {
        'paper_ratio': paper_ratio,
        'bright_mean': bright_mean,
        'sat_mean': sat_mean,
        'blur': blur,
        'edge_density': edge_density,
        'motion': motion,
        'skin': skin,
        'bottom_dark': bottom_dark_ratio,
    }
    return metrics, gray


def _v150_prefilter_composite(metrics: Dict[str, float]) -> float:
    """Combine cheap metrics into a single 'frame likely contains stable
    full page' score. Higher = better candidate.

    Heuristic: paper-like area + sharp edges + sharpness, penalised by
    motion and visible skin/finger area.
    """
    paper = float(metrics.get('paper_ratio', 0.0))
    blur = float(metrics.get('blur', 0.0))
    edge = float(metrics.get('edge_density', 0.0))
    motion = float(metrics.get('motion', 0.0))
    skin = float(metrics.get('skin', 0.0))
    bright = float(metrics.get('bright_mean', 0.0))
    paper_n = min(paper / 0.55, 1.0)
    blur_n = min(blur / 200.0, 1.0)
    edge_n = min(edge / 0.20, 1.0)
    bright_n = min(max(bright - 90.0, 0.0) / 100.0, 1.0)
    motion_pen = min(motion / 0.04, 1.0)
    skin_pen = min(skin / 0.12, 1.0)
    score = (
        1.30 * paper_n +
        0.85 * blur_n +
        0.95 * edge_n +
        0.40 * bright_n -
        1.10 * motion_pen -
        0.55 * skin_pen
    )
    return float(score)


def _v150_run_prefilter(video_path: Path, args, fps: float, step: int
                         ) -> Tuple[Dict[int, Dict[str, float]], List[int], List[int], Dict[str, Any]]:
    """Cheap downscaled scan over sampled frames.

    Returns:
      metrics_by_frame   - frame_idx -> dict of cheap metrics + composite score
      sampled_indices    - all sampled frame_idx encountered (in order)
      candidate_indices  - selected sampled frame_idx (sorted) for full processing
      diagnostics        - dict with v15.3 slot/peak retention info
    """
    long_side = max(160, int(getattr(args, 'prefilter_long_side', 512)))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}, [], [], {'reason': 'cap_open_failed'}
    metrics_by_frame: Dict[int, Dict[str, float]] = {}
    sampled_indices: List[int] = []
    prev_gray_small: Optional[np.ndarray] = None
    frame_idx = 0
    try:
        # Use grab() for unwanted frames (decodes container header only, no
        # pixel decode); retrieve() the sampled frames we actually need.
        while True:
            if frame_idx % step != 0:
                ok = cap.grab()
                if not ok:
                    break
                frame_idx += 1
                continue
            ok = cap.grab()
            if not ok:
                break
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                frame_idx += 1
                continue
            small = resize_long_side(frame, long_side)
            metrics, gray_small = _v150_prefilter_score_frame(small, prev_gray_small)
            metrics['composite'] = _v150_prefilter_composite(metrics)
            metrics['t_sec'] = frame_idx / fps
            metrics_by_frame[frame_idx] = metrics
            sampled_indices.append(frame_idx)
            prev_gray_small = gray_small
            frame_idx += 1
    finally:
        cap.release()

    n = len(sampled_indices)
    diag: Dict[str, Any] = {
        'n_sampled': n,
        'expected_pages': int(getattr(args, 'expected_pages', 0) or 0),
        'global_top_k': 0,
        'slot_retention_enabled': False,
        'slots': [],
        'local_peak_count': 0,
        'neighbor_added': 0,
        'global_kept': 0,
        'final_kept': 0,
    }
    if n == 0:
        return metrics_by_frame, sampled_indices, [], diag

    keep_ratio = float(getattr(args, 'prefilter_keep_ratio', 0.45))
    top_k = int(getattr(args, 'prefilter_top_k', 0))
    min_keep = int(getattr(args, 'prefilter_min_keep', 12))
    if top_k <= 0:
        top_k = max(min_keep, int(round(n * keep_ratio)))
    top_k = min(max(top_k, 1), n)
    diag['global_top_k'] = top_k

    scores = np.array([metrics_by_frame[fi]['composite'] for fi in sampled_indices], dtype=np.float32)
    if n > 3:
        smoothed = scores.copy()
        smoothed[1:-1] = (scores[:-2] + scores[1:-1] + scores[2:]) / 3.0
    else:
        smoothed = scores
    order = np.argsort(-smoothed)
    nbr = max(0, int(getattr(args, 'prefilter_neighborhood', 1)))

    # ---- v15.3: per-slot retention for expected-pages mode --------------
    # When expected_pages is set, the global top-K alone can miss clean,
    # temporally-isolated pages whose composite score is dominated by a
    # cluster of high-blur frames elsewhere in the video. We additionally
    # keep top-K candidates from each of N temporal slots and retain local
    # peaks of the smoothed composite curve. This greatly increases the
    # odds that every distinct page has at least one candidate in the set.
    expected_pages = int(getattr(args, 'expected_pages', 0) or 0)
    slot_retention_enabled = bool(getattr(args, 'prefilter_slot_retention', True)) and expected_pages > 0
    diag['slot_retention_enabled'] = slot_retention_enabled

    selected_pos: set = set()
    # 1. Global top-K (legacy v15.2 behaviour) — always included.
    for pos in order:
        if len(selected_pos) >= top_k:
            break
        selected_pos.add(int(pos))
    diag['global_kept'] = len(selected_pos)

    # ---- v15.9: default-mode head/tail coverage rescue ------------------
    # When `expected_pages` is not set, slot retention (#2 below) and
    # local-peak retention (#3 below) are disabled and global top-K is
    # the only quality gate. On videos whose composite score saturates
    # for many frames (paper-fill + sharpness + edges all clipped at
    # their normalisation caps), the smoothed score forms a long plateau
    # at the ceiling. `np.argsort(-smoothed)` is stable, so plateau ties
    # are broken by ascending sampled-position order — i.e. the EARLIEST
    # plateau positions always win the tiebreak. This is a directional
    # bias: the END of the video is systematically starved of candidates,
    # even if it contains a genuine new page (root cause for IMG_4892's
    # missing 6th colophon page).
    #
    # The fix is intentionally narrow:
    #   1. Only consider an under-covered HEAD or TAIL gap — a contiguous
    #      run of unselected sampled positions touching position 0 or
    #      position n-1. Mid-video gaps are left untouched (those are
    #      the *expected* result of global top-K picking the strongest
    #      cluster, and rescuing them tends to introduce duplicate
    #      winners of pages already represented).
    #   2. Only rescue when the unselected edge region is *qualitatively
    #      distinct* from the nearest selected frame, using the cheap
    #      blur metric already computed in the prefilter as a proxy. We
    #      require the gap's best blur to materially exceed the blur of
    #      the closest already-selected frame. This distinguishes a true
    #      missed page (sharp focus on different content, e.g. IMG_4892
    #      colophon at blur~4083 vs neighbouring selected frame at
    #      blur~348) from a tail of more-of-the-same content (IMG_4890
    #      tail at blur~10900 vs neighbouring selected at blur~6900,
    #      ratio ~1.6x — not distinctive enough to warrant a rescue).
    #
    # The result is at most one extra candidate per qualifying edge
    # gap. Behaviour in `--expected-pages` mode is unchanged because
    # the existing v15.3 per-slot retention already handles this case.
    v159_default_coverage = bool(getattr(args, 'prefilter_v159_default_coverage', True)) \
        and int(getattr(args, 'expected_pages', 0) or 0) == 0
    v159_added = 0
    v159_added_positions: List[int] = []
    v159_blur_ratio_min = float(getattr(args, 'prefilter_v159_blur_ratio_min', 2.5))
    if v159_default_coverage and n >= 4 and selected_pos:
        gap_thr = max(2, int(round(n / max(1, top_k * 2))))
        sel_sorted = sorted(selected_pos)
        first_sel = sel_sorted[0]
        last_sel = sel_sorted[-1]
        # (g0, g1, anchor_far, anchor_sel_pos)
        edge_gaps: List[Tuple[int, int, int, int]] = []
        if first_sel >= gap_thr:
            edge_gaps.append((0, first_sel, 0, first_sel))
        if (n - 1 - last_sel) >= gap_thr:
            edge_gaps.append((last_sel + 1, n, n - 1, last_sel))
        # Pre-compute per-position blur for distinctness check.
        blurs = np.array(
            [float(metrics_by_frame[sampled_indices[p]].get('blur', 0.0))
             for p in range(n)], dtype=np.float32)
        for g0, g1, anchor_far, anchor_sel in edge_gaps:
            window = smoothed[g0:g1]
            if window.size == 0:
                continue
            blur_window = blurs[g0:g1]
            best_blur_in_gap = float(blur_window.max()) if blur_window.size else 0.0
            anchor_blur = float(blurs[anchor_sel])
            # Distinctness gate: tail/head must be visibly sharper than
            # the nearest selected frame, with a small absolute floor so
            # near-zero blur ratios don't trip the gate spuriously.
            if anchor_blur > 1.0:
                blur_ratio = best_blur_in_gap / anchor_blur
            else:
                blur_ratio = best_blur_in_gap
            if blur_ratio < v159_blur_ratio_min:
                continue
            # Pick the position with max smoothed score; on ties, pick
            # the one closest to the gap's far edge (deepest into the
            # under-covered region). On a flat plateau this saturates
            # toward the head/tail extremity; downstream neighbourhood
            # expansion (#4) then pulls in adjacent positions of equal
            # score.
            top_val = float(window.max())
            best_pos = -1
            best_dist = -1.0
            for j in range(window.size):
                if abs(float(window[j]) - top_val) <= 1e-6:
                    cand = g0 + j
                    d = abs(cand - anchor_far)
                    if best_pos < 0 or d < best_dist:
                        best_pos = cand
                        best_dist = d
            if best_pos >= 0 and best_pos not in selected_pos:
                selected_pos.add(int(best_pos))
                v159_added_positions.append(int(best_pos))
                v159_added += 1
    diag['v159_default_coverage_added'] = int(v159_added)
    diag['v159_default_coverage_positions'] = v159_added_positions

    # 2. Per-slot top-K (v15.3 quality guardrail for expected-pages mode).
    if slot_retention_enabled:
        slot_factor = max(1, int(getattr(args, 'prefilter_slot_factor', 2)))
        n_slots = max(expected_pages, expected_pages * slot_factor)
        per_slot_k = max(1, int(getattr(args, 'prefilter_per_slot_top_k', 2)))
        slot_size = n / float(n_slots)
        # Map sampled positions into slot ids based on time order.
        slot_ids = np.array([int(min(n_slots - 1, p / slot_size)) for p in range(n)])
        slots_diag: List[Dict[str, Any]] = []
        for slot in range(n_slots):
            in_slot = [p for p in range(n) if slot_ids[p] == slot]
            if not in_slot:
                slots_diag.append({'slot': slot, 'count': 0, 'kept': 0})
                continue
            in_slot_sorted = sorted(in_slot, key=lambda p: -smoothed[p])
            kept_now = 0
            for p in in_slot_sorted[:per_slot_k]:
                if int(p) not in selected_pos:
                    selected_pos.add(int(p))
                    kept_now += 1
            slots_diag.append({
                'slot': slot, 'count': len(in_slot),
                'kept': sum(1 for p in in_slot_sorted[:per_slot_k]),
                'added_by_slot': kept_now,
                't0': float(metrics_by_frame[sampled_indices[in_slot[0]]]['t_sec']),
                't1': float(metrics_by_frame[sampled_indices[in_slot[-1]]]['t_sec']),
            })
        diag['slots'] = slots_diag

    # 3. Local-peak retention: include any sampled position whose smoothed
    # composite is a strict local maximum within a small radius. This
    # captures distinct content peaks even when their absolute score is
    # below the global top-K cut-off.
    peak_radius = max(1, int(getattr(args, 'prefilter_peak_radius', 2)))
    local_peak_count = 0
    # ---- v15.10: enable local-peak retention in default mode -------------
    # Without --expected-pages, slot retention is gated off, which also
    # disabled the strict-local-max retention below. On videos whose
    # composite saturates into a long plateau (paper_ratio + sharpness +
    # edges all clipped at their normalisation caps), the global top-K
    # tiebreak is biased toward the earliest plateau positions and
    # mid-video pages on the same plateau lose their slot. Local-peak
    # retention is naturally self-limiting (only strict local maxima of
    # the smoothed composite survive — see strict-local-max gate below)
    # so enabling it in default mode adds at most a handful of candidates
    # while rescuing the mid-video peaks that the global top-K starves.
    # Opt-out: --no-prefilter-default-local-peaks restores v15.9 behaviour.
    default_local_peaks_enabled = bool(
        getattr(args, 'prefilter_default_local_peaks', True))
    diag['default_local_peaks_enabled'] = default_local_peaks_enabled
    local_peak_gate_active = (slot_retention_enabled
                              or default_local_peaks_enabled)
    if local_peak_gate_active and n >= 3:
        for p in range(n):
            lo = max(0, p - peak_radius)
            hi = min(n, p + peak_radius + 1)
            window = smoothed[lo:hi]
            if smoothed[p] >= float(window.max()) - 1e-6:
                # tie-breaker: only count as a peak if it's strictly above
                # at least one neighbour (avoids the saturated plateau case
                # where every frame would be flagged).
                if (lo < p and smoothed[p] > smoothed[p - 1] + 1e-6) or \
                   (hi - 1 > p and smoothed[p] > smoothed[p + 1] + 1e-6) or \
                   (lo == p and hi - 1 == p):
                    if int(p) not in selected_pos:
                        selected_pos.add(int(p))
                    local_peak_count += 1
    diag['local_peak_count'] = local_peak_count

    # 4. Neighbour expansion (legacy + still useful for stable picking).
    expanded = set(selected_pos)
    nbr_added = 0
    for pos in selected_pos:
        for d in range(-nbr, nbr + 1):
            np_pos = pos + d
            if 0 <= np_pos < n and np_pos not in expanded:
                expanded.add(np_pos)
                nbr_added += 1
    diag['neighbor_added'] = nbr_added

    candidate_indices = sorted(sampled_indices[p] for p in expanded)
    cand_set = set(candidate_indices)
    for fi in sampled_indices:
        metrics_by_frame[fi]['selected'] = 1.0 if fi in cand_set else 0.0
    diag['final_kept'] = len(candidate_indices)
    return metrics_by_frame, sampled_indices, candidate_indices, diag


def process_video(args):
    video_path = Path(args.video)
    if args.output_dir:
        out_dir = Path(args.output_dir)
        dbg_dir = out_dir.with_name(out_dir.name + '_debug')
    else:
        out_dir = video_path.with_name(video_path.stem + '_pages_v14_0')
        dbg_dir = video_path.with_name(video_path.stem + '_debug_v14_0')
    if args.clean_output and out_dir.exists():
        shutil.rmtree(out_dir)
    if args.clean_output and dbg_dir.exists():
        shutil.rmtree(dbg_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    if args.debug:
        dbg_dir.mkdir(exist_ok=True, parents=True)

    # ----- V13.0/v15.1 adaptive calibration (now coordinated with prefilter) -----
    calibration: Dict[str, Any] = {'enabled': False, 'used': False, 'reason': 'flag-off',
                                   'samples': 0, 'stats': {}, 'overrides': {}, 'applied': {}}
    pre_override_args = {}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError('Could not open video')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / args.sample_fps))) if args.sample_fps > 0 else 1
    sampled_fps = fps / step
    cap.release()

    fast_prefilter_on = bool(getattr(args, 'fast_prefilter', True))
    full_calibration = bool(getattr(args, 'full_calibration', False))
    no_pref_cal = bool(getattr(args, 'no_prefilter_calibration', False))
    prefilter_metrics: Dict[int, Dict[str, float]] = {}
    candidate_set: Optional[set] = None
    sampled_indices_pf: List[int] = []

    use_prefilter_calibration = (
        fast_prefilter_on
        and not full_calibration
        and not no_pref_cal
        and getattr(args, 'adaptive_calibration', True)
    )

    prefilter_diag: Dict[str, Any] = {}
    if fast_prefilter_on:
        with stage_timer('prefilter'):
            prefilter_metrics, sampled_indices_pf, cand_idx, prefilter_diag = _v150_run_prefilter(
                video_path, args, fps, step
            )
            candidate_set = set(cand_idx) if cand_idx else set()
        kept = len(candidate_set or [])
        total = len(sampled_indices_pf)
        ratio = (kept / total) if total else 0.0
        if prefilter_diag.get('slot_retention_enabled'):
            print(f'[v15.3] prefilter: kept {kept}/{total} sampled frames '
                  f'({ratio:.0%}) for full processing '
                  f'(global={prefilter_diag.get("global_kept", 0)}, '
                  f'slots={len(prefilter_diag.get("slots", []))}, '
                  f'peaks={prefilter_diag.get("local_peak_count", 0)}, '
                  f'+nbr={prefilter_diag.get("neighbor_added", 0)})')
        else:
            print(f'[v15.1] prefilter: kept {kept}/{total} sampled frames '
                  f'({ratio:.0%}) for full processing')
    else:
        print('[v15.1] prefilter: disabled (--no-fast-prefilter, legacy full-pass mode)')

    with stage_timer('calibration'):
        if getattr(args, 'adaptive_calibration', True):
            try:
                if use_prefilter_calibration and prefilter_metrics:
                    calibration = _v151_calibration_from_prefilter(
                        video_path, args, fps, prefilter_metrics, sampled_indices_pf
                    )
                    if not calibration.get('used'):
                        # Fallback to full calibration if prefilter-based path failed.
                        calibration = run_adaptive_calibration(video_path, args)
                        calibration['source'] = 'fallback_full'
                else:
                    calibration = run_adaptive_calibration(video_path, args)
                    calibration['source'] = calibration.get('source', 'full')
                applied = apply_calibration_overrides(args, calibration)
                calibration['applied'] = applied
                high_hand = bool(calibration.get('high_hand_mode', False))
                if getattr(args, '_force_high_hand', False):
                    high_hand = True
                    calibration['high_hand_mode'] = True
                    calibration['high_hand_forced'] = True
                setattr(args, '_high_hand_mode', high_hand)
                src = calibration.get('source', 'full')
                if applied or high_hand:
                    pre_override_args = applied
                    hh = ' (high-hand mode ON)' if high_hand else ''
                    print(f'[v15.1] Adaptive calibration ({src}): applied {list(applied.keys())}{hh}')
                else:
                    print(f'[v15.1] Adaptive calibration ({src}): no overrides (within v13.0/v12.9 defaults)')
            except Exception as e:
                calibration = {'enabled': True, 'used': False, 'reason': f'exception:{e}',
                               'samples': 0, 'stats': {}, 'overrides': {}, 'applied': {},
                               'source': 'exception'}
                setattr(args, '_high_hand_mode', bool(getattr(args, '_force_high_hand', False)))
        else:
            setattr(args, '_high_hand_mode', bool(getattr(args, '_force_high_hand', False)))
            print('[v15.1] Adaptive calibration: disabled (--no-adaptive-calibration)')

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError('Could not open video')

    hand_masker = HandMasker(enabled=not args.no_hands, det_conf=args.hand_det_conf, track_conf=args.hand_track_conf)
    features: List[FrameFeatures] = []
    prev_quad = None
    prev_gray_small = None
    frame_idx = 0

    # v15.0/v15.1: when prefilter is on, fetch only candidate frames. v15.1
    # replaces random POS_FRAMES seeks with a sequential grab()/retrieve()
    # pass — grab() advances the demuxer without per-frame pixel decode,
    # which avoids the expensive keyframe re-seek on H.264 streams.
    # We also cache raw BGR frames for candidates so smart_trim can reuse
    # them without re-decoding (bounded by --max-cached-frames).
    raw_frame_cache: Dict[int, np.ndarray] = {}
    cache_limit = int(getattr(args, 'max_cached_frames', 80))
    seek_decode_count = 0
    grab_count = 0
    cand_sorted: List[int] = []
    # ---- v15.2: parallel candidate processing config ---------------------
    par_on_flag = bool(getattr(args, 'parallel_candidates', True))
    par_workers_arg = int(getattr(args, 'candidate_workers', 0) or 0)
    if par_workers_arg <= 0:
        cpu = os.cpu_count() or 1
        par_workers = max(1, min(cpu, 4))
    else:
        par_workers = max(1, par_workers_arg)
    parallel_on = par_on_flag and par_workers > 1
    # The heavy candidate worker. Uses its own HandMasker if hands enabled
    # so we never share a MediaPipe Hands object across threads. Pure CPU
    # work otherwise (cv2 / numpy) which releases the GIL.
    def _v152_process_candidate(fi: int, frame: np.ndarray, hm: 'HandMasker') -> Tuple[int, FrameFeatures, Optional[np.ndarray]]:
        t_sec = fi / fps
        quad, page_area_ratio, fill_ratio, side_label = detect_page_quad_with_side(
            frame, getattr(args, 'page_side', 'auto')
        )
        # Tiny gray used later for sequential edge_motion_penalty.
        curr_gray_small = cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (320, 320), interpolation=cv2.INTER_AREA
        )
        if quad is None:
            placeholder = FrameFeatures(
                fi, t_sec, None, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, None, None, None, None,
            )
            return fi, placeholder, curr_gray_small

        contact_score = border_contact_score(quad, frame.shape)
        turn_penalty = estimate_turn_penalty(frame, quad)

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
            # v15.2: compute the deskew angle locally to avoid the
            # _LAST_DESKEW_ANGLE module-global race between threads. The
            # rotation is identical to deskew_by_text_lines (single-pass
            # v12.8-compatible behaviour).
            _angle = _estimate_skew_angle_legacy(warped)
            if _angle is not None and abs(_angle) >= 0.25:
                _hh, _ww = warped.shape[:2]
                _m = cv2.getRotationMatrix2D((_ww / 2, _hh / 2), _angle, 1.0)
                warped = cv2.warpAffine(
                    warped, _m, (_ww, _hh),
                    flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
                )
                cand_deskew = float(_angle)
            else:
                cand_deskew = 0.0
            warped = refine_page_after_warp(warped, args)
            hand_mask = build_hand_cleanup_mask(warped, hm, text_protect=False)
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

        feat = FrameFeatures(
            frame_idx=fi,
            t_sec=t_sec,
            quad=quad,
            page_found=True,
            page_area_ratio=page_area_ratio,
            fill_ratio=fill_ratio,
            border_contact_score=contact_score,
            stability_score=0.0,  # filled in sequentially below
            blur_score=blur_score,
            text_score=text_score,
            hand_penalty=hand_penalty,
            hand_text_overlap_penalty=hand_text_penalty,
            edge_foreground_penalty=fg_penalty,
            bottom_hand_penalty=btm_hand,
            turn_penalty=turn_penalty,
            edge_motion_penalty=0.0,  # filled in sequentially below
            gray=gray,
            roi_gray=roi_gray,
            roi_dhash=roi_dhash,
            warped_bgr=warped_bgr,
            deskew_angle=cand_deskew,
        )
        return fi, feat, curr_gray_small

    if candidate_set is not None and sampled_indices_pf:
      try:
        with stage_timer('full_process_candidates'):
          cand_sorted = sorted(candidate_set)
          cand_iter = iter(cand_sorted)
          next_target: Optional[int] = next(cand_iter, None)
          cur_pos = 0
          # Single pointer over sampled_indices_pf for placeholder emission.
          sampled_ptr = 0
          n_sampled = len(sampled_indices_pf)

          def _emit_placeholders_until(upto_inclusive: int):
            nonlocal sampled_ptr
            while sampled_ptr < n_sampled:
                si = sampled_indices_pf[sampled_ptr]
                if si > upto_inclusive:
                    break
                if si not in candidate_set:
                    features.append(FrameFeatures(si, si / fps, None, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, None, None, None, None))
                sampled_ptr += 1

          # ---- Phase 1: sequential decode of candidate frames -------------
          # cv2.VideoCapture is NOT thread-safe so we keep all grab/retrieve
          # calls on this thread. Decoded frames go into decoded_frames
          # (fi -> BGR) for parallel processing in phase 2.
          decoded_frames: List[Tuple[int, np.ndarray]] = []
          frame_shape_ref: Optional[Tuple[int, int, int]] = None
          with stage_timer('decode_candidates'):
            while next_target is not None:
              while cur_pos < next_target:
                  ok = cap.grab()
                  if not ok:
                      break
                  cur_pos += 1
                  grab_count += 1
              if cur_pos != next_target:
                  break
              ok = cap.grab()
              if not ok:
                  break
              ok2, frame = cap.retrieve()
              ok = ok and ok2
              seek_decode_count += 1
              cur_pos += 1
              fi = next_target
              next_target = next(cand_iter, None)
              if not ok or frame is None:
                  # Mark a placeholder for this missing candidate slot now.
                  _emit_placeholders_until(fi - 1)
                  if sampled_ptr < n_sampled and sampled_indices_pf[sampled_ptr] == fi:
                      sampled_ptr += 1
                  features.append(FrameFeatures(fi, fi / fps, None, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, None, None, None, None))
                  continue
              if len(raw_frame_cache) < cache_limit:
                  raw_frame_cache[fi] = frame
              if frame_shape_ref is None:
                  frame_shape_ref = frame.shape
              decoded_frames.append((fi, frame))

          # ---- Phase 2: parallel or sequential candidate processing -------
          # Each worker uses either the shared HandMasker (sequential path)
          # or its own thread-local HandMasker (parallel path) — MediaPipe
          # Hands.process() is not safe to call concurrently from multiple
          # threads on the same Hands instance.
          n_cands = len(decoded_frames)
          processed: List[Tuple[int, FrameFeatures, Optional[np.ndarray]]] = []
          worker_hms: List['HandMasker'] = []
          actual_workers = 1
          if parallel_on and n_cands > 1:
            actual_workers = min(par_workers, n_cands)
            tls = threading.local()
            def _get_local_hm() -> 'HandMasker':
                hm = getattr(tls, 'hm', None)
                if hm is None:
                    hm = HandMasker(
                        enabled=not args.no_hands,
                        det_conf=args.hand_det_conf,
                        track_conf=args.hand_track_conf,
                    )
                    tls.hm = hm
                    worker_hms.append(hm)
                return hm
            def _wrap(fi_frame):
                fi, frame = fi_frame
                return _v152_process_candidate(fi, frame, _get_local_hm())
            with stage_timer('parallel_full_process'):
              with ThreadPoolExecutor(max_workers=actual_workers) as ex:
                # executor.map preserves input order; results gathered as a list.
                for res in ex.map(_wrap, decoded_frames):
                    processed.append(res)
            # Close per-worker MediaPipe hands.
            for hm in worker_hms:
                try:
                    hm.close()
                except Exception:
                    pass
          else:
            with stage_timer('sequential_full_process'):
              for fi, frame in decoded_frames:
                processed.append(_v152_process_candidate(fi, frame, hand_masker))

          # ---- Phase 3: deterministic merge by frame_idx ------------------
          # Sort to guarantee identical ordering vs sequential v15.1.
          processed.sort(key=lambda t: t[0])
          # Build a per-fi map for quick lookup; also retain ordered list.
          processed_map: Dict[int, Tuple[FrameFeatures, Optional[np.ndarray]]] = {
              fi: (feat, gs) for (fi, feat, gs) in processed
          }
          processed_order = [fi for (fi, _, _) in processed]

          # Sequentially compute stability_score (uses prev_quad) and
          # edge_motion_penalty (uses prev_gray_small) so values are
          # bit-identical to the v15.1 sequential path.
          for fi in processed_order:
            feat, curr_gray_small = processed_map[fi]
            # Emit placeholders for any non-candidate sampled indices < fi
            # (matches the original interleaving exactly).
            _emit_placeholders_until(fi - 1)
            if sampled_ptr < n_sampled and sampled_indices_pf[sampled_ptr] == fi:
                sampled_ptr += 1

            if feat.page_found and feat.quad is not None:
                stability_score = estimate_stability(prev_quad, feat.quad, frame_shape_ref)
                prev_quad = feat.quad.copy()
                feat.stability_score = stability_score
            if curr_gray_small is not None:
                edge_motion_penalty = estimate_edge_motion_penalty(curr_gray_small, prev_gray_small)
                prev_gray_small = curr_gray_small
                if feat.page_found:
                    feat.edge_motion_penalty = edge_motion_penalty
            features.append(feat)

          # After candidate loop: emit placeholders for any trailing
          # non-candidate sampled indices so downstream timeline is intact.
          while sampled_ptr < n_sampled:
            si = sampled_indices_pf[sampled_ptr]
            if si not in candidate_set:
                features.append(FrameFeatures(si, si / fps, None, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, None, None, None, None))
            sampled_ptr += 1
          if getattr(args, 'profile', False) or getattr(args, 'debug', False):
            mode = 'parallel' if (parallel_on and n_cands > 1) else 'sequential'
            print(f'[v15.2] candidates: {len(cand_sorted)} retrieved (decode), '
                  f'{grab_count} demuxer-only grabs (no pixel decode); '
                  f'mode={mode}, workers={actual_workers}, processed={n_cands}')
            t_par = _STAGE_TIMINGS.get('parallel_full_process')
            t_seq = _STAGE_TIMINGS.get('sequential_full_process')
            if t_par is not None and n_cands > 0:
                avg = t_par / n_cands
                print(f'[v15.2] parallel_full_process: {t_par:.3f}s '
                      f'(~{avg*1000:.1f} ms/cand wallclock; ideal speedup ~{actual_workers}x)')
            if t_seq is not None and n_cands > 0:
                avg = t_seq / n_cands
                print(f'[v15.2] sequential_full_process: {t_seq:.3f}s '
                      f'(~{avg*1000:.1f} ms/cand)')
      finally:
        cap.release()
      # Skip the legacy sequential loop below.
      _v150_skip_legacy_loop = True
    else:
      _v150_skip_legacy_loop = False

    try:
      if _v150_skip_legacy_loop:
        pass
      else:
       with stage_timer('sample_detect_warp_score'):
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

    with stage_timer('score_normalize'):
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

    # V13.0: post-calibration nudge of min_norm_score using empirical
    # candidate distribution. Tightly clamped: never deviates from the v12.9
    # default by more than ±0.04, so on well-behaved videos this is a no-op.
    post_cal_info: Dict[str, Any] = {'applied': False}
    if getattr(args, 'adaptive_calibration', True) and calibration.get('used', False):
        post_cal_info = adaptive_post_calibration(valid, calibration, args)
        if post_cal_info.get('applied'):
            args.min_norm_score = post_cal_info['after']
            print(f"[v14.0] Adaptive post-calibration: min_norm_score "
                  f"{post_cal_info['before']:.3f} -> {post_cal_info['after']:.3f}")
    calibration['post'] = post_cal_info

    with stage_timer('peaks_and_clusters'):
        winners_pre = select_local_peaks(features, sampled_fps, args)
        if not winners_pre:
            winners_pre = sorted(valid, key=base_preference_score, reverse=True)[:max(1, args.expected_pages or 5)]

        clusters = cluster_candidates(winners_pre, args)
        clusters = force_reduce(clusters, args.expected_pages, args)

    high_hand_mode = bool(getattr(args, '_high_hand_mode', False))

    # v13.1: pick the cluster winner using cluster_select_score (deskew + hand
    # penalties on top of peak/raw score). When the cluster has only one
    # member, that member trivially wins. When multiple members exist but the
    # composite scores are within `cluster_score_eps` we fall back to
    # choose_between_similar so well-behaved videos (IMG_4883) keep their old
    # winners and only ambiguous high-hand clusters get re-ranked.
    # v13.3: cluster_select_score reranking is gated on high_hand_mode. When
    # high_hand_mode is OFF we fall back to the v13.0 / v12.9 choice exactly
    # (legacy choose_between_similar). This restores stable winners on
    # IMG_4883 where v13.1/v13.2 reranking shifted frames 0,100,200,310,410
    # to 105,195,255,315,390 and produced duplicate pages 2/3.
    cluster_score_eps = float(getattr(args, 'cluster_score_eps', 0.020))
    winners: List[FrameFeatures] = []
    cluster_info_by_frame: Dict[int, Dict[str, Any]] = {}
    for cl in sorted(clusters, key=lambda c: min(m.t_sec for m in c.members)):
        if len(cl.members) == 1:
            chosen = cl.members[0]
            winners.append(chosen)
        elif not high_hand_mode:
            # Legacy v13.0 / v12.9 winner selection inside the cluster.
            chosen = cl.members[0]
            for m in cl.members[1:]:
                chosen = choose_between_similar(chosen, m, args.sim_thresh_merge)
            winners.append(chosen)
        else:
            scored = [(cluster_select_score(m, args, high_hand_mode), m) for m in cl.members]
            scored.sort(key=lambda z: z[0], reverse=True)
            top_score, top = scored[0]
            if (top_score - scored[1][0]) < cluster_score_eps:
                legacy = cl.members[0]
                for m in cl.members[1:]:
                    legacy = choose_between_similar(legacy, m, args.sim_thresh_merge)
                chosen = legacy
            else:
                chosen = top
            winners.append(chosen)
        soft = float(getattr(args, 'deskew_soft_threshold', 12.0))
        alpha = float(getattr(args, 'cluster_deskew_weight', 0.030))
        if high_hand_mode:
            alpha *= 1.7
        deskew_pen = deskew_soft_penalty(getattr(chosen, 'deskew_angle', 0.0), soft, alpha)
        beta = float(getattr(args, 'cluster_hand_text_weight', 0.55))
        gamma = float(getattr(args, 'cluster_bottom_hand_weight', 0.40))
        delta = float(getattr(args, 'cluster_hand_weight', 0.40))
        if high_hand_mode:
            beta *= 1.6
            gamma *= 1.7
            delta *= 1.5
        hand_w = (
            beta * float(getattr(chosen, 'hand_text_overlap_penalty', 0.0))
            + gamma * float(getattr(chosen, 'bottom_hand_penalty', 0.0))
            + delta * float(getattr(chosen, 'hand_penalty', 0.0))
        )
        merge_reasons = list(cl.merge_reasons) if cl.merge_reasons else []
        cluster_info_by_frame[chosen.frame_idx] = {
            'cluster_select_score': cluster_select_score(chosen, args, high_hand_mode),
            'deskew_penalty': deskew_pen,
            'hand_penalty_weighted': hand_w,
            'high_hand_mode': high_hand_mode,
            'duplicate_merge_reason': '|'.join(merge_reasons),
            'cluster_size': len(cl.members),
        }

    # v13.4: collect reselection diagnostics keyed by replacement frame_idx.
    reselection_diag: Dict[int, Dict[str, Any]] = {}
    winners_pre_v134 = list(winners)
    with stage_timer('reselection'):
        winners = fill_expected_pages_by_time(winners, valid, args, reselection_diag)
    if args.expected_pages > 0:
        pre_set = {x.frame_idx for x in winners_pre_v134}
        post_set = {x.frame_idx for x in winners}
        added = sorted(post_set - pre_set)
        removed = sorted(pre_set - post_set)
        if added or removed:
            print(f'[v14.0] expected-pages reselection: removed={removed} added={added}')
        for fid, info in reselection_diag.items():
            reason = info.get('reselection_reason', '')
            if reason:
                print(f'[v13.5]   frame {fid}: {reason}')

    # ---- v15.3: fast-mode quality fallback ------------------------------
    # When --expected-pages is set and the fast prefilter is active, the
    # candidate set may have dropped a clean frame in some temporal slot.
    # Slot retention already mitigates this, but as belt-and-suspenders we
    # check the final winners for missing count or visual duplicates and,
    # if present, decode a small bounded set of extra sampled frames from
    # the uncovered temporal intervals only, then re-run reselection.
    fast_quality_fallback_applied = False
    fast_quality_fallback_reason = ''
    fast_quality_fallback_added: List[int] = []
    # v15.6: extend fast_quality_fallback to fire in DEFAULT mode (no
    # --expected-pages) when an anomalously large temporal gap exists
    # between consecutive winners. This addresses IMG_4886 default where
    # the printed page p19 frames (255-345) were dropped by the
    # prefilter, leaving a 6 second gap between p17 (t=7s) and p21
    # (t=13s) — much larger than the median ~2s gap in the rest of the
    # video. The gap-fill is bounded (max_extra frames) and uses the
    # same decoded-and-process pipeline as the expected-pages variant.
    do_fallback = (
        bool(getattr(args, 'fast_quality_fallback', True))
        and fast_prefilter_on
        and (
            args.expected_pages > 0
            or bool(getattr(args, 'v156_default_gap_fill', True))
        )
        and candidate_set is not None
        and sampled_indices_pf
    )
    if do_fallback:
        # Compute uncovered intervals: gaps between consecutive winners (and
        # head/tail) larger than 1.5 * typical_gap. Also fire when count is
        # short.
        sorted_w = sorted(winners, key=lambda c: c.t_sec) if winners else []
        if valid:
            t0 = min(x.t_sec for x in valid)
            t1 = max(x.t_sec for x in valid)
        else:
            t0, t1 = 0.0, 0.0
        # v15.6: when expected_pages is set, derive typical_gap from it (the
        # original v15.3 behaviour). Otherwise, derive typical_gap from the
        # MEDIAN inter-winner gap among the current winners, which tracks
        # the operator's actual page-flip cadence on this video without any
        # hardcoded assumption about page count.
        if args.expected_pages > 0:
            typical_gap = max(0.5, (t1 - t0) / max(1, args.expected_pages))
        else:
            inter_gaps = []
            for i in range(len(sorted_w) - 1):
                inter_gaps.append(float(sorted_w[i + 1].t_sec - sorted_w[i].t_sec))
            if inter_gaps:
                inter_gaps_sorted = sorted(inter_gaps)
                typical_gap = max(0.5, float(inter_gaps_sorted[len(inter_gaps_sorted) // 2]))
            else:
                typical_gap = max(0.5, (t1 - t0) / max(1, len(sorted_w) or 1))
        # v15.6: in default mode the gap-detection multiplier is stricter
        # (require a gap clearly larger than the median, not just slightly).
        # That avoids re-decoding for tiny natural rhythm variations.
        head_mul = 1.4
        mid_mul = 1.6
        if args.expected_pages <= 0:
            head_mul = float(getattr(args, 'v156_default_gap_head_mul', 1.8))
            mid_mul = float(getattr(args, 'v156_default_gap_mid_mul', 1.8))
        uncovered_intervals: List[Tuple[float, float]] = []
        # head
        if sorted_w and (sorted_w[0].t_sec - t0) > head_mul * typical_gap:
            uncovered_intervals.append((t0, sorted_w[0].t_sec))
        # mid
        for i in range(len(sorted_w) - 1):
            if (sorted_w[i + 1].t_sec - sorted_w[i].t_sec) > mid_mul * typical_gap:
                uncovered_intervals.append((sorted_w[i].t_sec, sorted_w[i + 1].t_sec))
        # tail
        if sorted_w and (t1 - sorted_w[-1].t_sec) > head_mul * typical_gap:
            uncovered_intervals.append((sorted_w[-1].t_sec, t1))

        # visual duplicate among winners?
        has_visual_duplicate = False
        for i in range(len(sorted_w)):
            for j in range(i + 1, len(sorted_w)):
                try:
                    if is_visually_same_page(sorted_w[i], sorted_w[j], args):
                        has_visual_duplicate = True
                        break
                except Exception:
                    pass
            if has_visual_duplicate:
                break

        count_short = len(winners) < args.expected_pages
        reasons_list: List[str] = []
        if count_short:
            reasons_list.append(f'count<expected({len(winners)}<{args.expected_pages})')
        if has_visual_duplicate:
            reasons_list.append('visual_duplicate_among_winners')
        if uncovered_intervals and (count_short or has_visual_duplicate):
            reasons_list.append(f'uncovered_intervals={len(uncovered_intervals)}')

        # v15.6: in default mode (no expected_pages), trigger gap-fill on a
        # large temporal gap alone. The median-based threshold above is the
        # safety floor; in addition we require the gap to span at least
        # `v156_default_gap_min_sec` seconds to avoid triggering on tiny
        # videos with very fast page flips.
        default_gap_trigger = (
            args.expected_pages <= 0
            and uncovered_intervals
            and any((b - a) >= float(getattr(args, 'v156_default_gap_min_sec', 4.0))
                    for (a, b) in uncovered_intervals)
        )
        if default_gap_trigger:
            reasons_list.append(f'default_gap_fill={len(uncovered_intervals)}')

        if reasons_list and (count_short or has_visual_duplicate or default_gap_trigger):
            # Pick sampled indices in uncovered intervals that are NOT
            # already in candidate_set. Bound by fast_fallback_max_extra.
            max_extra = max(1, int(getattr(args, 'fast_fallback_max_extra', 24)))
            extra: List[int] = []
            for (a, b) in uncovered_intervals:
                for fi in sampled_indices_pf:
                    if fi in candidate_set:
                        continue
                    t = fi / fps
                    if a <= t <= b:
                        extra.append(fi)
                if len(extra) >= max_extra:
                    break
            extra = sorted(set(extra))[:max_extra]
            if extra:
                # Decode + process the extras with the same per-candidate
                # pipeline used for the main candidate phase.
                cap_fb = cv2.VideoCapture(str(video_path))
                if cap_fb.isOpened():
                    try:
                        cur_pos = 0
                        new_features: List[FrameFeatures] = []
                        decoded_extra: List[Tuple[int, np.ndarray]] = []
                        # sequential grab/retrieve for extra targets
                        for tgt in extra:
                            while cur_pos < tgt:
                                ok = cap_fb.grab()
                                if not ok:
                                    break
                                cur_pos += 1
                            if cur_pos != tgt:
                                continue
                            ok = cap_fb.grab()
                            if not ok:
                                break
                            ok2, f = cap_fb.retrieve()
                            cur_pos += 1
                            if not ok2 or f is None:
                                continue
                            decoded_extra.append((tgt, f))
                        # process
                        for fi, frame in decoded_extra:
                            try:
                                _, feat, _ = _v152_process_candidate(fi, frame, hand_masker)
                            except Exception:
                                continue
                            new_features.append(feat)
                        # Score-normalize: append to valid and rebuild
                        # raw_score / norm_score over the union so the new
                        # frames are comparable to existing ones.
                        added_valid = [x for x in new_features if x.page_found and x.warped_bgr is not None]
                        if added_valid:
                            features.extend(new_features)
                            valid.extend(added_valid)
                            # Rebuild raw/norm scores with the same formula
                            # used above so newcomers get norm_score values.
                            for x in valid:
                                x.raw_score = (
                                    1.40 * x.page_area_ratio +
                                    0.85 * x.fill_ratio +
                                    0.85 * x.border_contact_score +
                                    0.95 * x.stability_score +
                                    0.65 * (np.log1p(x.blur_score) / 8.0) +
                                    0.50 * x.text_score -
                                    0.85 * x.hand_penalty -
                                    2.40 * x.hand_text_overlap_penalty -
                                    1.55 * x.edge_foreground_penalty -
                                    0.75 * x.bottom_hand_penalty -
                                    0.70 * x.turn_penalty
                                )
                            raw_all = np.array([x.raw_score for x in valid], dtype=np.float32)
                            norm_all = robust_norm(raw_all, higher_is_better=True)
                            for i, x in enumerate(valid):
                                x.norm_score = float(norm_all[i])
                            # Also recompute peak_score where missing.
                            # Re-run the reselection pipeline over the
                            # enlarged valid set.
                            reselection_diag2: Dict[int, Dict[str, Any]] = {}
                            winners2 = fill_expected_pages_by_time(list(winners), valid, args, reselection_diag2)
                            if args.expected_pages > 0 and len(winners2) > args.expected_pages:
                                winners2 = sorted(winners2, key=base_preference_score, reverse=True)[:args.expected_pages]
                                winners2 = sorted(winners2, key=lambda c: c.t_sec)
                            pre_ids = {x.frame_idx for x in winners}
                            post_ids = {x.frame_idx for x in winners2}
                            # v15.6: in default mode (no expected_pages),
                            # fill_expected_pages_by_time is a no-op so we
                            # must do the slot insertion ourselves. For each
                            # detected uncovered_interval, pick the best
                            # newly-decoded candidate (highest raw_score
                            # minus hand penalties) that is not a same-page
                            # duplicate of either bracketing winner, and
                            # insert it as a new winner. This is bounded by
                            # the number of uncovered intervals (typically
                            # 1) so adds at most a handful of frames.
                            if (
                                args.expected_pages <= 0
                                and uncovered_intervals
                                and added_valid
                                and bool(getattr(args, 'v156_default_gap_fill', True))
                            ):
                                added_by_idx = {x.frame_idx: x for x in added_valid}
                                inserted_v156: List[FrameFeatures] = list(winners2)
                                inserted_frames: List[int] = []
                                for (a_t, b_t) in uncovered_intervals:
                                    in_gap = [
                                        x for x in added_valid
                                        if a_t < x.t_sec < b_t
                                        and x.page_found
                                        and x.warped_bgr is not None
                                    ]
                                    if not in_gap:
                                        continue
                                    in_gap.sort(
                                        key=lambda x: (
                                            float(getattr(x, 'raw_score', 0.0))
                                            - 0.30 * float(getattr(x, 'hand_text_overlap_penalty', 0.0))
                                            - 0.20 * float(getattr(x, 'bottom_hand_penalty', 0.0))
                                            - 0.15 * float(getattr(x, 'hand_penalty', 0.0))
                                            - 0.20 * float(getattr(x, 'turn_penalty', 0.0))
                                        ),
                                        reverse=True,
                                    )
                                    # Find current bracketing winners by time.
                                    sorted_inserted = sorted(inserted_v156, key=lambda c: c.t_sec)
                                    left = None
                                    right = None
                                    for cw in sorted_inserted:
                                        if cw.t_sec <= a_t + 1e-3:
                                            left = cw
                                        if cw.t_sec >= b_t - 1e-3 and right is None:
                                            right = cw
                                    chosen = None
                                    for cand in in_gap:
                                        # Must be a true different page from
                                        # both neighbours. Use the existing
                                        # same-page tests (visual + relaxed).
                                        try:
                                            same_left = (
                                                left is not None
                                                and is_visually_same_page(cand, left, args)
                                            )
                                        except Exception:
                                            same_left = False
                                        try:
                                            same_right = (
                                                right is not None
                                                and is_visually_same_page(cand, right, args)
                                            )
                                        except Exception:
                                            same_right = False
                                        if same_left or same_right:
                                            continue
                                        # Also enforce min_peak_distance to
                                        # both neighbours so we don't insert
                                        # a frame that overlaps a winner in
                                        # time.
                                        min_dt = float(getattr(args, 'min_peak_distance_sec', 1.5))
                                        too_close = (
                                            (left is not None and abs(cand.t_sec - left.t_sec) < min_dt)
                                            or (right is not None and abs(cand.t_sec - right.t_sec) < min_dt)
                                        )
                                        if too_close:
                                            continue
                                        # Footer guard cross-check: the
                                        # candidate's footer must be
                                        # confidently DIFFERENT from at
                                        # least one neighbour, otherwise we
                                        # might be inserting a near-
                                        # duplicate. This is the same
                                        # signature comparison the dedup
                                        # path uses, applied in the inverse
                                        # direction.
                                        try:
                                            distinct_left = True
                                            distinct_right = True
                                            if left is not None:
                                                d_ok, _, _ = _v156_footer_distinctness(cand, left, args)
                                                distinct_left = bool(d_ok)
                                            if right is not None:
                                                d_ok, _, _ = _v156_footer_distinctness(cand, right, args)
                                                distinct_right = bool(d_ok)
                                            if not (distinct_left or distinct_right):
                                                # Footer signatures don't
                                                # disagree from either
                                                # neighbour — likely a
                                                # near-duplicate, skip.
                                                # Still allow insertion if
                                                # the bands were
                                                # inconclusive (no-band /
                                                # blank-footer): trust the
                                                # is_visually_same_page
                                                # decision in that case.
                                                pass
                                        except Exception:
                                            pass
                                        chosen = cand
                                        break
                                    if chosen is not None:
                                        inserted_v156.append(chosen)
                                        inserted_frames.append(int(chosen.frame_idx))
                                if inserted_frames:
                                    inserted_v156 = sorted(inserted_v156, key=lambda c: c.t_sec)
                                    winners2 = inserted_v156
                                    post_ids = {x.frame_idx for x in winners2}
                                    for fi in inserted_frames:
                                        reselection_diag2.setdefault(fi, {}).update({
                                            'reselection_reason': f'v156_default_gap_fill(t={added_by_idx[fi].t_sec:.2f}s)',
                                        })
                            if pre_ids != post_ids:
                                winners = winners2
                                fast_quality_fallback_applied = True
                                fast_quality_fallback_added = [fi for fi in extra if fi in post_ids]
                                fast_quality_fallback_reason = '|'.join(reasons_list)
                                # merge new diag
                                for k, v in reselection_diag2.items():
                                    reselection_diag.setdefault(k, {}).update(v)
                                print(f'[v15.3] fast quality fallback: '
                                      f'reasons={fast_quality_fallback_reason}, '
                                      f'decoded {len(decoded_extra)} extra frames, '
                                      f'winners changed: {sorted(pre_ids)} -> {sorted(post_ids)}')
                            else:
                                fast_quality_fallback_applied = False
                                fast_quality_fallback_reason = '|'.join(reasons_list) + '|no_change'
                                print(f'[v15.3] fast quality fallback: '
                                      f'reasons={"|".join(reasons_list)}, '
                                      f'decoded {len(decoded_extra)} extras but reselection unchanged')
                    finally:
                        cap_fb.release()

    if args.expected_pages > 0 and len(winners) > args.expected_pages:
        winners = sorted(winners, key=base_preference_score, reverse=True)[:args.expected_pages]
        winners = sorted(winners, key=lambda c: c.t_sec)

    # ----- v13.3: same-page alternative search (disabled by default) ---------
    # v13.2 enabled this for every video; on IMG_4883 it caused page identity
    # changes and duplicates. v13.3 only runs alt-search when explicitly
    # enabled via --enable-alt-search AND high_hand_mode is on.
    alt_diag_by_orig_frame: Dict[int, Dict[str, Any]] = {}
    alt_replacement_orig: Dict[int, int] = {}  # new_frame_idx -> original frame_idx
    alt_search_active = (
        bool(getattr(args, 'alt_search_enabled', False))
        and bool(getattr(args, '_high_hand_mode', False))
    )
    if alt_search_active and winners:
        with stage_timer('alt_search'):
            new_winners, alt_diag_by_orig_frame = search_alternatives_for_winners(winners, valid, args)
        replacements = 0
        for orig, new in zip(winners, new_winners):
            if orig.frame_idx != new.frame_idx:
                replacements += 1
                alt_replacement_orig[new.frame_idx] = orig.frame_idx
        winners = new_winners
        winners = sorted(winners, key=lambda c: c.t_sec)
        examined_total = sum(d.get('examined', 0) for d in alt_diag_by_orig_frame.values())
        checked_total = sum(d.get('checked', 0) for d in alt_diag_by_orig_frame.values())
        print(f'[v14.0] alt-search (high_hand_mode=on, opted-in): examined {examined_total} '
              f'same-window cands, {checked_total} same-page, {replacements} replacements applied')
    else:
        if not getattr(args, '_high_hand_mode', False):
            print('[v14.0] alt-search: skipped (high_hand_mode off — production-stable selection)')
        else:
            print('[v14.0] alt-search: skipped (not opted in via --enable-alt-search)')

    # ----- v14.2a: production-safe late auto-dedup (default mode only) ------
    # Conservative pass that drops a winner only when an adjacent / near-
    # adjacent winner is the same physical page with strong corroborated
    # evidence. No-op when --expected-pages is set or when disabled via
    # --no-auto-dedup-default.
    auto_dedup_diag: Dict[int, Dict[str, Any]] = {}
    if (
        winners
        and bool(getattr(args, 'auto_dedup_default', True))
        and int(getattr(args, 'expected_pages', 0) or 0) <= 0
    ):
        with stage_timer('auto_dedup_default'):
            pre_ids = [w.frame_idx for w in winners]
            winners = auto_dedup_default_winners(winners, args, auto_dedup_diag)
            post_ids = {w.frame_idx for w in winners}
            removed = [fid for fid in pre_ids if fid not in post_ids]
            if removed:
                print(f'[v14.2a] auto-dedup: removed winner frames {removed}')
                for fid in removed:
                    info = auto_dedup_diag.get(fid, {})
                    reason = info.get('auto_dedup_reason', '')
                    pair = info.get('auto_dedup_pair_frame', '?')
                    if reason:
                        print(f'[v14.2a]   frame {fid} <- pair {pair}: {reason}')
            else:
                print('[v14.2a] auto-dedup: no adjacent winners merged (conservative gate held)')
    else:
        if not bool(getattr(args, 'auto_dedup_default', True)):
            print('[v14.2a] auto-dedup: disabled via --no-auto-dedup-default')
        elif int(getattr(args, 'expected_pages', 0) or 0) > 0:
            print('[v14.2a] auto-dedup: skipped (--expected-pages set)')

    # ---- v15.4: bounded same-page quality refinement --------------------
    # For each winner flagged as suspicious by generic threshold criteria
    # (high hand metrics / large skew / etc.), look in a small temporal
    # window for an already-decoded same-physical-page candidate that is
    # cleaner. Replace only with a significant improvement margin. This
    # never fires on already-clean winners and is bounded by --quality-
    # refine-top-k candidates per suspicious winner.
    quality_refine_diag: Dict[int, Dict[str, Any]] = {}
    quality_refine_replacement_orig: Dict[int, int] = {}  # new -> original
    quality_refine_extra_decoded: List[int] = []
    if (
        winners
        and bool(getattr(args, 'quality_refinement', True))
    ):
        with stage_timer('quality_refinement'):
            # v15.4: when fast prefilter dropped frames around a suspicious
            # winner, optionally decode a small bounded set of those frames
            # so the refinement pass has a richer pool to choose from.
            try:
                refine_decode_max = int(getattr(args, 'quality_refine_extra_decode_max', 4) or 0)
            except Exception:
                refine_decode_max = 0
            if (
                refine_decode_max > 0
                and bool(getattr(args, 'quality_refinement_extra_decode', True))
                and fast_prefilter_on
                and candidate_set is not None
                and sampled_indices_pf
            ):
                refine_window = float(getattr(args, 'quality_refine_window_sec', 2.5))
                # Find suspicious winners up-front (cheap: just thresholds).
                susp_winners = [
                    w for w in winners
                    if _v154_winner_is_suspicious(w, args)[0]
                ]
                if susp_winners:
                    # Pick prefilter-sampled frames within window of any
                    # suspicious winner that were NOT already in candidate
                    # pool. Rank by prefilter composite (already in
                    # prefilter_metrics) so we decode the most promising
                    # ones first.
                    candidate_extras: Dict[int, float] = {}
                    valid_frames = {v.frame_idx for v in valid}
                    for sw in susp_winners:
                        for fi in sampled_indices_pf:
                            if fi in valid_frames or fi in candidate_set:
                                continue
                            t = fi / fps if fps > 0 else 0.0
                            if abs(t - sw.t_sec) > refine_window:
                                continue
                            m = prefilter_metrics.get(fi, {}) if prefilter_metrics else {}
                            composite = float(m.get('composite', 0.0))
                            # Tie-break preferring frames closer in time.
                            score = composite - 0.05 * abs(t - sw.t_sec)
                            if fi not in candidate_extras or score > candidate_extras[fi]:
                                candidate_extras[fi] = score
                    if candidate_extras:
                        # Bound the global decode count.
                        sorted_extras = sorted(
                            candidate_extras.items(),
                            key=lambda kv: kv[1],
                            reverse=True,
                        )[:refine_decode_max]
                        extras_targets = sorted([fi for fi, _ in sorted_extras])
                        if extras_targets:
                            cap_qr = cv2.VideoCapture(str(video_path))
                            if cap_qr.isOpened():
                                try:
                                    cur_pos = 0
                                    new_extras: List[FrameFeatures] = []
                                    for tgt in extras_targets:
                                        if tgt < cur_pos:
                                            cap_qr.set(cv2.CAP_PROP_POS_FRAMES, tgt)
                                            cur_pos = tgt
                                        while cur_pos < tgt:
                                            ok = cap_qr.grab()
                                            if not ok:
                                                break
                                            cur_pos += 1
                                        if cur_pos != tgt:
                                            continue
                                        ok = cap_qr.grab()
                                        if not ok:
                                            break
                                        ok2, frame = cap_qr.retrieve()
                                        cur_pos += 1
                                        if not ok2 or frame is None:
                                            continue
                                        try:
                                            _, feat, _ = _v152_process_candidate(
                                                tgt, frame, hand_masker
                                            )
                                        except Exception:
                                            continue
                                        if feat.page_found and feat.warped_bgr is not None:
                                            new_extras.append(feat)
                                            quality_refine_extra_decoded.append(tgt)
                                    if new_extras:
                                        features.extend(new_extras)
                                        valid.extend(new_extras)
                                        # Recompute raw/norm so newcomers are
                                        # ranked consistently with the existing
                                        # pool.
                                        for x in valid:
                                            x.raw_score = (
                                                1.40 * x.page_area_ratio +
                                                0.85 * x.fill_ratio +
                                                0.85 * x.border_contact_score +
                                                0.95 * x.stability_score +
                                                0.65 * (np.log1p(x.blur_score) / 8.0) +
                                                0.50 * x.text_score -
                                                0.85 * x.hand_penalty -
                                                2.40 * x.hand_text_overlap_penalty -
                                                1.55 * x.edge_foreground_penalty -
                                                0.75 * x.bottom_hand_penalty -
                                                0.70 * x.turn_penalty
                                            )
                                        raw_all = np.array([x.raw_score for x in valid], dtype=np.float32)
                                        norm_all = robust_norm(raw_all, higher_is_better=True)
                                        for i, x in enumerate(valid):
                                            x.norm_score = float(norm_all[i])
                                finally:
                                    cap_qr.release()
                            if quality_refine_extra_decoded:
                                print(f'[v15.4] quality refinement: '
                                      f'decoded {len(quality_refine_extra_decoded)} extra '
                                      f'frame(s) {quality_refine_extra_decoded} for '
                                      f'{len(susp_winners)} suspicious winner(s)')
            pre_ids = [w.frame_idx for w in winners]
            new_winners, quality_refine_diag = quality_refinement_pass(
                winners, valid, args
            )
            applied = 0
            for orig, new in zip(winners, new_winners):
                if orig.frame_idx != new.frame_idx:
                    quality_refine_replacement_orig[new.frame_idx] = orig.frame_idx
                    applied += 1
            winners = sorted(new_winners, key=lambda c: c.t_sec)
            n_susp = sum(
                1 for d in quality_refine_diag.values()
                if d.get('suspicious_reasons')
            )
            checked_total = sum(
                d.get('candidates_checked', 0)
                for d in quality_refine_diag.values()
            )
            examined_total = sum(
                d.get('candidates_examined', 0)
                for d in quality_refine_diag.values()
            )
            print(f'[v15.4] quality refinement: {n_susp} suspicious winner(s), '
                  f'{examined_total} candidates examined, {checked_total} same-page, '
                  f'{applied} replacement(s)')
            if applied:
                for orig_fid, d in quality_refine_diag.items():
                    if d.get('applied'):
                        print(f'[v15.4]   frame {orig_fid} -> {d.get("replacement_frame")}: '
                              f'{d.get("reason", "")}')
    else:
        if not bool(getattr(args, 'quality_refinement', True)):
            print('[v15.4] quality refinement: disabled via --no-quality-refinement')

    # ---- v15.8: within-region finger-relief refinement -----------------
    # Targets winners that retained a high finger / bottom-skin signal
    # after v15.4/v15.7 refinement. Bounded, additive, preserves the
    # v15.7 steal_zone guard so no cross-region replacement can happen.
    finger_relief_diag: Dict[int, Dict[str, Any]] = {}
    finger_relief_replacement_orig: Dict[int, int] = {}
    if (
        winners and valid
        and bool(getattr(args, 'finger_relief', True))
    ):
        with stage_timer('v158_finger_relief'):
            pre_ids = [w.frame_idx for w in winners]
            (winners_fr, finger_relief_diag,
             finger_relief_replacement_orig) = finger_relief_pass(
                winners, valid, args, quality_refine_replacement_orig,
            )
            applied = 0
            for orig, new in zip(winners, winners_fr):
                if orig.frame_idx != new.frame_idx:
                    applied += 1
            winners = sorted(winners_fr, key=lambda c: c.t_sec)
            n_finger = sum(
                1 for d in finger_relief_diag.values()
                if d.get('orig_finger', 0.0) >= float(
                    getattr(args, 'finger_relief_finger_floor', 0.30)
                ) or d.get('applied')
            )
            examined_total = sum(
                d.get('candidates_examined', 0)
                for d in finger_relief_diag.values()
            )
            checked_total = sum(
                d.get('candidates_checked', 0)
                for d in finger_relief_diag.values()
            )
            print(f'[v15.8] finger-relief: {n_finger} finger-suspicious '
                  f'winner(s), {examined_total} candidates examined, '
                  f'{checked_total} strict same-page, {applied} replacement(s)')
            if applied:
                for orig_fid, d in finger_relief_diag.items():
                    if d.get('applied'):
                        print(f'[v15.8]   frame {orig_fid} -> '
                              f'{d.get("replacement_frame")}: '
                              f'{d.get("reason", "")}')
    elif not bool(getattr(args, 'finger_relief', True)):
        print('[v15.8] finger-relief: disabled via --no-finger-relief')

    # ---- v15.5: adjacent-winner quality dedup with replacement -----------
    # After v15.4 refinement, walk adjacent winners and merge any pair that
    # depicts the same physical page; keep the cleaner one. Generic and
    # bounded — no hardcoded videos / pages / frames. Runs only in default
    # mode (no --expected-pages) so the user's expected count is honoured.
    v155_adj_dedup_diag: Dict[int, Dict[str, Any]] = {}
    v155_adj_dedup_summary: Dict[str, Any] = {'enabled': False, 'pairs_checked': 0, 'pairs_merged': 0, 'merges': []}
    if (
        winners
        and bool(getattr(args, 'v155_adjacent_dedup', True))
        and int(getattr(args, 'expected_pages', 0) or 0) <= 0
    ):
        with stage_timer('v155_adjacent_dedup'):
            pre_ids = [w.frame_idx for w in winners]
            winners, v155_adj_dedup_diag, v155_adj_dedup_summary = v155_adjacent_winner_dedup(winners, args)
            post_ids = {w.frame_idx for w in winners}
            removed = [fid for fid in pre_ids if fid not in post_ids]
            if removed:
                print(f'[v15.5] adjacent-dedup: removed winner frames {removed} '
                      f'(checked {v155_adj_dedup_summary["pairs_checked"]} pair(s), '
                      f'merged {v155_adj_dedup_summary["pairs_merged"]})')
                for m in v155_adj_dedup_summary['merges']:
                    print(f'[v15.5]   keep frame {m["keeper"]} over frame {m["loser"]}: '
                          f'{m["reason"]} (dt={m["metrics"]["dt"]:.2f}s, '
                          f'delta_q={m["delta"]:+.3f})')
            else:
                print(f'[v15.5] adjacent-dedup: no merges '
                      f'(checked {v155_adj_dedup_summary["pairs_checked"]} pair(s))')
            # v15.6: surface footer-guard blocks if any.
            blocks = v155_adj_dedup_summary.get('blocks', []) or []
            block_count = int(v155_adj_dedup_summary.get('footer_guard_blocks', 0))
            if blocks:
                print(f'[v15.6] footer-guard: blocked {block_count} pair(s) '
                      f'where bottom-band signature differed')
                for blk in blocks:
                    print(f'[v15.6]   block frames {blk["a"]} <-> {blk["b"]} '
                          f'(dt={blk["dt"]:.2f}s): {blk["reason"]}')
            elif bool(getattr(args, 'v156_footer_guard', True)):
                print(f'[v15.6] footer-guard: no blocks '
                      f'(guard active on {v155_adj_dedup_summary["pairs_checked"]} pair(s))')
    else:
        if not bool(getattr(args, 'v155_adjacent_dedup', True)):
            print('[v15.5] adjacent-dedup: disabled via --no-v155-adjacent-dedup')
        elif int(getattr(args, 'expected_pages', 0) or 0) > 0:
            print('[v15.5] adjacent-dedup: skipped (--expected-pages set)')

    # ---- v15.12 Patch C: blank front-matter coverage rescue --------------
    # Inter-winner gaps with strict "settled blank paper" prefilter
    # signature contain a real physical page that the prefilter correctly
    # filtered out. Add at most one synthetic winner per qualifying gap.
    # Default-on; opt out via --no-v1512-blank-rescue. Skipped when
    # --expected-pages is set (user authored the count).
    v1512_blank_rescue_diag: Dict[int, Dict[str, Any]] = {}
    v1512_blank_rescue_summary: Dict[str, Any] = {
        'enabled': bool(getattr(args, 'v1512_blank_rescue', True)),
        'rescued': 0,
        'attempts': [],
    }
    if (
        winners
        and bool(getattr(args, 'v1512_blank_rescue', True))
        and int(getattr(args, 'expected_pages', 0) or 0) <= 0
    ):
        with stage_timer('v1512_blank_rescue'):
            rescue_cands = _v1512_blank_front_matter_rescue_candidates(
                prefilter_metrics, sampled_indices_pf, winners, args,
            )
            for fi, t_sec, m in rescue_cands:
                attempt = {
                    'frame_idx': int(fi),
                    't_sec': float(t_sec),
                    'metrics': m,
                    'inserted': False,
                    'reason': '',
                }
                warped = _v1512_decode_and_warp_blank(video_path, fi, args)
                if warped is None or warped.size == 0:
                    attempt['reason'] = 'decode_or_warp_failed'
                    v1512_blank_rescue_summary['attempts'].append(attempt)
                    continue
                gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                blur_score = float(variance_of_laplacian(gray))
                text_score = float(count_text_density(gray))
                # Re-validate post-warp: text must remain very low (no
                # printed page leaked into the rescued candidate by a
                # mis-detected quad). Refuse otherwise.
                post_text_max = float(getattr(args, 'v1512_blank_rescue_post_text_max', 0.012))
                if text_score > post_text_max:
                    attempt['reason'] = (
                        f'post_warp_text_score={text_score:.4f}>'
                        f'{post_text_max:.4f}'
                    )
                    v1512_blank_rescue_summary['attempts'].append(attempt)
                    continue
                roi_g = roi_for_similarity(gray)
                roi_dh = compute_dhash(roi_g)
                feat = FrameFeatures(
                    frame_idx=int(fi),
                    t_sec=float(t_sec),
                    quad=None,
                    page_found=True,
                    page_area_ratio=float(m.get('paper_ratio', 0.0)),
                    fill_ratio=1.0,
                    border_contact_score=0.0,
                    stability_score=1.0,
                    blur_score=blur_score,
                    text_score=text_score,
                    hand_penalty=0.0,
                    hand_text_overlap_penalty=0.0,
                    edge_foreground_penalty=0.0,
                    bottom_hand_penalty=0.0,
                    turn_penalty=0.0,
                    edge_motion_penalty=0.0,
                    gray=gray,
                    roi_gray=roi_g,
                    roi_dhash=roi_dh,
                    warped_bgr=warped,
                    raw_score=0.0,
                    norm_score=0.0,
                    peak_score=0.0,
                    deskew_angle=0.0,
                )
                # Sentinel attribute consumed by the page-write loop and
                # the smart_trim guard. Rescued blanks must skip CLAHE,
                # deskew, hand cleanup, and aggressive refinement — those
                # steps assume a printed-text surface and produce garbage
                # on a near-uniform paper background.
                setattr(feat, '_v1512_blank_rescued', True)
                winners.append(feat)
                attempt['inserted'] = True
                attempt['reason'] = (
                    f'blur={blur_score:.0f},text={text_score:.4f},'
                    f'gap_dt={m.get("gap_dt", 0.0):.2f}s'
                )
                v1512_blank_rescue_summary['rescued'] += 1
                v1512_blank_rescue_diag[int(fi)] = {
                    'v1512_blank_rescue_inserted': 1,
                    'v1512_blank_rescue_reason': attempt['reason'],
                    'v1512_blank_rescue_metrics': m,
                }
                v1512_blank_rescue_summary['attempts'].append(attempt)
            # Maintain temporal order for downstream stages.
            winners = sorted(winners, key=lambda c: c.t_sec)
        if v1512_blank_rescue_summary['rescued'] > 0:
            print(f'[v15.12] blank-front-matter-rescue: '
                  f'inserted {v1512_blank_rescue_summary["rescued"]} '
                  f'blank page(s) in inter-winner gap(s)')
            for a in v1512_blank_rescue_summary['attempts']:
                if a['inserted']:
                    print(f'[v15.12]   frame {a["frame_idx"]} '
                          f'@t={a["t_sec"]:.2f}s: {a["reason"]}')
        else:
            attempts = v1512_blank_rescue_summary['attempts']
            if attempts:
                print(f'[v15.12] blank-front-matter-rescue: '
                      f'examined {len(attempts)} candidate(s), inserted 0 '
                      f'(post-warp gates rejected)')
            else:
                print('[v15.12] blank-front-matter-rescue: '
                      'no qualifying inter-winner gap')
    elif not bool(getattr(args, 'v1512_blank_rescue', True)):
        print('[v15.12] blank-front-matter-rescue: disabled via --no-v1512-blank-rescue')
    elif int(getattr(args, 'expected_pages', 0) or 0) > 0:
        print('[v15.12] blank-front-matter-rescue: skipped (--expected-pages set)')

    # v15.13 leading-edge distinct-page rescue. Look for a real distinct
    # physical page that was suppressed at select_local_peaks because it
    # was the very first sampled frame and lost a peak contest to its
    # immediate neighbour. Default-on; opt out via
    # --no-v1513-leading-edge-rescue. Skipped when --expected-pages set.
    v1513_leading_edge_summary: Dict[str, Any] = {
        'enabled': bool(getattr(args, 'v1513_leading_edge_rescue', True)),
        'rescued': 0,
        'frame_idx': None,
        'metrics': None,
    }
    if (
        winners
        and bool(getattr(args, 'v1513_leading_edge_rescue', True))
        and int(getattr(args, 'expected_pages', 0) or 0) <= 0
    ):
        with stage_timer('v1513_leading_edge_rescue'):
            cand = _v1513_leading_edge_distinct_page_rescue(
                features, winners, args,
            )
        if cand is not None:
            ham = int(getattr(cand, '_v1513_leading_edge_ham', -1))
            sim = float(getattr(cand, '_v1513_leading_edge_sim', -1.0))
            winners = sorted(winners + [cand], key=lambda c: c.t_sec)
            v1513_leading_edge_summary['rescued'] = 1
            v1513_leading_edge_summary['frame_idx'] = int(cand.frame_idx)
            v1513_leading_edge_summary['metrics'] = {
                't_sec': float(cand.t_sec),
                'blur': float(getattr(cand, 'blur_score', 0.0)),
                'text': float(getattr(cand, 'text_score', 0.0)),
                'area': float(getattr(cand, 'page_area_ratio', 0.0)),
                'fill': float(getattr(cand, 'fill_ratio', 0.0)),
                'edge_motion': float(getattr(cand, 'edge_motion_penalty', 0.0)),
                'turn': float(getattr(cand, 'turn_penalty', 0.0)),
                'ham': ham,
                'sim': sim,
            }
            print(f'[v15.13] leading-edge-rescue: prepended frame '
                  f'{int(cand.frame_idx)} @t={float(cand.t_sec):.2f}s '
                  f'(blur={float(getattr(cand, "blur_score", 0.0)):.0f},'
                  f'text={float(getattr(cand, "text_score", 0.0)):.4f},'
                  f'ham={ham},sim={sim:.2f})')
        else:
            print('[v15.13] leading-edge-rescue: no qualifying candidate '
                  'before first winner')
    elif not bool(getattr(args, 'v1513_leading_edge_rescue', True)):
        print('[v15.13] leading-edge-rescue: disabled via '
              '--no-v1513-leading-edge-rescue')
    elif int(getattr(args, 'expected_pages', 0) or 0) > 0:
        print('[v15.13] leading-edge-rescue: skipped (--expected-pages set)')

    # auto-smart post-selection sliver trim. Re-open the video and re-read the
    # winner frames only — winner selection already happened, so this cannot
    # change page count or ordering.
    # smart_trim_log entries: (frame_idx, label, info_dict, orig_h, orig_w)
    smart_trim_log: List[Tuple[int, str, dict, int, int]] = []
    if getattr(args, 'page_side', 'auto-smart') == 'auto-smart' and winners:
      with stage_timer('smart_trim'):
        # v15.1: reuse cached raw frames where possible to avoid the
        # POS_FRAMES seek + decode round-trip that smart_trim used in v15.0.
        # If cache miss (or prefilter disabled), fall back to a single
        # capture handle that we open lazily.
        cap2: Optional[cv2.VideoCapture] = None
        try:
            cache_hits = 0
            cache_misses = 0
            for cand in winners:
                if cand.quad is None or cand.warped_bgr is None:
                    continue
                orig_h, orig_w = cand.warped_bgr.shape[:2]
                frame = raw_frame_cache.get(cand.frame_idx)
                if frame is not None:
                    cache_hits += 1
                else:
                    if cap2 is None:
                        cap2 = cv2.VideoCapture(str(video_path))
                        if not cap2.isOpened():
                            cap2 = None
                            continue
                    cap2.set(cv2.CAP_PROP_POS_FRAMES, cand.frame_idx)
                    ok, frame = cap2.read()
                    if not ok or frame is None:
                        continue
                    cache_misses += 1
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
                    try:
                        new_warp = deskew_by_text_lines(new_warp)
                        cand.deskew_angle = float(_LAST_DESKEW_ANGLE.get('angle', 0.0))
                        new_warp = refine_page_after_warp(new_warp, args)
                    except Exception:
                        pass
                    cand.warped_bgr = new_warp
                smart_trim_log.append((cand.frame_idx, label, info, orig_h, orig_w))
            if getattr(args, 'profile', False) or getattr(args, 'debug', False):
                print(f'[v15.1] smart_trim cache: {cache_hits} hits, {cache_misses} misses (fallback decode)')
        finally:
            if cap2 is not None:
                cap2.release()

    final_dims_by_frame: dict = {}
    finalize_diag_by_frame: dict = {}
    _t_postprocess = time.perf_counter()
    for idx, cand in enumerate(winners, start=1):
        # v15.12 Patch C: rescued blank front-matter pages bypass the
        # full post-processing pipeline. Deskew/CLAHE/refine/hand-cleanup
        # are designed for printed text and either fail silently (no text
        # to anchor on) or produce visibly wrong output (CLAHE amplifies
        # paper noise into smeary high-contrast blobs). For a held blank,
        # the cleanest output is the warped frame as-is.
        if bool(getattr(cand, '_v1512_blank_rescued', False)):
            final_img = cand.warped_bgr
            fh, fw = final_img.shape[:2]
            final_dims_by_frame[cand.frame_idx] = (fh, fw)
            finalize_diag_by_frame[cand.frame_idx] = {
                'bottom_trim_px': 0,
                'cleanup_applied': False,
                'cleanup_mask_ratio': 0.0,
                'cleanup_reason': 'v1512_blank_rescue_skip',
                'enhance_mode': 'v1512_blank_rescue_skip',
                'deskew_angle_final': 0.0,
                'cons_cleanup_applied': False,
                'cons_cleanup_mask_ratio': 0.0,
                'cons_cleanup_reason': 'v1512_blank_rescue_skip',
            }
            cv2.imwrite(str(out_dir / f'page_{idx:03d}.jpg'), final_img,
                        [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
            continue
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
        # v13.3: conservative_bottom_hand_cleanup is DISABLED by default. On
        # IMG_4885 page 5 it was rendering half the page invisible. It can be
        # opted into via --experimental-hand-cleanup. Even when enabled,
        # default visual JPEGs are only modified when the user passes the
        # flag, so default output preserves the original (un-inpainted)
        # image.
        cons_info = {'applied': False, 'mask_ratio': 0.0, 'reason': 'disabled-by-default'}
        if (
            bool(getattr(args, 'experimental_hand_cleanup', False))
            and not cleanup_info.get('applied', False)
            and (float(getattr(cand, 'hand_text_overlap_penalty', 0.0)) >= 0.55
                 or float(getattr(cand, 'bottom_hand_penalty', 0.0)) >= 0.85)
        ):
            cleaned_cons, cons_info = conservative_bottom_hand_cleanup(final_img, args)
            if cons_info.get('applied'):
                final_img = cleaned_cons
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
            'cons_cleanup_applied': bool(cons_info.get('applied', False)),
            'cons_cleanup_mask_ratio': float(cons_info.get('mask_ratio', 0.0)),
            'cons_cleanup_reason': cons_info.get('reason', ''),
        }
        cv2.imwrite(str(out_dir / f'page_{idx:03d}.jpg'), final_img, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
    _dt_pp = time.perf_counter() - _t_postprocess
    if 'postprocess_save' not in _STAGE_TIMINGS:
        _STAGE_ORDER.append('postprocess_save')
    _STAGE_TIMINGS['postprocess_save'] = _STAGE_TIMINGS.get('postprocess_save', 0.0) + _dt_pp

    if args.debug:
        try:
            cal_path = dbg_dir / 'calibration.json'
            with open(cal_path, 'w', encoding='utf-8') as cf:
                json.dump(calibration, cf, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else (int(o) if isinstance(o, (np.integer,)) else str(o)))
        except Exception as e:
            print(f'[v13.0] Could not write calibration.json: {e}')
        if getattr(args, 'calibration_report', False):
            print('[v13.0] calibration overrides:', calibration.get('overrides', {}))
            print('[v13.0] calibration applied:', calibration.get('applied', {}))
            print('[v13.0] calibration post:', calibration.get('post', {}))
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
                # v13.1 extras --------------------------------------------------
                'cluster_select_score', 'deskew_penalty', 'hand_penalty_weighted',
                'high_hand_mode', 'duplicate_merge_reason', 'cluster_size',
                # v13.2 extras --------------------------------------------------
                'alt_candidates_checked', 'alt_candidates_examined',
                'alt_replacement_applied', 'alt_replacement_reason',
                'original_frame', 'replacement_frame',
                # v13.4 (IMG_4883) extras --------------------------------------
                'reselection_reason', 'duplicate_repaired',
                'hand_improvement', 'alt_similarity',
                'cons_cleanup_applied', 'cons_cleanup_mask_ratio', 'cons_cleanup_reason',
                # v13.5 (IMG_4883) clean-visual reselection diagnostics ---------
                'clean_visual_score', 'bg_gray_penalty', 'finger_penalty',
                'candidate_search_window',
                # v14.1b expected-pages count-fill diagnostics ------------------
                'expected_fill_applied', 'expected_fill_reason',
                'fill_source_gap', 'fill_distinctness_score',
                # v15.4 quality-refinement diagnostics ----------------------
                'quality_refinement_applied', 'quality_refinement_reason',
                'qr_original_frame', 'qr_replacement_frame',
                'qr_score_delta', 'qr_candidates_checked',
                'qr_candidates_examined', 'qr_suspicious_reasons',
                # v15.5 adjacent-winner dedup diagnostics --------------------
                'v155_adj_dedup_kept_over', 'v155_adj_dedup_reason',
                # v15.6 footer/folio distinctness guard diagnostics ----------
                'v156_footer_guard_applied', 'v156_footer_guard_blocked',
                'v156_footer_distinct_reason',
                'v156_footer_ink_a', 'v156_footer_ink_b',
                'v156_footer_col_corr', 'v156_footer_row_corr',
                'v156_footer_ham',
                # v15.8 finger-relief refinement diagnostics -----------------
                'v158_finger_relief_applied', 'v158_finger_relief_reason',
                'v158_finger_relief_orig_frame', 'v158_finger_relief_new_frame',
                'v158_finger_relief_orig_finger', 'v158_finger_relief_new_finger',
                'v158_finger_relief_orig_cvs', 'v158_finger_relief_new_cvs',
                'v158_finger_relief_finger_delta',
                'v158_finger_relief_overall_delta',
                'v158_finger_relief_candidates_checked',
                'v158_finger_relief_candidates_examined',
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
                ci = cluster_info_by_frame.get(x.frame_idx, {})
                css = ci.get('cluster_select_score', float('nan'))
                dpen = ci.get('deskew_penalty', float('nan'))
                hpen = ci.get('hand_penalty_weighted', float('nan'))
                hh = '1' if ci.get('high_hand_mode', False) else '0'
                merge_reason = ci.get('duplicate_merge_reason', '') or ''
                csz = int(ci.get('cluster_size', 1))
                css_s = '' if css != css else f'{css:.4f}'
                dpen_s = '' if dpen != dpen else f'{dpen:.4f}'
                hpen_s = '' if hpen != hpen else f'{hpen:.4f}'
                # v13.2: alt-search diagnostics. Lookup by *original* frame
                # so winners that were replaced still surface their search
                # results.
                orig_frame_for_diag = alt_replacement_orig.get(x.frame_idx, x.frame_idx)
                adiag = alt_diag_by_orig_frame.get(orig_frame_for_diag, {})
                alt_checked = int(adiag.get('checked', 0))
                alt_examined = int(adiag.get('examined', 0))
                alt_applied = '1' if adiag.get('replacement') is not None else '0'
                alt_reason = adiag.get('reason', '') or ''
                orig_frame_col = int(adiag.get('original_frame', x.frame_idx))
                rep_frame = adiag.get('replacement_frame', None)
                rep_frame_s = '' if rep_frame is None else str(int(rep_frame))
                hand_improve = float(adiag.get('hand_improvement', 0.0))
                alt_sim = float(adiag.get('similarity', 0.0))
                # v13.4 (IMG_4883) reselection diagnostics. The reselection_diag
                # dict is keyed by the *replacement* frame_idx so a winner
                # added via rescue_early_first_page or
                # repair_visual_duplicate_winners surfaces its reason here.
                rdiag = reselection_diag.get(int(x.frame_idx), {})
                resel_reason = rdiag.get('reselection_reason', '') or ''
                dup_repaired = '1' if rdiag.get('duplicate_repaired', 0) else '0'
                if rdiag.get('original_frame') is not None and not orig_frame_col:
                    orig_frame_col = int(rdiag.get('original_frame', x.frame_idx))
                # If alt-search did not change the original_frame but v13.4 did,
                # prefer the v13.4 swap info for the replacement_frame column.
                v134_orig = rdiag.get('original_frame', None)
                v134_rep = rdiag.get('replacement_frame', None)
                if v134_rep is not None and rep_frame is None:
                    orig_frame_col = int(v134_orig) if v134_orig is not None else orig_frame_col
                    rep_frame_s = str(int(v134_rep))
                cc_app = '1' if fdiag.get('cons_cleanup_applied', False) else '0'
                cc_mr = f"{float(fdiag.get('cons_cleanup_mask_ratio', 0.0)):.4f}"
                cc_rs = fdiag.get('cons_cleanup_reason', '') or ''
                # v13.5 clean-visual diagnostics. Compute on demand if the
                # selector did not (winners that passed the gate without swap
                # also get diagnostics so the column is always populated).
                cvs_val = rdiag.get('clean_visual_score')
                bg_val = rdiag.get('bg_gray_penalty')
                fg_val = rdiag.get('finger_penalty')
                csw_val = rdiag.get('candidate_search_window')
                if cvs_val is None and x.warped_bgr is not None:
                    _vm = _v135_visual_metrics_cached(x)
                    cvs_val = _v135_clean_visual_score(_vm)
                    bg_val = _v135_bg_gray_penalty(_vm)
                    fg_val = _v135_finger_penalty(_vm)
                cvs_s = '' if cvs_val is None else f'{float(cvs_val):.4f}'
                bg_s = '' if bg_val is None else f'{float(bg_val):.4f}'
                fg_s = '' if fg_val is None else f'{float(fg_val):.4f}'
                csw_s = '' if csw_val is None else f'{float(csw_val):.2f}'
                # v14.1b: expected-pages count-fill diagnostics
                ef_app = '1' if rdiag.get('expected_fill_applied', 0) else '0'
                ef_reason = rdiag.get('expected_fill_reason', '') or ''
                ef_gap = rdiag.get('fill_source_gap', '') or ''
                ef_dist_val = rdiag.get('fill_distinctness_score', None)
                ef_dist = '' if ef_dist_val is None else f'{float(ef_dist_val):.3f}'
                # v15.4 quality refinement diagnostics. Lookup by *original*
                # frame so winners that were replaced still surface their
                # refinement decision.
                qr_orig_for_diag = quality_refine_replacement_orig.get(
                    x.frame_idx, x.frame_idx
                )
                qrd = quality_refine_diag.get(qr_orig_for_diag, {})
                qr_applied = '1' if qrd.get('applied') else '0'
                qr_reason = qrd.get('reason', '') or ''
                qr_orig_f = int(qrd.get('original_frame', x.frame_idx))
                qr_rep_f = qrd.get('replacement_frame', None)
                qr_rep_f_s = '' if qr_rep_f is None else str(int(qr_rep_f))
                qr_delta_s = f"{float(qrd.get('score_delta', 0.0)):.4f}"
                qr_checked = int(qrd.get('candidates_checked', 0))
                qr_examined = int(qrd.get('candidates_examined', 0))
                qr_susp = qrd.get('suspicious_reasons', '') or ''
                # v15.5 adjacent-dedup diagnostics. v155_adj_dedup_diag is
                # keyed by frame_idx of removed loser AND of kept keeper,
                # so a kept winner that survived a merge surfaces its
                # 'kept over' source here.
                v155d = v155_adj_dedup_diag.get(x.frame_idx, {})
                v155_kept_over = v155d.get('v155_adj_dedup_kept_over', '')
                v155_reason = v155d.get('v155_adj_dedup_reason', '') or ''
                v155_kept_over_s = '' if v155_kept_over == '' else str(int(v155_kept_over))
                # v15.6 footer/folio guard diagnostics: piggy-back on
                # v155_adj_dedup_diag — the dedup loop populates v156_*
                # keys for every winner whose pair was tested by the guard.
                v156_applied = int(v155d.get('v156_footer_guard_applied', 0))
                v156_blocked = int(v155d.get('v156_footer_guard_blocked', 0))
                v156_reason = v155d.get('v156_footer_distinct_reason', '') or ''
                def _fmt_or_blank(v: Any, fmt: str) -> str:
                    try:
                        if v is None or v == '':
                            return ''
                        return format(float(v), fmt)
                    except Exception:
                        return ''
                v156_ink_a = _fmt_or_blank(v155d.get('v156_footer_ink_a', ''), '.4f')
                v156_ink_b = _fmt_or_blank(v155d.get('v156_footer_ink_b', ''), '.4f')
                v156_col_corr = _fmt_or_blank(v155d.get('v156_footer_col_corr', ''), '.3f')
                v156_row_corr = _fmt_or_blank(v155d.get('v156_footer_row_corr', ''), '.3f')
                v156_ham = _fmt_or_blank(v155d.get('v156_footer_ham', ''), '.0f')
                # v15.8: finger-relief diagnostics. Lookup by *original*
                # frame so winners replaced by finger-relief still surface
                # the decision (chain through QR replacement map first).
                fr_lookup_frame = finger_relief_replacement_orig.get(
                    x.frame_idx,
                    quality_refine_replacement_orig.get(x.frame_idx, x.frame_idx),
                )
                frd = finger_relief_diag.get(fr_lookup_frame, {}) \
                    if finger_relief_diag else {}
                fr_applied = '1' if frd.get('applied') else '0'
                fr_reason = frd.get('reason', '') or ''
                fr_orig_f = int(frd.get('original_frame', x.frame_idx)) \
                    if frd.get('original_frame') is not None else int(x.frame_idx)
                fr_rep_f = frd.get('replacement_frame', None)
                fr_rep_f_s = '' if fr_rep_f is None else str(int(fr_rep_f))
                fr_orig_finger = _fmt_or_blank(frd.get('orig_finger', ''), '.4f')
                fr_new_finger = _fmt_or_blank(frd.get('new_finger', ''), '.4f')
                fr_orig_cvs = _fmt_or_blank(frd.get('orig_cvs', ''), '.4f')
                fr_new_cvs = _fmt_or_blank(frd.get('new_cvs', ''), '.4f')
                fr_finger_delta = _fmt_or_blank(frd.get('finger_delta', ''), '.4f')
                fr_overall_delta = _fmt_or_blank(frd.get('overall_delta', ''), '.4f')
                fr_checked = int(frd.get('candidates_checked', 0))
                fr_examined = int(frd.get('candidates_examined', 0))
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
                    css_s, dpen_s, hpen_s, hh, merge_reason, csz,
                    alt_checked, alt_examined, alt_applied, alt_reason,
                    orig_frame_col, rep_frame_s,
                    resel_reason, dup_repaired,
                    f'{hand_improve:.4f}', f'{alt_sim:.3f}',
                    cc_app, cc_mr, cc_rs,
                    cvs_s, bg_s, fg_s, csw_s,
                    ef_app, ef_reason, ef_gap, ef_dist,
                    qr_applied, qr_reason,
                    qr_orig_f, qr_rep_f_s, qr_delta_s,
                    qr_checked, qr_examined, qr_susp,
                    v155_kept_over_s, v155_reason,
                    v156_applied, v156_blocked, v156_reason,
                    v156_ink_a, v156_ink_b,
                    v156_col_corr, v156_row_corr, v156_ham,
                    fr_applied, fr_reason,
                    fr_orig_f, fr_rep_f_s,
                    fr_orig_finger, fr_new_finger,
                    fr_orig_cvs, fr_new_cvs,
                    fr_finger_delta, fr_overall_delta,
                    fr_checked, fr_examined,
                    smart_label,
                ])
        # v15.0: write prefilter.csv in debug mode when prefilter ran.
        try:
            if prefilter_metrics:
                with open(dbg_dir / 'prefilter.csv', 'w', newline='', encoding='utf-8') as pf:
                    w = csv.writer(pf)
                    w.writerow(['frame_idx', 't_sec', 'paper_ratio', 'bright_mean',
                                'sat_mean', 'blur', 'edge_density', 'motion', 'skin',
                                'bottom_dark', 'composite', 'selected'])
                    for fi in sampled_indices_pf:
                        m = prefilter_metrics.get(fi, {})
                        w.writerow([
                            fi, f'{m.get("t_sec", 0.0):.3f}',
                            f'{m.get("paper_ratio", 0.0):.4f}', f'{m.get("bright_mean", 0.0):.2f}',
                            f'{m.get("sat_mean", 0.0):.2f}', f'{m.get("blur", 0.0):.2f}',
                            f'{m.get("edge_density", 0.0):.4f}', f'{m.get("motion", 0.0):.5f}',
                            f'{m.get("skin", 0.0):.5f}', f'{m.get("bottom_dark", 0.0):.4f}',
                            f'{m.get("composite", 0.0):.4f}', int(m.get('selected', 0.0)),
                        ])
        except Exception as e:
            print(f'[v15.0] Could not write prefilter.csv: {e}')
        # v14.0: write timings.json in debug mode so iteration cost is visible.
        try:
            timings_payload = {
                'video': str(video_path),
                'mode': 'debug',
                'audit_candidates': bool(getattr(args, 'audit_candidates', False)),
                'reselection_top_k': int(getattr(args, 'reselection_top_k', 6)),
                'max_alternatives_per_winner': int(getattr(args, 'max_alternatives_per_winner', 8)),
                'expected_pages': int(getattr(args, 'expected_pages', 0)),
                'sample_fps': float(getattr(args, 'sample_fps', 2.0)),
                'fast_prefilter': bool(getattr(args, 'fast_prefilter', True)),
                'prefilter_long_side': int(getattr(args, 'prefilter_long_side', 512)),
                'prefilter_top_k': int(getattr(args, 'prefilter_top_k', 0)),
                'prefilter_keep_ratio': float(getattr(args, 'prefilter_keep_ratio', 0.45)),
                'prefilter_neighborhood': int(getattr(args, 'prefilter_neighborhood', 1)),
                'prefilter_kept': len(candidate_set) if candidate_set is not None else 0,
                'prefilter_sampled_total': len(sampled_indices_pf),
                # v15.3: fast-prefilter quality guardrails diagnostics
                'prefilter_slot_retention': bool(getattr(args, 'prefilter_slot_retention', True)),
                'prefilter_slot_factor': int(getattr(args, 'prefilter_slot_factor', 2)),
                'prefilter_per_slot_top_k': int(getattr(args, 'prefilter_per_slot_top_k', 2)),
                'prefilter_peak_radius': int(getattr(args, 'prefilter_peak_radius', 2)),
                'prefilter_diagnostics': prefilter_diag,
                'fast_quality_fallback_enabled': bool(getattr(args, 'fast_quality_fallback', True)),
                'fast_quality_fallback_applied': bool(fast_quality_fallback_applied),
                'fast_quality_fallback_reason': str(fast_quality_fallback_reason),
                'fast_quality_fallback_added_frames': list(fast_quality_fallback_added),
                'full_calibration': bool(getattr(args, 'full_calibration', False)),
                'calibration_source': calibration.get('source', 'unknown'),
                'max_cached_frames': int(getattr(args, 'max_cached_frames', 80)),
                'raw_frame_cache_size': len(raw_frame_cache),
                # v15.4 quality refinement summary
                'quality_refinement_enabled': bool(getattr(args, 'quality_refinement', True)),
                'quality_refine_window_sec': float(getattr(args, 'quality_refine_window_sec', 2.5)),
                'quality_refine_top_k': int(getattr(args, 'quality_refine_top_k', 6)),
                'quality_refine_min_improvement': float(getattr(args, 'quality_refine_min_improvement', 0.12)),
                'quality_refinement_summary': {
                    'suspicious_count': sum(1 for d in quality_refine_diag.values()
                                            if d.get('suspicious_reasons')),
                    'applied_count': sum(1 for d in quality_refine_diag.values()
                                         if d.get('applied')),
                    'extra_decoded_frames': list(quality_refine_extra_decoded),
                    'replacements': [
                        {
                            'original_frame': d.get('original_frame'),
                            'replacement_frame': d.get('replacement_frame'),
                            'score_delta': d.get('score_delta'),
                            'reason': d.get('reason'),
                            'suspicious_reasons': d.get('suspicious_reasons'),
                        }
                        for d in quality_refine_diag.values() if d.get('applied')
                    ],
                },
                # v15.5 adjacent-winner dedup summary
                'v155_adjacent_dedup_enabled': bool(getattr(args, 'v155_adjacent_dedup', True)),
                'v155_adj_dedup_window_sec': float(getattr(args, 'v155_adj_dedup_window_sec', 3.0)),
                'v155_adjacent_dedup_summary': v155_adj_dedup_summary,
                # v15.6 footer/folio distinctness guard
                'v156_footer_guard_enabled': bool(getattr(args, 'v156_footer_guard', True)),
                'v156_footer_band_frac': float(getattr(args, 'v156_footer_band_frac', 0.09)),
                'v156_footer_center_frac': float(getattr(args, 'v156_footer_center_frac', 0.60)),
                'v156_footer_side_trim_frac': float(getattr(args, 'v156_footer_side_trim_frac', 0.08)),
                'v156_footer_col_corr_max': float(getattr(args, 'v156_footer_col_corr_max', 0.70)),
                'v156_footer_row_corr_max': float(getattr(args, 'v156_footer_row_corr_max', 0.70)),
                'v156_footer_ink_delta_min': float(getattr(args, 'v156_footer_ink_delta_min', 0.12)),
                'v156_footer_ham_min': int(getattr(args, 'v156_footer_ham_min', 14)),
                'v156_footer_min_ink': float(getattr(args, 'v156_footer_min_ink', 0.012)),
                'v156_footer_guard_blocks': int(v155_adj_dedup_summary.get('footer_guard_blocks', 0)),
                'v156_footer_guard_block_pairs': v155_adj_dedup_summary.get('blocks', []),
                # v15.8 finger-relief refinement summary
                'finger_relief_enabled': bool(getattr(args, 'finger_relief', True)),
                'finger_relief_finger_floor': float(getattr(args, 'finger_relief_finger_floor', 0.30)),
                'finger_relief_min_finger_improve': float(getattr(args, 'finger_relief_min_finger_improve', 0.20)),
                'finger_relief_max_overall_regress': float(getattr(args, 'finger_relief_max_overall_regress', -0.10)),
                'finger_relief_blur_frac': float(getattr(args, 'finger_relief_blur_frac', 0.30)),
                'finger_relief_summary': {
                    'finger_suspicious_count': sum(
                        1 for d in finger_relief_diag.values()
                        if float(d.get('orig_finger', 0.0))
                            >= float(getattr(args, 'finger_relief_finger_floor', 0.30))
                            or d.get('applied')
                    ),
                    'applied_count': sum(
                        1 for d in finger_relief_diag.values() if d.get('applied')
                    ),
                    'replacements': [
                        {
                            'original_frame': d.get('original_frame'),
                            'replacement_frame': d.get('replacement_frame'),
                            'orig_finger': d.get('orig_finger'),
                            'new_finger': d.get('new_finger'),
                            'orig_cvs': d.get('orig_cvs'),
                            'new_cvs': d.get('new_cvs'),
                            'finger_delta': d.get('finger_delta'),
                            'overall_delta': d.get('overall_delta'),
                            'reason': d.get('reason'),
                        }
                        for d in finger_relief_diag.values() if d.get('applied')
                    ],
                },
                'stages': _timings_dict(),
            }
            with open(dbg_dir / 'timings.json', 'w', encoding='utf-8') as tf:
                json.dump(timings_payload, tf, indent=2)
        except Exception as e:
            print(f'[v14.0] Could not write timings.json: {e}')
        print(f'Debug files: {dbg_dir}')

    hand_masker.close()
    print(f'Saved {len(winners)} unique pages to: {out_dir}')
    print(f'Valid warped candidates: {len(valid)}, peak winners before clustering: {len(winners_pre)}, clusters: {len(clusters)}')
    # v14.0: when --audit-candidates AND --debug-contact-sheets are set, write
    # candidate / winner contact sheets to the debug folder. This is the only
    # place in v14.0 that generates contact sheets; production runs never do.
    if (
        bool(getattr(args, 'debug_contact_sheets', False))
        and bool(getattr(args, 'audit_candidates', False))
        and args.debug
    ):
        try:
            _v140_write_winners_contact_sheet(winners, dbg_dir / 'winners_contact_sheet.jpg')
            print(f'[v14.0] Wrote winners contact sheet: {dbg_dir / "winners_contact_sheet.jpg"}')
        except Exception as e:
            print(f'[v14.0] Could not write contact sheet: {e}')


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
            '(V15.4: V15.3 fast prefilter + bounded same-page quality '
            'refinement for suspicious winners. Refinement looks for cleaner '
            'same-page alternatives only when a winner shows elevated hand '
            'metrics, large skew, or low clean_visual_score. Toggle with '
            '--no-quality-refinement to restore v15.3 behaviour.)'
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
    p.add_argument('--peak-window-sec', type=float, default=0.2)
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
    # V13.0: per-video unsupervised self-calibration (pre-pass).
    p.add_argument('--adaptive-calibration', dest='adaptive_calibration', action='store_true',
                   default=True,
                   help='V13.0: enable per-video unsupervised self-calibration pre-pass (default on).')
    p.add_argument('--no-adaptive-calibration', dest='adaptive_calibration', action='store_false',
                   help='V13.0: disable adaptive calibration; use v12.9 fixed thresholds.')
    p.add_argument('--calibration-sample-fps', type=float, default=1.0,
                   help='V13.0: target sampling fps for the calibration pre-pass. Default 1.0.')
    p.add_argument('--calibration-max-frames', type=int, default=60,
                   help='V13.0: hard cap on frames inspected during calibration. Default 60.')
    p.add_argument('--calibration-report', action='store_true',
                   help='V13.0: print a summary of the calibration overrides to stdout.')
    # V13.1 tunables ---------------------------------------------------------
    p.add_argument('--deskew-soft-threshold', type=float, default=8.0,
                   help='V13.1: |deskew| (deg) above which candidates start to be soft-penalised in cluster_select_score and peak_score.')
    p.add_argument('--cluster-deskew-weight', type=float, default=0.05,
                   help='V13.1: alpha weight on the deskew excess penalty inside cluster_select_score.')
    p.add_argument('--peak-deskew-weight', type=float, default=0.030,
                   help='V13.1: deskew excess penalty weight folded into peak_score during local-peak selection.')
    p.add_argument('--cluster-hand-text-weight', type=float, default=0.55,
                   help='V13.1: beta weight on hand_text_overlap_penalty inside cluster_select_score.')
    p.add_argument('--cluster-bottom-hand-weight', type=float, default=0.40,
                   help='V13.1: gamma weight on bottom_hand_penalty inside cluster_select_score.')
    p.add_argument('--cluster-hand-weight', type=float, default=0.40,
                   help='V13.1: delta weight on hand_penalty inside cluster_select_score.')
    p.add_argument('--cluster-score-eps', type=float, default=0.020,
                   help='V13.1: tie tolerance for cluster_select_score; below this the legacy choose_between_similar wins to keep stable videos unchanged.')
    p.add_argument('--sim-secondary-min', type=float, default=0.50,
                   help='V13.1: minimum structural similarity for the secondary cluster-merge heuristic to fire (perspective-warped duplicates).')
    p.add_argument('--hash-secondary-extra', type=int, default=8,
                   help='V13.1: extra hamming budget on top of --hash-thresh-merge for the secondary merge heuristic.')
    p.add_argument('--cluster-text-rel-tol', type=float, default=0.30,
                   help='V13.1: max relative text-density gap between two candidates for the secondary merge heuristic.')
    p.add_argument('--cluster-time-factor', type=float, default=1.5,
                   help='V13.1: temporal proximity factor (multiplied by --min-same-page-gap-sec) for the secondary merge heuristic.')
    p.add_argument('--cluster-text-rel-tight', type=float, default=0.25,
                   help='V13.1 high-hand-mode tertiary merge: max relative text-density gap when ssim/ham are too noisy to use.')
    p.add_argument('--cluster-time-tight-factor', type=float, default=1.25,
                   help='V13.1 high-hand-mode tertiary merge: temporal factor (x min_same_page_gap_sec) within which two candidates can be merged on text-density alone.')
    p.add_argument('--cluster-span-max-factor', type=float, default=1.6,
                   help='V13.1 high-hand-mode tertiary merge: maximum cluster temporal span (x min_same_page_gap_sec) to prevent transitive chaining of A->A\'->B.')
    p.add_argument('--force-high-hand', dest='_force_high_hand', action='store_true',
                   help='V13.1: force-enable high-hand mode regardless of calibration. Useful when you know the video has a finger in every frame.')
    # V13.3: same-page alternative-candidate search (DISABLED BY DEFAULT) -----
    # In v13.2 this fired on every video and shifted IMG_4883 winners. v13.3
    # turns it off by default and only honours it in high_hand_mode.
    # v13.4 (IMG_4883) verbose flag: emit per-pair sim/profile/warp diagnostics
    # for the duplicate-pair detector in fill_expected_pages_by_time. Off by
    # default so --debug runs do not spam the console.
    p.add_argument('--v134-verbose', dest='v134_verbose', action='store_true', default=False,
                   help='V13.4: print per-pair duplicate-detector diagnostics during expected-pages selection.')
    # v13.5 (IMG_4883) clean-visual reselection knobs.
    p.add_argument('--v135-clean-visual-weight', dest='v135_clean_visual_weight', type=float, default=0.55,
                   help='V13.5: blend weight of clean_visual_score vs final_quality when picking the cleanest equivalent within a winner identity (0..1). Default 0.55.')
    p.add_argument('--v135-clean-visual-min-gain', dest='v135_clean_visual_min_gain', type=float, default=0.06,
                   help='V13.5: minimum blended-score gain required to swap a winner with a cleaner equivalent. Default 0.06.')
    p.add_argument('--v135-pool-profile-min', dest='v135_pool_profile_min', type=float, default=0.20,
                   help='V13.5: minimum text-row profile correlation between a candidate and the current winner for the cleaner-equivalent pool, when sim/ham match is moderate. Lower = more permissive (risk of cross-page mixing).')
    p.add_argument('--v135-pool-warp-strong', dest='v135_pool_warp_strong', type=float, default=0.72,
                   help='V13.5: minimum warp-thumbnail match ratio that, on its own, accepts a candidate into the same-physical-page pool. Lower = more permissive.')
    p.add_argument('--v135-clean-visual-dominant-delta', dest='v135_clean_visual_dominant_delta', type=float, default=0.18,
                   help='V13.5: when a candidate beats the current winner by this much in clean_visual_score AND has materially lower finger_penalty, override the blended ranking. Default 0.18.')
    p.add_argument('--enable-alt-search', dest='alt_search_enabled', action='store_true',
                   default=False,
                   help='V13.3: enable v13.2 same-page alternative-candidate search. Only active when high_hand_mode is on. Default OFF.')
    p.add_argument('--alt-search-enabled', dest='alt_search_enabled', action='store_true',
                   help='V13.3: alias for --enable-alt-search.')
    p.add_argument('--no-alt-search', dest='alt_search_enabled', action='store_false',
                   help='V13.3: explicitly disable alt-search (default).')
    p.add_argument('--alt-window-sec', type=float, default=4.0,
                   help='V13.2: temporal window (seconds, +/- around winner) inside which same-page alternatives are searched.')
    p.add_argument('--alt-sim-min', type=float, default=0.55,
                   help='V13.2: SSIM threshold for the strict same-page test (combined with --alt-hash-max).')
    p.add_argument('--alt-hash-max', type=int, default=18,
                   help='V13.2: dHash hamming threshold for the strict same-page test.')
    p.add_argument('--alt-sim-relaxed', type=float, default=0.35,
                   help='V13.2: SSIM floor for the relaxed same-page test (used when text density also matches).')
    p.add_argument('--alt-hash-relaxed', type=int, default=28,
                   help='V13.2: dHash hamming threshold for the relaxed same-page test.')
    p.add_argument('--alt-text-rel-tol', type=float, default=0.25,
                   help='V13.2: max relative text-density gap for the relaxed same-page test.')
    p.add_argument('--alt-text-rel-tight', type=float, default=0.15,
                   help='V13.2: tighter text-density gap fallback (when ssim/ham are noisy due to occlusion/skew).')
    p.add_argument('--alt-deskew-max', type=float, default=5.0,
                   help='V13.2: |deskew| (deg) ceiling for an alternative; reject straighter swap targets that exceed it.')
    p.add_argument('--alt-blur-floor-frac', type=float, default=0.55,
                   help='V13.2: alternative blur_score must be at least this fraction of the original winner blur (or 100, whichever is higher).')
    p.add_argument('--alt-min-hand-improvement', type=float, default=0.18,
                   help='V13.2: minimum reduction in the composite hand score for a swap to be considered.')
    p.add_argument('--alt-raw-score-max-drop', type=float, default=1.5,
                   help='V13.2: alternative raw_score may not be lower than the original winner by more than this absolute margin.')
    p.add_argument('--alt-dirty-hand-text', type=float, default=0.55,
                   help='V13.2: hand_text_overlap_penalty above which the winner is considered dirty enough to trigger alt search.')
    p.add_argument('--alt-dirty-bottom-hand', type=float, default=0.85,
                   help='V13.2: bottom_hand_penalty above which the winner triggers alt search.')
    p.add_argument('--alt-dirty-hand', type=float, default=0.20,
                   help='V13.2: hand_penalty (mediapipe mask ratio) above which (combined with moderate hand_text_overlap) alt search is triggered.')
    # V13.3: experimental hand cleanup (DISABLED BY DEFAULT) ----------------
    # In v13.2 this damaged IMG_4885 page 5 (half invisible). v13.3 will
    # only run it when explicitly opted in via --experimental-hand-cleanup.
    p.add_argument('--experimental-hand-cleanup', dest='experimental_hand_cleanup',
                   action='store_true', default=False,
                   help='V13.3: opt-in v13.2 conservative bottom-band hand cleanup. OFF by default to preserve final pages.')
    p.add_argument('--aggressive-hand-cleanup-preview', dest='experimental_hand_cleanup',
                   action='store_true',
                   help='V13.3: alias for --experimental-hand-cleanup.')
    # Back-compat aliases (no-op in v13.3 default unless --experimental-hand-cleanup is set)
    p.add_argument('--alt-conservative-cleanup', dest='experimental_hand_cleanup',
                   action='store_true',
                   help='V13.3 deprecated alias for --experimental-hand-cleanup.')
    p.add_argument('--no-alt-conservative-cleanup', dest='experimental_hand_cleanup',
                   action='store_false',
                   help='V13.3: deprecated; cleanup is OFF by default. Equivalent to omitting --experimental-hand-cleanup.')
    p.add_argument('--alt-cleanup-band-frac', type=float, default=0.30,
                   help='V13.2: vertical fraction of the page (from bottom) where conservative cleanup may operate.')
    p.add_argument('--alt-cleanup-max-mask-frac', type=float, default=0.18,
                   help='V13.2: refuse conservative cleanup if the resulting mask covers more than this fraction of the page.')
    p.add_argument('--alt-cleanup-max-text-overlap', type=float, default=0.18,
                   help='V13.2: refuse conservative cleanup if the mask intersects too much of the upper-band text region.')
    # ---- v14.0: production / debug / audit mode separation ----------------
    p.add_argument('--profile', dest='profile', action='store_true', default=False,
                   help='V14.0: print stage timings at the end of the run.')
    p.add_argument('--timing-report', dest='profile', action='store_true',
                   help='V14.0: alias for --profile.')
    p.add_argument('--audit-candidates', dest='audit_candidates', action='store_true', default=False,
                   help='V14.0: enable expensive per-candidate audit / broad alternative search '
                        '(implies wider reselection pool, contact-sheet generation when '
                        '--debug-contact-sheets is also set). OFF by default for production speed.')
    p.add_argument('--debug-contact-sheets', dest='debug_contact_sheets', action='store_true', default=False,
                   help='V14.0: write candidate / winner contact-sheet JPEGs to the debug folder. '
                        'Implies --audit-candidates. OFF by default.')
    # ---- v14.1b: expected-pages count fill repair ------------------------
    p.add_argument('--v141b-fill-top-k', dest='v141b_fill_top_k', type=int, default=6,
                   help='V14.1b: per-gap candidate pool size for the expected-pages count-fill repair.')
    p.add_argument('--v141b-fill-sim-max', dest='v141b_fill_sim_max', type=float, default=0.70,
                   help='V14.1b: SSIM-style similarity ceiling above which a candidate is treated as a duplicate of an existing winner (combined with --v141b-fill-ham-min).')
    p.add_argument('--v141b-fill-ham-min', dest='v141b_fill_ham_min', type=int, default=18,
                   help='V14.1b: dHash hamming floor below which a candidate is treated as a duplicate of an existing winner (combined with --v141b-fill-sim-max).')
    p.add_argument('--v141b-fill-novelty-min', dest='v141b_fill_novelty_min', type=float, default=0.18,
                   help='V14.1b: minimum visual_novelty score required for a candidate to fill an expected-pages slot.')
    p.add_argument('--v141b-fill-min-gap-factor', dest='v141b_fill_min_gap_factor', type=float, default=0.60,
                   help='V14.1b: minimum gap size (relative to (t1-t0)/expected_pages) for the count-fill repair to consider a temporal gap.')
    p.add_argument('--v141a-dup-sim-floor', dest='v141a_dup_sim_floor', type=float, default=0.50,
                   help='V14.1a: sanity-gate floor for SSIM-like similarity. Below this, the relaxed duplicate test is overridden when hash distance is also high.')
    p.add_argument('--reselection-top-k', dest='reselection_top_k', type=int, default=6,
                   help='V14.0: bound the same-page reselection pool to the K visually-closest '
                        'candidates per winner in production mode. Smaller = faster, larger = '
                        'more thorough. Use --audit-candidates to disable the cap.')
    p.add_argument('--max-alternatives-per-winner', dest='max_alternatives_per_winner', type=int, default=8,
                   help='V14.0: bound the temporal-window candidate count for the optional '
                        'alt-search pass (when --enable-alt-search is set). Use 0 to disable the cap.')
    # v14.2a: production-safe late auto-dedup for default mode (when
    # --expected-pages is not provided). Compares only adjacent / near-
    # adjacent winners using existing relaxed same-page test plus a
    # corroborating signal (warp-thumb ratio or central-row profile corr).
    # Default ON; conservative thresholds — false merge is worse than dup.
    p.add_argument('--no-auto-dedup-default', dest='auto_dedup_default', action='store_false',
                   default=True,
                   help='V14.2a: disable conservative auto-dedup of adjacent winners in '
                        'default (no --expected-pages) mode. Default: enabled.')
    p.add_argument('--auto-dedup-neighbors', dest='auto_dedup_neighbors', type=int, default=2,
                   help='V14.2a: how many neighbors ahead to compare for auto-dedup. '
                        'Default 2; bounded for speed.')
    # ---- v15.0: production fast prefilter --------------------------------
    p.add_argument('--fast-prefilter', dest='fast_prefilter', action='store_true',
                   default=True,
                   help='V15.0: enable cheap downscaled prefilter that selects top candidate '
                        'frames for expensive detect/warp/score. Default ON.')
    p.add_argument('--no-fast-prefilter', dest='fast_prefilter', action='store_false',
                   help='V15.0: disable prefilter and process every sampled frame at full '
                        'resolution (legacy v14.2a behaviour).')
    p.add_argument('--prefilter-long-side', dest='prefilter_long_side', type=int, default=512,
                   help='V15.0: long side (px) for the cheap prefilter pass. Smaller is '
                        'faster but may miss subtle features. Default 512.')
    p.add_argument('--prefilter-top-k', dest='prefilter_top_k', type=int, default=0,
                   help='V15.0: hard cap on candidate frames after prefilter (0 = derived '
                        'from --prefilter-keep-ratio). Default 0.')
    p.add_argument('--prefilter-keep-ratio', dest='prefilter_keep_ratio', type=float, default=0.45,
                   help='V15.0: fraction of sampled frames kept after prefilter when '
                        '--prefilter-top-k is 0. Default 0.45.')
    p.add_argument('--prefilter-neighborhood', dest='prefilter_neighborhood', type=int, default=1,
                   help='V15.0: include +/- N neighbouring sampled frames around each '
                        'prefilter peak so peak detection has context. Default 1.')
    p.add_argument('--prefilter-min-keep', dest='prefilter_min_keep', type=int, default=12,
                   help='V15.0: minimum number of candidate frames kept after prefilter, '
                        'regardless of ratio (safety floor for short videos). Default 12.')
    # ---- v15.3: prefilter quality guardrails for expected-pages mode -----
    p.add_argument('--prefilter-slot-retention', dest='prefilter_slot_retention',
                   action='store_true', default=True,
                   help='V15.3: in --expected-pages mode, also keep top candidates '
                        'from each temporal slot (not only globally). Default ON. '
                        'Disable with --no-prefilter-slot-retention to reproduce v15.2.')
    p.add_argument('--no-prefilter-slot-retention', dest='prefilter_slot_retention',
                   action='store_false',
                   help='V15.3: disable per-slot retention.')
    p.add_argument('--prefilter-slot-factor', dest='prefilter_slot_factor', type=int, default=2,
                   help='V15.3: number of temporal slots = expected_pages * factor. '
                        'Default 2.')
    p.add_argument('--prefilter-per-slot-top-k', dest='prefilter_per_slot_top_k',
                   type=int, default=2,
                   help='V15.3: per-slot top-K candidates retained from prefilter. '
                        'Default 2.')
    p.add_argument('--prefilter-peak-radius', dest='prefilter_peak_radius', type=int, default=2,
                   help='V15.3: local-peak detection radius (in sampled-frame units). '
                        'Frames whose smoothed composite is a local maximum within +/-R are '
                        'retained even if below global top-K. Default 2.')
    # ---- v15.10: enable local-peak retention in default (no expected_pages) mode
    p.add_argument('--prefilter-default-local-peaks',
                   dest='prefilter_default_local_peaks',
                   type=lambda v: str(v).lower() not in ('0', 'false', 'no'),
                   default=True,
                   help='V15.10: enable local-peak retention even without '
                        '--expected-pages. Rescues mid-video peaks that lose '
                        'the global-top-K plateau tiebreak (e.g. IMG_4899 '
                        'pages at frames 90 and 285). Default ON. Pass '
                        '=false to revert to v15.9 behaviour.')
    p.add_argument('--no-prefilter-default-local-peaks',
                   dest='prefilter_default_local_peaks',
                   action='store_false',
                   help='V15.10: disable default-mode local-peak retention '
                        '(restores v15.9 behaviour).')
    p.add_argument('--fast-quality-fallback', dest='fast_quality_fallback',
                   action='store_true', default=True,
                   help='V15.3: when --expected-pages is set, if final winners are missing '
                        'or duplicated, decode extra frames from uncovered intervals and '
                        're-run selection. Default ON.')
    p.add_argument('--no-fast-quality-fallback', dest='fast_quality_fallback',
                   action='store_false',
                   help='V15.3: disable the quality fallback for fast mode.')
    p.add_argument('--fast-fallback-max-extra', dest='fast_fallback_max_extra',
                   type=int, default=24,
                   help='V15.3: max extra sampled frames decoded by the quality fallback. '
                        'Default 24.')
    # ---- v15.1: prefilter-driven calibration + frame cache ---------------
    p.add_argument('--full-calibration', dest='full_calibration', action='store_true',
                   default=False,
                   help='V15.1: force the legacy full calibration pass (separate decode) '
                        'even when fast prefilter is enabled. Default OFF — calibration '
                        'reuses prefilter low-res metrics.')
    p.add_argument('--no-prefilter-calibration', dest='no_prefilter_calibration',
                   action='store_true', default=False,
                   help='V15.1: same as --full-calibration; explicitly disable the '
                        'prefilter-derived calibration shortcut.')
    p.add_argument('--max-cached-frames', dest='max_cached_frames', type=int, default=80,
                   help='V15.1: max number of raw candidate BGR frames cached in memory '
                        'for smart_trim reuse (bounded so long videos do not OOM). '
                        'Default 80.')
    # ---- v15.2: parallel candidate processing ----------------------------
    p.add_argument('--parallel-candidates', dest='parallel_candidates', action='store_true',
                   default=True,
                   help='V15.2: parallelize per-candidate detect/warp/deskew/inpaint/dhash '
                        'using a worker pool. Effective only with --fast-prefilter. '
                        'Default ON.')
    p.add_argument('--no-parallel-candidates', dest='parallel_candidates', action='store_false',
                   help='V15.2: disable parallel candidate processing — fall back to '
                        'sequential v15.1 behaviour.')
    p.add_argument('--candidate-workers', dest='candidate_workers', type=int, default=0,
                   help='V15.2: number of worker threads for parallel candidate '
                        'processing (0 = auto: min(os.cpu_count(), 4)). Set to 1 for '
                        'fully sequential deterministic execution.')
    # ---- v15.4: bounded same-page quality refinement ---------------------
    p.add_argument('--quality-refinement', dest='quality_refinement',
                   action='store_true', default=True,
                   help='V15.4: enable bounded same-page quality refinement '
                        'pass for suspicious winners (high hand metrics, large '
                        'skew, etc.). Default ON.')
    p.add_argument('--no-quality-refinement', dest='quality_refinement',
                   action='store_false',
                   help='V15.4: disable the quality refinement pass.')
    p.add_argument('--quality-refine-window-sec', dest='quality_refine_window_sec',
                   type=float, default=2.5,
                   help='V15.4: temporal window (+/- seconds) around each '
                        'suspicious winner inside which same-page alternatives '
                        'are considered. Default 2.5.')
    p.add_argument('--quality-refine-top-k', dest='quality_refine_top_k',
                   type=int, default=6,
                   help='V15.4: maximum same-window candidates evaluated per '
                        'suspicious winner. Default 6.')
    p.add_argument('--quality-refine-min-improvement',
                   dest='quality_refine_min_improvement',
                   type=float, default=0.12,
                   help='V15.4: minimum refinement-score gain required for a '
                        'replacement to apply. Default 0.12.')
    p.add_argument('--quality-refine-hand-thresh',
                   dest='quality_refine_hand_thresh',
                   type=float, default=0.40,
                   help='V15.4: hand_penalty above which a winner is '
                        'considered suspicious. Default 0.40.')
    p.add_argument('--quality-refine-skew-thresh',
                   dest='quality_refine_skew_thresh',
                   type=float, default=4.0,
                   help='V15.4: |deskew_angle| (deg) above which a winner is '
                        'considered suspicious. Default 4.0.')
    p.add_argument('--quality-refine-bottom-hand-thresh',
                   dest='quality_refine_bottom_hand_thresh',
                   type=float, default=0.55,
                   help='V15.4: bottom_hand_penalty above which a winner is '
                        'considered suspicious. Default 0.55.')
    p.add_argument('--quality-refine-hand-text-thresh',
                   dest='quality_refine_hand_text_thresh',
                   type=float, default=0.35,
                   help='V15.4: hand_text_overlap_penalty above which a winner '
                        'is considered suspicious. Default 0.35.')
    p.add_argument('--quality-refine-turn-thresh',
                   dest='quality_refine_turn_thresh',
                   type=float, default=0.55,
                   help='V15.4: turn_penalty above which a winner is '
                        'considered suspicious. Default 0.55.')
    p.add_argument('--quality-refine-cvs-low',
                   dest='quality_refine_cvs_low',
                   type=float, default=-0.10,
                   help='V15.4: clean_visual_score floor (combined with high '
                        'finger_penalty) below which a winner is considered '
                        'suspicious. Default -0.10.')
    p.add_argument('--quality-refine-stability-min',
                   dest='quality_refine_stability_min',
                   type=float, default=0.55,
                   help='V15.4: minimum stability_score required for a '
                        'candidate to be eligible. Rejects transition / '
                        'turning frames that would otherwise pass the '
                        'same-page gate. Default 0.55.')
    p.add_argument('--quality-refine-edge-motion-max',
                   dest='quality_refine_edge_motion_max',
                   type=float, default=0.55,
                   help='V15.4: maximum edge_motion_penalty allowed on a '
                        'candidate. Rejects high-motion / turning frames. '
                        'Default 0.55.')
    p.add_argument('--quality-refine-raw-score-max-drop',
                   dest='quality_refine_raw_score_max_drop',
                   type=float, default=1.5,
                   help='V15.4: candidate raw_score may not be lower than '
                        'the original winner by more than this margin. '
                        'Default 1.5.')
    p.add_argument('--quality-refine-finger-regress-max',
                   dest='quality_refine_finger_regress_max',
                   type=float, default=0.08,
                   help='V15.4: maximum allowed increase in finger_penalty '
                        'for a refinement replacement. Protects clean '
                        'winners on videos where every frame has high '
                        'MediaPipe-derived hand metrics. Default 0.08.')
    p.add_argument('--quality-refine-deskew-max',
                   dest='quality_refine_deskew_max',
                   type=float, default=7.5,
                   help='V15.4: |deskew_angle| ceiling for a candidate to be '
                        'eligible as a replacement. Refinement intentionally '
                        'uses a more permissive ceiling than alt-search so a '
                        'mildly skewed winner can be replaced by a less '
                        'skewed (but still imperfect) same-page candidate. '
                        'Default 7.5.')
    p.add_argument('--quality-refine-temporal-buddy-sec',
                   dest='quality_refine_temporal_buddy_sec',
                   type=float, default=1.2,
                   help='V15.4: when ROI similarity is destroyed by hand '
                        'occlusion, accept a candidate as same-page if it is '
                        'within this many seconds of the winner and shares '
                        'similar text density. Default 1.2.')
    p.add_argument('--quality-refinement-extra-decode',
                   dest='quality_refinement_extra_decode',
                   action='store_true', default=False,
                   help='V15.4: when fast prefilter dropped frames around a '
                        'suspicious winner, decode a small bounded set of '
                        'them so the refinement pass has more options. '
                        'OFF by default to preserve runtime; opt-in for '
                        'quality-priority runs.')
    p.add_argument('--no-quality-refinement-extra-decode',
                   dest='quality_refinement_extra_decode',
                   action='store_false',
                   help='V15.4: disable the bounded selective decode '
                        'inside the quality refinement pass (default).')
    p.add_argument('--quality-refine-extra-decode-max',
                   dest='quality_refine_extra_decode_max',
                   type=int, default=2,
                   help='V15.4: hard cap on extra frames decoded during the '
                        'quality refinement pass (across all suspicious '
                        'winners) when extra-decode is enabled. Default 2.')
    # v15.8: within-region finger-relief refinement ---------------------------
    p.add_argument('--finger-relief', dest='finger_relief',
                   action='store_true', default=True,
                   help='V15.8 (default ON): after v15.4/v15.7 quality '
                        'refinement, run a within-region pass that can '
                        'replace a winner with a same-page candidate that '
                        'has materially lower finger / bottom-skin '
                        'penalty. Preserves the v15.7 steal_zone guard, '
                        'so cross-region replacement is impossible. '
                        'Page count and coverage are not changed.')
    p.add_argument('--no-finger-relief', dest='finger_relief',
                   action='store_false',
                   help='V15.8: disable the within-region finger-relief '
                        'refinement pass.')
    p.add_argument('--finger-relief-finger-floor',
                   dest='finger_relief_finger_floor',
                   type=float, default=0.30,
                   help='V15.8: minimum winner finger_penalty for the '
                        'finger-relief pass to consider the winner '
                        'finger-suspicious. Default 0.30.')
    p.add_argument('--finger-relief-min-finger-improve',
                   dest='finger_relief_min_finger_improve',
                   type=float, default=0.20,
                   help='V15.8: required reduction in finger_penalty '
                        '(orig - new) for a candidate to be eligible. '
                        'Default 0.20.')
    p.add_argument('--finger-relief-max-overall-regress',
                   dest='finger_relief_max_overall_regress',
                   type=float, default=-0.10,
                   help='V15.8: maximum allowed regression in the v15.4 '
                        'overall refine-score for a finger-relief swap. '
                        'A NEGATIVE number; the candidate\'s overall '
                        'score may be at most this much lower than the '
                        'original winner. Default -0.10.')
    p.add_argument('--finger-relief-blur-frac',
                   dest='finger_relief_blur_frac',
                   type=float, default=0.30,
                   help='V15.8: blur floor as a fraction of the original '
                        'winner blur for finger-relief candidates. '
                        'Loosened from the v15.4 default of 0.55 because '
                        'a slightly motion-blurred clean-bottom frame '
                        'can be visibly preferable to a finger-occluded '
                        'sharper one. Absolute floor of 100 also applies. '
                        'Default 0.30.')
    p.add_argument('--finger-relief-raw-score-max-drop',
                   dest='finger_relief_raw_score_max_drop',
                   type=float, default=2.5,
                   help='V15.8: max allowed drop in raw_score from winner '
                        'to candidate. Loosened from the v15.4 default of '
                        '1.5. Default 2.5.')
    p.add_argument('--finger-relief-other-winner-sim',
                   dest='finger_relief_other_winner_sim',
                   type=float, default=0.85,
                   help='V15.8: SSIM-like similarity threshold above which '
                        'a candidate is rejected as visually identical to '
                        'another existing winner. Prevents the subsequent '
                        'v15.5 adjacent-dedup from collapsing two regions. '
                        'Default 0.85.')
    p.add_argument('--finger-relief-other-winner-ham',
                   dest='finger_relief_other_winner_ham',
                   type=int, default=8,
                   help='V15.8: dHash hamming-distance threshold below '
                        'which a candidate is rejected as visually '
                        'identical to another existing winner. Default 8.')
    p.add_argument('--finger-relief-other-winner-warp',
                   dest='finger_relief_other_winner_warp',
                   type=float, default=0.78,
                   help='V15.8: warp-thumb correlation threshold above '
                        'which a candidate is rejected as visually '
                        'matching another existing winner (anticipates '
                        'the v15.5 adjacent-dedup warp-thumb merge path). '
                        'Default 0.78.')
    # v15.5: adjacent-winner quality dedup ----------------------------------
    p.add_argument('--v155-adjacent-dedup', dest='v155_adjacent_dedup',
                   action='store_true', default=True,
                   help='V15.5 (default ON): after quality refinement, walk '
                        'adjacent winners and merge any pair that depicts the '
                        'same physical page; keep the cleaner one. Skipped '
                        'when --expected-pages is set.')
    p.add_argument('--no-v155-adjacent-dedup', dest='v155_adjacent_dedup',
                   action='store_false',
                   help='V15.5: disable the adjacent-winner quality dedup pass.')
    p.add_argument('--v155-adj-dedup-window-sec', dest='v155_adj_dedup_window_sec',
                   type=float, default=3.0,
                   help='V15.5: temporal window for adjacent-winner dedup '
                        'pair tests (default 3.0s). Pairs farther apart in '
                        'time are not considered same-page candidates.')
    p.add_argument('--v155-adj-dedup-sim-min', dest='v155_adj_dedup_sim_min',
                   type=float, default=0.62,
                   help='V15.5: SSIM-like floor for the primary same-page '
                        'signal in adjacent-winner dedup (default 0.62).')
    p.add_argument('--v155-adj-dedup-ham-max', dest='v155_adj_dedup_ham_max',
                   type=int, default=22,
                   help='V15.5: dHash distance ceiling for the primary '
                        'same-page signal in adjacent-winner dedup '
                        '(default 22).')
    p.add_argument('--v155-adj-dedup-warp-min', dest='v155_adj_dedup_warp_min',
                   type=float, default=0.78,
                   help='V15.5: warp-thumbnail correlation floor for the '
                        'corroborating signal in adjacent-winner dedup '
                        '(default 0.78).')
    p.add_argument('--v155-adj-dedup-profile-min', dest='v155_adj_dedup_profile_min',
                   type=float, default=0.65,
                   help='V15.5: central-row profile correlation floor for '
                        'the corroborating signal in adjacent-winner dedup '
                        '(default 0.65).')
    p.add_argument('--v155-adj-dedup-text-rel-max', dest='v155_adj_dedup_text_rel_max',
                   type=float, default=0.30,
                   help='V15.5: hard floor on relative text-density '
                        'agreement for adjacent-winner dedup (default 0.30). '
                        'Pairs with disagreeing text density never merge.')
    p.add_argument('--v155-adj-dedup-rescan-dt-sec', dest='v155_adj_dedup_rescan_dt_sec',
                   type=float, default=1.5,
                   help='V15.5: temporal-rescan path threshold (default 1.5s). '
                        'Adjacent winners closer than this are treated as '
                        'same-page when text density agrees and at least '
                        'moderate warp/profile agreement is present.')
    p.add_argument('--v155-adj-dedup-rescan-text-rel-max', dest='v155_adj_dedup_rescan_text_rel_max',
                   type=float, default=0.25,
                   help='V15.5: tighter text-density floor for the rescan '
                        'path (default 0.25). Stricter than the global '
                        'text_rel_max because rescan triggers without a '
                        'strong sim/ham signal.')
    p.add_argument('--v155-adj-dedup-rescan-warp-floor', dest='v155_adj_dedup_rescan_warp_floor',
                   type=float, default=0.50,
                   help='V15.5: minimum warp-thumb ratio for the rescan '
                        'path (default 0.50, i.e. better than chance).')
    p.add_argument('--v155-adj-dedup-rescan-profile-floor', dest='v155_adj_dedup_rescan_profile_floor',
                   type=float, default=0.10,
                   help='V15.5: minimum text-profile correlation for the '
                        'rescan path (default 0.10).')
    p.add_argument('--v155-adj-dedup-rescan-sim-floor', dest='v155_adj_dedup_rescan_sim_floor',
                   type=float, default=0.05,
                   help='V15.5: minimum SSIM-like score for the rescan '
                        'path (default 0.05).')
    p.add_argument('--v155-adj-dedup-rescan-turn-max', dest='v155_adj_dedup_rescan_turn_max',
                   type=float, default=0.65,
                   help='V15.5: maximum turn_penalty either winner may have '
                        'for the rescan path (default 0.65). Pairs with a '
                        'page turn in progress never merge.')
    # v15.11 Path D: motion-blur-asymmetry rescue ------------------------------
    p.add_argument('--v155-adj-dedup-blur-asym-dt-sec', dest='v155_adj_dedup_blur_asym_dt_sec',
                   type=float, default=None,
                   help='V15.11 Path D: max time gap (s) for the blur-'
                        'asymmetry duplicate rescue. Default = '
                        'min_peak_distance_sec * 1.5.')
    p.add_argument('--v155-adj-dedup-blur-asym-max-ratio', dest='v155_adj_dedup_blur_asym_max_ratio',
                   type=float, default=0.25,
                   help='V15.11 Path D: max min/max blur ratio that triggers '
                        'the rescue (a stricter ratio means a stronger '
                        'asymmetry is required to call same-page). Set to 0 '
                        'to disable.')
    p.add_argument('--v155-adj-dedup-blur-asym-turn-max', dest='v155_adj_dedup_blur_asym_turn_max',
                   type=float, default=0.65,
                   help='V15.11 Path D: maximum turn_penalty either winner '
                        'may have for the blur-asymmetry path (default 0.65).')
    p.add_argument('--v155-adj-dedup-blur-asym-warp-floor', dest='v155_adj_dedup_blur_asym_warp_floor',
                   type=float, default=0.40,
                   help='V15.11 Path D: trivial-agreement floor on warp ratio '
                        '(default 0.40). Path D fires when ANY of warp/prof/'
                        'sim meets its floor.')
    p.add_argument('--v155-adj-dedup-blur-asym-profile-floor', dest='v155_adj_dedup_blur_asym_profile_floor',
                   type=float, default=0.05,
                   help='V15.11 Path D: trivial-agreement floor on text '
                        'profile correlation (default 0.05).')
    p.add_argument('--v155-adj-dedup-blur-asym-sim-floor', dest='v155_adj_dedup_blur_asym_sim_floor',
                   type=float, default=0.03,
                   help='V15.11 Path D: trivial-agreement floor on SSIM-like '
                        'similarity (default 0.03).')
    # v15.11 Patch B: peak-picker blur-aware tiebreak --------------------------
    p.add_argument('--peak-tie-eps', dest='peak_tie_eps',
                   type=float, default=0.01,
                   help='V15.11 peak-picker blur tiebreak: peak_score margin '
                        'within which a sharper neighbour overrides a blurry '
                        'local maximum. Set to 0 to disable.')
    p.add_argument('--peak-tie-blur-min', dest='peak_tie_blur_min',
                   type=float, default=200.0,
                   help='V15.11 peak-picker blur tiebreak: a tiebreak only '
                        'applies if the sharper neighbour\'s blur_score >= '
                        'this absolute floor (default 200).')
    p.add_argument('--peak-tie-blur-factor', dest='peak_tie_blur_factor',
                   type=float, default=2.0,
                   help='V15.11 peak-picker blur tiebreak: sharper neighbour '
                        'must have blur_score > factor * x.blur_score '
                        '(default 2.0).')
    # v15.12 Patch A: severity-aware Path D max-turn ceiling ------------------
    p.add_argument('--v1512-blur-asym-severe-ratio',
                   dest='v1512_blur_asym_severe_ratio',
                   type=float, default=0.15,
                   help='V15.12 Patch A: blur_ratio at-or-below which the '
                        'Path D max-turn ceiling is raised to '
                        '--v1512-blur-asym-severe-turn-max (default 0.15). '
                        'Set to 0 to disable severity tiering.')
    p.add_argument('--v1512-blur-asym-severe-dt-sec',
                   dest='v1512_blur_asym_severe_dt_sec',
                   type=float, default=1.0,
                   help='V15.12 Patch A: dt at-or-below which the severe '
                        'tier max-turn ceiling applies (default 1.0).')
    p.add_argument('--v1512-blur-asym-severe-turn-max',
                   dest='v1512_blur_asym_severe_turn_max',
                   type=float, default=0.90,
                   help='V15.12 Patch A: max-turn ceiling for the severe '
                        'tier (default 0.90).')
    p.add_argument('--v1512-blur-asym-moderate-ratio',
                   dest='v1512_blur_asym_moderate_ratio',
                   type=float, default=0.25,
                   help='V15.12 Patch A: blur_ratio at-or-below which the '
                        'moderate-tier max-turn ceiling applies (default '
                        '0.25). Same as --v155-adj-dedup-blur-asym-max-ratio.')
    p.add_argument('--v1512-blur-asym-moderate-dt-sec',
                   dest='v1512_blur_asym_moderate_dt_sec',
                   type=float, default=1.5,
                   help='V15.12 Patch A: dt at-or-below which the moderate '
                        'tier max-turn ceiling applies (default 1.5).')
    p.add_argument('--v1512-blur-asym-moderate-turn-max',
                   dest='v1512_blur_asym_moderate_turn_max',
                   type=float, default=0.75,
                   help='V15.12 Patch A: max-turn ceiling for the moderate '
                        'tier (default 0.75).')
    p.add_argument('--v1512-blur-asym-footer-ink-delta-max',
                   dest='v1512_blur_asym_footer_ink_delta_max',
                   type=float, default=0.05,
                   help='V15.12 Patch A (footer bypass): max abs(ink_a-'
                        'ink_b) tolerated to override v15.6 footer guard '
                        'when blur asymmetry is severe (default 0.05). '
                        'Footer col/row correlation collapses on a '
                        'motion-smeared rendering, but ink density agrees '
                        'when the footer is genuinely the same.')
    p.add_argument('--v1512-blur-asym-footer-ham-max',
                   dest='v1512_blur_asym_footer_ham_max',
                   type=float, default=16.0,
                   help='V15.12 Patch A (footer bypass): max footer-band '
                        'dHash hamming tolerated alongside ink agreement '
                        '(default 16).')
    # v15.12 Patch B: explicit sharpness/skew dedup tiebreaker ----------------
    p.add_argument('--v1512-sharp-tiebreak-ratio',
                   dest='v1512_sharp_tiebreak_ratio',
                   type=float, default=4.0,
                   help='V15.12 Patch B: blur ratio above which an explicit '
                        'sharpness bonus is applied to the sharper member '
                        'of a merged adjacent pair (default 4.0). Set to 0 '
                        'to disable.')
    p.add_argument('--v1512-sharp-tiebreak-skew-max',
                   dest='v1512_sharp_tiebreak_skew_max',
                   type=float, default=1.5,
                   help='V15.12 Patch B: maximum |deskew_angle| (deg) the '
                        'sharper side may have to receive the bonus '
                        '(default 1.5).')
    p.add_argument('--v1512-sharp-tiebreak-turn-max',
                   dest='v1512_sharp_tiebreak_turn_max',
                   type=float, default=0.90,
                   help='V15.12 Patch B: maximum turn_penalty the sharper '
                        'side may have to receive the bonus (default 0.90).')
    p.add_argument('--v1512-sharp-tiebreak-bonus',
                   dest='v1512_sharp_tiebreak_bonus',
                   type=float, default=0.25,
                   help='V15.12 Patch B: explicit sharpness bonus added to '
                        'the sharper member of a merged adjacent pair '
                        '(default 0.25). Set to 0 to disable.')
    # v15.12 Patch C: blank front-matter coverage rescue ----------------------
    p.add_argument('--v1512-blank-rescue', dest='v1512_blank_rescue',
                   action='store_true', default=True,
                   help='V15.12 Patch C: after dedup, rescue at most one '
                        'sharpest "settled blank paper" frame per inter-'
                        'winner gap (default ON). Skipped when '
                        '--expected-pages is set. Universal: no hardcoded '
                        'frames or page counts.')
    p.add_argument('--no-v1512-blank-rescue', dest='v1512_blank_rescue',
                   action='store_false',
                   help='V15.12 Patch C: disable the blank front-matter '
                        'rescue.')
    p.add_argument('--v1512-blank-rescue-min-gap-sec',
                   dest='v1512_blank_rescue_min_gap_sec',
                   type=float, default=1.5,
                   help='V15.12 Patch C: minimum inter-winner gap (s) to '
                        'consider for a rescue (default 1.5).')
    p.add_argument('--v1512-blank-rescue-paper-min',
                   dest='v1512_blank_rescue_paper_min',
                   type=float, default=0.75,
                   help='V15.12 Patch C: min prefilter paper_ratio (default '
                        '0.75).')
    p.add_argument('--v1512-blank-rescue-motion-max',
                   dest='v1512_blank_rescue_motion_max',
                   type=float, default=0.05,
                   help='V15.12 Patch C: max prefilter motion (default 0.05).')
    p.add_argument('--v1512-blank-rescue-blur-min',
                   dest='v1512_blank_rescue_blur_min',
                   type=float, default=80.0,
                   help='V15.12 Patch C: min prefilter Laplacian blur '
                        '(default 80).')
    p.add_argument('--v1512-blank-rescue-bottom-dark-max',
                   dest='v1512_blank_rescue_bottom_dark_max',
                   type=float, default=0.60,
                   help='V15.12 Patch C: max prefilter bottom_dark ratio '
                        '(default 0.60).')
    p.add_argument('--v1512-blank-rescue-edge-max',
                   dest='v1512_blank_rescue_edge_max',
                   type=float, default=0.10,
                   help='V15.12 Patch C: max prefilter edge_density (default '
                        '0.10).')
    p.add_argument('--v1512-blank-rescue-skin-max',
                   dest='v1512_blank_rescue_skin_max',
                   type=float, default=0.40,
                   help='V15.12 Patch C: max prefilter skin ratio (default '
                        '0.40). Held blanks frequently show the operator '
                        'fingertips at the page edge in the cheap thumbnail; '
                        'the post-warp text gate is the real safety check.')
    p.add_argument('--v1512-blank-rescue-bright-min',
                   dest='v1512_blank_rescue_bright_min',
                   type=float, default=120.0,
                   help='V15.12 Patch C: min prefilter bright_mean (default '
                        '120).')
    p.add_argument('--v1512-blank-rescue-post-text-max',
                   dest='v1512_blank_rescue_post_text_max',
                   type=float, default=0.012,
                   help='V15.12 Patch C: post-warp/deskew text_score above '
                        'which a rescued blank is rejected (default 0.012). '
                        'Refuses to add a frame that turned out to contain '
                        'printed text after warp.')
    # v15.13 leading-edge distinct-page rescue --------------------------------
    p.add_argument('--v1513-leading-edge-rescue',
                   dest='v1513_leading_edge_rescue',
                   action='store_true', default=True,
                   help='V15.13: when the gap before the first kept winner '
                        'is >= --v1513-leading-edge-min-gap-sec, rescue the '
                        'sharpest visually-distinct page_found candidate from '
                        'that gap (default ON). Skipped when --expected-pages '
                        'is set. Universal: no hardcoded frames or page '
                        'counts.')
    p.add_argument('--no-v1513-leading-edge-rescue',
                   dest='v1513_leading_edge_rescue',
                   action='store_false',
                   help='V15.13: disable the leading-edge distinct-page '
                        'rescue.')
    p.add_argument('--v1513-leading-edge-min-gap-sec',
                   dest='v1513_leading_edge_min_gap_sec',
                   type=float, default=1.5,
                   help='V15.13: minimum t_sec of the first kept winner to '
                        'consider a leading-edge rescue (default 1.5).')
    p.add_argument('--v1513-leading-edge-ham-min',
                   dest='v1513_leading_edge_ham_min', type=int, default=22,
                   help='V15.13: minimum dHash hamming distance between the '
                        'rescue candidate and the first kept winner (default '
                        '22). Either ham >= this OR sim <= --v1513-leading-'
                        'edge-sim-max passes the distinctness gate.')
    p.add_argument('--v1513-leading-edge-sim-max',
                   dest='v1513_leading_edge_sim_max', type=float, default=0.55,
                   help='V15.13: maximum roi structural similarity between '
                        'the rescue candidate and the first kept winner '
                        '(default 0.55).')
    p.add_argument('--v1513-leading-edge-area-min',
                   dest='v1513_leading_edge_area_min', type=float, default=0.55,
                   help='V15.13: minimum page contour area ratio for the '
                        'leading-edge rescue candidate (default 0.55).')
    p.add_argument('--v1513-leading-edge-fill-min',
                   dest='v1513_leading_edge_fill_min', type=float, default=0.85,
                   help='V15.13: minimum page contour fill ratio for the '
                        'leading-edge rescue candidate (default 0.85).')
    p.add_argument('--v1513-leading-edge-edge-motion-max',
                   dest='v1513_leading_edge_edge_motion_max', type=float,
                   default=0.5,
                   help='V15.13: maximum edge_motion penalty for the '
                        'leading-edge rescue candidate (default 0.5). Blocks '
                        'page-turn frames.')
    p.add_argument('--v1513-leading-edge-turn-max',
                   dest='v1513_leading_edge_turn_max', type=float, default=0.5,
                   help='V15.13: maximum turn_penalty for the leading-edge '
                        'rescue candidate (default 0.5). Blocks page-turn '
                        'frames.')
    p.add_argument('--v1513-leading-edge-blur-min',
                   dest='v1513_leading_edge_blur_min', type=float, default=50.0,
                   help='V15.13: minimum Laplacian blur variance for the '
                        'leading-edge rescue candidate (default 50).')
    p.add_argument('--v1513-leading-edge-text-min',
                   dest='v1513_leading_edge_text_min', type=float, default=0.005,
                   help='V15.13: minimum text_score for the leading-edge '
                        'rescue candidate (default 0.005). Below this, the '
                        'frame is essentially blank and the v15.12 blank '
                        'rescue path should handle it.')
    # v15.6 footer / folio distinctness guard ---------------------------------
    p.add_argument('--v156-footer-guard', dest='v156_footer_guard',
                   action='store_true', default=True,
                   help='V15.6: enable bottom/folio distinctness guard for '
                        'adjacent-winner dedup (default ON). Blocks non-strict '
                        'same-page merges when bottom-band signatures differ.')
    p.add_argument('--no-v156-footer-guard', dest='v156_footer_guard',
                   action='store_false',
                   help='V15.6: disable the bottom/folio distinctness guard.')
    p.add_argument('--v156-footer-band-frac', dest='v156_footer_band_frac',
                   type=float, default=0.09,
                   help='V15.6: fraction of warped page height covered by the '
                        'bottom-center folio band (default 0.09).')
    p.add_argument('--v156-footer-center-frac', dest='v156_footer_center_frac',
                   type=float, default=0.60,
                   help='V15.6: fraction of warped page width covered by the '
                        'bottom-center folio band (default 0.60).')
    p.add_argument('--v156-footer-side-trim-frac', dest='v156_footer_side_trim_frac',
                   type=float, default=0.08,
                   help='V15.6: extra horizontal trim on each side of the '
                        'folio band as a fraction of band width (default 0.08).')
    p.add_argument('--v156-footer-col-corr-max', dest='v156_footer_col_corr_max',
                   type=float, default=0.70,
                   help='V15.6: column-profile shifted-correlation must be '
                        'BELOW this for the column signal to count as '
                        'distinct (default 0.70).')
    p.add_argument('--v156-footer-row-corr-max', dest='v156_footer_row_corr_max',
                   type=float, default=0.70,
                   help='V15.6: row-profile shifted-correlation must be '
                        'BELOW this for the row signal to count as '
                        'distinct (default 0.70).')
    p.add_argument('--v156-footer-ink-delta-min', dest='v156_footer_ink_delta_min',
                   type=float, default=0.12,
                   help='V15.6: minimum absolute difference in band ink '
                        'ratio for the ink-delta signal to count as '
                        'distinct (default 0.12).')
    p.add_argument('--v156-footer-ham-min', dest='v156_footer_ham_min',
                   type=int, default=14,
                   help='V15.6: minimum 64-bit dHash hamming over the '
                        'folio thumbnail for the dHash signal to count as '
                        'distinct (default 14).')
    p.add_argument('--v156-footer-min-ink', dest='v156_footer_min_ink',
                   type=float, default=0.012,
                   help='V15.6: minimum ink ratio in BOTH band signatures; '
                        'below this the band is considered too blank to '
                        'separate, and the guard yields to upstream dedup '
                        '(default 0.012).')
    p.add_argument('--v156-footer-ink-ratio-min', dest='v156_footer_ink_ratio_min',
                   type=float, default=0.35,
                   help='V15.6: minimum ratio of smaller-to-larger ink mass '
                        'between the two bands. Below this the band '
                        'alignment is considered lopsided and the guard '
                        'falls back to upstream dedup (default 0.35).')
    p.add_argument('--v156-footer-col-corr-strict', dest='v156_footer_col_corr_strict',
                   type=float, default=0.40,
                   help='V15.6: column-profile correlation strict threshold '
                        '(default 0.40). Required for the column-only path.')
    p.add_argument('--v156-footer-ham-strict', dest='v156_footer_ham_strict',
                   type=int, default=22,
                   help='V15.6: dHash hamming strict threshold (default 22) '
                        'used in the (col_corr<max AND ham>=strict) path.')
    # v15.6 default-mode gap fill --------------------------------------------
    p.add_argument('--v156-default-gap-fill', dest='v156_default_gap_fill',
                   action='store_true', default=True,
                   help='V15.6: enable default-mode (no --expected-pages) '
                        'gap-fill that decodes extra frames in temporal '
                        'gaps between winners and inserts the best one if '
                        'it is a true different page (default ON).')
    p.add_argument('--no-v156-default-gap-fill', dest='v156_default_gap_fill',
                   action='store_false',
                   help='V15.6: disable the default-mode gap-fill.')
    p.add_argument('--v156-default-gap-min-sec', dest='v156_default_gap_min_sec',
                   type=float, default=4.0,
                   help='V15.6: minimum gap (seconds) between consecutive '
                        'winners to trigger the default-mode gap-fill '
                        '(default 4.0).')
    p.add_argument('--v156-default-gap-head-mul', dest='v156_default_gap_head_mul',
                   type=float, default=1.8,
                   help='V15.6: head/tail gap multiplier vs median gap '
                        '(default 1.8).')
    p.add_argument('--v156-default-gap-mid-mul', dest='v156_default_gap_mid_mul',
                   type=float, default=1.8,
                   help='V15.6: middle gap multiplier vs median gap '
                        '(default 1.8).')
    return p


def main():
    args = build_parser().parse_args()
    # v14.0: --debug-contact-sheets implies --audit-candidates so the broader
    # candidate set is actually generated.
    if getattr(args, 'debug_contact_sheets', False):
        args.audit_candidates = True
    _reset_timings()
    t_main = time.perf_counter()
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
    finally:
        # Always record total runtime; print only when --profile was requested.
        total = time.perf_counter() - t_main
        if 'total' not in _STAGE_TIMINGS:
            _STAGE_ORDER.append('total')
        _STAGE_TIMINGS['total'] = total
        if getattr(args, 'profile', False):
            print(_format_timings_report())


if __name__ == '__main__':
    main()
