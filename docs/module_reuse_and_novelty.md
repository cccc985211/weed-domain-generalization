# Existing modules, academic integrity, and proposed novelty

**Working policy:** All modules are hypotheses until implemented, validated and compared. Using a published idea is normal only if the related paper is cited, code license and attribution requirements are respected, and the paper is transparent about what is borrowed versus new.

## Prior art that must be compared

- Lee et al. (2024), Object-Aware Domain Generalization for Object Detection, AAAI, DOI 10.1609/aaai.v38i4.28076. OA-Mix + OA-Loss. https://ojs.aaai.org/index.php/AAAI/article/view/28076
- Zdraveska et al. (2026), A Multi-Dataset Benchmark of YOLO-Based Weed Detection in Precision Agriculture, arXiv:2609.33991. Multiple cross-dataset weed benchmarks and a unified one-class protocol. https://arxiv.org/abs/2609.33991
- Alfalfa-Yolo (2026), A Boundary-Aware Lightweight Framework for Real-Time Weed Detection in Alfalfa Fields, Frontiers in Plant Science. Strict cross-dataset transfer to CottonWeedDet12 and ImageWeeds is reported. https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2026.1955970/full

## Conservative 3-component proposal

A / OPBR: prior-art-inspired object-preserving background perturbation with a distinct margin/tapering treatment for sparse weed boxes. Treat as adapted existing technique unless an independently justified departure is established. Compare to OA-Mix where technically feasible.

B / PSAS: prior-art-inspired progress-controlled augmentation scheduler; test carefully against fixed strength with matched integrated augmentation intensity. Do not claim merely scheduling is novel.

C / SOFC: potential primary method contribution: scale-aware per-weed-region feature consistency on identical-geometry dual views. This is a hypothesis only: search literature for feature-ROI consistency and scale weighting before any novelty claim. A/B only become contributions if they contain substantive novel, measurable modifications.

## Academic integrity requirements

- Do not copy paper paragraphs, figures, tables or experimental results; cite the source and redraw original diagrams as needed.
- For reused code, capture upstream repository URL, commit, license, modified file list, and attribution/NOTICE requirements. Do not assume GitHub public code is freely relicensable.
- Report the exact contribution boundary in the method section: 'based on X / modified Y / newly proposed Z'.
- Avoid presenting the number of modules as evidence of novelty. If method C does not outperform matched controls, revisit the research claim.
- Verify whether a benchmark and competing method permit fair comparison; never imply external numbers are directly comparable when protocols differ.

## Evaluation rules

- First establish label compatibility and source-only model-selection. Target domain metrics must remain frozen until the method is fixed.
- At least 8 factorial ablations for A/B/C if feasible, with single-module baselines, strong standard augmentation comparison, and matched computational-budget controls for two-view loss.
- Report full source and target metrics and honest failure cases, GPU time and multiple seeds for shortlisted candidates.
