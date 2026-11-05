# Agent instructions: reproducible scientific work

You are a local coding/execution Agent for a **research experiment**, not an autonomous scientific author. Prioritize rigor and reproducibility over finding a high score.

## Scope
- Work only inside this newly-created repository, plus local dataset/output locations explicitly supplied by the human user.
- For EXP-001, only investigate datasets, run ≤2 epoch smoke tests, and prepare proposals; do not train full baselines or implement three modules until an explicit GO from the researcher.
- Keep all dataset and checkpoint folders out of Git and outside the public GitHub repository.
- Confirm before any destructive, external, billing, security-sensitive or publicly visible change. A Git commit/PR may be requested, but inspect staged files and secrets before pushing.

## Mandatory safeguards
- Never change original dataset files or ground truth. Use versioned conversion scripts writing to a separate location.
- No leakage: group adjacent/near-duplicate images and preserve group boundaries between splits. Show duplicate detection method (pHash, file hashes, acquisition metadata when present).
- Do not train on or optimize against frozen target-domain test metrics. Do not decide hyperparameters using target test results.
- Do not delete failed runs; label failures and include logs. Do not invent metrics, papers or output files.
- Do not include secrets, `.env`, tokens, personal credentials, dataset ZIPs, raw images, `*.pt`, or large binary outputs in public commits.
- Every run must have an immutable experiment ID, git commit, seed, exact config, package versions, GPU information, data manifest hash and timestamps.
- Seed and deterministic flags must be recorded; nondeterminism must be disclosed.
- Prevent parallel GPU training unless the operator explicitly approves. On Windows start `workers=0`, use guarded Python main, AMP only after smoke checks.
- Use locked datasets and per-split manifests; never modify a test split after seeing performance without declaring a new protocol.

## Gates
- `PASS`: all deliverables verifiable.
- `CONDITIONAL`: work correct but a known bounded issue remains and is documented.
- `BLOCKED`: dataset, permission, dependency or scientific comparability blocker.
- `FAILED`: attempted task could not be reliably completed.

At task completion, create `reports/EXP-NNN/report.md` and machine-readable metrics/artifacts when applicable, then stop. Never claim an experiment succeeded solely because the process exited zero; validate outputs and logs.
