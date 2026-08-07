# lab-02 rerun of the v1 matching method

These two files are a later lab-02 run of the **superseded v1** matching method
(`cross_arch_matching.py`) against the fully-trained 50k-step checkpoints, rather
than the partial ones the original v1 run used. Their checkpoint paths point at
`/Users/lab-02/...`.

They are preserved here only so the run is not lost. **Do not cite them.** The v1
method has three defects documented in `CORRECTIONS.md` §1 — many-to-one matching,
unclosed triangles, and an uncalibrated threshold that sits below the permutation
noise floor. Running it on better checkpoints does not fix any of those.

The current result is `data/sae-analysis/matching-v2/`, produced by
`cross_arch_matching_v2.py` on the same fully-trained checkpoints.
