# Sources

Where this skill's facts come from and how much of each source was read. Checked means: Full, Partial, Abstract only, or Installed (read from the installed package). Per-paper links and venues are in `papers.md`; per-repo commits are at the end of `jax-stack.md` and `pytorch-gpu-sim.md`.

| Source | Date checked | Checked | Supports |
|---|---|---|---|
| The papers in `papers.md` (arXiv record, proceedings page or DOI) | 2026-10-27 | Record page; body as cited in each reference file | Links, venues, one-line takeaways |
| 22 repos (PureJaxRL, purejaxql, Craftax, Kinetix, Brax / Playground, Isaac Lab, rsl_rl, mjlab, FastTD3, holosoma, FlashSAC, SAPG, PufferLib and others), shallow clones of the default branch | 2026-10-27 | Partial: the files cited as file:line | Shipped defaults and implementation facts in `jax-stack.md` and `pytorch-gpu-sim.md` |
| Zheng et al. 2026, arXiv 2610.33420 (plain-text extraction) | 2026-11-24 | Full for every number the skill uses | Every exponent, per-10x factor and validation number in `scaling-laws.md` and the SKILL.md defaults table (citation audit, `AUDIT.md`) |
| A 206-paper RL corpus (plain-text extractions of arXiv PDFs: curricula, UED, resets, continual and multi-task RL, parallel robot RL) | 2026-11-24 | Triage of every paper (abstract, intro, conclusion); deep reads of the papers cited in the reference files | The v1.1.0 additions: `reset-and-task-sampling.md`, staggered resets, CPO, plasticity and determinism rows, 20 `papers.md` rows. Every added number was grepped against its paper, then independently reviewed |
| gymnax 0.0.9 and 1.0.1 | 2026-11-24 | Installed, via `inspect` and by running resets | The version split in `jax-stack.md` (step tuple, `is_truncated`, time limit in `done`), MinAtar start-state counts |
| jax 0.11.2 and 0.12.0.1, flax 0.12.12, optax 0.2.8, distrax 0.1.9 | 2026-11-24 | Installed; test suite run on each pairing | The test-command and the compatibility line |
