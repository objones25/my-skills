# Reading guide and critique

The orchestrator writes these paper-wide pieces in `globals.json` (SKILL.md, step 7); chunk workers read the Soundness audit to know what `critique_notes` and `audit` entries need.

## Contents
- Reading route
- Soundness audit: the domain critique
- Domain critique checklists

## Reading route

The guide page opens with a `route`: 3-6 steps saying what to read first and how
deeply, using the paper's own sections. Classify the paper (empirical system,
method, measurement, SoK/survey, theory, retrospective, article) and point the
reader at the real contribution. Give a Keshav-style depth per step ("pass 1:
abstract + intro", "pass 2: section 6, the method").

## Soundness audit: the domain critique

A paper is an argument to weigh. Choose the `domain` (security, ml, rl, ir,
systems, general, article) and apply its checklist (below, under Domain critique checklists), recording at
least four items as `{check, verdict, where, note}` plus the basis field the
verdict requires:

| Verdict | Use it when | Required field |
|---|---|---|
| `ok` | The paper does this; you can point to it. | `evidence`: what in the paper satisfies the check, with page. |
| `concern` | A weakness that limits how far a claim reaches, **including evidence the paper omits for a claim it makes.** | `claim`: the paper's claim whose support this weakens, with page. |
| `flaw` | The evidence cannot support (or contradicts) a central claim. | `claim`, as above. |
| `cant-tell` | The evidence may exist but you could not see it (a supplementary appendix you could not open, an artifact that is offline). | `missing`: what you could not see and where it is. |
| `not-applicable` | The check does not fit this kind of paper. | `why_na`. |

**The mapping rule.** If the paper makes a claim and does not report the evidence
that claim needs (no baseline, no variance, single run, no ablation), the verdict
is `concern`, or `flaw` when the claim is central. It is not `cant-tell`: you
*can* tell, and what you can tell is that the support is missing. The build
rejects a `cant-tell` whose note says the paper omits something, and an `ok`
whose note reports a problem.

> Wrong: `{"check": "Fuzzing rigor: multiple trials, statistical test?", "verdict":
> "cant-tell", "note": "No trials, no variance, single-run counts."}`
>
> Right: `{"verdict": "concern", "claim": "p.6: 'made Driller highly successful in
> finding bugs'", "note": "The effectiveness claim rests on single competition
> counts with no trials or variance; the Driller paper (ref 2) has the real
> evaluation."}`

A `flaw` or `concern` should also appear as a `subtle` note in the gutter where it
bites.

Separately, every figure, table, equation, and algorithm gets an `audit` entry
recording the concrete check you ran: for an equation, instantiate the smallest
case and confirm shapes and units; for a figure or table, confirm the number the
text cites and that the comparison is fair. A `flagged` audit needs a `subtle`
note. Run anything computable in Python and give the reader a hand-checkable
instance.

## Domain critique checklists

Pick the `domain` that fits the paper and apply its items (and the general ones).
Each item is phrased as what to check; record a verdict and what the paper does.

### article (journalism, blog post, essay, vendor write-up)
- Is the author in a position to know (participant, primary-source reporter,
  vendor selling the fix)? Flag commercial bias.
- Are factual claims tied to primary documents (court filings, advisories,
  postmortems, papers), and are those linked?
- Are numbers attributed to a source, and are they the source's own or repeated
  from elsewhere?
- Date and currency: has the story changed since (later postmortems, corrections)?
- Is reporting separated from speculation; are anonymous sources or hedges doing
  the work?
- For technical explainers: is the mechanism right when checked against docs or
  standards?

### general (any paper)
- Are the claimed contributions stated, and does each have evidence of the scope
  it claims? (Lipton & Steinhardt 2018: separate explanation from speculation.)
- Are assumptions stated, and is the result's scope honest about them?
- Is the baseline/prior-work comparison fair (same data, same budget)?
- Are results reported with variability (seeds, splits, CIs), or single numbers?
- Is there an ablation or analysis showing *where* the gain comes from?
- Reproducibility: code, data, and settings released?
Sources: Lipton & Steinhardt, "Troubling Trends in ML Scholarship"
(https://arxiv.org/abs/1807.03341); NeurIPS checklist.

### security
- Is the **threat model** stated (adversary goal, capability, knowledge), and
  does the evaluation match it? (Carlini et al. 2019, https://arxiv.org/abs/1902.06705)
- **Systems benchmarking**: subset presented as "overall"? CPU saturated for
  overhead? geometric mean for ratios? variance reported? (van der Kouwe et al.
  2019.)
- **ML-for-security pitfalls**: sampling bias, data snooping, label inaccuracy,
  spurious correlations, wrong metric for class imbalance, lab-only evaluation,
  no adaptive attacker. (Arp et al. 2022, https://www.usenix.org/system/files/sec22-arp.pdf)
- **Fuzzing**: multiple trials with a statistical test (Mann-Whitney U) and
  effect size; bugs counted as ground-truth bugs not "unique crashes"; seeds
  disclosed; >= 24h or justified; established benchmarks. (Klees et al. 2018,
  https://arxiv.org/abs/1808.09700; Schloegel et al. 2024, https://arxiv.org/abs/2405.10220)
- **Vuln detection / repair**: can the model tell a vulnerable function from its
  patch? plausible vs correct patches separated? chronological split, dedup?
  (Risse & Böhme 2024, https://www.usenix.org/system/files/usenixsecurity24-risse.pdf; Smith et al. 2015)
- **LLM-agent benchmarks**: cost reported with accuracy? tasks actually solvable
  and tests sufficient? contamination checked? (Kapoor et al. 2024,
  https://arxiv.org/abs/2407.01502)

### ml
- Explanation vs speculation separated; "mathiness" not standing in for clarity.
- Sources of gains isolated (ablations); baselines tuned to the same budget.
  (Dodge et al. 2019, https://aclanthology.org/D19-1224/)
- **Leakage**: a real held-out set; preprocessing/feature-selection fit on train
  only; no duplicates across split; no features unavailable at prediction time;
  no temporal leakage; train/test independence; test distribution matches the
  claim. (Kapoor & Narayanan 2023, https://arxiv.org/abs/2207.07048)
- Error bars present, with the source of variability named (seed/split/init).
  (NeurIPS checklist.)

### rl
- How many seeds? (Two sets of 5 seeds of the *same* algorithm differed
  significantly: Henderson et al. 2018, https://arxiv.org/abs/1709.06560.)
- Hyperparameters, architecture, reward scaling, and codebase reported for the
  proposed method *and* baselines?
- Interval estimates, not point estimates: stratified bootstrap CIs, the
  interquartile mean, performance profiles, probability of improvement. (Agarwal
  et al. 2021, https://arxiv.org/abs/2108.13264)

### ir
- Are BM25/QL/RM3 baselines tuned, not off-the-shelf? Is a strong
  query-expansion baseline included? (Lin 2018,
  https://cs.uwaterloo.ca/~jimmylin/publications/Lin_SIGIRForum2018.pdf)
- Is the baseline at least at the collection's known level, and compared with the
  best published number, not just the paper's own? (Armstrong et al. 2009; Yang
  et al. 2019, https://arxiv.org/abs/1904.09171)
- Do neural gains survive on top of a strong baseline and out of domain? Are
  unjudged documents handled? latency/cost reported? (BEIR: Thakur et al. 2021,
  https://arxiv.org/abs/2104.08663)
- Is a significance test reported and appropriate? (Dror et al. 2018,
  https://aclanthology.org/P18-1128/)

### systems
- Same as the security "systems benchmarking" item: honest subsets, saturation,
  geometric mean for ratios, absolute numbers with a proper baseline, variance,
  platform/versions stated. (van der Kouwe et al. 2019.)
