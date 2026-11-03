# Robot RL experiment plan (template)

Fill this in before spending compute. Each section is a slide or a paragraph; define every symbol where it first appears.

1. Motivation: one concrete situation where the problem bites. Examples: GPU simulation gives thousands of robots per GPU, but more envs do not automatically mean faster learning (`references/training/on-policy-at-scale.md` case study); exploration that is exponentially hard online becomes polynomial with a simulator you can reset (combination lock: A^H episodes vs about H(A+1)/2 steps, `examples/exploration_chain.py`).
2. Problem statement: the interaction protocol (online, local simulator, generative model, demonstrations), the objective, what is unknown, what you measure.
3. Method: the algorithm as a diagram plus 3 to 5 named steps, and every choice that departs from the source paper (constants, network sizes, reset semantics, observation set).
4. Related work (table): paper, access model, assumptions, result, gap this experiment addresses (`references/papers/README.md`).
5. Question and hypothesis: what result would change your mind.
6. Experimental design: environments from simplest up; the one variable you sweep; baselines (an exact or known-good answer where one exists); the fixed evaluator and its metrics (`references/training/evaluation.md`); seeds; the batch arithmetic (`scripts/batch_math.py`); compute and cost (wall-clock × $/h).
7. Milestones: a minimum viable result first (smallest env, one seed, end to end), then the sweep, then stretch goals.
8. Risks and fallbacks: what you will report if the main experiment fails, and which negative result would still be informative.
