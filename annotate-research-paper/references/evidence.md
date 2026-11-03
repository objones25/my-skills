# Evidence behind the rules

Read this to justify or adjust a rule; not needed for an ordinary run.

## Contents
- Evidence table
- References

## Evidence table

Read this to justify or adjust a rule. No controlled study has tested expert-style
marginalia or explainers on CS papers directly; the rules extrapolate from nearby findings, mostly
with undergraduates on short lessons, so for a graduate reader expect the effects of added support to
be smaller (expertise reversal).
Strength: S1 controlled/quasi-experimental or a review/meta-analysis of such; S2
systematic audit, reproduction, or user study (N >= 19); S3 small qualitative
(N <= 17); S4 expert guidance or venue checklist.

| Rule | Finding | Source | Strength |
|---|---|---|---|
| Open with a classify-and-route guide | Keshav's first pass answers the five Cs; experts rarely read sequentially and jump to results. | Keshav 2007; Bazerman 1988 | S4, S3 |
| Tag each claim, point to evidence, state scope | 100% of faculty vs 18% of students evaluated claims while reading. Claims should match the generalization the results support. | Nelms & Segura-Totten 2019; NeurIPS checklist | S3, S4 |
| Lead with the implication; never restate | 75% of faculty notes were analytic or evaluative vs 21% of students'. Elaborative interrogation (asking why) and self-explanation rate moderate utility; rereading and summarization low. Prompted self-explanation: 22.6% vs 12.5% on hard inference items. | Nelms & Segura-Totten 2019; Dunlosky et al. 2013; Chi et al. 1994 | S3, S1 |
| Figures: message first (Metric -> Baseline -> Takeaway) | "Identify your message" is rule 2 for figures; a figure exists to convey one idea. Skill with results sections develops slowly; novices find results hardest. | Rougier et al. 2014; Hubbard & Dunbar 2017; Carey et al. 2020 | S4, S2 |
| Check statistics in results figures | Researchers misjudge CI/SE overlap (N=473); 85.6% of 703 papers used bar graphs for continuous data; y-axis truncation inflates perceived effects. | Belia et al. 2005; Weissgerber et al. 2015; Correll et al. 2020 | S1, S2 |
| Name assumptions, mark the load-bearing ones | Third pass: challenge every assumption; state strong assumptions and robustness to violations. | Keshav 2007; NeurIPS checklist | S4 |
| Define terms at point of use | Readers delay lookups and accumulate confusion; just-in-time definitions made tasks ~45 s faster (N=27). | Head et al. 2021 | S2 |
| Answer before seeing the answer (upside-down answer under the question) | Attempting to answer before studying improved later performance even for items answered wrong at pretest, and the effect held after a week. Feedback, immediate or delayed, raised later correct recall and reduced errors picked up from multiple-choice lures, compared with no feedback. | Richland, Kornell & Kao 2009; Butler & Roediger 2008 | S1 |
| Answers explain why | Elaborated feedback effect size 0.49 vs 0.32 for giving the correct answer and 0.05 for right/wrong only; elaborated feedback is especially better for higher-order outcomes. Immediate feedback tended to help lower-order and delayed higher-order learning, but the interaction was not significant (abstract; full text access-controlled). Explanation feedback beat correct-answer feedback on new inference questions (45% vs 30%). | Van der Kleij, Feskens & Eggen 2015; Butler, Godbole & Marsh 2013 | S1 |
| Record how each link was checked; never claim a fetch | 55% of GPT-3.5 and 18% of GPT-4 citations were fabricated; 43% / 24% of the real ones had substantive errors. Frontier models got only 50.9% of BibTeX entries fully correct; deterministic retrieval followed by revision raised that to 78.3%, and separating search from revision cut regressions (0.8% vs 4.8%). | Walters & Wilder 2023; Rao & Callison-Burch 2026 | S2 |
| Critique in proportion; note field norms | Only 1 of 50 systems-security papers had no benchmarking flaw; frame field-wide flaws as systemic. | van der Kouwe et al. 2019; Carey et al. 2020 | S2, S4 |
| Highlight every key point, each with a reason, within a density band | Highlighting alone rates low utility; later readers of used textbooks valued interpretive notes most. So each highlight carries a hover reason, and the cap keeps emphasis meaningful. | Dunlosky et al. 2013; Marshall 1997 | S1, S3 |
| Explainers select; never paraphrase passage by passage or repeat the margins | Two sources carrying the same information cost effort without adding understanding (redundancy effect). Proofs with line-by-line commentary (e-Proofs) did no better at once and worse after two weeks (N = 49), while students liked them more. Summaries a fifth the length of textbook chapters matched or beat the chapters, at 20 minutes and at 6-12 months. | Sweller et al. 2019; Roy, Inglis & Alcock 2017; Reder & Anderson 1980; Mayer et al. 1996 | S1, S2, S1 |
| Break down only the steps that are load-bearing and hard to see; background gets a notation row, not a section | Support that helps novices loses value or hurts as knowledge grows (expertise reversal, >2,200 students; e.g. d = 1.20 novices vs -0.36 experts). Provided explanations often fail to help, and help conceptual more than procedural knowledge (meta-analysis of 21 studies, abstract only). Load comes from how many elements must be held at once, so support pays off on high-interaction steps. | Kalyuga 2007; Wittwer & Renkl 2008, 2010; Sweller et al. 2019 | S1 |
| Big picture first; the proof as a chain of modules | Pre-training on key concepts (5 of 5 tests, d = 0.85) and advance organizers (small positive effect over 135 studies) help. Proof comprehension has a holistic side (key idea, modules, how they connect) distinct from line-by-line justification. Experts reading proofs infer the unstated warrants between lines. | Mayer 2009; Luiten et al. 1980; Mejía-Ramos et al. 2012; Inglis & Alcock 2012 | S1, S1, S4, S2 |
| Check questions ask the reader to produce the inference | Self-explanation prompts beat provided explanations (g = .35, k = 6) and no prompt (g = .67); self-explanation training for proofs gave d = 0.95 and lasted. Explanations on demand reduced learners' own explaining. | Bisra et al. 2018; Hodds, Alcock & Inglis 2014; Schworm & Renkl 2002 | S1 |
| One hand-sized example, tied back to the general form | Concrete-then-faded beat concrete-only and abstract-only on transfer (d = 0.6-0.8, N = 73); other studies favor abstract-only, so the example must be tied back explicitly. | McNeil & Fyfe 2012; Kaminski et al. 2008 | S1 (mixed) |
| Diagram the mechanism and the proof chain; no printed code | Text plus diagrams is among the largest multimedia effects; interesting but extraneous material (seductive details) reliably hurts. Mayer's own medians are larger than an independent meta-analysis (g = 0.37), so expect modest gains. | Mayer 2017; Rey 2012; Cromley & Chen 2025 | S1 |
| A fresh reader reviews each explainer against its goal | Learners' sense of having learned does not track learning (active instruction: +0.46 SD learning, -0.56 SD felt learning). A build that checks only form passed ten walkthroughs that one reader's audit found unhelpful: 2.5x the paper's length, main idea 5-15 pages in, 13-23 undefined terms in the first 4 pages. | Deslauriers et al. 2019; audit of this skill's output, Oct 2026 | S1, S3 |

Rules without a study behind them, kept because the user requires them or because
they follow from how language models fail: never guess about existence or events
(a model does not know what happened after its training data ends); choose
standalone teaching resources after inspecting their content (a mid-series episode
assumes context the reader lacks; a channel's reputation says nothing about one
video's fit); go page by page and give every key point a decision (a run that
covered a paper's densest section with two notes is the failure this prevents);
give each chunk of at most 5 pages its own context, floors, and blind verifier (a
single-context run on a 44-page survey put 22 of its 26 notes on the first 21
pages);
verify every quote against the paper (models paraphrase and then put quotation
marks around the paraphrase); run demos and show their real output.

## References

- Bazerman (1988). Physicists Reading Physics. Ch. 8 in *Shaping Written Knowledge*. https://wac.colostate.edu/books/landmarks/bazerman-shaping/
- Belia, Fidler, Williams & Cumming (2005). Researchers misunderstand confidence intervals and standard error bars. *Psychological Methods* 10(4). doi:10.1037/1082-989X.10.4.389
- Butler & Roediger (2008). Feedback enhances the positive effects and reduces the negative effects of multiple-choice testing. *Memory & Cognition* 36(3):604-616. doi:10.3758/MC.36.3.604
- Butler, Godbole & Marsh (2013). Explanation feedback is better than correct answer feedback for promoting transfer of learning. *J. Educational Psychology* 105(2). https://bpb-us-e2.wpmucdn.com/sites.wustl.edu/dist/8/805/files/2026/06/Butler-et-al.-2013-Explanation-feedback-is-better-than-correct-answer-feedback-for-promoting-transfer-of-learning.pdf
- Carey, Steiner & Petri (2020). Ten simple rules for reading a scientific paper. *PLOS Comp Biol* 16(7). https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008032
- Chi, de Leeuw, Chiu & LaVancher (1994). Eliciting self-explanations improves understanding. *Cognitive Science* 18(3). https://pcl.sitehost.iu.edu/rgoldsto/courses/cogscilearning/chiselfexplanation.pdf
- Correll, Bertini & Franconeri (2020). Truncating the Y-Axis: Threat or Menace? *CHI 2020*. https://arxiv.org/abs/1907.02035
- Dunlosky, Rawson, Marsh, Nathan & Willingham (2013). Improving students' learning with effective learning techniques. *PSPI* 14(1). doi:10.1177/1529100612453266
- Head et al. (2021). Augmenting Scientific Papers with Just-in-Time, Position-Sensitive Definitions (ScholarPhi). *CHI 2021*. https://arxiv.org/abs/2009.14237
- Hubbard & Dunbar (2017). Perceptions of scientific research literature ... depend on academic career stage. *PLOS ONE* 12(12). doi:10.1371/journal.pone.0189753
- Keshav (2007). How to Read a Paper. *ACM SIGCOMM CCR* 37(3). https://www.cs.princeton.edu/courses/archive/fall10/cos597B/papers/howtoread-keshav.pdf
- Marshall (1997). Annotation: from paper books to the digital library. *ACM DL '97*. doi:10.1145/263690.263806
- Nelms & Segura-Totten (2019). Expert-novice comparison ... analysis of primary literature. *CBE Life Sci Educ* 18:ar56. https://www.lifescied.org/doi/10.1187/cbe.18-05-0077
- NeurIPS Paper Checklist. https://neurips.cc/public/guides/PaperChecklist
- Rao & Callison-Burch (2026). BibTeX Citation Hallucinations in Scientific Publishing Agents: Evaluation and Mitigation. https://arxiv.org/abs/2604.03159
- Richland, Kornell & Kao (2009). The pretesting effect: Do unsuccessful retrieval attempts enhance learning? *J. Experimental Psychology: Applied*. https://learninglab.uchicago.edu/Pre-Testing_files/RichlandKornellKao.pdf
- Rougier, Droettboom & Bourne (2014). Ten Simple Rules for Better Figures. *PLOS Comp Biol* 10(9). https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003833
- van der Kouwe et al. (2019). SoK: Benchmarking Flaws in Systems Security. *IEEE EuroS&P*. https://www.trustworthy.systems/publications/full_text/vanderKouwe_HABG_19.pdf
- Van der Kleij, Feskens & Eggen (2015). Effects of feedback in a computer-based learning environment on students' learning outcomes: a meta-analysis. *Review of Educational Research* 85(4):475-511. Abstract: https://acuresearchbank.acu.edu.au/item/86y84/effects-of-feedback-in-a-computer-based-learning-environment-on-students-learning-outcomes-a-meta-analysis
- Walters & Wilder (2023). Fabrication and errors in the bibliographic citations generated by ChatGPT. *Scientific Reports* 13. https://pmc.ncbi.nlm.nih.gov/articles/PMC10484980
- Weissgerber, Milic, Winham & Garovic (2015). Beyond Bar and Line Graphs. *PLOS Biology* 13(4). doi:10.1371/journal.pbio.1002128
- Bisra, Liu, Nesbit, Salimi & Winne (2018). Inducing self-explanation: a meta-analysis. *Educ Psychol Rev* 30. https://www.gwern.net/doc/psychology/spaced-repetition/2018-bisra.pdf
- Cromley & Chen (2025). Meta-analysis of Mayer's multimedia principles (abstract). https://par.nsf.gov/biblio/10637927
- Deslauriers, McCarty, Miller, Callaghan & Kestin (2019). Measuring actual learning versus feeling of learning. *PNAS* 116(39). https://www.pnas.org/doi/10.1073/pnas.1821936116
- Hodds, Alcock & Inglis (2014). Self-explanation training improves proof comprehension. *JRME* 45(1). https://eric.ed.gov/?id=EJ1028348
- Inglis & Alcock (2012). Expert and novice approaches to reading mathematical proofs. *JRME* 43(4). https://eric.ed.gov/?id=EJ977681
- Kalyuga (2007). Expertise reversal effect and its implications for learner-tailored instruction. *Educ Psychol Rev* 19. https://www.uky.edu/~gmswan3/EDC608/Kalyuga2007_Article_ExpertiseReversalEffectAndItsI.pdf
- Kaminski, Sloutsky & Heckler (2008). The advantage of abstract examples in learning math. *Science* 320. doi:10.1126/science.1154659
- Luiten, Ames & Ackerson (1980). A meta-analysis of the effects of advance organizers. https://eric.ed.gov/?id=EJ233231
- Mayer (2009). *Multimedia Learning*, 2nd ed. (chapter summaries). https://www.cambridge.org/core/books/multimedia-learning/segmenting-principle/37240877DDA0362355ADB39936027982
- Mayer (2017). Using multimedia for e-learning. *J Computer Assisted Learning* 33. https://eric.ed.gov/?id=EJ1153516
- Mayer, Bove, Bryman, Mars & Tapangco (1996). When less is more. *J Educ Psychol* 88(1). https://eric.ed.gov/?id=EJ526858
- McNeil & Fyfe (2012). "Concreteness fading" promotes transfer of mathematical knowledge. *Learning and Instruction* 22. https://pcl.sitehost.iu.edu/rgoldsto/courses/cogscilearning/McNeilFyfe12.pdf
- Mejía-Ramos, Fuller, Weber, Rhoads & Samkoff (2012). An assessment model for proof comprehension. *Educ Stud Math* 79. https://eric.ed.gov/?id=EJ948400
- Reder & Anderson (1980). A comparison of texts and their summaries. *J Verbal Learning and Verbal Behavior* 19. https://www.andrew.cmu.edu/user/reder/publications/80_lmr_jra.pdf
- Rey (2012). A review of research and a meta-analysis of the seductive detail effect. *Educ Res Rev* 7. https://eric.ed.gov/?id=EJ986386
- Roy, Inglis & Alcock (2017). Multimedia resources designed to support learning from written proofs. *Educ Stud Math* 96. https://link.springer.com/article/10.1007/s10649-017-9754-7
- Schworm & Renkl (2002). Learning by solved example problems: instructional explanations reduce self-explanation activity. *CogSci 2002*. https://escholarship.org/content/qt00q0t4h1/qt00q0t4h1.pdf
- Sweller, van Merriënboer & Paas (2019). Cognitive architecture and instructional design: 20 years later. *Educ Psychol Rev* 31. https://link.springer.com/article/10.1007/s10648-019-09465-5
- Wittwer & Renkl (2008). Why instructional explanations often do not work. *Educational Psychologist* 43(1). https://eric.ed.gov/?id=EJ784018
- Wittwer & Renkl (2010). How effective are instructional explanations in example-based learning? A meta-analytic review. *Educ Psychol Rev* 22. https://eric.ed.gov/?id=EJ906661
