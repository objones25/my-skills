#!/usr/bin/env python3
"""
Reference losses for post-training, each next to the property that tells it apart:

    preference  DPO, IPO, SimPO                       (offline pairs)
    RL          GRPO, Dr. GRPO, DAPO, GSPO           (groups of sampled responses, 0/1 rewards)
    helpers     group advantages, dynamic-sampling filter, k3 KL estimator

    python post_training_losses.py --verify     # property tests with real output
    python post_training_losses.py --explain grpo-length-bias

Notation used everywhere:

    G        responses sampled per prompt (a "group")
    T        max response length in tokens; mask (G, T) marks real tokens
    logp     (G, T) log pi_theta(token) under the policy being trained
    old_logp (G, T) the same tokens under the policy that sampled them
    R        (G,) scalar reward per response
    A        advantage: GRPO uses (R - mean R) / std R for every token of response i

Sources (equations checked against the papers' text, Nov 2026): DPO [2305.18290 Eq. 7],
IPO [2310.12036], SimPO [2405.14734], GRPO [2402.03300 Eq. 3], Dr. GRPO [2503.20783 §3.2],
DAPO [2503.14476 Eq. 10; eps_low 0.2, eps_high 0.28 in §4.1], GSPO [2507.18071 Eqs. 5-7],
k3 estimator (Schulman, "Approximating KL divergence", 2020, as used by GRPO Eq. 4).
"""

from __future__ import annotations

import argparse
import math
import sys

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Preference losses. Inputs are summed log-probs of whole responses, shape (batch,).
# ---------------------------------------------------------------------------


def dpo_loss(pc, pr, rc, rr, beta: float = 0.1):
    """-log sigmoid(beta * ((pc - rc) - (pr - rr))). Returns (loss, chosen_reward, rejected_reward);
    the implicit rewards beta*(policy - reference) are what to log, alongside raw pc and pr."""
    chosen, rejected = beta * (pc - rc), beta * (pr - rr)
    return -F.logsigmoid(chosen - rejected).mean(), chosen.detach(), rejected.detach()


def ipo_loss(pc, pr, rc, rr, tau: float = 0.1):
    """(h - 1/(2 tau))^2 with h the same log-ratio difference DPO uses. Squared loss pulls h to a
    finite target, so the optimum stays regularized when preferences are near-deterministic."""
    h = (pc - rc) - (pr - rr)
    return (h - 1.0 / (2 * tau)).pow(2).mean()


def simpo_loss(pc, pr, len_c, len_r, beta: float, gamma: float):
    """Reference-free, length-normalized: -log sigmoid(beta*pc/|yc| - beta*pr/|yr| - gamma)."""
    return -F.logsigmoid(beta * pc / len_c - beta * pr / len_r - gamma).mean()


# ---------------------------------------------------------------------------
# RL helpers
# ---------------------------------------------------------------------------


def group_advantages(R: torch.Tensor, normalize_std: bool = True, eps: float = 1e-6) -> torch.Tensor:
    """(G,) -> (G,). GRPO divides by the group std; Dr. GRPO does not (std reweights prompts by
    difficulty: groups that are almost all right or all wrong get the largest weight)."""
    A = R - R.mean()
    return A / (R.std() + eps) if normalize_std else A


def keep_group(R: torch.Tensor) -> bool:
    """DAPO dynamic sampling: drop groups where every response got the same reward,
    because their advantages are all zero and they contribute no gradient."""
    return bool(R.max() > R.min())


def k3_kl(logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    """Per-token estimator of KL(pi || ref) from samples of pi: ratio - log ratio - 1 with
    ratio = ref/pi. Non-negative for every sample and unbiased when the samples come from
    the current policy pi. With samples from an older policy its gradient is biased;
    DeepSeek-V3.2 multiplies it by the importance ratio pi/pi_old to correct this."""
    log_ratio = ref_logp - logp
    return log_ratio.exp() - log_ratio - 1


def _clipped(ratio, A, lo, hi):
    return torch.minimum(ratio * A, ratio.clamp(1 - lo, 1 + hi) * A)


def policy_loss(logp, old_logp, A, mask, variant: str = "grpo", eps: float = 0.2,
                eps_high: float | None = None, max_tokens: int | None = None) -> torch.Tensor:
    """
    Negative surrogate objective for one group. A is (G,), broadcast to every token.

    grpo     (1/G) sum_i (1/|o_i|) sum_t clip(r_it) A_i        per-response length average
    dr_grpo  (1/G) sum_i (1/T_max) sum_t clip(r_it) A_i        constant denominator
    dapo     (1/sum_i |o_i|) sum_i sum_t clip(r_it) A_i         every token weighted equally,
             with eps_low / eps_high ("clip-higher")
    gspo     (1/G) sum_i clip(s_i) A_i,  s_i = exp(mean_t log r_it)   sequence-level ratio
    """
    log_r = (logp - old_logp) * mask
    A_tok = A[:, None].expand_as(logp)
    lengths = mask.sum(-1)
    if variant == "gspo":
        s = (log_r.sum(-1) / lengths).exp()                                # (G,)
        return -_clipped(s, A, eps, eps if eps_high is None else eps_high).mean()
    hi = eps if eps_high is None else eps_high
    per_tok = _clipped(log_r.exp(), A_tok, eps, hi) * mask                 # (G, T)
    if variant == "grpo":
        return -(per_tok.sum(-1) / lengths).mean()
    if variant == "dr_grpo":
        return -(per_tok.sum(-1) / (max_tokens or mask.shape[-1])).mean()
    if variant == "dapo":
        return -per_tok.sum() / lengths.sum()
    raise ValueError(variant)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

RESULTS: list[bool] = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def _token_grad(variant, lengths, A, **kw):
    """Gradient of the loss w.r.t. each token's logp, at ratio = 1, for responses of given lengths."""
    G, T = len(lengths), max(lengths)
    mask = torch.zeros(G, T)
    for i, n in enumerate(lengths):
        mask[i, :n] = 1
    logp = torch.zeros(G, T, requires_grad=True)
    policy_loss(logp, torch.zeros(G, T), torch.tensor(A, dtype=torch.float32), mask, variant, **kw).backward()
    return logp.grad


def verify() -> int:
    RESULTS.clear()
    torch.manual_seed(0)

    print("\nDPO")
    z = torch.zeros(4)
    loss, _, _ = dpo_loss(z, z, z, z)
    check("loss = log 2 when policy == reference", abs(loss.item() - math.log(2)) < 1e-6,
          f"{loss.item():.4f} vs {math.log(2):.4f}")
    pc = torch.zeros(1, requires_grad=True)
    pr = torch.zeros(1, requires_grad=True)
    dpo_loss(pc, pr, torch.zeros(1), torch.zeros(1))[0].backward()
    check("gradient raises chosen and lowers rejected", pc.grad.item() < 0 < pr.grad.item(),
          f"dL/dchosen {pc.grad.item():+.3f}, dL/drejected {pr.grad.item():+.3f}")
    before = dpo_loss(torch.tensor([-50.0]), torch.tensor([-60.0]), torch.tensor([-50.0]), torch.tensor([-60.0]))
    after = dpo_loss(torch.tensor([-53.0]), torch.tensor([-70.0]), torch.tensor([-50.0]), torch.tensor([-60.0]))
    check("loss can fall while the chosen log-prob falls (likelihood displacement)",
          after[0] < before[0], f"chosen logp -50 -> -53, loss {before[0].item():.3f} -> {after[0].item():.3f}; "
          "log raw logprobs, not just the margin")

    print("\nIPO and SimPO")
    h_target = 1 / (2 * 0.1)
    check("IPO is minimized at h = 1/(2 tau), not at h -> infinity",
          ipo_loss(torch.tensor([h_target]), z[:1], z[:1], z[:1]).item() < 1e-12
          and ipo_loss(torch.tensor([10 * h_target]), z[:1], z[:1], z[:1]).item() > 1)
    a = simpo_loss(torch.tensor([-20.0]), torch.tensor([-40.0]), torch.tensor([10.0]), torch.tensor([40.0]), 2.0, 0.5)
    b = simpo_loss(torch.tensor([-20.0]), torch.tensor([-40.0]), torch.tensor([10.0]), torch.tensor([10.0]), 2.0, 0.5)
    check("SimPO compares per-token log-probs: a long rejected answer is judged per token", a > b,
          f"rejected 40 tokens: {a.item():.3f}; 10 tokens: {b.item():.3f}")

    print("\nGroup advantages")
    for R in (torch.ones(8), torch.zeros(8)):
        A = group_advantages(R)
        check(f"all rewards {int(R[0])}: advantages are all zero (no gradient)", bool((A == 0).all()))
    check("DAPO dynamic sampling drops those groups", not keep_group(torch.ones(8)) and keep_group(torch.tensor([1.0, 0, 0, 0])))
    easy = group_advantages(torch.tensor([1.0, 1, 1, 1, 1, 1, 1, 0]))
    mid = group_advantages(torch.tensor([1.0, 1, 1, 1, 0, 0, 0, 0]))
    check("std normalization gives the one wrong answer in an easy group a larger weight",
          easy.abs().max() > mid.abs().max(), f"max |A| easy {easy.abs().max():.2f} vs balanced {mid.abs().max():.2f}")
    easy_dr = group_advantages(torch.tensor([1.0, 1, 1, 1, 1, 1, 1, 0]), normalize_std=False)
    check("Dr. GRPO's unnormalized advantage keeps |A| <= 1 for 0/1 rewards", easy_dr.abs().max() <= 1)

    print("\nLength bias (gradient per token at ratio 1, one short and one long wrong answer)")
    lens, A = [10, 100], [-1.0, -1.0]
    g = _token_grad("grpo", lens, A)
    short, long_ = g[0, 0].item(), g[1, 0].item()
    check("GRPO: a token in the long wrong answer is penalized 10x less", abs(short / long_ - 10) < 1e-4,
          f"per-token grad {short:+.4f} vs {long_:+.5f}; long wrong answers are cheap, so length grows")
    for variant, kw in (("dr_grpo", {"max_tokens": 128}), ("dapo", {})):
        g = _token_grad(variant, lens, A, **kw)
        check(f"{variant}: every token weighted the same", abs(g[0, 0] - g[1, 0]).item() < 1e-8,
              f"{g[0, 0].item():+.5f} vs {g[1, 0].item():+.5f}")
    g = _token_grad("dapo", [10, 100], [1.0, -1.0])
    check("DAPO: weights tokens equally across the group, so the long answer dominates the group's gradient",
          g[1].abs().sum() > 5 * g[0].abs().sum())

    print("\nClipping")
    G, T = 1, 4
    mask = torch.ones(G, T)
    old = torch.zeros(G, T)
    lp = torch.full((G, T), math.log(1.25), requires_grad=True)
    policy_loss(lp, old, torch.tensor([1.0]), mask, "grpo", eps=0.2).backward()
    clipped_grad = lp.grad.abs().sum().item()
    lp2 = torch.full((G, T), math.log(1.25), requires_grad=True)
    policy_loss(lp2, old, torch.tensor([1.0]), mask, "dapo", eps=0.2, eps_high=0.28).backward()
    check("ratio 1.25, A > 0: symmetric eps 0.2 stops the gradient, clip-higher 0.28 keeps it",
          clipped_grad == 0 and lp2.grad.abs().sum() > 0,
          f"grads {clipped_grad:.3f} vs {lp2.grad.abs().sum().item():.3f}")

    print("\nGSPO")
    logp = torch.log(torch.tensor([[1.1, 0.9, 1.3, 1.0]]))
    s = ((logp * mask).sum(-1) / mask.sum(-1)).exp()
    geo = torch.tensor([1.1 * 0.9 * 1.3 * 1.0]).pow(1 / 4)
    check("sequence ratio = geometric mean of token ratios", torch.allclose(s, geo), f"{s.item():.4f}")
    lp = logp.clone().requires_grad_(True)
    policy_loss(lp, old, torch.tensor([1.0]), mask, "gspo", eps=0.2).backward()
    check("every token in a response gets the same gradient weight", torch.allclose(lp.grad, lp.grad[0, :1].expand_as(lp.grad)))
    lp = torch.log(torch.tensor([[1.5, 1.5, 1.5, 1.5]])).requires_grad_(True)
    policy_loss(lp, old, torch.tensor([1.0]), mask, "gspo", eps=0.2).backward()
    check("a response whose sequence ratio is outside the clip range drops out entirely",
          lp.grad.abs().sum().item() == 0)

    print("\nk3 KL estimator")
    torch.manual_seed(0)
    p = torch.tensor([0.6, 0.3, 0.1])
    q = torch.tensor([0.4, 0.4, 0.2])
    true_kl = (p * (p / q).log()).sum().item()
    x = torch.multinomial(p, 200_000, replacement=True)
    est = k3_kl(p.log()[x], q.log()[x])
    check("never negative on any sample", bool((est >= 0).all()))
    check("mean matches KL(pi || ref)", abs(est.mean().item() - true_kl) < 3e-3,
          f"{est.mean().item():.4f} vs exact {true_kl:.4f}")

    failed = RESULTS.count(False)
    print(f"\n{'all checks passed' if not failed else f'{failed} checks FAILED'}")
    return int(failed > 0)


EXPLAIN = {
    "grpo-length-bias": """
GRPO averages each response's token losses by that response's own length, 1/|o_i|. For a wrong
answer (A < 0), every token's push-down is divided by |o_i|, so a 1000-token wrong answer is
penalized per token 10x less than a 100-token one. The policy learns that being wrong at length
is cheap, and responses grow without getting more correct. Dr. GRPO divides by a constant (the
generation budget); DAPO divides by the total tokens in the group. Common PPO libraries are
length-normalized too (OpenRLHF per response, trl and verl per batch), which Dr. GRPO also
counts as biased [Dr. GRPO §3.1, Table 2, Listing 1].""",
    "grpo-std-bias": """
Dividing advantages by the group's reward std reweights prompts. With 0/1 rewards, a group that is
7/8 right has a small std, so its one wrong answer gets |A| = 2.47 while a 4/8 group's answers get
|A| = 0.94. Prompts the model already nearly solves (or nearly never solves) dominate the update.
Dr. GRPO drops the std; batch-level normalization (across all prompts) avoids the per-prompt skew.""",
    "zero-variance-groups": """
If all G responses to a prompt get the same reward, R - mean(R) = 0 for every one of them, so the
whole group contributes nothing but noise in the denominator. As training progresses more prompts
become all-correct. DAPO over-samples and keeps only groups with 0 < correct < G, so each batch has
a fixed number of prompts that actually carry gradient.""",
    "likelihood-displacement": """
DPO's loss depends only on the margin (chosen - rejected log-ratio). The margin can grow while
the chosen response's own log-probability falls, as long as the rejected one falls faster. The
mass goes to responses that are in neither set. Log pc and pr themselves, and treat a falling pc
as a warning even when the loss and reward accuracy look healthy.""",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--explain", choices=sorted(EXPLAIN))
    a = ap.parse_args()
    if a.explain:
        print(EXPLAIN[a.explain].strip())
        return 0
    return verify()


if __name__ == "__main__":
    sys.exit(main())
