# 06: "My MLA port's generation diverges from its own forward pass"

*Illustrative scenario.* A from-scratch MLA implementation matches loss curves in training. With the latent cache at decode, outputs drift from the full forward pass after the first few tokens.

## Verdict

RoPE is being applied to keys reconstructed from the latent. That breaks the absorbed decode path, which assumes nothing position-dependent sits between W_uk and the cached latent. Keep RoPE on the separate, decoupled key (shared across heads) and leave the latent-derived keys unrotated.

## The check

```
$ python scripts/modern_blocks.py --verify
Multi-head latent attention
  [PASS] absorbed scores (from cache only) == explicit per-head scores  max diff 1.2e-07
  [PASS] latent-cache decode == full forward  max diff 1.2e-07
  [PASS] cache holds 24+8 = 32 numbers per token, not 160
...
  [PASS] mla-rope-on-latent: caught  absorbed and explicit scores differ by 0.657
```

`python scripts/modern_blocks.py --bug mla-rope-on-latent` prints the derivation: q . (W_uk c) = (W_uk^T q) . c holds, q . (R_t W_uk c) cannot be rewritten that way.

## Why training did not show it

Training uses the explicit path, which is internally consistent with or without the extra rotation. Only the absorbed decode path, which multiplies through W_uk^T, notices. Test decode against forward on every port, for every attention variant.

## Related checks in the same script

Router bias in the gate weights, top-k renormalization mismatches against a checkpoint, and a sink implemented as a logit offset (which does nothing) are the other silent bugs it detects.
