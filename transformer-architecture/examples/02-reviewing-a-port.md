# 02: "My port trains fine. Is it right?"

*Illustrative scenario.* Someone ported `repeat_kv` and `RMSNorm` into their own codebase. Loss goes down. They want to load a public checkpoint next. The port is `examples/buggy_port.py`.

## Verdict

Two bugs, neither visible in training curves. `repeat_kv` uses `.repeat()`, which assigns query heads to the wrong KV heads; it is harmless from scratch and breaks every external checkpoint and fused kernel. `RMSNorm` adds eps to the sqrt instead of inside it, which gives a NaN gradient on any all-zero row.

## The run

```
$ python scripts/reference_block.py --verify --impl examples/buggy_port.py
testing your repeat_kv, RMSNorm from examples/buggy_port.py

RMSNorm
  [PASS] matches x/sqrt(mean(x^2)+eps) with unit gain  max abs diff 1.67e-06
  [PASS] mean is NOT removed (unlike LayerNorm)  output mean +0.0542
  [PASS] an all-zero vector stays finite (eps inside the sqrt)
  [FAIL] ...and gets a finite gradient (catches x / (sqrt(ms) + eps))  sqrt has an infinite derivative at 0; eps must be under it
...
GQA
  [FAIL] repeat_kv maps kv head j to query heads [j*n_rep, (j+1)*n_rep)  got [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]
  [skip] ...and .repeat() is exactly a query-head permutation  (a property of the reference repeat_kv; not meaningful for yours)
...
2 CHECK(S) FAILED
```

## Reading it

- The RMSNorm forward check passes at 1.67e-06: `sqrt(ms) + eps` and `sqrt(ms + eps)` differ by about eps on non-zero inputs. Only the backward pass on an all-zero row separates them, which is why forward-only tests (and the previous version of this verifier) missed it. Zero rows are common: padding, a zero-initialized branch at step 0.
- The GQA layout `[0, 1, 2, 3, 0, 1, 2, 3]` means query head 1 reads KV head 1 instead of KV head 0. It is exactly a relabelling of query heads, so a model trained from scratch learns around it. Say that precisely in a review: it is an interop bug, not a quality bug.

## Fix

`x.repeat_interleave(n_rep, dim=1)` (or expand then reshape), and `x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)`. Re-run until it prints `all checks passed`.
