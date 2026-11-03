# Quantizing for CPU inference

*Illustrative scenario; the output is real (torch 2.15.0, torchao 0.18.0, CPU).*

**Question.** "Our model serves on CPU. A blog post says int4 weight-only quantization gives
the biggest win. How do I apply it?"

## Verdict

Use int8 (`Int8WeightOnlyConfig`, or `Int8DynamicActivationInt8WeightConfig` for dynamic
activation quantization), not int4. With torchao 0.18 on CPU, `Int4WeightOnlyConfig` fails
with `ImportError: Requires mslk >= 1.0.0`; the int4 kernels the blog post benchmarks are
CUDA (or XPU) kernels. Int8 cut a test model's checkpoint 3.9x at 45 dB SQNR.

## Evidence

From the walkthrough in `references/14-quantization.md`, executed by
`tests/test_walkthroughs.py`:

```
returns: None | weight type: Int8Tensor | bias type: Parameter
checkpoint: 3.9x smaller
SQNR: 45 dB
```

And the int4 attempt on the same install:

```python
quantize_(model, Int4WeightOnlyConfig())    # torchao 0.18, CPU
# ImportError: Requires mslk >= 1.0.0
```

SQNR is the ratio of output signal power to quantization error power in decibels; 45 dB
means the error carries about 1/30,000 of the output's power. It is a cheap proxy; the
decision should rest on the task metric on held-out data.

## What would change the answer

A CUDA deployment makes int4 the right first try for decode-bound LLM serving (see
`references/14-quantization.md` for why weight-only quantization speeds up decode and not
prefill). A newer torchao may ship CPU int4 kernels; `dir(torchao.quantization)` and a
one-layer trial on the target machine settle it in a minute.
