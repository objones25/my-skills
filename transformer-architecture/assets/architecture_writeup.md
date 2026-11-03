# <Model name>: architecture

<!-- Template for a README section, design doc, or interview answer that defends a config.
     Fill every <...>. Numbers come from scripts/config_budget.py; paste its tables, do not retype them. -->

## Constraints

| Constraint | Value | Source |
|---|---|---|
| Training compute | <GPU type x count x hours, or FLOPs> | <budget> |
| Tokens available | <N> | <dataset> |
| Inference target | <batch, context, latency, VRAM per replica> | <deployment> |
| Vocabulary / languages | <tokenizer, size> | <tokenizer> |

## Config

| field | value | | field | value |
|---|---|---|---|---|
| d_model | | | vocab | |
| n_layers | | | d_ff (or expert d_ff) | |
| n_heads / n_kv_heads | | | context / rope_theta | |
| head_dim | | | tied embeddings | |
| MoE: experts / active / shared | <or "dense"> | | attention variant | <GQA / MLA / sliding pattern / gated> |

## Budget (from `config_budget.py`)

<paste the Parameter budget and Compute and memory tables>

## The load-bearing choices

<!-- Two or three. Each: the choice, the number that forced it, what would change it. -->

1. **<choice>.** <why, with the number>. Would change if <condition>.
2. **<choice>.** ...

## Conventions (inside the observed range, no defense needed)

<norm, activation, init, optimizer, schedule; one line each, cite references/reference-configs.md>

## Verified

- `python scripts/config_budget.py <args>`: <total> parameters, validator: <findings>
- `python scripts/reference_block.py --verify --impl <our file>`: <result>
- `python scripts/modern_blocks.py --verify` (if MoE / MLA / gating / sinks): <result>

## Open questions

<what is genuinely unsettled for this design, with the paper that disagrees>
