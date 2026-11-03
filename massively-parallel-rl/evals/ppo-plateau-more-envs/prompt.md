---
tags: [massively-parallel-rl]
max_turns: 40
timeout_seconds: 1500
allowed_tools: [Read, Glob, Grep, Skill, Write, Edit, Bash]
---

my PPO on Craftax-Symbolic-v1 (purejaxrl style, jitted, one A100) was improving fine with NUM_ENVS=1024, NUM_STEPS=64, UPDATE_EPOCHS=4, NUM_MINIBATCHES=8, LR=2.5e-4. I bumped NUM_ENVS to 16384 to use the GPU better and kept everything else the same, and now at the same number of env steps the return is lower and it seems to plateau earlier. what's going on and what exactly should I change? give me the configs to run
