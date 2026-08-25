"""OPD-Zoom: privileged-prompt distillation for extreme generative super-resolution.

Stages (see CLAUDE.md §1):
  teacher  (a) privileged oracle prompter: same VLM + GT-HR patch + thumbnail
  sft      (b) warm-start the blind student on oracle prompts
  train    (c) hybrid SDPO+GRPO in verl
  eval     (d) NR-IQA + faithfulness + latency harness
  distill  (e) small-student prompter
  common   shared prompts / io / config
"""
