"""(a) Privileged oracle prompter.

Run Qwen2.5-VL-3B as a privileged teacher: feed GT-HR patch + full-image thumbnail
(+ x_{i-1}, x_{i-2}) and generate "oracle prompts". Contrast with the blind student, which
sees only [prev_SR_output, bicubic_zoom] (ref/coz/inference_coz.py:337).

This stage is framework-agnostic (plain VLM inference) and powers the Gate-1 pilot.
"""
