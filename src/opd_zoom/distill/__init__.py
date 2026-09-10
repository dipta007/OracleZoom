"""(e) Small-student distillation.

Distill the trained prompter into a 0.5B-2B VLM to cut prompt-extraction latency (the CoZ
bottleneck: PE dominates SR at high scale). Optional; cut first if behind schedule.
"""
