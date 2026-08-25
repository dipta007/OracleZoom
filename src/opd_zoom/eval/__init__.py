"""(d) Evaluation harness.

NR-IQA (NIQE, MUSIQ, MANIQA-pipal, CLIPIQA) at 4x/16x/64x/256x on DIV2K + DIV8K, matching
CoZ Table 1. Plus faithfulness at 4x (PSNR/SSIM/LPIPS/DISTS + DINOv2/CLIP cosine + MLLM
hallucination rate) and latency/params.

PIN one pyiqa version and record it (metric internals drift across releases). Protocol:
512x512 center crop, resize-by-4 per recursion, 4 recursions.
"""
