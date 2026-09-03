"""Latent-diffusion inverse pipeline (the diffusion track).

Builds the pieces described in reports/diffusion_plan.pdf, 2D first:

    Phase 0  data2d.py          cache 2D Perm/Por slices + normalizer + splits
    Phase 1  autoencoder2d.py   compress a field to a small latent, reconstruct
    Phase 2  ldm2d.py           latent DDPM prior (sample plausible fields)
    Phase 3  measurement.py     sensor operator H(field) -> readings (loc, time)

Everything is config-driven and reuses the project's FieldNormalizer so the
log10(Perm)+standardize convention matches the existing GRU-FNO model exactly.
Phases 4-6 (ESMDA assimilation, validation, 3D) come after these gates pass.
"""
