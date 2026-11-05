# nnU-Net v2 (modified)

This folder is a modified copy of [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) (v2.6.4),
released under the Apache License 2.0 (see `LICENSE`).

Modifications for the multi-object topology-aware anatomical loss:

- `nnunetv2/training/loss/anatomical_prior.py` — new persistent-homology-based anatomical topology loss
- `nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py` — adds the topology loss (with an optional warm-up period) to the training/validation objective; `num_epochs` set to 350
- `nnunetv2/training/logging/nnunet_logger.py` — logs and plots the topology loss
- `pyproject.toml` — adds the `cripser` dependency

See the repository-level `README.md` for usage. For general nnU-Net documentation, see `documentation/`
or the original repository.
