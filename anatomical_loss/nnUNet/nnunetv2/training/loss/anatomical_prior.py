import json
import os

import numpy as np
import torch
from torch import nn
import cripser as crip
import tcripser as trip


def crip_wrapper(X, D):
    return crip.computePH(X, maxdim=D)


def trip_wrapper(X, D):
    return trip.computePH(X, maxdim=D)


def get_roi(X: torch.Tensor, thresh: float = 0.03):

    true_points = torch.nonzero(X >= thresh)

    if true_points.numel() == 0:
        return [slice(None, None) for _ in range(X.ndim)]

    corner1 = true_points.min(dim=0)[0]
    corner2 = true_points.max(dim=0)[0]

    roi = [slice(int(c1), int(c2) + 1) for c1, c2 in zip(corner1, corner2)]
    return roi


def get_differentiable_barcode(tensor: torch.Tensor, barcode: np.ndarray):
    
    if barcode.size == 0:
        return None, [tensor.new_zeros((0,)) for _ in range(tensor.ndim)]

    inf = barcode[barcode[:, 2] == np.finfo(barcode.dtype).max]
    fin = barcode[barcode[:, 2] < np.finfo(barcode.dtype).max]

    inf_birth = None
    if len(inf) > 0:
        inf_birth = tensor[tuple(inf[:, 3:3 + tensor.ndim].astype(np.int64).T)]

    if len(fin) == 0:
        return inf_birth, [tensor.new_zeros((0,)) for _ in range(tensor.ndim)]

    births = tensor[tuple(fin[:, 3:3 + tensor.ndim].astype(np.int64).T)]
    deaths = tensor[tuple(fin[:, 6:6 + tensor.ndim].astype(np.int64).T)]
    delta_p = deaths - births

    delta_p = [delta_p[fin[:, 0] == d] for d in range(tensor.ndim)]
    delta_p = [
        torch.sort(v, descending=True)[0] if v.numel() > 0 else tensor.new_zeros((0,))
        for v in delta_p
    ]

    return inf_birth, delta_p



class AnatomicalPriorLoss(nn.Module):
    def __init__(
        self,
        config_path=None,
        construction='0',
        roi_thresh=0.03,
        use_softmax=True,
        reduction='none'
    ):
        super().__init__()

        self.cfg = {
            "enabled": True,
            "weight": 1.0,
            "roi_thresh": roi_thresh,
            "construction": construction,
            "use_softmax": use_softmax,
            "exclude_priors": [],
        }

        if config_path is None:
            config_path = os.environ.get("NNUNET_ANATOMICAL_CONFIG", None)

        if config_path is not None:
            with open(config_path, "r") as f:
                user_cfg = json.load(f)
            self.cfg.update(user_cfg)

        self.construction = self.cfg["construction"]
        self.roi_thresh = self.cfg["roi_thresh"]
        self.use_softmax = self.cfg["use_softmax"]
        self.reduction = reduction

        self.prior_specs = self._build_prior_specs(self.cfg.get("exclude_priors", []))
        self.prior_names = [s["name"] for s in self.prior_specs]

        self.PH = {
            '0': crip_wrapper,
            'N': trip_wrapper
        }
    
    def _build_prior_specs(self, exclude_priors):
        specs = [
            {"name": "SR_single",  "labels": (1,),   "betti": (2, 0, 0)},
            {"name": "LR_single",  "labels": (2,),   "betti": (2, 0, 0)},
            {"name": "MR_single",  "labels": (3,),   "betti": (2, 0, 0)},
            {"name": "IR_single",  "labels": (4,),   "betti": (2, 0, 0)},
            {"name": "ON_single",  "labels": (5,),   "betti": (2, 0, 0)},
            {"name": "EB_single",  "labels": (9,),   "betti": (2, 0, 0)},
            {"name": "ON_EB_pair", "labels": (5, 9), "betti": (2, 0, 0)},
        ]

        exclude_priors = set(exclude_priors or [])
        specs = [s for s in specs if s["name"] not in exclude_priors]
        return specs

    def _topo_loss_per_sample_per_prior(self, probs_b: torch.Tensor) -> torch.Tensor:
        device = probs_b.device
        spatial_ndim = probs_b.ndim - 1

        if self.roi_thresh is not None:
            fg_prob = probs_b[1:].sum(0)
            roi = get_roi(fg_prob, self.roi_thresh)
            probs_b = probs_b[(slice(None), *roi)]

        loss_list = []

        for spec in self.prior_specs:
            class_tuple = spec["labels"]
            expected_betti = spec["betti"]

            combo_prob = probs_b[list(class_tuple)].sum(0)
            combo = 1.0 - combo_prob

            combo_arr = combo.detach().cpu().numpy().astype(np.float64)
            max_dim = len(expected_betti)
            bcode = self.PH[self.construction](combo_arr, max_dim)

            max_features = bcode.shape[0] if getattr(bcode, "ndim", 0) > 1 else 1

            bcodes = torch.zeros(
                [spatial_ndim, max_features],
                dtype=probs_b.dtype,
                device=device
            )

            _, fin = get_differentiable_barcode(combo, bcode)
            for dim in range(spatial_ndim):
                if dim < len(fin) and fin[dim].numel() > 0:
                    bcodes[dim, :len(fin[dim])] = fin[dim]

            expected = torch.tensor(expected_betti, device=device, dtype=torch.long).clone()
            expected[0] = torch.clamp(expected[0] - 1, min=0)

            matching = torch.zeros_like(bcodes, dtype=torch.bool, device=device)
            for dim in range(min(spatial_ndim, len(expected))):
                n_keep = int(expected[dim].item())
                if n_keep > 0:
                    matching[dim, :n_keep] = True

            A = (1.0 - bcodes[matching]).sum() if matching.any() else probs_b.new_zeros(())
            Z = bcodes[~matching].sum() if (~matching).any() else probs_b.new_zeros(())

            loss_list.append(A + Z)

        return torch.stack(loss_list, dim=0)   # [P]
    
    def forward(self, logits_hr: torch.Tensor, case_ids=None) -> torch.Tensor:
        if self.use_softmax:
            probs = torch.softmax(logits_hr, dim=1)
        else:
            probs = logits_hr

        per_sample = []
        for b in range(probs.shape[0]):
            loss_bp = self._topo_loss_per_sample_per_prior(probs[b])   # [P]
            per_sample.append(loss_bp)

        per_prior_loss = torch.stack(per_sample, dim=0)   # [B, P]

        if self.reduction == 'mean':
            return per_prior_loss.mean()
        elif self.reduction == 'sum':
            return per_prior_loss.sum()
        return per_prior_loss