"""The same named matrix selection and AdamW fallback in every study arm."""
import torch
from torch import nn
from pytorch_opt import KFAC, Muon, Shampoo, SOAP, matrix_param_groups

OPTIMIZERS = ("adamw", "muon", "shampoo", "kfac")


def optimizer_for(model, name, *, lr, fallback_lr, weight_decay=0.1,
                  refresh=10, selection="all", damping=1e-3):
    if name not in (*OPTIMIZERS, "soap"):
        raise ValueError(f"Unknown optimizer: {name}")
    if selection not in ("all", "mamba_output"):
        raise ValueError("selection must be all or mamba_output")
    selected = [n for n, m in model.named_modules()
                if isinstance(m, nn.Linear) and n != "lm_head"
                and (selection == "all" or n.endswith("mixer.out_proj"))]
    if not selected:
        raise ValueError("No selected matrices; mamba_output requires a Mamba model")
    groups = matrix_param_groups(model, selected, lr=lr, adamw_lr=fallback_lr,
                                 weight_decay=weight_decay, adamw_wd=weight_decay)
    common = dict(adamw_betas=(0.9, 0.95), adamw_eps=1e-8)
    if name == "adamw":
        if lr != fallback_lr:
            raise ValueError("The AdamW control uses one peak LR on all groups")
        opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8,
                                foreach=False, fused=False)
    elif name == "muon":
        opt = Muon(groups, momentum=0.95, ns_steps=5, lr_adjust="spectral", **common)
    elif name == "shampoo":
        opt = Shampoo(groups, max_preconditioner_dim=8192,
                       precondition_frequency=refresh, root_dtype=torch.float64,
                       track_cond=False, **common)
    elif name == "soap":
        opt = SOAP(groups, max_preconditioner_dim=8192,
                    precondition_frequency=refresh, **common)
    else:
        opt = KFAC(model, params=groups, fisher_mode="sampled", damping=damping,
                   stats_every=refresh, inv_every=refresh, **common)
    routing = [dict(selected=g["use_preconditioner"], names=g["param_names"],
                    parameters=sum(p.numel() for p in g["params"]),
                    peak_lr=g["lr"], weight_decay=g["weight_decay"])
               for g in opt.param_groups]
    return opt, routing
