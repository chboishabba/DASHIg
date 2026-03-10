from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from phase2_validation.leech_modular_benchmark import TASKS, OPTIMIZERS, make_split


torch.set_num_threads(2)


@dataclass
class PlainBenchmarkConfig:
    p: int = 97
    task: str = "mul"
    optimizer: str = "adamw"
    seed: int = 0
    train_frac: float = 0.3
    epochs: int = 30000
    log_every: int = 20
    lr: float = 1e-3
    weight_decay: float = 0.25
    momentum: float = 0.9
    d_model: int = 192
    n_heads: int = 8
    n_layers: int = 2
    ff_mult: int = 2
    dropout: float = 0.0
    grok_thr: float = 0.97
    grok_patience_logs: int = 5
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"unsupported task: {self.task}")
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"unsupported optimizer: {self.optimizer}")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")


class PlainBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            bias=False,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + self.dropout(attn_out)
        x = x + self.ff(self.ln2(x))
        return x


class PlainModularClassifier(nn.Module):
    def __init__(self, cfg: PlainBenchmarkConfig) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(cfg.p, cfg.d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, 3, cfg.d_model))
        self.blocks = nn.ModuleList(
            [PlainBlock(cfg.d_model, cfg.n_heads, cfg.ff_mult, cfg.dropout) for _ in range(cfg.n_layers)]
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.p, bias=False)
        self.loss_fn = nn.CrossEntropyLoss()
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)

    def forward(self, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, steps = idx.shape
        if steps != 2:
            raise ValueError("expected modular pair inputs of shape [B, 2]")
        tok = self.token_emb(idx)
        cls = self.cls_token.expand(bsz, -1, -1)
        x = torch.cat([cls, tok], dim=1) + self.pos_emb[:, : 1 + steps]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.head(x[:, 0, :])
        return logits, x


@torch.no_grad()
def eval_metrics(model: PlainModularClassifier, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    model.eval()
    logits, hidden = model(x)
    loss = model.loss_fn(logits, y)
    preds = logits.argmax(-1)
    acc = float((preds == y).float().mean().item())
    hidden_norm = float(hidden.norm(dim=-1).mean().item())
    return {
        "acc": acc,
        "loss": float(loss.detach().cpu().item()),
        "total_loss": float(loss.detach().cpu().item()),
        "geo_loss": 0.0,
        "mean_resonance": 0.0,
        "max_resonance": hidden_norm,
    }


def first_epoch_at_or_none(epochs_logged: Iterable[int], values: Iterable[float], thr: float) -> int | None:
    for epoch, value in zip(epochs_logged, values):
        if value >= thr:
            return epoch
    return None


def make_optimizer(cfg: PlainBenchmarkConfig, model: nn.Module) -> torch.optim.Optimizer:
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            momentum=cfg.momentum,
        )
    raise ValueError(f"unsupported optimizer: {cfg.optimizer}")


def run_one(cfg: PlainBenchmarkConfig) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    x_train, y_train, x_test, y_test = make_split(cfg.p, cfg.train_frac, cfg.seed, cfg.task, cfg.device)
    model = PlainModularClassifier(cfg).to(cfg.device)
    optimizer = make_optimizer(cfg, model)

    epochs_logged: List[int] = []
    train_acc_log: List[float] = []
    test_acc_log: List[float] = []
    train_loss_log: List[float] = []
    test_loss_log: List[float] = []
    train_total_loss_log: List[float] = []
    test_total_loss_log: List[float] = []
    train_res_log: List[float] = []
    test_res_log: List[float] = []
    stop_epoch: int | None = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x_train)
        total_loss = model.loss_fn(logits, y_train)
        total_loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % cfg.log_every == 0:
            train_metrics = eval_metrics(model, x_train, y_train)
            test_metrics = eval_metrics(model, x_test, y_test)
            epochs_logged.append(epoch)
            train_acc_log.append(train_metrics["acc"])
            test_acc_log.append(test_metrics["acc"])
            train_loss_log.append(train_metrics["loss"])
            test_loss_log.append(test_metrics["loss"])
            train_total_loss_log.append(train_metrics["total_loss"])
            test_total_loss_log.append(test_metrics["total_loss"])
            train_res_log.append(train_metrics["max_resonance"])
            test_res_log.append(test_metrics["max_resonance"])

            if len(test_acc_log) >= cfg.grok_patience_logs:
                if min(test_acc_log[-cfg.grok_patience_logs :]) >= cfg.grok_thr:
                    stop_epoch = epoch
                    break

    summary = {
        "model_name": "plain_modular_transformer",
        "task": cfg.task,
        "optimizer": cfg.optimizer,
        "seed": cfg.seed,
        "p": cfg.p,
        "weight_decay": float(cfg.weight_decay),
        "epochs": cfg.epochs,
        "train_frac": cfg.train_frac,
        "lr": cfg.lr,
        "momentum": cfg.momentum if cfg.optimizer == "sgd" else "",
        "d_model": cfg.d_model,
        "n_heads": cfg.n_heads,
        "n_layers": cfg.n_layers,
        "ff_mult": cfg.ff_mult,
        "lambda_geo": "",
        "grok_thr": cfg.grok_thr,
        "grok_patience_logs": cfg.grok_patience_logs,
        "t_fit": first_epoch_at_or_none(epochs_logged, train_acc_log, 0.99),
        "t50": first_epoch_at_or_none(epochs_logged, test_acc_log, 0.50),
        "t95": first_epoch_at_or_none(epochs_logged, test_acc_log, 0.95),
        "stop_epoch": stop_epoch if stop_epoch is not None else epochs_logged[-1],
        "stopped_early": stop_epoch is not None,
        "final_train_loss": train_loss_log[-1],
        "final_test_loss": test_loss_log[-1],
        "final_train_total_loss": train_total_loss_log[-1],
        "final_test_total_loss": test_total_loss_log[-1],
        "final_train_acc": train_acc_log[-1],
        "final_test_acc": test_acc_log[-1],
        "final_train_resonance": train_res_log[-1],
        "final_test_resonance": test_res_log[-1],
    }

    trajectory: List[Dict[str, object]] = []
    for (
        epoch,
        train_loss,
        test_loss,
        train_total_loss,
        test_total_loss,
        train_acc,
        test_acc,
        train_res,
        test_res,
    ) in zip(
        epochs_logged,
        train_loss_log,
        test_loss_log,
        train_total_loss_log,
        test_total_loss_log,
        train_acc_log,
        test_acc_log,
        train_res_log,
        test_res_log,
    ):
        trajectory.append(
            {
                "model_name": "plain_modular_transformer",
                "task": cfg.task,
                "optimizer": cfg.optimizer,
                "seed": cfg.seed,
                "p": cfg.p,
                "weight_decay": float(cfg.weight_decay),
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_total_loss": train_total_loss,
                "test_total_loss": test_total_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "train_resonance": train_res,
                "test_resonance": test_res,
            }
        )

    return summary, trajectory
