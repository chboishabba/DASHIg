from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


torch.set_num_threads(2)


TASKS = {"mul", "add"}
OPTIMIZERS = {"adamw", "sgd"}


@dataclass
class LeechBenchmarkConfig:
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
    lambda_geo: float = 0.01
    resonance_threshold: float = 0.95
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
        head_dim = self.d_model // self.n_heads
        if head_dim % 24 != 0:
            raise ValueError("head_dim must be a multiple of 24")


def generate_leech_kernel(dim: int = 24) -> torch.Tensor:
    base = np.zeros((dim, dim), dtype=np.float32)
    for i in range(dim - 1):
        base[i, i] = 2.0
        base[i, i + 1] = 2.0
    base[-1, -1] = 2.0
    base[-1, 0] = -2.0
    q, _ = np.linalg.qr(base)
    return torch.from_numpy(q.astype(np.float32))


def modular_labels(ab: np.ndarray, p: int, task: str) -> np.ndarray:
    if task == "mul":
        return (ab[:, 0] * ab[:, 1]) % p
    if task == "add":
        return (ab[:, 0] + ab[:, 1]) % p
    raise ValueError(f"unsupported task: {task}")


def make_split(p: int, train_frac: float, seed: int, task: str, device: str) -> Tuple[torch.Tensor, ...]:
    pairs = np.array([(a, b) for a in range(p) for b in range(p)], dtype=np.int64)
    labels = modular_labels(pairs, p, task)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    n_train = int(train_frac * len(pairs))
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]
    x_train = torch.tensor(pairs[train_idx], device=device)
    y_train = torch.tensor(labels[train_idx], device=device)
    x_test = torch.tensor(pairs[test_idx], device=device)
    y_test = torch.tensor(labels[test_idx], device=device)
    return x_train, y_train, x_test, y_test


class LeechSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.num_blocks = self.head_dim // 24
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        kernel = generate_leech_kernel(24)
        self.register_buffer("leech_kernel", kernel)

    def _apply_kernel(self, x: torch.Tensor) -> torch.Tensor:
        bsz, heads, steps, dim = x.shape
        x = x.view(bsz, heads, steps, self.num_blocks, 24)
        x = torch.einsum("...i,ij->...j", x, self.leech_kernel)
        return x.reshape(bsz, heads, steps, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, steps, _ = x.shape
        qkv = self.qkv(x).reshape(bsz, steps, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self._apply_kernel(q)
        k = self._apply_kernel(k)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(bsz, steps, -1)
        return self.out(out)


class LeechBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = LeechSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class LeechResonanceHead(nn.Module):
    def __init__(self, d_model: int, lambda_geo: float) -> None:
        super().__init__()
        if d_model % 24 != 0:
            raise ValueError("d_model must be divisible by 24")
        self.lambda_geo = lambda_geo
        self.register_buffer("basis", generate_leech_kernel(24))
        self.ce = nn.CrossEntropyLoss()

    def resonance_stats(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, steps, dim = hidden.shape
        blocks = hidden.view(bsz, steps, dim // 24, 24)
        blocks = F.normalize(blocks, dim=-1)
        basis = F.normalize(self.basis, dim=-1)
        sim = torch.matmul(blocks, basis.T)
        max_sim = torch.max(sim, dim=-1).values
        return max_sim.mean(), max_sim.max()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        ce_loss = self.ce(logits, targets)
        mean_res, max_res = self.resonance_stats(hidden)
        geo_loss = 1.0 - mean_res
        total = ce_loss + self.lambda_geo * geo_loss
        return total, {
            "ce_loss": float(ce_loss.detach().cpu().item()),
            "geo_loss": float(geo_loss.detach().cpu().item()),
            "mean_resonance": float(mean_res.detach().cpu().item()),
            "max_resonance": float(max_res.detach().cpu().item()),
        }


class LeechModularClassifier(nn.Module):
    def __init__(self, cfg: LeechBenchmarkConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.p, cfg.d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, 3, cfg.d_model))
        self.blocks = nn.ModuleList(
            [LeechBlock(cfg.d_model, cfg.n_heads, cfg.ff_mult, cfg.dropout) for _ in range(cfg.n_layers)]
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.p, bias=False)
        self.resonance = LeechResonanceHead(cfg.d_model, cfg.lambda_geo)
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
def eval_metrics(model: LeechModularClassifier, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    model.eval()
    logits, hidden = model(x)
    total_loss, extras = model.resonance(logits, y, hidden)
    preds = logits.argmax(-1)
    acc = float((preds == y).float().mean().item())
    return {
        "acc": acc,
        "loss": extras["ce_loss"],
        "total_loss": float(total_loss.detach().cpu().item()),
        "geo_loss": extras["geo_loss"],
        "mean_resonance": extras["mean_resonance"],
        "max_resonance": extras["max_resonance"],
    }


def first_epoch_at_or_none(epochs_logged: Iterable[int], values: Iterable[float], thr: float) -> int | None:
    for epoch, value in zip(epochs_logged, values):
        if value >= thr:
            return epoch
    return None


def make_optimizer(cfg: LeechBenchmarkConfig, model: nn.Module) -> torch.optim.Optimizer:
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


def run_one(cfg: LeechBenchmarkConfig) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    x_train, y_train, x_test, y_test = make_split(cfg.p, cfg.train_frac, cfg.seed, cfg.task, cfg.device)
    model = LeechModularClassifier(cfg).to(cfg.device)
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
        logits, hidden = model(x_train)
        total_loss, _ = model.resonance(logits, y_train, hidden)
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
            train_res_log.append(train_metrics["mean_resonance"])
            test_res_log.append(test_metrics["mean_resonance"])

            if len(test_acc_log) >= cfg.grok_patience_logs:
                if min(test_acc_log[-cfg.grok_patience_logs :]) >= cfg.grok_thr:
                    stop_epoch = epoch
                    break

    summary = {
        "model_name": "leech_modular_classifier",
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
        "lambda_geo": cfg.lambda_geo,
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
                "model_name": "leech_modular_classifier",
                "task": cfg.task,
                "optimizer": cfg.optimizer,
                "seed": cfg.seed,
                "p": cfg.p,
                "weight_decay": float(cfg.weight_decay),
                "grok_thr": cfg.grok_thr,
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
