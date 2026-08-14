"""CNN training loop, checkpointing, and OOS inference."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import ADAM_LR, BATCH_SIZE, EARLY_STOP_PATIENCE, MAX_EPOCHS
from models.cnn import PriceTrendCNN, flatten_feature_size
from models.dataset import ImageLabelDataset, SampleRef


def _collate_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor, SampleRef]],
) -> tuple[torch.Tensor, torch.Tensor, list[SampleRef]]:
    xs, ys, refs = zip(*batch)
    return torch.stack(xs), torch.stack(ys), list(refs)


@dataclass
class TrainConfig:
    image_days: int
    horizon: int
    seed: int
    pixel_mean: float
    pixel_std: float
    flatten_size: int
    n_train: int
    n_val: int
    n_test: int


def _device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_epoch(
    model: PriceTrendCNN,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    n_batches = 0
    for x, y, _ref in loader:
        x = x.to(device)
        y = y.to(device)
        if is_train:
            optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        if is_train:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


def save_checkpoint(
    path: Path,
    model: PriceTrendCNN,
    config: TrainConfig,
    val_loss: float,
    epoch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(config),
            "val_loss": val_loss,
            "epoch": epoch,
        },
        path,
    )
    meta_path = path.with_suffix(".json")
    meta_path.write_text(
        json.dumps(
            {
                **asdict(config),
                "val_loss": val_loss,
                "epoch": epoch,
            },
            indent=2,
        )
    )


def load_checkpoint(path: Path, device: torch.device) -> tuple[PriceTrendCNN, TrainConfig]:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = TrainConfig(**payload["config"])
    model = PriceTrendCNN(config.image_days)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    return model, config


def train_model(
    train_ds: ImageLabelDataset,
    val_ds: ImageLabelDataset,
    *,
    image_days: int,
    horizon: int,
    seed: int,
    pixel_mean: float,
    pixel_std: float,
    n_test: int,
    checkpoint_path: Path,
    device: str | None = None,
    batch_size: int = BATCH_SIZE,
    lr: float = ADAM_LR,
    patience: int = EARLY_STOP_PATIENCE,
    max_epochs: int = MAX_EPOCHS,
    log_fn=print,
) -> PriceTrendCNN:
    dev = _device(device)
    torch.manual_seed(seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate_batch
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate_batch
    )

    model = PriceTrendCNN(image_days).to(dev)
    flat = flatten_feature_size(image_days)
    log_fn(
        f"model I{image_days} flatten={flat} params={sum(p.numel() for p in model.parameters())}"
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    config = TrainConfig(
        image_days=image_days,
        horizon=horizon,
        seed=seed,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
        flatten_size=flat,
        n_train=len(train_ds),
        n_val=len(val_ds),
        n_test=n_test,
    )

    best_val = float("inf")
    best_epoch = -1
    stale_epochs = 0
    best_state = None

    for epoch in range(max_epochs):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, dev)
        val_loss = run_epoch(model, val_loader, criterion, None, dev)
        log_fn(
            f"epoch={epoch + 1} train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
        )
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            stale_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                log_fn(f"early stop at epoch={epoch + 1} best_epoch={best_epoch}")
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    save_checkpoint(checkpoint_path, model, config, best_val, best_epoch)
    log_fn(f"saved checkpoint epoch={best_epoch} val_loss={best_val:.6f} -> {checkpoint_path}")
    return model


@torch.no_grad()
def predict_dataset(
    model: PriceTrendCNN,
    dataset: ImageLabelDataset,
    device: str | None = None,
    batch_size: int = BATCH_SIZE,
) -> pd.DataFrame:
    dev = _device(device)
    model.eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate_batch
    )
    rows: list[dict] = []
    for x, _y, refs in loader:
        x = x.to(dev)
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        for prob, ref in zip(probs, refs):
            ref: SampleRef
            rows.append(
                {
                    "PERMNO": ref.permno,
                    "Date": ref.date,
                    "p_up": float(prob),
                    "label": ref.label,
                }
            )
    return pd.DataFrame(rows)
