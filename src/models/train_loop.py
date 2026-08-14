"""CNN training loop, checkpointing, and OOS inference."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from config import ADAM_LR, BATCH_SIZE, EARLY_STOP_PATIENCE, MAX_EPOCHS
from models.cnn import PriceTrendCNN, copy_conv_blocks_from, flatten_feature_size
from models.dataset import ImageLabelDataset, SampleRef

LAST_CKPT = "last.pt"
BEST_CKPT = "best.pt"


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
    init_from_image_days: int | None = None


def _device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ckpt_dir_paths(ckpt_dir: Path) -> tuple[Path, Path]:
    return ckpt_dir / LAST_CKPT, ckpt_dir / BEST_CKPT


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _write_meta(path: Path, payload: dict) -> None:
    meta = {k: v for k, v in payload.items() if k not in ("model_state", "optimizer_state")}
    if "config" in meta and isinstance(meta["config"], dict):
        pass
    else:
        meta = dict(meta)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))


def training_is_finished(ckpt_dir: Path) -> bool:
    last_path, best_path = ckpt_dir_paths(ckpt_dir)
    if last_path.is_file():
        payload = torch.load(last_path, map_location="cpu", weights_only=False)
        return bool(payload.get("finished", False))
    return best_path.is_file()


def iter_index_batches(n: int, batch_size: int, shuffle: bool):
    """Index batches matching PyTorch DataLoader RandomSampler + BatchSampler.

    DataLoader(num_workers=0) draws _base_seed when the iterator is created,
    then RandomSampler draws its private generator seed on the first batch.
    Dropout during training uses the global RNG after those draws, so this
    order must stay the same.
    """
    torch.empty((), dtype=torch.int64).random_().item()
    if shuffle:
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator()
        generator.manual_seed(seed)
        perm = torch.randperm(n, generator=generator).numpy()
        for start in range(0, n, batch_size):
            yield perm[start : start + batch_size]
        return
    for start in range(0, n, batch_size):
        yield np.arange(start, min(start + batch_size, n), dtype=np.int64)


def _move_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    pin: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pin:
        x = x.pin_memory()
        y = y.pin_memory()
    return x.to(device, non_blocking=pin), y.to(device, non_blocking=pin)


def run_epoch(
    model: PriceTrendCNN,
    dataset: ImageLabelDataset,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    batch_size: int,
    shuffle: bool,
) -> float:
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    pin = device.type == "cuda"
    total_loss = 0.0
    n_batches = 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for indices in iter_index_batches(len(dataset), batch_size, shuffle):
            x, y = dataset.batch_xy(indices)
            x, y = _move_batch(x, y, device, pin)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
    return total_loss / max(n_batches, 1)


def save_best_checkpoint(
    path: Path,
    model: PriceTrendCNN,
    config: TrainConfig,
    val_loss: float,
    epoch: int,
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "config": asdict(config),
        "val_loss": val_loss,
        "epoch": epoch,
        "finished": True,
    }
    _atomic_torch_save(payload, path)
    _write_meta(path, payload)


def save_training_state(
    path: Path,
    *,
    model: PriceTrendCNN,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    epoch: int,
    train_loss: float,
    val_loss: float,
    best_val: float,
    best_epoch: int,
    stale_epochs: int,
    finished: bool,
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": asdict(config),
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val": best_val,
        "best_epoch": best_epoch,
        "stale_epochs": stale_epochs,
        "finished": finished,
    }
    _atomic_torch_save(payload, path)
    _write_meta(path, payload)


def load_checkpoint(path: Path, device: torch.device) -> tuple[PriceTrendCNN, TrainConfig]:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = TrainConfig(**payload["config"])
    model = PriceTrendCNN(config.image_days)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    return model, config


def init_model_from_transfer(
    model: PriceTrendCNN,
    init_checkpoint: Path,
    device: torch.device,
) -> None:
    payload = torch.load(init_checkpoint, map_location=device, weights_only=False)
    source_config = TrainConfig(**payload["config"])
    copy_conv_blocks_from(model, payload["model_state"], source_config.image_days)


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
    ckpt_dir: Path,
    init_from_checkpoint: Path | None = None,
    init_from_image_days: int | None = None,
    device: str | None = None,
    batch_size: int = BATCH_SIZE,
    lr: float = ADAM_LR,
    patience: int = EARLY_STOP_PATIENCE,
    max_epochs: int = MAX_EPOCHS,
    log_fn=print,
) -> PriceTrendCNN:
    dev = _device(device)
    last_path, best_path = ckpt_dir_paths(ckpt_dir)

    flat = flatten_feature_size(image_days)
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
        init_from_image_days=init_from_image_days,
    )

    resume_payload: dict | None = None
    if last_path.is_file():
        resume_payload = torch.load(last_path, map_location=dev, weights_only=False)
        if resume_payload.get("finished", False):
            log_fn(f"training finished; load {best_path}")
            model, _ = load_checkpoint(best_path, dev)
            return model

    if resume_payload is not None:
        log_fn(f"resume training from {last_path} epoch={resume_payload['epoch']}")
        model = PriceTrendCNN(image_days).to(dev)
        model.load_state_dict(resume_payload["model_state"])
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer.load_state_dict(resume_payload["optimizer_state"])
        start_epoch = int(resume_payload["epoch"])
        best_val = float(resume_payload["best_val"])
        best_epoch = int(resume_payload["best_epoch"])
        stale_epochs = int(resume_payload["stale_epochs"])
    else:
        if best_path.is_file():
            log_fn(f"checkpoint exists, skip training: {best_path}")
            model, _ = load_checkpoint(best_path, dev)
            return model

        torch.manual_seed(seed)
        model = PriceTrendCNN(image_days).to(dev)
        log_fn(
            f"model I{image_days} flatten={flat} "
            f"params={sum(p.numel() for p in model.parameters())}"
        )
        if init_from_checkpoint is not None:
            log_fn(f"init conv blocks from {init_from_checkpoint}")
            init_model_from_transfer(model, init_from_checkpoint, dev)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        start_epoch = 0
        best_val = float("inf")
        best_epoch = -1
        stale_epochs = 0

    for epoch in range(start_epoch, max_epochs):
        train_loss = run_epoch(
            model, train_ds, criterion, optimizer, dev, batch_size, True
        )
        val_loss = run_epoch(
            model, val_ds, criterion, None, dev, batch_size, False
        )
        epoch_num = epoch + 1
        log_fn(
            f"epoch={epoch_num} train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch_num
            stale_epochs = 0
            save_best_checkpoint(best_path, model, config, best_val, best_epoch)
            log_fn(f"new best epoch={best_epoch} val_loss={best_val:.6f} -> {best_path}")
        else:
            stale_epochs += 1

        finished = False
        if stale_epochs >= patience:
            log_fn(f"early stop at epoch={epoch_num} best_epoch={best_epoch}")
            finished = True
        elif epoch_num >= max_epochs:
            finished = True

        save_training_state(
            last_path,
            model=model,
            optimizer=optimizer,
            config=config,
            epoch=epoch_num,
            train_loss=train_loss,
            val_loss=val_loss,
            best_val=best_val,
            best_epoch=best_epoch,
            stale_epochs=stale_epochs,
            finished=finished,
        )

        if finished:
            break

    if best_epoch < 0:
        raise RuntimeError("training finished without any validation epoch")

    model, _ = load_checkpoint(best_path, dev)
    log_fn(f"training done best_epoch={best_epoch} val_loss={best_val:.6f}")
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
    pin = dev.type == "cuda"
    rows: list[dict] = []
    for indices in iter_index_batches(len(dataset), batch_size, False):
        x, _y = dataset.batch_xy(indices)
        x, _y = _move_batch(x, _y, dev, pin)
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        refs = dataset.batch_refs(indices)
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
