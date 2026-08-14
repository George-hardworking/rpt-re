"""CLI: train panel CNN on memmap images and write OOS P(up) predictions."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from config import BATCH_SIZE, MODELS_ROOT, N_ENSEMBLE, TRAIN_VAL_SPLIT_SEED, WINDOW_DAYS
from models.dataset import (
    ImageLabelDataset,
    collect_samples,
    compute_pixel_mean_std,
    model_run_dir,
    split_samples,
)
from models.train_loop import load_checkpoint, predict_dataset, train_model


def log(msg: str) -> None:
    print(msg, flush=True)


def run_dir(models_root: Path, image_days: int, horizon: int) -> Path:
    return models_root / f"I{image_days}_R{horizon}"


def ensemble_pred_path(models_root: Path, image_days: int, horizon: int) -> Path:
    return run_dir(models_root, image_days, horizon) / "ensemble_pred.feather"


def seed_pred_path(models_root: Path, image_days: int, horizon: int, seed: int) -> Path:
    return run_dir(models_root, image_days, horizon) / f"seed{seed}_pred.feather"


def merge_ensemble(
    models_root: Path,
    image_days: int,
    horizon: int,
    seeds: list[int],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for seed in seeds:
        path = seed_pred_path(models_root, image_days, horizon, seed)
        if not path.exists():
            raise FileNotFoundError(f"missing seed prediction: {path}")
        frame = pd.read_feather(path)[["PERMNO", "Date", "p_up"]].rename(
            columns={"p_up": f"p_seed{seed}"}
        )
        frames.append(frame)
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["PERMNO", "Date"])
    p_cols = [f"p_seed{seed}" for seed in seeds]
    merged["p_up"] = merged[p_cols].mean(axis=1)
    return merged[["PERMNO", "Date", "p_up"]]


def prepare_datasets(
    images_root: Path,
    image_days: int,
    horizon: int,
    *,
    split_seed: int,
    year_limit: int | None,
    max_samples: int | None,
) -> tuple[ImageLabelDataset, ImageLabelDataset, ImageLabelDataset, int]:
    samples = collect_samples(
        images_root,
        image_days,
        horizon,
        year_limit=year_limit,
        max_samples=max_samples,
    )
    train_samples, val_samples, test_samples = split_samples(samples, seed=split_seed)
    log(
        f"I{image_days}/R{horizon} split_seed={split_seed} samples "
        f"train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}"
    )

    pixel_mean, pixel_std = compute_pixel_mean_std(images_root, image_days, train_samples)
    log(f"pixel_mean={pixel_mean:.6f} pixel_std={pixel_std:.6f}")

    train_ds = ImageLabelDataset(images_root, image_days, train_samples, pixel_mean, pixel_std)
    val_ds = ImageLabelDataset(images_root, image_days, val_samples, pixel_mean, pixel_std)
    test_ds = ImageLabelDataset(images_root, image_days, test_samples, pixel_mean, pixel_std)
    return train_ds, val_ds, test_ds, len(test_samples)


def train_one_seed(
    *,
    train_ds: ImageLabelDataset,
    val_ds: ImageLabelDataset,
    test_ds: ImageLabelDataset,
    n_test: int,
    models_root: Path,
    image_days: int,
    horizon: int,
    seed: int,
    device: str | None,
    batch_size: int,
    fresh: bool,
    skip_train: bool,
    skip_predict: bool,
) -> None:
    ckpt_dir = model_run_dir(models_root, image_days, horizon, seed)
    ckpt_path = ckpt_dir / "best.pt"
    pred_path = seed_pred_path(models_root, image_days, horizon, seed)

    if fresh and ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    if fresh and pred_path.exists():
        pred_path.unlink()

    log(f"I{image_days}/R{horizon} train_seed={seed}")

    if skip_train and ckpt_path.exists():
        log(f"skip train; load {ckpt_path}")
        from models.train_loop import _device

        model, _cfg = load_checkpoint(ckpt_path, _device(device))
    elif ckpt_path.exists():
        log(f"checkpoint exists, skip training: {ckpt_path}")
        from models.train_loop import _device

        model, _cfg = load_checkpoint(ckpt_path, _device(device))
    else:
        model = train_model(
            train_ds,
            val_ds,
            image_days=image_days,
            horizon=horizon,
            seed=seed,
            pixel_mean=train_ds.pixel_mean,
            pixel_std=train_ds.pixel_std,
            n_test=n_test,
            checkpoint_path=ckpt_path,
            device=device,
            batch_size=batch_size,
            log_fn=log,
        )

    if skip_predict and pred_path.exists():
        log(f"skip predict; existing {pred_path}")
        return

    pred_df = predict_dataset(model, test_ds, device=device, batch_size=batch_size)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_feather(pred_path)
    log(f"saved OOS predictions n={len(pred_df)} -> {pred_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN on OHLC images")
    parser.add_argument("--images", type=Path, default=None, help="processed/images root")
    parser.add_argument("--models", type=Path, default=MODELS_ROOT)
    parser.add_argument("--image-days", type=int, required=True, choices=WINDOW_DAYS)
    parser.add_argument("--horizon", type=int, required=True, choices=WINDOW_DAYS)
    parser.add_argument("--seed", type=int, default=0, help="optimization seed (init / shuffle)")
    parser.add_argument(
        "--split-seed",
        type=int,
        default=TRAIN_VAL_SPLIT_SEED,
        help="fixed 70/30 train/val split seed (shared across ensemble members)",
    )
    parser.add_argument(
        "--all-seeds",
        action="store_true",
        help=f"train/predict seeds 0..{N_ENSEMBLE - 1} and write ensemble_pred.feather",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--year-limit", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-predict", action="store_true")
    args = parser.parse_args()

    from config import IMAGES_ROOT

    images_root = args.images if args.images is not None else IMAGES_ROOT
    seeds = list(range(N_ENSEMBLE)) if args.all_seeds else [args.seed]

    if args.fresh:
        target = run_dir(args.models, args.image_days, args.horizon)
        if target.exists():
            shutil.rmtree(target)

    train_ds, val_ds, test_ds, n_test = prepare_datasets(
        images_root,
        args.image_days,
        args.horizon,
        split_seed=args.split_seed,
        year_limit=args.year_limit,
        max_samples=args.max_samples,
    )

    for seed in seeds:
        train_one_seed(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            n_test=n_test,
            models_root=args.models,
            image_days=args.image_days,
            horizon=args.horizon,
            seed=seed,
            device=args.device,
            batch_size=args.batch_size,
            fresh=False,
            skip_train=args.skip_train,
            skip_predict=args.skip_predict,
        )

    if args.all_seeds:
        ensemble = merge_ensemble(args.models, args.image_days, args.horizon, seeds)
        out_path = ensemble_pred_path(args.models, args.image_days, args.horizon)
        ensemble.to_feather(out_path)
        log(f"ensemble n={len(ensemble)} -> {out_path}")


if __name__ == "__main__":
    main()
