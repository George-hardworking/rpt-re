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
from models.cnn import cnn_num_blocks
from models.dataset import (
    ImageLabelDataset,
    collect_samples,
    compute_pixel_mean_std,
    model_run_dir,
    model_run_tag,
    split_samples,
)
from models.train_loop import (
    BEST_CKPT,
    ckpt_dir_paths,
    load_checkpoint,
    predict_dataset,
    train_model,
    training_is_finished,
)


def log(msg: str) -> None:
    print(msg, flush=True)


def run_dir(
    models_root: Path,
    image_days: int,
    horizon: int,
    *,
    init_from_image_days: int | None = None,
) -> Path:
    return models_root / model_run_tag(
        image_days, horizon, init_from_image_days=init_from_image_days
    )


def ensemble_pred_path(
    models_root: Path,
    image_days: int,
    horizon: int,
    *,
    init_from_image_days: int | None = None,
) -> Path:
    return run_dir(
        models_root, image_days, horizon, init_from_image_days=init_from_image_days
    ) / "ensemble_pred.feather"


def seed_pred_path(
    models_root: Path,
    image_days: int,
    horizon: int,
    seed: int,
    *,
    init_from_image_days: int | None = None,
) -> Path:
    return run_dir(
        models_root, image_days, horizon, init_from_image_days=init_from_image_days
    ) / f"seed{seed}_pred.feather"


def validate_init_from(image_days: int, init_from_image_days: int) -> None:
    if init_from_image_days not in WINDOW_DAYS:
        raise ValueError(f"unsupported --init-from-image-days={init_from_image_days}")
    if cnn_num_blocks(init_from_image_days) + 1 != cnn_num_blocks(image_days):
        raise ValueError(
            f"layer transfer requires source blocks + 1 = target blocks; "
            f"I{init_from_image_days} has {cnn_num_blocks(init_from_image_days)} blocks, "
            f"I{image_days} has {cnn_num_blocks(image_days)}"
        )


def init_checkpoint_path(
    models_root: Path,
    init_from_image_days: int,
    horizon: int,
    seed: int,
) -> Path:
    src_dir = model_run_dir(models_root, init_from_image_days, horizon, seed)
    return src_dir / BEST_CKPT


def merge_ensemble(
    models_root: Path,
    image_days: int,
    horizon: int,
    seeds: list[int],
    *,
    init_from_image_days: int | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for seed in seeds:
        path = seed_pred_path(
            models_root,
            image_days,
            horizon,
            seed,
            init_from_image_days=init_from_image_days,
        )
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
    init_from_image_days: int | None,
) -> None:
    ckpt_dir = model_run_dir(
        models_root,
        image_days,
        horizon,
        seed,
        init_from_image_days=init_from_image_days,
    )
    _, best_path = ckpt_dir_paths(ckpt_dir)
    pred_path = seed_pred_path(
        models_root,
        image_days,
        horizon,
        seed,
        init_from_image_days=init_from_image_days,
    )

    if fresh and ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    if fresh and pred_path.exists():
        pred_path.unlink()

    tag = model_run_tag(image_days, horizon, init_from_image_days=init_from_image_days)
    log(f"{tag} train_seed={seed}")

    init_from_checkpoint: Path | None = None
    if init_from_image_days is not None:
        init_from_checkpoint = init_checkpoint_path(
            models_root, init_from_image_days, horizon, seed
        )
        if not init_from_checkpoint.is_file():
            raise FileNotFoundError(
                f"missing init checkpoint for layer transfer: {init_from_checkpoint}"
            )

    if skip_train and training_is_finished(ckpt_dir):
        log(f"skip train; load {best_path}")
        from models.train_loop import _device

        model, _cfg = load_checkpoint(best_path, _device(device))
    elif training_is_finished(ckpt_dir):
        log(f"training finished; load {best_path}")
        from models.train_loop import _device

        model, _cfg = load_checkpoint(best_path, _device(device))
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
            ckpt_dir=ckpt_dir,
            init_from_checkpoint=init_from_checkpoint,
            init_from_image_days=init_from_image_days,
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
    parser.add_argument(
        "--init-from-image-days",
        type=int,
        default=None,
        choices=WINDOW_DAYS,
        help="layer transfer: init conv blocks from I{src}/R{horizon} best.pt (5->20, 20->60)",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--year-limit", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-predict", action="store_true")
    args = parser.parse_args()

    if args.init_from_image_days is not None:
        validate_init_from(args.image_days, args.init_from_image_days)

    from config import IMAGES_ROOT

    images_root = args.images if args.images is not None else IMAGES_ROOT
    seeds = list(range(N_ENSEMBLE)) if args.all_seeds else [args.seed]

    if args.fresh:
        target = run_dir(
            args.models,
            args.image_days,
            args.horizon,
            init_from_image_days=args.init_from_image_days,
        )
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
            init_from_image_days=args.init_from_image_days,
        )

    if args.all_seeds:
        ensemble = merge_ensemble(
            args.models,
            args.image_days,
            args.horizon,
            seeds,
            init_from_image_days=args.init_from_image_days,
        )
        out_path = ensemble_pred_path(
            args.models,
            args.image_days,
            args.horizon,
            init_from_image_days=args.init_from_image_days,
        )
        ensemble.to_feather(out_path)
        log(f"ensemble n={len(ensemble)} -> {out_path}")


if __name__ == "__main__":
    main()
