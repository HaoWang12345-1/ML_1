"""
Train Gaussian process surrogate models for cumulative PEG-NR datasets.

Examples
--------
python scripts/train_gp_models.py --data data/Rubber_PEG_ML_data_clean_table_final_35_backup.csv --tag GP0_n35 --expected-n 35 --set-default
python scripts/train_gp_models.py --data data/Rubber_PEG_ML_data_round4_LCBHVI_updated_50_for_GP.csv --tag GP4_n50 --expected-n 50 --set-default
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

import config
from src.gp_model import (
    load_gp_dataset,
    check_gp_dataset,
    loocv_all_targets,
    fit_gp_models,
    save_gp_models,
    save_gp_evaluation,
    TARGET_GP_SETTINGS,
)


DEFAULT_DATA_FILE = Path("data/Rubber_PEG_ML_data_clean_table_final.csv")
FAST_LOOCV = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GP LOOCV and final GP training for a cumulative PEG-NR dataset."
    )
    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="Input cumulative measured dataset CSV.",
    )
    parser.add_argument(
        "--tag",
        required=True,
        help="Version tag for outputs, for example GP0_n35 or GP4_n50.",
    )
    parser.add_argument(
        "--expected-n",
        required=True,
        type=int,
        help="Expected number of training rows. The script stops if this does not match.",
    )
    parser.add_argument(
        "--set-default",
        action="store_true",
        help="Copy --data to data/Rubber_PEG_ML_data_clean_table_final.csv for Step 3 compatibility.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fewer optimizer restarts for a quick test.",
    )
    return parser.parse_args()


def maybe_fast_settings(fast: bool) -> None:
    if fast or FAST_LOOCV:
        for target in TARGET_GP_SETTINGS:
            TARGET_GP_SETTINGS[target]["n_restarts_loo"] = 3
            TARGET_GP_SETTINGS[target]["n_restarts_final"] = 5


def validate_dataset(df: pd.DataFrame, expected_n: int) -> None:
    if len(df) != expected_n:
        raise ValueError(f"Expected {expected_n} rows, got {len(df)}")

    if "sample_id" not in df.columns:
        raise ValueError("Missing required column: sample_id")

    if df["sample_id"].duplicated().any():
        dup = df.loc[df["sample_id"].duplicated(), "sample_id"].tolist()
        raise ValueError(f"Duplicated sample_id detected: {dup}")

    if df[config.X_COLS].duplicated().any():
        dup = df.loc[df[config.X_COLS].duplicated(keep=False), ["sample_id"] + config.X_COLS]
        raise ValueError(f"Duplicated input X design detected:\n{dup}")

    if {"c_peg", "peg_wt_pct"}.issubset(df.columns):
        max_err = (df["c_peg"].astype(float) - df["peg_wt_pct"].astype(float) / 70.0).abs().max()
        print(f"c_peg = peg_wt_pct / 70 max error: {max_err:.3e}")
        if max_err > 1e-5:
            raise ValueError("c_peg encoding mismatch. Expected c_peg = peg_wt_pct / 70.")


def plot_parity(parity_df: pd.DataFrame, tag: str) -> None:
    fig_dir = config.OUTPUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for target, sub in parity_df.groupby("target"):
        fig, ax = plt.subplots(figsize=(4.8, 4.5))
        ax.errorbar(
            sub["y_true"],
            sub["y_pred"],
            yerr=sub["y_std"],
            fmt="o",
            capsize=2,
            alpha=0.8,
        )
        xy_min = min(sub["y_true"].min(), sub["y_pred"].min())
        xy_max = max(sub["y_true"].max(), sub["y_pred"].max())
        pad = (xy_max - xy_min) * 0.08 if xy_max > xy_min else 1.0
        ax.plot([xy_min - pad, xy_max + pad], [xy_min - pad, xy_max + pad], linestyle="--")
        ax.set_xlabel("Measured")
        ax.set_ylabel("LOOCV predicted")
        ax.set_title(f"GP LOOCV parity: {target} ({tag})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / f"gp_loocv_parity_{target}_{tag}.png", dpi=300)
        plt.close(fig)


def save_versioned_outputs(
    metrics_df: pd.DataFrame,
    parity_df: pd.DataFrame,
    gp_models: dict,
    data_file: Path,
    tag: str,
    expected_n: int,
) -> None:
    metrics_versioned = config.OUTPUT_DIR / f"gp_metrics_{tag}.csv"
    parity_versioned = config.OUTPUT_DIR / f"gp_parity_{tag}.csv"
    model_versioned = config.MODEL_DIR / f"gp_models_{tag}.pkl"
    meta_file = config.OUTPUT_DIR / f"gp_{tag}_metadata.txt"

    metrics_df.to_csv(metrics_versioned, index=False, encoding="utf-8-sig")
    parity_df.to_csv(parity_versioned, index=False, encoding="utf-8-sig")
    joblib.dump(gp_models, model_versioned)

    with meta_file.open("w", encoding="utf-8") as f:
        f.write("step: GP_sequential_unified\n")
        f.write(f"tag: {tag}\n")
        f.write(f"n_training_samples: {expected_n}\n")
        f.write(f"training_file: {data_file}\n")
        f.write("rule: cumulative measured dataset only; no virtual PSL candidates.\n")
        f.write("compatibility_outputs: gp_metrics_round_0.csv, gp_parity_round_0.csv, gp_models_round_0.pkl\n")

    print(f"Saved versioned metrics: {metrics_versioned}")
    print(f"Saved versioned parity:  {parity_versioned}")
    print(f"Saved versioned models:  {model_versioned}")
    print(f"Saved metadata:          {meta_file}")


def main() -> None:
    args = parse_args()
    maybe_fast_settings(args.fast)

    data_file = args.data
    tag = args.tag
    expected_n = args.expected_n

    print("=" * 80)
    print(f"Unified Step 2 GP: {tag}")
    print("=" * 80)

    if not data_file.exists():
        raise FileNotFoundError(f"Missing input data file: {data_file}")

    if args.set_default:
        shutil.copy2(data_file, DEFAULT_DATA_FILE)
        data_file = DEFAULT_DATA_FILE
        print(f"Copied input dataset to default Step 3 file: {DEFAULT_DATA_FILE}")

    df = load_gp_dataset(data_file)
    check_gp_dataset(df)
    validate_dataset(df, expected_n)

    print(f"\nTraining data file: {data_file}")
    print(f"Data shape: {df.shape}")
    print(f"Training rows: {len(df)}")

    print("\nRunning LOOCV...")
    metrics_df, parity_df = loocv_all_targets(df)

    metrics_file, parity_file = save_gp_evaluation(metrics_df, parity_df, round_id=0)
    print(f"Saved compatibility metrics: {metrics_file}")
    print(f"Saved compatibility parity:  {parity_file}")

    print("\nTraining final GP models on all current measured samples...")
    gp_models = fit_gp_models(df)
    model_file = save_gp_models(gp_models, round_id=0)
    print(f"Saved compatibility GP models: {model_file}")

    save_versioned_outputs(metrics_df, parity_df, gp_models, data_file, tag, expected_n)
    plot_parity(parity_df, tag)

    print("\nLOOCV summary:")
    show = ["target", "r2_loo", "rmse_loo", "mae_loo", "transform", "kernel_type", "matern_nu"]
    show = [c for c in show if c in metrics_df.columns]
    print(metrics_df[show].to_string(index=False))

    print("\nPASS: Unified GP training finished.")
    print("=" * 80)


if __name__ == "__main__":
    main()
