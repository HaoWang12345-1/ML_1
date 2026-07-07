"""
Run GP-guided Pareto set learning for the PEG-NR design space.

The script loads the current GP surrogate, generates preference-guided
candidate designs, and exports library predictions, Pareto candidates, and
summary figures.
"""

from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from config import INITIAL_DATA_FILE, OUTPUT_DIR, MODEL_DIR, N_PREFERENCE_VECTORS
from src.gp_model import load_gp_dataset, check_gp_dataset, fit_gp_models, save_gp_models
from src.psl_mlp_model import generate_psl_candidates_from_library, summarize_psl_candidates


def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_or_fit_gp_models(df, round_id=0):
    model_file = MODEL_DIR / f"gp_models_round_{round_id}.pkl"
    if model_file.exists():
        print(f"\nLoading existing GP models: {model_file}")
        return joblib.load(model_file)

    print("\nNo GP model file found. Fitting final GP models using all input data...")
    gp_models = fit_gp_models(df)
    saved = save_gp_models(gp_models, round_id=round_id)
    print(f"Saved GP models to: {saved}")
    return gp_models


def save_outputs(psl_df, library_pred_df, pareto_df, round_id=0):
    psl_csv = OUTPUT_DIR / f"psl_candidates_1000_round_{round_id}.csv"
    library_csv = OUTPUT_DIR / f"library_predictions_round_{round_id}.csv"
    pareto_csv = OUTPUT_DIR / f"predicted_pareto_front_round_{round_id}.csv"
    summary_csv = OUTPUT_DIR / f"psl_summary_round_{round_id}.csv"

    psl_df.to_csv(psl_csv, index=False, encoding="utf-8-sig")
    library_pred_df.to_csv(library_csv, index=False, encoding="utf-8-sig")
    pareto_df.to_csv(pareto_csv, index=False, encoding="utf-8-sig")

    summary_items = [
        {"item": "n_psl_candidates", "value": len(psl_df)},
        {"item": "n_unique_snapped_designs", "value": psl_df["design_key"].nunique() if "design_key" in psl_df.columns else "NA"},
        {"item": "n_full_library", "value": len(library_pred_df)},
        {"item": "n_predicted_pareto_front", "value": len(pareto_df)},
        {"item": "snapped_peg_wt_pct_min", "value": psl_df["peg_wt_pct"].min()},
        {"item": "snapped_peg_wt_pct_max", "value": psl_df["peg_wt_pct"].max()},
        {"item": "snapped_modulus_pred_min", "value": psl_df["modulus_mpa_pred"].min()},
        {"item": "snapped_modulus_pred_max", "value": psl_df["modulus_mpa_pred"].max()},
        {"item": "snapped_gc_pred_min", "value": psl_df["gc_j_m2_pred"].min()},
        {"item": "snapped_gc_pred_max", "value": psl_df["gc_j_m2_pred"].max()},
        {"item": "snapped_gth_pred_min", "value": psl_df["gth_app_j_m2_pred"].min()},
        {"item": "snapped_gth_pred_max", "value": psl_df["gth_app_j_m2_pred"].max()},
    ]

    if "modulus_mpa_pred_continuous" in psl_df.columns:
        summary_items += [
            {"item": "continuous_peg_wt_pct_min", "value": psl_df["peg_wt_pct_continuous"].min()},
            {"item": "continuous_peg_wt_pct_max", "value": psl_df["peg_wt_pct_continuous"].max()},
            {"item": "continuous_modulus_pred_min", "value": psl_df["modulus_mpa_pred_continuous"].min()},
            {"item": "continuous_modulus_pred_max", "value": psl_df["modulus_mpa_pred_continuous"].max()},
            {"item": "continuous_gc_pred_min", "value": psl_df["gc_j_m2_pred_continuous"].min()},
            {"item": "continuous_gc_pred_max", "value": psl_df["gc_j_m2_pred_continuous"].max()},
            {"item": "continuous_gth_pred_min", "value": psl_df["gth_app_j_m2_pred_continuous"].min()},
            {"item": "continuous_gth_pred_max", "value": psl_df["gth_app_j_m2_pred_continuous"].max()},
        ]

    summary = pd.DataFrame(summary_items)
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print("\nSaved Step 3 output tables:")
    print(f"  {psl_csv}")
    print(f"  {library_csv}")
    print(f"  {pareto_csv}")
    print(f"  {summary_csv}")


def plot_loss(round_id=0):
    loss_file = OUTPUT_DIR / f"psl_training_loss_round_{round_id}.csv"
    if not loss_file.exists():
        print("Loss file not found; skipping loss plot.")
        return

    loss_df = pd.read_csv(loss_file)
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(loss_df["step"], loss_df["psl_loss"], marker="o", markersize=3, linewidth=1)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Augmented Tchebycheff loss")
    ax.set_title("PSL training loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = OUTPUT_DIR / "figures" / f"figure_psl_loss_curve_round_{round_id}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved loss plot: {out}")



def lambda_simplex_xy(psl_df):
    x = psl_df["lambda_gc"] + 0.5 * psl_df["lambda_gth"]
    y = (3 ** 0.5 / 2.0) * psl_df["lambda_gth"]
    return x, y


def get_continuous_or_snapped_cols(psl_df):
    if "modulus_mpa_pred_continuous" in psl_df.columns:
        return {
            "design_x": "c_peg_continuous",
            "design_y": "i_ti_continuous",
            "design_z": "d_el_continuous",
            "modulus": "modulus_mpa_pred_continuous",
            "gc": "gc_j_m2_pred_continuous",
            "gth": "gth_app_j_m2_pred_continuous",
            "balanced": "balanced_obj_continuous" if "balanced_obj_continuous" in psl_df.columns else "balanced_obj",
            "label": "Continuous PSL manifold",
        }
    return {
        "design_x": "c_peg",
        "design_y": "i_ti",
        "design_z": "d_el",
        "modulus": "modulus_mpa_pred",
        "gc": "gc_j_m2_pred",
        "gth": "gth_app_j_m2_pred",
        "balanced": "balanced_obj",
        "label": "Snapped PSL candidates",
    }


def plot_composition_space(psl_df, train_df, round_id=0):
    cols = get_continuous_or_snapped_cols(psl_df)
    lx, ly = lambda_simplex_xy(psl_df)
    tri = mtri.Triangulation(lx, ly)

    fig = plt.figure(figsize=(6.4, 5.4))
    ax = fig.add_subplot(111, projection="3d")

    try:
        ax.plot_trisurf(
            psl_df[cols["design_x"]],
            psl_df[cols["design_y"]],
            psl_df[cols["design_z"]],
            triangles=tri.triangles,
            alpha=0.35,
            linewidth=0.15,
            antialiased=True,
        )
    except Exception:
        pass

    sc = ax.scatter(
        psl_df[cols["design_x"]],
        psl_df[cols["design_y"]],
        psl_df[cols["design_z"]],
        c=psl_df[cols["balanced"]] if cols["balanced"] in psl_df.columns else None,
        s=18,
        alpha=0.70,
        label=cols["label"],
    )
    ax.scatter(
        train_df["c_peg"], train_df["i_ti"], train_df["d_el"],
        s=45, marker="^", edgecolor="black", linewidth=0.6, label="Input data"
    )
    ax.set_xlabel("c_peg")
    ax.set_ylabel("i_ti")
    ax.set_zlabel("d_el")
    ax.set_title("Continuous PSL manifold in mechanism descriptor space")
    ax.legend(loc="best")
    if cols["balanced"] in psl_df.columns:
        cb = fig.colorbar(sc, ax=ax, shrink=0.70, pad=0.10)
        cb.set_label("Balanced objective")
    fig.tight_layout()
    out = OUTPUT_DIR / "figures" / f"figure_psl_continuous_design_surface_round_{round_id}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved continuous design-surface plot: {out}")


def plot_performance_space(psl_df, train_df, pareto_df, round_id=0):
    cols = get_continuous_or_snapped_cols(psl_df)
    lx, ly = lambda_simplex_xy(psl_df)
    tri = mtri.Triangulation(lx, ly)

    fig = plt.figure(figsize=(6.4, 5.4))
    ax = fig.add_subplot(111, projection="3d")

    try:
        ax.plot_trisurf(
            psl_df[cols["modulus"]],
            psl_df[cols["gc"]],
            psl_df[cols["gth"]],
            triangles=tri.triangles,
            alpha=0.35,
            linewidth=0.15,
            antialiased=True,
        )
    except Exception:
        pass

    sc = ax.scatter(
        psl_df[cols["modulus"]],
        psl_df[cols["gc"]],
        psl_df[cols["gth"]],
        c=psl_df[cols["gth"]],
        s=18,
        alpha=0.70,
        label=cols["label"],
    )

    ax.scatter(
        train_df["modulus_mpa"], train_df["gc_j_m2"], train_df["gth_app_j_m2"],
        s=45, marker="^", edgecolor="black", linewidth=0.6, label="Input data"
    )

    if pareto_df is not None and len(pareto_df) > 0:
        ax.scatter(
            pareto_df["modulus_mpa_pred"], pareto_df["gc_j_m2_pred"], pareto_df["gth_app_j_m2_pred"],
            s=55, marker="*", edgecolor="black", linewidth=0.6, label="Predicted feasible-library Pareto front"
        )

    ax.set_xlabel("Modulus (MPa, lower better)")
    ax.set_ylabel("Gc (J m$^{-2}$)")
    ax.set_zlabel("Gth_app (J m$^{-2}$)")
    ax.set_title("Continuous GP-predicted PSL performance surface")
    ax.legend(loc="best")
    cb = fig.colorbar(sc, ax=ax, shrink=0.70, pad=0.10)
    cb.set_label("Predicted Gth_app")
    fig.tight_layout()
    out = OUTPUT_DIR / "figures" / f"figure_psl_continuous_performance_surface_round_{round_id}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved continuous performance-surface plot: {out}")


def plot_pair_tradeoff(psl_df, train_df, round_id=0):
    cols = get_continuous_or_snapped_cols(psl_df)
    fig, ax = plt.subplots(figsize=(5.8, 4.5))
    sc = ax.scatter(
        psl_df[cols["gc"]], psl_df[cols["gth"]],
        c=psl_df[cols["modulus"]], s=24, alpha=0.75,
        label=cols["label"]
    )
    ax.scatter(
        train_df["gc_j_m2"], train_df["gth_app_j_m2"],
        s=48, marker="^", edgecolor="black", linewidth=0.6, label="Input data"
    )
    ax.set_xlabel("Gc (J m$^{-2}$)")
    ax.set_ylabel("Gth_app (J m$^{-2}$)")
    ax.set_title("Continuous damage-resistance trade-off")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Predicted modulus (MPa)")
    fig.tight_layout()
    out = OUTPUT_DIR / "figures" / f"figure_psl_continuous_gc_gth_tradeoff_round_{round_id}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved continuous Gc-Gth trade-off plot: {out}")


def plot_snapped_support(psl_df, round_id=0):
    if "design_key" not in psl_df.columns:
        return
    group_cols = ["peg_wt_pct", "thermal_condition", "locking_level"]
    groups = (
        psl_df.groupby(group_cols)
        .size()
        .reset_index(name="n_lambda")
        .sort_values("n_lambda", ascending=False)
    )
    groups["design"] = groups.apply(
        lambda r: f"PEG{int(r['peg_wt_pct'])}/{r['thermal_condition']}/{r['locking_level']}",
        axis=1,
    )
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar(groups["design"], groups["n_lambda"])
    ax.set_ylabel("Number of λ mapped to design")
    ax.set_xlabel("Snapped experimental design")
    ax.set_title("Snapped feasible designs: preference-vector support")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    out = OUTPUT_DIR / "figures" / f"figure_psl_snapped_support_round_{round_id}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved snapped-support plot: {out}")


def make_plots(psl_df, train_df, pareto_df, round_id=0):
    plot_loss(round_id=round_id)
    plot_composition_space(psl_df, train_df, round_id=round_id)
    plot_performance_space(psl_df, train_df, pareto_df, round_id=round_id)
    plot_pair_tradeoff(psl_df, train_df, round_id=round_id)
    plot_snapped_support(psl_df, round_id=round_id)


def main():
    ensure_dirs()

    print("=" * 80)
    print("Step 3: GP-guided Pareto set learning")
    print("=" * 80)

    train_df = load_gp_dataset(INITIAL_DATA_FILE)
    check_gp_dataset(train_df)

    gp_models = load_or_fit_gp_models(train_df, round_id=0)

    psl_df, library_pred_df, pareto_df = generate_psl_candidates_from_library(
        train_df=train_df,
        gp_models=gp_models,
        n_preference_vectors=N_PREFERENCE_VECTORS,
        random_seed=42,
        exclude_existing=True,
    )

    summarize_psl_candidates(psl_df)
    save_outputs(psl_df, library_pred_df, pareto_df, round_id=0)
    make_plots(psl_df, train_df, pareto_df, round_id=0)

    print("\nStep 3 completed.")
    print("Next: inspect psl_candidates_1000_round_0.csv and run Step 4.")


if __name__ == "__main__":
    main()
