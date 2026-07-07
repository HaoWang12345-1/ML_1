"""
Select active-learning Round 1 candidates using LCB/UCB-HVI.

The selection starts from the initial 35-sample GP0/PSL0 state, applies
hydration and performance constraints, and uses sequential greedy HVI to
choose a five-sample batch.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROUND_ID = 1
BATCH_SIZE = 5
BETA = 1.0
N_MC = 30000
RANDOM_SEED = 42

MIN_PEG_WT_PCT_HYDRATION = 50.0
MIN_D_EL_HIGH_LOCKING = 0.90
MIN_GC_PRED = 6000.0
MIN_GTH_PRED = 2000.0
MIN_GC_ROBUST = 5000.0
MIN_GTH_ROBUST = 1500.0
MAX_MODULUS_PRED = 0.75

PRACTICAL_PEG_STEP = 1.0
PRACTICAL_TEMP_STEP = 2.5
MIN_BATCH_DISTANCE = 0.008

DATA_FILE = Path("data/Rubber_PEG_ML_data_clean_table_final.csv")
PSL_CONTINUOUS_FILE = Path("outputs/psl_continuous_1000_round_0.csv")
PSL_SNAPPED_FILE = Path("outputs/psl_candidates_1000_round_0.csv")
LIBRARY_PRED_FILE = Path("outputs/library_predictions_round_0.csv")

OUTPUT_DIR = Path("outputs")
FIGURE_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

X_COLS = ["c_peg", "i_ti", "d_el"]
Y_PRED_COLS = ["modulus_mpa_pred", "gc_j_m2_pred", "gth_app_j_m2_pred"]
STD_COLS = ["modulus_mpa_std", "gc_j_m2_std", "gth_app_j_m2_std"]


def normalize_with_bounds(values, vmin, vmax):
    x = np.asarray(values, dtype=float)
    return (x - vmin) / (vmax - vmin + 1e-12)

def make_design_key(df):
    return (
        df["c_peg"].astype(float).round(6).astype(str)
        + "_"
        + df["i_ti"].astype(float).round(4).astype(str)
        + "_"
        + df["d_el"].astype(float).round(4).astype(str)
    )

def iti_to_temperature_C(i_ti):
    x = float(i_ti)
    if x <= 0:
        return np.nan
    if x <= 0.33:
        return 60.0
    if x <= 0.50:
        return 60.0 + (x - 0.33) / (0.50 - 0.33) * 5.0
    if x <= 0.67:
        return 65.0 + (x - 0.50) / (0.67 - 0.50) * 10.0
    if x <= 1.00:
        return 75.0 + (x - 0.67) / (1.00 - 0.67) * 15.0
    return 90.0

def add_practical_columns(df):
    out = df.copy()
    out["peg_wt_pct"] = out["c_peg"].astype(float) * 70.0
    out["rubber_wt_pct"] = 100.0 - out["peg_wt_pct"]
    out["hot_pressing_temperature_C_est"] = out["i_ti"].apply(iti_to_temperature_C)
    out["peg_wt_pct_practical"] = (out["peg_wt_pct"] / PRACTICAL_PEG_STEP).round() * PRACTICAL_PEG_STEP
    out["rubber_wt_pct_practical"] = 100.0 - out["peg_wt_pct_practical"]
    out["hot_pressing_temperature_C_practical"] = (
        out["hot_pressing_temperature_C_est"] / PRACTICAL_TEMP_STEP
    ).round() * PRACTICAL_TEMP_STEP
    out["locking_level_practical"] = np.where(out["d_el"].astype(float) >= 0.90, "high", "not_high")
    out["practical_design_key"] = (
        out["peg_wt_pct_practical"].round(3).astype(str)
        + "_"
        + out["hot_pressing_temperature_C_practical"].round(3).astype(str)
        + "_"
        + out["locking_level_practical"].astype(str)
    )
    return out

def ensure_candidate_columns(df, source):
    out = df.copy()
    aliases = {
        "c_peg": ["c_peg_continuous"],
        "i_ti": ["i_ti_continuous"],
        "d_el": ["d_el_continuous"],
        "modulus_mpa_pred": ["modulus_mpa_pred_continuous", "modulus_pred_continuous"],
        "gc_j_m2_pred": ["gc_j_m2_pred_continuous", "gc_pred_continuous"],
        "gth_app_j_m2_pred": ["gth_app_j_m2_pred_continuous", "gth_pred_continuous"],
        "modulus_mpa_std": ["modulus_mpa_std_continuous", "modulus_std_continuous"],
        "gc_j_m2_std": ["gc_j_m2_std_continuous", "gc_std_continuous"],
        "gth_app_j_m2_std": ["gth_app_j_m2_std_continuous", "gth_std_continuous"],
    }
    for target, names in aliases.items():
        if target not in out.columns:
            for name in names:
                if name in out.columns:
                    out[target] = out[name]
                    break
    missing = [c for c in X_COLS + Y_PRED_COLS if c not in out.columns]
    if missing:
        raise ValueError(f"{source} missing required columns {missing}. Available columns: {list(out.columns)}")
    for c in STD_COLS:
        if c not in out.columns:
            out[c] = 0.0
    out["candidate_source"] = source
    for c in X_COLS + Y_PRED_COLS + STD_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = add_practical_columns(out)
    out["design_key"] = make_design_key(out)
    return out

def load_candidate_pool():
    dfs = []
    if PSL_CONTINUOUS_FILE.exists():
        dfs.append(ensure_candidate_columns(pd.read_csv(PSL_CONTINUOUS_FILE), "PSL_continuous"))
    if PSL_SNAPPED_FILE.exists():
        dfs.append(ensure_candidate_columns(pd.read_csv(PSL_SNAPPED_FILE), "PSL_snapped"))
    if LIBRARY_PRED_FILE.exists():
        dfs.append(ensure_candidate_columns(pd.read_csv(LIBRARY_PRED_FILE), "full_library"))
    if not dfs:
        raise FileNotFoundError("No candidate pool found. Run Step 3 PSL first.")
    pool = pd.concat(dfs, ignore_index=True, sort=False)
    source_rank = {"PSL_continuous": 0, "PSL_snapped": 1, "full_library": 2}
    pool["_source_rank"] = pool["candidate_source"].map(source_rank).fillna(99)
    pool = pool.sort_values("_source_rank").drop_duplicates("design_key", keep="first")
    return pool.drop(columns=["_source_rank"]).reset_index(drop=True)

def measured_keys(train_df):
    train = add_practical_columns(train_df.copy())
    train["design_key"] = make_design_key(train)
    return set(train["design_key"]), set(train["practical_design_key"])

def compute_bounds(train_df, cand_df, beta=BETA):
    E_train = train_df["modulus_mpa"].to_numpy(float)
    Gc_train = train_df["gc_j_m2"].to_numpy(float)
    Gth_train = train_df["gth_app_j_m2"].to_numpy(float)
    E_lcb = np.maximum(cand_df["modulus_mpa_pred"].to_numpy(float) - beta * cand_df["modulus_mpa_std"].to_numpy(float), 1e-9)
    Gc_ucb = cand_df["gc_j_m2_pred"].to_numpy(float) + beta * cand_df["gc_j_m2_std"].to_numpy(float)
    Gth_ucb = cand_df["gth_app_j_m2_pred"].to_numpy(float) + beta * cand_df["gth_app_j_m2_std"].to_numpy(float)
    return {
        "E_min": float(np.nanmin(np.concatenate([E_train, E_lcb]))),
        "E_max": float(np.nanmax(np.concatenate([E_train, E_lcb]))),
        "Gc_min": float(np.nanmin(np.concatenate([Gc_train, Gc_ucb]))),
        "Gc_max": float(np.nanmax(np.concatenate([Gc_train, Gc_ucb]))),
        "Gth_min": float(np.nanmin(np.concatenate([Gth_train, Gth_ucb]))),
        "Gth_max": float(np.nanmax(np.concatenate([Gth_train, Gth_ucb]))),
    }

def measured_objectives(train_df, bounds):
    E = normalize_with_bounds(train_df["modulus_mpa"].to_numpy(float), bounds["E_min"], bounds["E_max"])
    Gc = normalize_with_bounds(train_df["gc_j_m2"].to_numpy(float), bounds["Gc_min"], bounds["Gc_max"])
    Gth = normalize_with_bounds(train_df["gth_app_j_m2"].to_numpy(float), bounds["Gth_min"], bounds["Gth_max"])
    return np.clip(np.vstack([1.0 - E, Gc, Gth]).T, 0.0, 1.0)

def lcb_ucb_candidate_objectives(cand_df, bounds, beta=BETA):
    E_lcb = np.maximum(cand_df["modulus_mpa_pred"].to_numpy(float) - beta * cand_df["modulus_mpa_std"].to_numpy(float), 1e-9)
    Gc_ucb = cand_df["gc_j_m2_pred"].to_numpy(float) + beta * cand_df["gc_j_m2_std"].to_numpy(float)
    Gth_ucb = cand_df["gth_app_j_m2_pred"].to_numpy(float) + beta * cand_df["gth_app_j_m2_std"].to_numpy(float)
    E_obj = 1.0 - normalize_with_bounds(E_lcb, bounds["E_min"], bounds["E_max"])
    Gc_obj = normalize_with_bounds(Gc_ucb, bounds["Gc_min"], bounds["Gc_max"])
    Gth_obj = normalize_with_bounds(Gth_ucb, bounds["Gth_min"], bounds["Gth_max"])
    return np.clip(np.vstack([E_obj, Gc_obj, Gth_obj]).T, 0.0, 1.0), E_lcb, Gc_ucb, Gth_ucb

def vectorized_dominance(Y, mc_points):
    return np.all(np.asarray(Y, dtype=np.float32)[:, None, :] >= np.asarray(mc_points, dtype=np.float32)[None, :, :], axis=2)

def min_distance_to_rows(row, selected_rows):
    if not selected_rows:
        return np.inf
    x = row[X_COLS].to_numpy(dtype=float)
    return min(float(np.linalg.norm(x - r[X_COLS].to_numpy(dtype=float))) for r in selected_rows)

def main():
    print("=" * 80)
    print("Step 4: LCB/UCB-HVI AL Round 1")
    print("=" * 80)

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing data file: {DATA_FILE}")

    train_df = pd.read_csv(DATA_FILE)
    if len(train_df) != 35:
        raise ValueError(f"Round 1 selection expects 35 initial rows, but current data has {len(train_df)} rows.")

    train_df["design_key"] = make_design_key(train_df)
    train_exact_keys, train_practical_keys = measured_keys(train_df)

    pool = load_candidate_pool()
    pool["gc_robust"] = pool["gc_j_m2_pred"] - BETA * pool["gc_j_m2_std"]
    pool["gth_robust"] = pool["gth_app_j_m2_pred"] - BETA * pool["gth_app_j_m2_std"]
    pool["modulus_robust"] = pool["modulus_mpa_pred"] + BETA * pool["modulus_mpa_std"]
    pool["hydration_pass"] = pool["peg_wt_pct"] >= MIN_PEG_WT_PCT_HYDRATION
    pool["high_locking_pass"] = pool["d_el"] >= MIN_D_EL_HIGH_LOCKING
    pool["performance_pass"] = (
        (pool["gc_j_m2_pred"] >= MIN_GC_PRED)
        & (pool["gth_app_j_m2_pred"] >= MIN_GTH_PRED)
        & (pool["gc_robust"] >= MIN_GC_ROBUST)
        & (pool["gth_robust"] >= MIN_GTH_ROBUST)
        & (pool["modulus_mpa_pred"] <= MAX_MODULUS_PRED)
    )
    pool["not_exactly_measured"] = ~pool["design_key"].isin(train_exact_keys)
    pool["not_practically_measured"] = ~pool["practical_design_key"].isin(train_practical_keys)

    hp = pool[
        pool["hydration_pass"]
        & pool["high_locking_pass"]
        & pool["performance_pass"]
        & pool["not_exactly_measured"]
        & pool["not_practically_measured"]
    ].copy().reset_index(drop=True)

    print("Candidate-pool audit")
    print("-" * 80)
    print(f"Training rows: {len(train_df)}")
    print(f"All candidate rows after merge/dedup: {len(pool)}")
    print(f"Hydration pass: {int(pool['hydration_pass'].sum())}")
    print(f"High-locking pass: {int(pool['high_locking_pass'].sum())}")
    print(f"Performance pass: {int(pool['performance_pass'].sum())}")
    print(f"High-performance feasible AL candidates: {len(hp)}")

    if len(hp) == 0:
        raise ValueError("No high-performance feasible AL candidates remain under Round 1 constraints.")

    bounds = compute_bounds(train_df, hp, beta=BETA)
    train_obj = measured_objectives(train_df, bounds)
    cand_obj, E_lcb, Gc_ucb, Gth_ucb = lcb_ucb_candidate_objectives(hp, bounds, beta=BETA)

    hp["modulus_lcb"] = E_lcb
    hp["gc_ucb"] = Gc_ucb
    hp["gth_ucb"] = Gth_ucb
    hp["softness_lcb_ucb_obj"] = cand_obj[:, 0]
    hp["gc_lcb_ucb_obj"] = cand_obj[:, 1]
    hp["gth_lcb_ucb_obj"] = cand_obj[:, 2]

    print("\nBuilding Monte Carlo dominance masks...")
    rng = np.random.default_rng(RANDOM_SEED)
    mc_points = rng.random((N_MC, 3)).astype(np.float32)

    train_cover = vectorized_dominance(train_obj, mc_points)
    current_dominated = np.any(train_cover, axis=0)
    hv_initial = float(current_dominated.mean())
    cand_cover = vectorized_dominance(cand_obj, mc_points)

    initial_gain_counts = np.logical_and(cand_cover, ~current_dominated[None, :]).sum(axis=1)
    hp["initial_hvi_raw"] = initial_gain_counts / float(N_MC)
    hp["initial_relative_hvi"] = hp["initial_hvi_raw"] / (hv_initial + 1e-12)

    selected_rows = []
    selected_indices = []
    selected_practical_keys = set()
    remaining = set(range(len(hp)))

    for rank in range(1, BATCH_SIZE + 1):
        hv_base = float(current_dominated.mean())
        eligible = []
        for idx in remaining:
            row = hp.iloc[idx]
            if row["practical_design_key"] in selected_practical_keys:
                continue
            if selected_rows and min_distance_to_rows(row, selected_rows) < MIN_BATCH_DISTANCE:
                continue
            eligible.append(idx)

        if not eligible:
            print(f"No eligible candidate remains at rank {rank}.")
            break

        eligible = np.asarray(eligible, dtype=int)
        gain_counts = np.logical_and(cand_cover[eligible], ~current_dominated[None, :]).sum(axis=1)
        best_pos = int(np.argmax(gain_counts))
        best_idx = int(eligible[best_pos])
        best_gain = float(gain_counts[best_pos] / float(N_MC))

        row = hp.iloc[best_idx].copy()
        row["al_rank"] = rank
        row["al_candidate_id"] = f"AL_R{ROUND_ID}_LCBHVI_{rank:02d}"
        row["selection_rule"] = "sequential_greedy_LCB_UCB_HVI"
        row["hvi_raw"] = best_gain
        row["relative_hvi_to_current_base"] = best_gain / (hv_base + 1e-12)
        row["hv_base_before_selection"] = hv_base
        row["distance_to_selected"] = min_distance_to_rows(row, selected_rows)
        row["posthoc_role"] = (
            "low_modulus_balanced" if row["modulus_mpa_pred"] <= hp["modulus_mpa_pred"].quantile(0.25)
            else "high_Gc_Gth_boundary_expansion"
        )

        selected_rows.append(row)
        selected_indices.append(best_idx)
        selected_practical_keys.add(row["practical_design_key"])
        current_dominated = np.logical_or(current_dominated, cand_cover[best_idx])
        remaining.remove(best_idx)

        print(
            f"Selected rank {rank}: {row['al_candidate_id']} | "
            f"PEG{row['peg_wt_pct_practical']:.1f}, T{row['hot_pressing_temperature_C_practical']:.1f}, "
            f"HVI={best_gain:.6f}, rel={row['relative_hvi_to_current_base']:.3%}"
        )

    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    hv_final = float(current_dominated.mean())
    batch_hvi = hv_final - hv_initial
    batch_relative_hvi = batch_hvi / (hv_initial + 1e-12)

    hp_sorted = hp.sort_values("initial_hvi_raw", ascending=False).reset_index(drop=True)

    summary = pd.DataFrame([
        {"item": "algorithm", "value": "High-performance-constrained LCB/UCB-HVI sequential greedy AL"},
        {"item": "round", "value": "AL_Round_1_from_initial_n35"},
        {"item": "weighted_score_used", "value": False},
        {"item": "training_samples", "value": len(train_df)},
        {"item": "candidate_pool_after_constraints", "value": len(hp)},
        {"item": "selected_batch_size", "value": len(selected)},
        {"item": "beta", "value": BETA},
        {"item": "mc_points", "value": N_MC},
        {"item": "hv_initial", "value": hv_initial},
        {"item": "hv_final_after_batch", "value": hv_final},
        {"item": "batch_hvi", "value": batch_hvi},
        {"item": "batch_relative_hvi", "value": batch_relative_hvi},
        {"item": "min_peg_wt_pct_hydration", "value": MIN_PEG_WT_PCT_HYDRATION},
        {"item": "min_gc_pred", "value": MIN_GC_PRED},
        {"item": "min_gth_pred", "value": MIN_GTH_PRED},
        {"item": "min_gc_robust", "value": MIN_GC_ROBUST},
        {"item": "min_gth_robust", "value": MIN_GTH_ROBUST},
        {"item": "max_modulus_pred", "value": MAX_MODULUS_PRED},
        {"item": "practical_peg_step", "value": PRACTICAL_PEG_STEP},
        {"item": "practical_temp_step", "value": PRACTICAL_TEMP_STEP},
    ])

    selected_path = OUTPUT_DIR / "al_selected_batch_round_1_lcb_hvi.csv"
    scored_path = OUTPUT_DIR / "al_scored_candidates_round_1_lcb_hvi.csv"
    summary_path = OUTPUT_DIR / "al_lcb_hvi_selection_summary_round_1.csv"
    xlsx_path = OUTPUT_DIR / "step4_AL_selection_round_1_lcb_hvi.xlsx"

    selected.to_csv(selected_path, index=False, encoding="utf-8-sig")
    hp_sorted.to_csv(scored_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(xlsx_path) as writer:
        selected.to_excel(writer, sheet_name="Selected_LCB_HVI_R1", index=False)
        hp_sorted.to_excel(writer, sheet_name="Scored_HP_Candidates", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)

    fig = plt.figure(figsize=(7.2, 5.8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(train_df["modulus_mpa"], train_df["gc_j_m2"], train_df["gth_app_j_m2"], s=36, alpha=0.45, label="Measured 35")
    ax.scatter(hp_sorted["modulus_mpa_pred"], hp_sorted["gc_j_m2_pred"], hp_sorted["gth_app_j_m2_pred"], s=18, alpha=0.20, label="HP feasible candidates")
    ax.scatter(selected["modulus_mpa_pred"], selected["gc_j_m2_pred"], selected["gth_app_j_m2_pred"], s=145, marker="*", label="AL R1 LCB/HVI")
    ax.set_xlabel("Modulus (MPa)")
    ax.set_ylabel("Gc (J/m²)")
    ax.set_zlabel("Gth_app (J/m²)")
    ax.set_title("AL Round 1: high-performance-constrained LCB/HVI")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure_step4_lcb_hvi_AL_round_1.png", dpi=300)
    plt.close(fig)

    show_cols = [
        "al_rank", "al_candidate_id", "candidate_source", "selection_rule", "posthoc_role",
        "peg_wt_pct", "peg_wt_pct_practical", "hot_pressing_temperature_C_est",
        "hot_pressing_temperature_C_practical", "d_el",
        "modulus_mpa_pred", "gc_j_m2_pred", "gth_app_j_m2_pred",
        "gc_robust", "gth_robust", "modulus_lcb", "gc_ucb", "gth_ucb",
        "hvi_raw", "relative_hvi_to_current_base",
    ]

    print("\nSelected AL Round 1 LCB/HVI batch:")
    print(selected[[c for c in show_cols if c in selected.columns]].to_string(index=False))

    print("\nSaved:")
    print(f"  {selected_path}")
    print(f"  {scored_path}")
    print(f"  {summary_path}")
    print(f"  {xlsx_path}")

    print("\nPASS: AL Round 1 used high-performance constraints and sequential greedy LCB/UCB-HVI.")

if __name__ == "__main__":
    main()
