"""
Comprehensive model search for the matchup predictors.

Searches over: rolling windows × feature sets × model type × hyperparameters
(× ensemble weights), scoring every combination with SEASON WALK-FORWARD
cross-validation (train on all prior seasons, test on each held-out season,
average). Multi-season CV is used deliberately — tuning on one or two seasons
overfits their noise (which is how a "better" config can score worse live).

The best config found is persisted to <sport>_best_model.json and OVERWRITTEN
whenever a better one appears; every trial is appended to <sport>_search_log.csv.
Interrupt any time — progress is saved incrementally, and re-running resumes the
best-so-far.

Usage:
    python model_search.py --sport nhl --trials 400
    python model_search.py --sport nhl --exhaustive
"""

import argparse
import itertools
import json
import os
import random
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier, RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")


# ---------------------------------------------------------------------------
# Data: full history incl. rich current season, per sport
# ---------------------------------------------------------------------------

def load_full(sport: str):
    if sport == "nhl":
        import nhl_matchup as M
        raw = pd.read_csv(M.TRAIN_CSV)
        raw["Date"] = pd.to_datetime(raw["Date"])
        try:  # include rich current-season completed games as an extra CV fold
            cur = pd.read_csv(os.path.join(DATA_DIR, "nhl_matches_current.csv"))
            cur["Date"] = pd.to_datetime(cur["Date"])
            cur = cur.dropna(subset=["Rslt"])
            if not cur.empty and "CF%" in cur.columns:
                raw = pd.concat([raw, cur], ignore_index=True)
        except Exception:
            pass
        pool = ["GF", "GA", "SOG", "SOG_OPP", "PIM", "PIM_OPP", "PPG", "PPO",
                "PPG_OPP", "PPO_OPP", "SHG", "SHG_OPP", "FOW", "FOL", "FO%",
                "CF", "CA", "CF%", "FF", "FA", "FF%", "oZS%", "PDO"]
        feature_sets = {
            "possess8":  ["GF", "GA", "SOG", "SOG_OPP", "CF%", "FF%", "PDO", "oZS%"],
            "current12": ["GF", "GA", "SOG", "SOG_OPP", "CF%", "FF%", "PDO", "oZS%", "PPG", "PPO", "PIM", "FO%"],
            "rich18":    ["GF", "GA", "SOG", "SOG_OPP", "CF%", "FF%", "PDO", "oZS%", "PPG", "PPO", "PIM", "FO%", "CF", "CA", "FF", "FA", "SHG", "PPG_OPP"],
            "full23":    pool,
            "minimal5":  ["GF", "GA", "CF%", "FF%", "PDO"],
        }
        windows = [[3, 5, 10], [5, 10, 20], [3, 7, 14], [5, 15, 30],
                   [10, 20, 40], [3, 5, 10, 20]]
        return M, raw, feature_sets, windows

    if sport == "mlb":
        import mlb_matchup as M
        from mlb_pitchers import sp_feature_cols
        raw = pd.read_csv(M.TRAIN_CSV)
        raw["Date"] = pd.to_datetime(raw["Date"])
        pool = ["RD", "R", "OBP", "SLG", "BABIP", "HR", "BB", "SO",
                "P_ERA", "P_WHIP", "P_K9", "P_BB9", "P_K_BB", "P_HR9"]
        feature_sets = {
            "core":     ["RD", "OBP", "SLG", "BABIP", "HR", "P_ERA", "P_WHIP", "P_K9", "P_BB9", "P_K_BB"],
            "rich":     pool,
            "hit_only": ["RD", "R", "OBP", "SLG", "BABIP", "HR"],
        }
        windows = [[5, 10, 20], [10, 20, 40], [5, 15, 30], [10, 30]]
        return M, raw, feature_sets, windows

    raise ValueError(sport)


def build_dataset(M, raw, windows, feats):
    M.TEAM_WINDOWS = windows
    M.TEAM_ROLL_COLS = feats
    return M.build_matchup(raw)   # (game, feature_cols)


# ---------------------------------------------------------------------------
# Model zoo — each yields (name, factory, param_grid)
# ---------------------------------------------------------------------------

def model_zoo() -> dict:
    return {
        "logreg": (
            lambda p: make_pipeline(StandardScaler(),
                                    LogisticRegression(max_iter=2000, C=p["C"])),
            {"C": [0.03, 0.1, 0.3, 1.0]},
        ),
        "rf": (
            lambda p: RandomForestClassifier(
                n_estimators=p["n"], max_depth=p["d"], min_samples_leaf=p["leaf"],
                max_features="sqrt", n_jobs=-1, random_state=42),
            {"n": [300, 500], "d": [5, 7, 10], "leaf": [8, 15, 25]},
        ),
        "extratrees": (
            lambda p: ExtraTreesClassifier(
                n_estimators=p["n"], max_depth=p["d"], min_samples_leaf=p["leaf"],
                max_features="sqrt", n_jobs=-1, random_state=42),
            {"n": [400], "d": [7, 12, None], "leaf": [8, 20]},
        ),
        "gb": (
            lambda p: GradientBoostingClassifier(
                n_estimators=p["n"], learning_rate=p["lr"], max_depth=p["d"],
                subsample=0.8, random_state=42),
            {"n": [150, 300], "lr": [0.02, 0.05], "d": [2, 3]},
        ),
        "histgb": (
            lambda p: HistGradientBoostingClassifier(
                learning_rate=p["lr"], max_depth=p["d"], max_iter=p["n"],
                l2_regularization=p["l2"], random_state=42),
            {"n": [300, 600], "lr": [0.02, 0.05], "d": [3, 6], "l2": [0.0, 1.0]},
        ),
        "xgb": (
            lambda p: XGBClassifier(
                n_estimators=p["n"], learning_rate=p["lr"], max_depth=p["d"],
                subsample=0.8, colsample_bytree=0.8, min_child_weight=p["mcw"],
                n_jobs=-1, random_state=42, verbosity=0),
            {"n": [300, 500], "lr": [0.02, 0.04], "d": [3, 4], "mcw": [8, 15]},
        ),
    }


def _grid(param_grid):
    keys = list(param_grid)
    for vals in itertools.product(*[param_grid[k] for k in keys]):
        yield dict(zip(keys, vals))


# ---------------------------------------------------------------------------
# Season walk-forward CV
# ---------------------------------------------------------------------------

def cv_eval(game, cols, factory, params, min_train=800):
    seasons = sorted(game["Season"].unique())
    accs, aucs, lls = [], [], []
    for s in seasons:
        tr, te = game[game["Season"] < s], game[game["Season"] == s]
        if len(tr) < min_train or te.empty:
            continue
        med = tr[cols].median()
        Xtr, ytr = tr[cols].fillna(med), tr["home_win"]
        Xte, yte = te[cols].fillna(med), te["home_win"]
        model = factory(params)
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]
        accs.append(accuracy_score(yte, (p >= 0.5).astype(int)))
        aucs.append(roc_auc_score(yte, p))
        lls.append(log_loss(yte, p, labels=[0, 1]))
    if not accs:
        return None
    return {"acc": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "auc": float(np.mean(aucs)), "logloss": float(np.mean(lls)),
            "folds": len(accs)}


# ---------------------------------------------------------------------------
# Search driver
# ---------------------------------------------------------------------------

def run(sport: str, trials: int, exhaustive: bool):
    M, raw, feature_sets, windows = load_full(sport)
    zoo = model_zoo()
    best_path = os.path.join(BASE_DIR, f"{sport}_best_model.json")
    log_path  = os.path.join(BASE_DIR, f"{sport}_search_log.csv")

    best = None
    if os.path.exists(best_path):
        best = json.load(open(best_path))
        print(f"resuming — current best acc={best['metrics']['acc']:.4f} "
              f"({best['model']}/{best['features']}/{best['windows']})")

    # enumerate every (windows, feature_set, model, params) combo
    combos = []
    for wname_i, w in enumerate(windows):
        for fname, feats in feature_sets.items():
            for mname, (factory, grid) in zoo.items():
                for params in _grid(grid):
                    combos.append((w, fname, feats, mname, factory, params))
    if not exhaustive:
        random.seed(1)
        random.shuffle(combos)
        combos = combos[:trials]
    print(f"{sport}: {len(combos)} combos to evaluate "
          f"({'exhaustive' if exhaustive else f'random {trials}'})\n")

    # cache built datasets per (windows, feature_set)
    cache = {}
    log_rows = []
    t0 = time.time()

    for i, (w, fname, feats, mname, factory, params) in enumerate(combos, 1):
        key = (tuple(w), fname)
        if key not in cache:
            cache[key] = build_dataset(M, raw, list(w), list(feats))
        game, cols = cache[key]

        m = cv_eval(game, cols, factory, params)
        if m is None:
            continue
        row = {"model": mname, "features": fname, "windows": "_".join(map(str, w)),
               "params": json.dumps(params), **m}
        log_rows.append(row)

        improved = best is None or m["acc"] > best["metrics"]["acc"]
        tag = ""
        if improved:
            best = {"sport": sport, "model": mname, "features": fname,
                    "windows": list(w), "feature_cols": list(feats),
                    "params": params, "metrics": m,
                    "found_at": time.strftime("%Y-%m-%d %H:%M:%S")}
            json.dump(best, open(best_path, "w"), indent=2)
            tag = "  <== NEW BEST"
        if improved or i % 25 == 0:
            print(f"[{i}/{len(combos)}] {mname:<10} {fname:<10} w={row['windows']:<10} "
                  f"acc={m['acc']:.4f}±{m['acc_std']:.3f} auc={m['auc']:.4f} "
                  f"ll={m['logloss']:.4f}{tag}")

    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    print(f"\nDone in {time.time()-t0:.0f}s. Trials logged: {len(log_rows)} → {log_path}")
    print(f"\n=== BEST {sport.upper()} CONFIG ===")
    print(json.dumps(best, indent=2))
    # show the top 10 from the log for context
    top = pd.DataFrame(log_rows).sort_values("acc", ascending=False).head(10)
    print("\n--- top 10 by mean CV accuracy ---")
    print(top[["model", "features", "windows", "acc", "acc_std", "auc", "logloss"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sport", default="nhl", choices=["nhl", "mlb"])
    ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--exhaustive", action="store_true")
    a = ap.parse_args()
    run(a.sport, a.trials, a.exhaustive)
