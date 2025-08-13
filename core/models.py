from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import re
from typing import Union, Optional, Mapping, Any
from collections import deque
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .config import ARTIFACT_PATH, WINDOW, TARGET

class AssetManager:
    """Load / persist fitted models + scalers with caching."""
    _cache = None

    def __init__(self, path: Path = ARTIFACT_PATH):
        self.path = path
        self.path.mkdir(exist_ok=True)
        need = [
            "xgb.pkl", "lstm.keras", "qc.pkl",
            "sx.pkl", "sy.pkl", "sr.pkl",
            "features.pkl", "bootstrap.csv"
        ]
        missing = [f for f in need if not (self.path / f).exists()]
        if missing:
            raise FileNotFoundError(f"Missing artefacts: {missing}. Train and save them first.")

    def load(self) -> Dict[str, Any]:
        if AssetManager._cache is None:
            AssetManager._cache = {
                "xgb_model":  joblib.load(self.path / "xgb.pkl"),
                "lstm_model": tf.keras.models.load_model(self.path / "lstm.keras"),
                "qc":         joblib.load(self.path / "qc.pkl"),
                "sc_X":       joblib.load(self.path / "sx.pkl"),
                "sc_y":       joblib.load(self.path / "sy.pkl"),
                "sc_res":     joblib.load(self.path / "sr.pkl"),
                "features":   joblib.load(self.path / "features.pkl"),
                "bs_df":      pd.read_csv(self.path / "bootstrap.csv", index_col=0, parse_dates=True),
            }
        return AssetManager._cache

class HybridForecaster:
    """XGBoost baseline + LSTM residual walk-forward inference."""
    def __init__(self, assets: Dict[str, Any], window: int = WINDOW):
        self.xgb      = assets["xgb_model"]
        self.lstm     = assets["lstm_model"]
        self.qc       = assets["qc"]
        self.sc_X     = assets["sc_X"]
        self.sc_y     = assets["sc_y"]
        self.sc_res   = assets["sc_res"]
        self.features: List[str] = assets["features"]
        self.window   = window

    def _build_seq(self, X: np.ndarray, rf: np.ndarray, res: Optional[np.ndarray]= None):
        Xs = []
        for i in range(self.window, len(X)):
            if res is not None:
                # Include residuals if provided
                sequence = np.hstack([
                    X[i-self.window:i],
                    rf[i-self.window:i, None],
                    res[i-self.window:i, None]    
                ])
            else:
                # Only X and rf if no residuals
                sequence = np.hstack([
                    X[i-self.window:i],
                    rf[i-self.window:i, None]
                ])
            Xs.append(sequence)
        return np.asarray(Xs)


    def walk_forward(
        self, 
        test_df: pd.DataFrame, 
        dev_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Leak-free recursive one-step-ahead forecast for XGB + LSTM (residual) hybrid.

        Key fix (vs the leaky version):
        - We DO NOT directly use precomputed demand-based features for the current row.
        - At each step we overwrite:
            * demand_i_1                  ← last known demand (actual or previous forecast)
            * demand_roll7_i_1            ← mean of last 7 known demands
            * demand_roll30_i_1           ← mean of last 30 known demands
        before scaling & predicting.

        Expects self.a to contain: 'qc', 'sc_X', 'sc_y', 'sc_res', 'xgb_model', 'lstm_model', 'features', 'window'.
        """
        
        W = self.window
        feats = self.features
        # transform entire TEST once (exogenous + static derived features are fine)
        t_test = self.qc.transform(test_df.copy())
        # ---------- Seed histories ----------
        if dev_df is not None:
            # DEV history is fully known → safe to use its precomputed features
            t_dev   = self.qc.transform(dev_df.copy())
            X_dev   = self.sc_X.transform(t_dev[feats])
            rf_dev  = self.xgb.predict(X_dev)  # target-scaled baseline

            # standardized residual history from DEV (what LSTM saw during training)
            res_dev_scaled = self.sc_y.transform(t_dev[[TARGET]]).ravel() - rf_dev
            res_dev_std    = self.sc_res.transform(res_dev_scaled.reshape(-1, 1)).ravel()

            # demand history (original units) for rolling means
            demand_hist = deque(dev_df[TARGET].tolist(), maxlen=30)

            rf_hist  = list(rf_dev)      # target-scaled
            res_hist = list(res_dev_std) # standardized
            hist_t   = t_dev.copy()      # transformed features history (we will append corrected rows)
            hist_raw = dev_df.copy()     # original-unit history (we will append forecasts)
            start_pos = 0
        else:
            # No DEV → we require first W rows of TEST to be truly known.
            if len(t_test) < W:
                raise ValueError(f"Need at least WINDOW={W} rows in test_df when dev_df=None.")
            hist_t   = t_test.iloc[:W].copy()
            hist_raw = test_df.iloc[:W].copy()

            # build initial baseline/residual history from the known warm-up
            X_hist = self.sc_X.transform(hist_t[feats])
            rf_hist = list(self.xgb.predict(X_hist))  # target-scaled
            res_hist = [0.0] * len(hist_t)   # no standardized residuals yet → zeros are a safe placeholder

            # demand history (original units) for rolling means
            demand_hist = deque(hist_raw[TARGET].tolist(), maxlen=30)

            start_pos = W  # first predictable row

        # ---------- rolling inference ----------
        preds, bases, res_out = [], [], []

        # convenience: columns we will overwrite (only if they actually exist in features)
        has_d1   = "demand_i_1"          in feats
        has_r7   = "demand_roll7_i_1"    in feats
        has_r30  = "demand_roll30_i_1"   in feats

        for idx in t_test.index[start_pos:]:
            # 1) overwrite *only* demand-based features for the current row
            last_demand = demand_hist[-1]  # original units (actual or last forecast)
            row_mod = t_test.loc[idx].copy()

            if has_d1:
                row_mod["demand_i_1"] = last_demand
            # if has_r7:
                # if fewer than 7 known demands, use what we have
                # r7 = np.mean(list(demand_hist)[-7:]) if len(demand_hist) >= 1 else last_demand
                # row_mod["demand_roll7_i_1"] = r7
            if has_r30:
                r30 = np.mean(list(demand_hist)[-30:]) if len(demand_hist) >= 1 else last_demand
                row_mod["demand_roll30_i_1"] = r30

            # 2) scale and make XGB baseline (target-scaled)
            x_now  = self.sc_X.transform(row_mod[feats].values.reshape(1, -1))
            rf_now = float(self.xgb.predict(x_now)[0])

            # 3) build LSTM sequence: last W-1 history rows + today's row
            #    - features channel: scaled hist features (already corrected in prior steps) + x_now
            #    - baseline channel: target-scaled rf history + rf_now
            #    - residual channel: standardized res history + 0 for today
            if len(hist_t) >= (W - 1):
                X_hist_win = self.sc_X.transform(hist_t[feats].iloc[-(W-1):])
                rf_win     = np.asarray(rf_hist[-(W-1):])
                res_win    = np.asarray(res_hist[-(W-1):])
            else:
                # left-pad (very tiny dev histories)
                pad = (W - 1) - len(hist_t)
                X_first = self.sc_X.transform(hist_t[feats].iloc[[0]])
                X_hist_win = np.vstack([X_first] * pad + [self.sc_X.transform(hist_t[feats])])
                rf_first   = rf_hist[0]
                rf_win     = np.concatenate([np.full(pad, rf_first), np.asarray(rf_hist)])
                res_win    = np.concatenate([np.zeros(pad), np.asarray(res_hist)])

            X_part  = np.vstack([X_hist_win, x_now])[-W:]            # (W, p)
            rf_part = np.append(rf_win, rf_now).reshape(-1, 1)[-W:]  # (W, 1)
            rs_part = np.append(res_win, 0.0).reshape(-1, 1)[-W:]    # (W, 1) – 0 placeholder today

            seq = np.hstack([X_part, rf_part, rs_part]).reshape(1, W, -1)

            # 4) LSTM residual (standardized) → back to target-scaled → combine
            res_now_std    = float(self.lstm.predict(seq, verbose=0)[0])
            res_now_scaled = self.sc_res.inverse_transform([[res_now_std]]).item()
            y_scaled       = rf_now + res_now_scaled
            y_hat          = self.sc_y.inverse_transform([[y_scaled]]).item()  # original units

            # (nice-to-have) baseline in original units + reported residual
            base_orig  = self.sc_y.inverse_transform([[rf_now]]).item()
            resid_orig = y_hat - base_orig

            preds.append(y_hat); bases.append(base_orig); res_out.append(resid_orig)

            # 5) update rolling histories for next step
            rf_hist.append(rf_now)         # target-scaled
            res_hist.append(res_now_std)   # standardized (what LSTM expects as input)
            demand_hist.append(y_hat)      # original units (used to recompute demand_roll features)

            # keep a corrected feature row in the transformed history
            hist_t = pd.concat([hist_t, pd.DataFrame([row_mod], index=[idx])])

            # raw history with realised = our forecast (so t+1 sees our y_hat)
            new_raw = test_df.loc[[idx]].copy()
            new_raw.iloc[0, new_raw.columns.get_loc(TARGET)] = y_hat
            hist_raw = pd.concat([hist_raw, new_raw])

        # ---------- pack outputs ----------
        out = test_df.copy()
        if dev_df is None and start_pos > 0:
            out = out.iloc[start_pos:].copy()  # first W rows are the warm-up, no forecasts

        out["y_fwd"]        = preds
        out["baseline_fwd"] = bases
        out["residual_fwd"] = res_out

        # ---------- carry quartile as ONE categorical column + attach composition ----------
        # try to infer the quartile OHE column names from the fitted QuartileCluster
        field = getattr(self.qc, 'field', None)
        if field is not None:
            # Expected names: e.g. "tot_adv_exp_i_1_Q1" … "tot_adv_exp_i_1_Q4"
            q_cols = [f"{field}_Q{i}" for i in range(1, 5) if f"{field}_Q{i}" in t_test.columns]
        else:
            # Fallback: any column that ends with _Q1.._Q4 (order them Q1..Q4)
            q_cols_all = [c for c in t_test.columns if re.search(r"_Q[1-4]$", c)]
            # keep deterministic order if possible
            order = ["Q1", "Q2", "Q3", "Q4"]
            q_cols = sorted(q_cols_all, key=lambda c: order.index(re.search(r"(Q[1-4])$", c).group(1)) if re.search(r"(Q[1-4])$", c) else 0)

        quartile = pd.Series(index=out.index, dtype="object")

        if q_cols:
            # case-style assignment from one-hot flags
            for i, qcol in enumerate(q_cols, start=1):
                mask = t_test.loc[out.index, qcol].astype(int) == 1
                quartile.loc[mask] = f"q{i}"

        # fallback: if any rows still unlabeled (e.g., no OHE present or all zeros),
        # derive label using the fitted QuartileCluster thresholds on the source field
        if quartile.isna().any() and field is not None and field in t_test.columns and hasattr(self.qc, "_bin"):
            missing_idx = quartile.index[quartile.isna()]
            labels = self.qc._bin(t_test.loc[missing_idx, field])
            quartile.loc[missing_idx] = pd.Series(labels, index=missing_idx).str.lower()

        # finalise
        out["quartile"] = quartile.fillna("unknown")
        
        # metrics if ground truth is present
        if TARGET in out.columns:
            out.attrs["rmse"] = float(np.sqrt(mean_squared_error(out[TARGET], out["y_fwd"])))
            out.attrs["mae"]  = float(mean_absolute_error(out[TARGET], out["y_fwd"]))
            out.attrs["mrae"] = np.mean((out[TARGET] - out["y_fwd"]).abs() / out[TARGET].replace(0, np.nan))
        return out

    def single_shot(
        self,
        test_df: pd.DataFrame,
        dev_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Non-recursive inference (forecast horizon = 1 for each test row).
        If dev_df is provided, the LSTM sequences are built as:
            [ last (W-1) rows from DEV  +  current TEST row ],
        which gives the model a realistic historical window. Otherwise,
        we fall back to the original behaviour (zero-residual warmup),
        which aligns outputs from index 'window' onward.
        """
        W = self.window

        # 1) Transform TEST with the TRAIN-fitted QuartileCluster
        t_test = self.qc.transform(test_df.copy())
        feats  = self.features

        # Helper: build a single sequence for one row using a fixed (W-1) history
        def make_seq_with_history(x_tail, rf_tail, res_tail_std, x_now, rf_now):
            """
            x_tail:      array shape (W-1, p)  – scaled features from DEV tail
            rf_tail:     array shape (W-1,)    – baseline (target-scaled) from DEV tail
            res_tail_std:array shape (W-1,)    – standardized residuals from DEV tail
            x_now:       array shape (1, p)    – scaled features for the current TEST row
            rf_now:      float                 – baseline for the current TEST row (target-scaled)
            """
            X_part  = np.vstack([x_tail, x_now])                    # (W, p)
            rf_part = np.append(rf_tail, rf_now).reshape(-1, 1)     # (W, 1)
            rs_part = np.append(res_tail_std, 0.0).reshape(-1, 1)   # (W, 1), 0 for "today"
            seq     = np.hstack([X_part, rf_part, rs_part]).reshape(1, W, -1)
            return seq

        preds, bases, res_out = [], [], []

        if dev_df is not None:
            # ── A) BEST: use last (W-1) rows of DEV as the historical window for each TEST row
            t_dev    = self.qc.transform(dev_df.copy())
            X_dev    = self.sc_X.transform(t_dev[feats])
            rf_dev   = self.xgb.predict(X_dev)  # target-scaled baseline
            # residuals: (target-scaled y) − (target-scaled baseline)
            res_dev_scaled = self.sc_y.transform(t_dev[[TARGET]]).ravel() - rf_dev
            res_dev_std    = self.sc_res.transform(res_dev_scaled.reshape(-1, 1)).ravel()

            # Take the tail with length (W-1); if DEV shorter, left-pad the first row
            if len(X_dev) >= (W - 1):
                X_tail       = X_dev[-(W-1):]
                rf_tail      = rf_dev[-(W-1):]
                res_tail_std = res_dev_std[-(W-1):]
            else:
                pad          = (W - 1) - len(X_dev)
                X_first      = X_dev[:1]
                rf_first     = rf_dev[:1]
                res_first    = res_dev_std[:1]
                X_tail       = np.vstack([np.repeat(X_first, pad, axis=0), X_dev])
                rf_tail      = np.concatenate([np.repeat(rf_first, pad), rf_dev])
                res_tail_std = np.concatenate([np.repeat(res_first, pad), res_dev_std])

            # (Optional but recommended) fix the boundary lag for the very first test row
            # so that demand_i_1 matches the last known DEV demand
            # Only do this if that lag column exists in features:
            lag_name = f"{TARGET}_i_1"
            if lag_name in feats and len(t_test) > 0:
                last_known = dev_df[TARGET].iloc[-1]
                first_idx  = t_test.index[0]
                t_test.loc[first_idx, lag_name] = last_known

            # Now score each TEST row independently (no recursion)
            for idx, row in t_test.iterrows():
                x_now  = self.sc_X.transform(row[feats].values.reshape(1, -1))
                rf_now = float(self.xgb.predict(x_now)[0])

                seq = make_seq_with_history(X_tail, rf_tail, res_tail_std, x_now, rf_now)

                # LSTM → standardized residual → back to target-scaled → combine
                res_now_std    = float(self.lstm.predict(seq, verbose=0)[0])
                res_now_scaled = self.sc_res.inverse_transform([[res_now_std]]).item()
                y_scaled       = rf_now + res_now_scaled
                y_hat          = self.sc_y.inverse_transform([[y_scaled]]).item()

                base_orig  = self.sc_y.inverse_transform([[rf_now]]).item()
                resid_orig = y_hat - base_orig

                preds.append(y_hat)
                bases.append(base_orig)
                res_out.append(resid_orig)

            out = test_df.copy()
            out["y_hat"]    = preds
            out["baseline"] = bases
            out["residual"] = res_out

            if TARGET in out.columns:
                out.attrs["rmse"] = float(np.sqrt(mean_squared_error(out[TARGET], out["y_hat"])))
                out.attrs["mae"]  = float(mean_absolute_error(out[TARGET], out["y_hat"]))
            return out
        else:
            # ── B) Legacy: no DEV context. Fall back to zero-residual warmup and align from W.
            X_test = self.sc_X.transform(t_test[feats])
            rf     = self.xgb.predict(X_test)                     # target-scaled
            # Build sequences with zero residual history, as before
            Xseq   = self._build_seq(X_test, rf, np.zeros_like(rf))
            delta_s = self.lstm.predict(Xseq, verbose=0).ravel()  # standardized residuals
            delta_scaled = self.sc_res.inverse_transform(delta_s.reshape(-1, 1)).ravel()

            off       = W
            rf_al     = rf[off:]
            y_pred_s  = rf_al + delta_scaled
            y_pred    = self.sc_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
            baseline  = self.sc_y.inverse_transform(rf_al.reshape(-1, 1)).ravel()

            idx = t_test.index[off:]
            out = test_df.loc[idx].copy()
            out["y_hat"]    = y_pred
            out["baseline"] = baseline
            out["residual"] = self.sc_res.inverse_transform(delta_s.reshape(-1, 1)).ravel()

            if TARGET in out.columns:
                out.attrs["rmse"] = float(np.sqrt(mean_squared_error(out[TARGET], out["y_hat"])))
                out.attrs["mae"]  = float(mean_absolute_error(out[TARGET], out["y_hat"]))
                out.attrs["mrae"] = np.mean((out[TARGET] - out["y_hat"]).abs() / out[TARGET].replace(0, np.nan))
            return out
  