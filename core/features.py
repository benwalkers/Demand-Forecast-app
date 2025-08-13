from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

# ──────────────────────────────────────────────────────────────
# Feature transformers
# ──────────────────────────────────────────────────────────────

class HolidayFlags(BaseEstimator, TransformerMixin):
    """Country + custom-day calendar dummies."""
    def __init__(self, date_col: str = "date", country: str = "China",
                 custom_days: Optional[Dict[Tuple[int,int], str]] = None):
        self.date_col = date_col
        self.country  = country
        self.custom_days = custom_days or {(11,11): "is_1111", (6,18): "is_618"}

    def fit(self, X, y=None):
        years = np.arange(X[self.date_col].dt.year.min(), X[self.date_col].dt.year.max() + 1)
        try:
            import holidays
            self.holidays_ = pd.Index(holidays.country_holidays(self.country, years=years).keys())
        except Exception:
            self.holidays_ = pd.Index([])
        return self

    def transform(self, X):
        df = X.copy()
        df[f"is_holiday_{self.country[:2].lower()}"] = df[self.date_col].isin(self.holidays_).astype(np.int8)
        for (m, d), name in self.custom_days.items():
            df[name] = ((df[self.date_col].dt.month == m) & (df[self.date_col].dt.day == d)).astype(np.int8)
        return df.drop(columns=[self.date_col], errors="ignore")  # drop date col

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
    def fit(self, X, y=None): return self
    def transform(self, X): return X.drop(columns=self.cols, errors="ignore")

class LagFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, media_half_life=7, demand_windows=(7, 30), price_windows=(7,), supply_window=3, target="demand"):
        self.alpha      = 1 - 0.5 ** (1 / media_half_life)
        self.demand_w   = demand_windows
        self.price_w    = price_windows
        self.supply_w   = supply_window
        self.target     = target

    def fit(self, X, y=None): return self

    def transform(self, X):
        df = X.copy()

        # demand rolling means
        if self.target in df.columns:
            for w in self.demand_w:
                df[f"{self.target}_roll{w}"] = df[self.target].rolling(w, min_periods=w).mean()

        # ad-stock for media
        media_cols = [c for c in df.columns if c.startswith(("adv_exp_", "grp_"))]
        for col in media_cols:
            df[f"{col}_adstock"] = df[col].ewm(alpha=self.alpha, adjust=False).mean()

        # price rolls
        price_cols = [c for c in df.columns if "price" in c or "discount" in c]
        for col in price_cols:
            for w in self.price_w:
                df[f"{col}_roll{w}"] = df[col].rolling(w, min_periods=w).mean()

        # supply shortfall signal
        if "pos_supply_data" in df.columns:
            min_supply = df["pos_supply_data"].rolling(self.supply_w, min_periods=self.supply_w).min()
            df["pos_supply_shortfall"] = (min_supply > df["pos_supply_data"].shift(1)).astype(int)

        # shift entire frame by +1 day to avoid leakage
        shifted = df.shift(1).dropna()
        shifted.columns = [f"{c}_i_1" for c in df.columns]
        # restore target as current (aligned)
        if self.target in df.columns:
            shifted[self.target] = df.loc[shifted.index, self.target]
        # restore date column for HolidayFlags
        shifted["date"] = shifted.index
        return shifted

class QuartileCluster(BaseEstimator, TransformerMixin):
    """
    One-hot categorical clusters Q1…Q4 on a numeric field; also outputs a 'quartile' label column.
    """
    def __init__(self, field="tot_adv_exp_i_1"):
        self.field = field
        #self.encoder = OneHotEncoder(sparse=False, dtype=np.float32, handle_unknown="ignore")
        try:
            self.encoder = OneHotEncoder(sparse_output=False, dtype=np.float32,
                                         handle_unknown="ignore")
        except TypeError:
            self.encoder = OneHotEncoder(sparse=False, dtype=np.float32,
                                         handle_unknown="ignore")

    def fit(self, X, y=None):
        q = X[self.field].quantile([0.25, 0.5, 0.75])
        self.qs_ = q.values  # q1, q2, q3
        labels = self._bin(X[self.field])
        self.encoder.fit(labels.reshape(-1, 1))
        return self

    def transform(self, X, y=None):
        labels = self._bin(X[self.field])
        ohe = self.encoder.transform(labels.reshape(-1, 1))
        ohe_df = pd.DataFrame(ohe, index=X.index, columns=self.encoder.get_feature_names_out([self.field]))
        out = pd.concat([X, ohe_df], axis=1)
        out["quartile"] = labels
        return out

    def _bin(self, s):
        q1, q2, q3 = self.qs_
        return np.where(s <= q1, "Q1", np.where(s <= q2, "Q2", np.where(s <= q3, "Q3", "Q4")))
