from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline

from .features import DropColumns, LagFeatures, HolidayFlags
from .config import TARGET

class DataLoader:
    _rename_map = {
        "DATE": "date",
        "DEMAND ": "demand",
        "Consumer Price Index (CPI)": "cpi",
        "Consumer Confidence Index(CCI)": "cci",
        "Producer Price Index (PPI)": "ppi",
        "Unit Price ($)": "unit_price",
        "POS/ Supply Data": "pos_supply_data",
        "SALES ($)": "sales",
        "Advertising Expenses (SMS)": "adv_exp_sms",
        "Advertising Expenses(Newspaper ads)": "adv_exp_newspaper",
        "Advertising Expenses(Radio)": "adv_exp_radio",
        "Advertising Expenses(TV)": "adv_exp_tv",
        "Advertising Expenses(Internet)": "adv_exp_internet",
        "GRP (NewPaper ads)": "grp_newspaper",
        "GRP(SMS)": "grp_sms",
        "GRP(Radio": "grp_radio",
        "GRP(Internet)": "grp_internet",
        "GRP(TV)": "grp_tv",
    }

    _pipe_r = Pipeline([
        ("drop_tvbrand", DropColumns(["TV Manufacturing Brand", "sales"]))
    ])
    _pipe_h = Pipeline([
        ("lag_feats", LagFeatures(target=TARGET)),
        ("holidays", HolidayFlags(date_col="date", country="China")),
    ])

    @staticmethod
    def load_csv(file_like_or_path) -> pd.DataFrame:
        df = (
            pd.read_csv(file_like_or_path)
            .rename(columns=DataLoader._rename_map)
            .assign(date=lambda d: pd.to_datetime(d["date"]))
        )
        df = DataLoader._pipe_r.fit_transform(df)
        df.set_index("date", inplace=True)

        # compound advertising spend
        adv_cols = [c for c in df.columns if c.startswith("adv_exp_")]
        df["tot_adv_exp"] = df[adv_cols].sum(axis=1)

        # heavy engineering
        df_eng = DataLoader._pipe_h.fit_transform(df.copy())
        return df_eng

def make_splits(df, ratios=(.7, .15, .15)):
    assert abs(sum(ratios) - 1) < 1e-6
    n = len(df); a = int(ratios[0] * n); b = a + int(ratios[1] * n)
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]
