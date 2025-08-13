import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error

def _parse_seasonal_lag(name: str) -> int:
    if name == "persistence":
        return 1
    m = re.fullmatch(r"seasonal_(\d+)", name)
    if m:
        return int(m.group(1))
    raise ValueError("unsupported baseline; use 'persistence' or 'seasonal_<k>'")

def skill_horizon_expanding(
        y_true: pd.Series,
        y_pred: pd.Series,
        *,
        baseline: str = "persistence",
        threshold: float = 0.0,
        history: pd.Series | None = None
    ) -> tuple[int, pd.DataFrame]:
    """Expanding-window RMSE skill horizon, optionally seeded with pre-test history."""
    if not y_true.index.equals(y_pred.index):
        raise ValueError("indices must match")

    L = _parse_seasonal_lag(baseline)

    if history is not None:
        hist = history.dropna()
        if len(hist) < L:
            raise ValueError(f"history must have at least {L} points for baseline '{baseline}'")
        y_concat = pd.concat([hist, y_true])
        y_base_all = y_concat.shift(L)
        y_base = y_base_all.loc[y_true.index]
    else:
        y_base = y_true.shift(L)

    df = pd.DataFrame({"y": y_true, "yhat": y_pred, "ybase": y_base}).copy()
    # trim minimal head if baseline has NaN
    if df["ybase"].isna().any():
        first_valid = int(np.argmax(~df["ybase"].isna().to_numpy()))
        df = df.iloc[first_valid:]

    n = len(df)
    rmse_model, rmse_base = np.empty(n), np.empty(n)

    for k in range(1, n + 1):
        rmse_model[k-1] = np.sqrt(mean_squared_error(df["y"].iloc[:k], df["yhat"].iloc[:k]))
        rmse_base[k-1]  = np.sqrt(mean_squared_error(df["y"].iloc[:k], df["ybase"].iloc[:k]))

    skill = 1 - rmse_model / rmse_base
    df["skill"]      = skill
    df["rmse_model"] = rmse_model
    df["rmse_base"]  = rmse_base

    below = df.loc[df["skill"] < threshold]
    horizon_days = (below.index[0] - df.index[0]).days if not below.empty else 0
    return horizon_days, df

def plot_skill_curve(skill_curve: pd.DataFrame | pd.Series,
                     h_days: int,
                     *,
                     title: str = "Expanding-Window Skill Curve vs. Naïve Baseline",
                     line_color: str = "royalblue") -> go.Figure:
    if isinstance(skill_curve, pd.DataFrame):
        s = skill_curve["skill"]
    else:
        s = skill_curve

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s.index, y=s.values, mode="lines+markers",
        name="Skill = 1 – RMSE_model / RMSE_baseline",
        line=dict(width=2, color=line_color), marker=dict(size=4)
    ))
    fig.add_hline(y=0, line=dict(color="firebrick", width=1.5, dash="dash"),
                  annotation_text="Model = Baseline", annotation_position="top right")

    if h_days > 0 and len(s) >= 1:
        horizon_date = s.index[0] + pd.Timedelta(days=h_days)
        if horizon_date in s.index:
            horizon_skill = s.loc[horizon_date]
        else:
            ix = s.index.get_indexer([horizon_date], method="nearest")[0]
            horizon_date = s.index[ix]
            horizon_skill = s.iloc[ix]

        fig.add_shape(type="line", x0=horizon_date, x1=horizon_date,
                      y0=min(0, float(s.min())), y1=float(s.max()) * 1.1,
                      line=dict(color="seagreen", width=2, dash="dot"))
        fig.add_annotation(x=horizon_date, y=float(s.max()) * 0.9,
                           text=f"Horizon ≈ {h_days} days<br>Skill: {horizon_skill:.3f}",
                           showarrow=True, arrowhead=2, arrowcolor="seagreen",
                           font=dict(color="seagreen", size=10),
                           bgcolor="rgba(255,255,255,0.85)", bordercolor="seagreen", borderwidth=1)
        fig.add_trace(go.Scatter(x=[horizon_date], y=[horizon_skill], mode="markers",
                                 marker=dict(size=8, symbol="diamond"), name=f"Horizon ({h_days} d)", showlegend=False))

    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Skill score",
                      hovermode="x unified", template="plotly_white",
                      margin=dict(t=60), height=500,
                      legend=dict(x=0.31, y=0.99, xanchor="center", yanchor="top"))
    fig.update_xaxes(type="date")
    return fig

