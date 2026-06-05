import pandas as pd
import math
import numpy as np
import yfinance as yf
from dataclasses import dataclass
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------- SVI --------
@dataclass
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float


def svi_total_variance(k: np.ndarray, p: SVIParams) -> np.ndarray:
    return p.a + p.b * (p.rho * (k - p.m) + np.sqrt((k - p.m)**2 + p.sigma**2))


def svi_iv(k: np.ndarray, T: float, p: SVIParams) -> np.ndarray:
    w = np.maximum(svi_total_variance(k, p), 1e-12)
    return np.sqrt(w / max(T, 1e-12))


SVI_SHAPES = {
    "equity":      dict(b=0.06, rho=-0.55, m=0.00, sigma=0.25),
    "commodities": dict(b=0.04, rho=-0.20, m=0.00, sigma=0.22),
    "fx":          dict(b=0.03, rho= 0.00, m=0.00, sigma=0.20),
}

def build_svi_params_by_preset(hv_atm: float, days: int,
                               preset: str = "equity",
                               skew_strength: float = 1.0,
                               center_shift: float = 0.0,
                               sigma_scale: float = 1.0) -> SVIParams:

    if preset not in SVI_SHAPES:
        raise ValueError(f"Unknown preset '{preset}'. Use one of {list(SVI_SHAPES)}")
    T = days / 365.0
    w_atm = (hv_atm ** 2) * T

    shape = SVI_SHAPES[preset].copy()
    b = max(1e-6, shape["b"] * skew_strength)
    rho = np.clip(shape["rho"], -0.999, 0.999)
    m = shape["m"] + center_shift
    sigma = max(1e-6, shape["sigma"] * sigma_scale)

    w0_shape = b * (rho * (-m) + np.sqrt(m**2 + sigma**2))
    a = w_atm - w0_shape

    return SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)


def round_strikes(K_array, step=5):
    return np.round(K_array / step) * step


def round_to_step(x, step):
    return np.round(x / step) * step


def make_smile_df(
    F: float, hv_atm: float, days: int,
    preset: str = "equity",
    skew_strength: float = 1.0,
    center_shift: float = 0.0,
    sigma_scale: float = 1.0,
    strike_low: float = 0.8,
    strike_high: float = 1.2,
    strike_step: int = 5,
    rho_override: float | None = None,
    align_min_to_center: bool = True,
) -> pd.DataFrame:
    p = build_svi_params_by_preset(
        hv_atm, days, preset,
        skew_strength, center_shift, sigma_scale,
        rho_override=rho_override,
        align_min_to_center=align_min_to_center
    )
    T = days / 365.0

    low_abs  = F * strike_low
    high_abs = F * strike_high
    start = math.ceil(low_abs  / strike_step) * strike_step
    stop  = math.floor(high_abs / strike_step) * strike_step
    if stop < start:
        start = math.floor(low_abs / strike_step) * strike_step
        stop  = start + strike_step

    K_grid = np.arange(start, stop + strike_step, strike_step, dtype=float)
    k_grid = np.log(K_grid / F)
    iv = svi_iv(k_grid, T, p)

    return pd.DataFrame({
        "strike": K_grid,
        "Moneyness": K_grid / F,
        "k = ln(K/F)": k_grid,
        "IV": iv
    })


def _adjust_m_for_min_at(center_shift: float, rho: float, sigma: float) -> float:

    denom = np.sqrt(max(1.0 - rho**2, 1e-12))
    return center_shift + rho * sigma / denom


def build_svi_params_by_preset(hv_atm: float, days: int,
                               preset: str = "equity",
                               skew_strength: float = 1.0,
                               center_shift: float = 0.0,
                               sigma_scale: float = 1.0,
                               rho_override: float | None = None,
                               align_min_to_center: bool = True) -> SVIParams:

    if preset not in SVI_SHAPES:
        raise ValueError(f"Unknown preset '{preset}'. Use one of {list(SVI_SHAPES)}")
    T = days / 365.0
    w_atm = (hv_atm ** 2) * T

    shape = SVI_SHAPES[preset].copy()
    b = max(1e-6, shape["b"] * skew_strength)
    rho = np.clip(rho_override if rho_override is not None else shape["rho"], -0.999, 0.999)
    sigma = max(1e-6, shape["sigma"] * sigma_scale)

    if align_min_to_center:
        m = _adjust_m_for_min_at(center_shift, rho, sigma)
    else:
        m = shape["m"] + center_shift

    w0_shape = b * (rho * (-m) + np.sqrt(m**2 + sigma**2))
    a = w_atm - w0_shape
    return SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)


_PRESET_SIGMA = {"equity": 0.25, "commodities": 0.22, "fx": 0.20}
_PRESET_RHO   = {"equity": -0.55, "commodities": -0.20, "fx": 0.00}


def suggest_svi_tuning(L, R, peak_shift_log=None, base_preset=None):
    L = float(L); R = float(R)
    if L <= 0 or R <= 0:
        raise ValueError("L и R должны быть > 0 (лог-длины хвостов).")

    ratio = L / R
    rho = (1.0 - ratio) / (1.0 + ratio)
    rho = float(np.clip(rho, -0.95, 0.95))

    if base_preset is None:
        if rho < -0.1:
            preset = "commodities" if abs(rho) < 0.35 else "equity"
        elif rho > 0.1:
            preset = "fx"
        else:
            preset = "commodities"
    else:
        preset = base_preset

    avg_span = 0.5 * (L + R)
    sigma0 = _PRESET_SIGMA[preset]
    sigma_scale = max(0.5, min(2.5, avg_span / sigma0))

    skew_strength = 1.0 + 0.6 * abs(ratio - 1.0)
    skew_strength = float(np.clip(skew_strength, 0.6, 2.0))

    center_shift = float(peak_shift_log) if peak_shift_log is not None else 0.0

    return {
        "preset": preset,
        "rho_hint": rho,
        "preset_rho_default": _PRESET_RHO[preset],
        "sigma_scale": round(sigma_scale, 3),
        "skew_strength": round(skew_strength, 3),
        "center_shift": round(center_shift, 6),
        "diagnostics": {
            "L": L, "R": R, "ratio_L_over_R": ratio, "avg_span": avg_span, "preset_sigma0": sigma0
        }
    }


def yang_zhang_volatility(dates, open_prices, close_prices, high_prices, low_prices, window_size=10):
    datetime_list = []
    hv_list = []

    for i in range(window_size, len(close_prices)):
        try:
            open_subset = open_prices[i-window_size:i]
            close_subset = close_prices[i-window_size:i]
            high_subset = high_prices[i-window_size:i]
            low_subset = low_prices[i-window_size:i]

            n = len(close_subset)
            if n < 2:
                raise ValueError("Not enough data.")

            r_o = np.log(np.array(open_subset[1:]) / np.array(close_subset[:-1]))
            r_c = np.log(np.array(close_subset) / np.array(open_subset))
            r_k = np.log(np.array(close_subset[1:]) / np.array(close_subset[:-1]))

            epsilon = 1e-6
            mask = (np.abs(r_k) > epsilon) & (np.abs(r_o) > epsilon)
            r_o = r_o[mask]
            r_k = r_k[mask]

            if len(r_o) < 2 or len(r_k) < 2:
                hv = np.nan
            else:
                mean_ro = np.mean(r_o)
                mean_rc = np.mean(r_c)

                sigma_o2 = np.var(r_o, ddof=1)
                sigma_c2 = np.var(r_c, ddof=1) if len(r_c) > 1 else 0
                var_rk = np.var(r_k, ddof=1)

                k = 1 / n * np.sum((1 - r_o / (r_k + epsilon)) ** 2)

                yz_volatility = sigma_o2 + k * sigma_c2 + (1 - k) * var_rk

                hv = np.sqrt(yz_volatility) if yz_volatility >= 0 else np.nan

            datetime_list.append(dates[i])
            hv_list.append(hv)
        except Exception as e:
            datetime_list.append(dates[i])
            hv_list.append(np.nan)

    return datetime_list, hv_list


def calculate_yang_zhang_volatility_series(open_prices, close_prices, high_prices, low_prices, window_size=10):
    yz_volatilities = []

    for i in range(window_size, len(close_prices)):
        yz_volatilities.append(
            yang_zhang_volatility(
                open_prices[i-window_size:i],
                close_prices[i-window_size:i],
                high_prices[i-window_size:i],
                low_prices[i-window_size:i],
            )
        )

    return yz_volatilities


def exponential_moving_average(values, window):
    ema = []
    alpha = 2 / (window + 1)
    for i, value in enumerate(values):
        if i == 0:
            ema.append(value)
        else:
            ema.append(alpha * value + (1 - alpha) * ema[-1])
    return ema


def hv_dataframe(df, symbol, days):
    days_working = int(days / 7 * 5)
    df['change'] = df['Close'][symbol].pct_change(periods=days_working)
    df = df.dropna()

    window = days_working
    window_up = int(window * 1.33)
    window_down = int(window * 0.66)
    w_main, w_up, w_down = 0.5, 0.2, 0.3
    open_prices = list(df[("Open", symbol)])
    high_prices = list(df[("High", symbol)])
    low_prices = list(df[("Low", symbol)])
    close_prices = list(df[("Close", symbol)])
    date_index = 'Date'
    dates = list(df.reset_index()[date_index])
    dates_list, sigma_list = yang_zhang_volatility(dates, open_prices, close_prices, high_prices, low_prices)
    hv_df = pd.DataFrame({})
    hv_df[date_index] = dates_list
    hv_df['Day Sigma'] = sigma_list
    hv_df = hv_df.dropna()
    hv_list = list(exponential_moving_average(list(hv_df['Day Sigma']), window))
    hv_up_list = list(exponential_moving_average(list(hv_df['Day Sigma']), window_up))
    hv_down_list = list(exponential_moving_average(list(hv_df['Day Sigma']), window_down))
    hv_df['Day Mean Sigma'] = hv_list
    hv_df['Day Mean Sigma UP'] = hv_up_list
    hv_df['Day Mean Sigma DOWN'] = hv_down_list
    day_var_blend = (
        (hv_df['Day Mean Sigma']      ** 2) * w_main +
        (hv_df['Day Mean Sigma UP']   ** 2) * w_up   +
        (hv_df['Day Mean Sigma DOWN'] ** 2) * w_down
    )
    hv_df['Day Sigma Blend'] = np.sqrt(day_var_blend)
    hv_df['HV'] = hv_df['Day Sigma Blend'] * np.sqrt(252)
    ma_window = min(max(window * 10, 60), 250)
    hv_df['HV Mean'] = hv_df['HV'].rolling(window=ma_window).mean()
    hv_df = hv_df.set_index(date_index)

    return hv_df


def rs_variance(O, H, L, C):
    # Rogers–Satchell
    return (np.log(H/O) * np.log(H/C) + np.log(L/O) * np.log(L/C))

def ewma_from_variance(v_series, halflife=4):
    # RiskMetrics: sigma_t^2 = (1-λ)*v_{t} + λ*sigma_{t-1}^2, λ = exp(-ln2/halflife)
    lam = float(np.exp(-np.log(2) / max(halflife, 0.5)))
    out = np.empty(len(v_series))
    out[:] = np.nan
    s2 = np.nan
    for i, v in enumerate(v_series):
        if np.isnan(v):
            out[i] = s2
            continue
        if np.isnan(s2):
            s2 = v
        else:
            s2 = lam * s2 + (1 - lam) * v
        out[i] = s2
    return np.sqrt(out)

def fast_vol_channel(df, symbol, ovn_weight=1.0, halflife=4):
    O = df[("Open",  symbol)].values
    H = df[("High",  symbol)].values
    L = df[("Low",   symbol)].values
    C = df[("Close", symbol)].values

    rs_var = rs_variance(O, H, L, C)

    ovn = np.empty_like(C)
    ovn[:] = np.nan
    ovn[1:] = np.log(O[1:] / C[:-1])**2

    v_inst = rs_var + ovn_weight * ovn

    sigma_fast = ewma_from_variance(v_inst, halflife=halflife)
    return sigma_fast


def fast_vol_channel(df, symbol, ovn_weight=1.0, cc_weight=1.0, halflife=4):
    O = df[("Open",  symbol)].values.astype(float)
    H = df[("High",  symbol)].values.astype(float)
    L = df[("Low",   symbol)].values.astype(float)
    C = df[("Close", symbol)].values.astype(float)

    rs_var = rs_variance(O, H, L, C)

    # overnight jump: Open_t vs Close_{t-1}
    ovn = np.empty_like(C)
    ovn[:] = np.nan
    ovn[1:] = np.log(O[1:] / C[:-1]) ** 2

    # close-to-close directional move: Close_t vs Close_{t-1}
    cc = np.empty_like(C)
    cc[:] = np.nan
    cc[1:] = np.log(C[1:] / C[:-1]) ** 2

    v_inst = rs_var + ovn_weight * ovn + cc_weight * cc

    sigma_fast = ewma_from_variance(v_inst, halflife=halflife)
    return sigma_fast


def hv_dataframe_fast(
        df, symbol, days,
        halflife_fast=4, 
        ovn_weight=1.0,
        cc_weight=0.0,
        mix_mode="max", 
        beta=0.5,
        index_coeff=1
    ):

    base = hv_dataframe(df.copy(), symbol, days)

    sigma_fast = fast_vol_channel(
        df, 
        symbol, 
        ovn_weight=ovn_weight,
        cc_weight=cc_weight,
        halflife=halflife_fast
    )

    out = base.copy()
    dates = out.index

    if 'Date' in df.columns:
        idx = df['Date']
    else:
        idx = df.index

    fast_aligned = pd.Series(sigma_fast, index=idx).reindex(dates)

    ann_fast = fast_aligned * np.sqrt(252)
    ann_slow = out['HV']

    if mix_mode == "max":
        ann_blend = np.fmax(ann_fast.values, ann_slow.values)
    else:
        b = float(np.clip(beta, 0.0, 1.0))
        ann_blend = (1 - b) * ann_fast.values + b * ann_slow.values

    out['HV Fast'] = ann_fast
    out['HV Blend'] = ann_blend
    
    out['ind'] = (out['HV Fast'] - out['HV']) / out['HV']
    out["EmergencyRegime"] = (
        out['ind'] >= index_coeff
    ).astype(int)

    close_col = None

    if isinstance(df.columns, pd.MultiIndex):
        if ('Close', symbol) in df.columns:
            close_col = df[('Close', symbol)]
        elif 'Close' in df.columns.get_level_values(0):
            close_col = df.xs('Close', level=0, axis=1).iloc[:, 0]
    elif 'Close' in df.columns:
        close_col = df['Close']

    if close_col is not None:
        close_series = pd.Series(close_col.squeeze().values, index=idx)
        out['Close'] = close_series.reindex(dates)

    return out


def iv_smiles(symbol, df_yahoo, expiration_list):
    basis_price = df_yahoo['Close'][symbol].iloc[-1]
    ticker = yf.Ticker(symbol)
    
    df_call = ticker.option_chain(expiration_list[0]).calls
    df_put = ticker.option_chain(expiration_list[0]).puts
    
    df_call = df_call[df_call['strike'] > basis_price]
    df_put = df_put[df_put['strike'] < basis_price]
    
    df_main = pd.concat([df_put, df_call]).reset_index(drop=True)[['strike', 'impliedVolatility']]
    df_main.rename(columns={'impliedVolatility': expiration_list[0]}, inplace=True)
    
    for expiration in expiration_list[1:]:
        try:
            df_call = ticker.option_chain(expiration).calls
            df_put = ticker.option_chain(expiration).puts
        except ValueError:
            continue
        
        df_call = df_call[df_call['strike'] > basis_price]
        df_put = df_put[df_put['strike'] < basis_price]
        
        df = pd.concat([df_put, df_call]).reset_index(drop=True)[['strike', 'impliedVolatility']]
        df.rename(columns={'impliedVolatility': expiration}, inplace=True)
    
        df_main = pd.merge(df_main, df, on='strike', how='outer')
    
    return df_main


def hv_iv_fig(symbol, move_mult):
    df_yahoo = yf.download(symbol, start='2019-01-01', interval='1d')
    hv_df = hv_dataframe_fast(df_yahoo, symbol, 40, halflife_fast=2, ovn_weight=1.0, cc_weight=0.25, mix_mode="blend", beta=0.3, index_coeff=1)
    hv_df = hv_df.reset_index()
    hv_df = hv_df[['Date', 'HV Blend', 'HV Fast', 'EmergencyRegime']]
    hv_df['HV Blend'] = hv_df['HV Blend'] * 100
    hv_df['HV Fast'] = hv_df['HV Fast'] * 100
    
    df_yahoo_move = yf.download('^MOVE', start='2019-01-01', interval='1d')['Close']
    df_yahoo_move = df_yahoo_move.reset_index()
    df_yahoo_move['^MOVE'] = df_yahoo_move['^MOVE'] / move_mult + 0.2
    
    df_yahoo_move['IV Change'] = df_yahoo_move['^MOVE'].diff(9)
    
    merged_df = hv_df.merge(df_yahoo_move, on='Date')
    # merged_df = merged_df[-750:]
    
    merged_df = merged_df.rename(columns={
        'HV Blend': 'HV',
        '^MOVE': 'IV'
    })
    
    merged_df['HV mean'] = merged_df['HV'].rolling(window=9).mean()
    merged_df['IV shift'] = merged_df['IV'].shift(9)
    merged_df['IV/HV diff'] = merged_df['IV shift'] - merged_df['HV mean']
    
    merged_df['Result'] = merged_df['IV/HV diff'] + -merged_df['IV Change']
    
    y = merged_df['IV/HV diff']
    y_pos = y.where(y >= 0)
    y_neg = y.where(y < 0)
    
    fig_hv22 = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.8, 0.2],
        vertical_spacing=0.02
    )
    
    fig_hv22.update_layout(
        title=f'{symbol[:2]} Yang-Zhang HV 20: {round(merged_df["HV"].iloc[-1], 2)}% vs ATM IV 22: {round(merged_df["IV"].iloc[-1], 2)}%',
        width=1500,
        height=700,
    )
    
    fig_hv22.add_trace(
        go.Scatter(
            x=merged_df['Date'],
            y=merged_df['IV'],
            mode='lines',
            name='IV'
        ),
        row=1, col=1
    )
    
    fig_hv22.add_trace(
        go.Scatter(
            x=merged_df['Date'],
            y=merged_df['HV'],
            mode='lines',
            name='HV'
        ),
        row=1, col=1
    )
    
    fig_hv22.add_trace(
        go.Scatter(
            x=merged_df['Date'],
            y=y_pos,
            fill='tozeroy',
            mode='lines',
            name='IV/HV diff > 0'
        ),
        row=2, col=1
    )
    
    fig_hv22.add_trace(
        go.Scatter(
            x=merged_df['Date'],
            y=y_pos,
            fill='tozeroy',
            fillcolor='rgba(0, 200, 0, 0.3)',
            line=dict(color='rgba(0, 200, 0, 1)'),
            mode='lines',
            name='IV/HV diff > 0'
        ),
        row=2, col=1
    )
    
    fig_hv22.add_trace(
        go.Scatter(
            x=merged_df['Date'],
            y=y_neg,
            fill='tozeroy',
            fillcolor='rgba(200, 0, 0, 0.3)',
            line=dict(color='rgba(200, 0, 0, 1)'),
            mode='lines',
            name='IV/HV diff < 0',
        ),
        row=2, col=1
    )
    
    fig_hv22.add_hline(y=0, row=2, col=1)
    
    hv_df_fig = px.line(
        hv_df[-252:],
        x='Date',
        y=['HV Blend', 'HV Fast'],
        width=1500,
        height=700,
        title=f'{symbol[:2]} Yang-Zhang HV 20: {round(hv_df["HV Blend"].iloc[-1], 2)}%, Fast: {round(hv_df["HV Fast"].iloc[-1], 2)}%'
    )
    
    cur_hv = hv_df["HV Blend"].iloc[-1] / 100
    cur_fut = df_yahoo['Close'][symbol].iloc[-1]
    cur_iv = df_yahoo_move['^MOVE'].iloc[-1]
    
    return fig_hv22, hv_df_fig, cur_hv, cur_fut, df_yahoo, cur_iv, merged_df, hv_df


def float_to_cme_price(value: float) -> str:
    points = int(value)

    frac_128 = round((value - points) * 128)

    frac_32 = frac_128 // 4
    q = frac_128 % 4
    
    if q == 1:
        q = 2
    if q == 2:
        q = 5
    if q == 3:
        q = 7

    return f"{points}-{frac_32:02d}'{q}"





