"""
BRT Intraday Strategy v2 — Single Month Backtester
===================================================
Runs the strategy on a single month's 1‑minute data and outputs daily P&L.

Usage:
    python brt_month_backtest.py --csv data_2024-01.csv --capital 100000 --full 75 --half 1

Output:
    Daily results printed to console and saved to CSV.
    Monthly summary printed.
"""

import argparse
import sys
import math
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass

# --- Configuration ------------------------------------------------------------
@dataclass
class Config:
    monthly_capital: float = 100_000
    full_size: float = 75.0
    half_size: float = 1.0
    risk_free_annual: float = 0.065

# --- Helper functions ---------------------------------------------------------
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    # Find datetime column
    dt_col = None
    for c in df.columns:
        if "datetime" in c or "date" in c or "time" in c or "timestamp" in c:
            dt_col = c
            break
    if dt_col is None:
        raise ValueError(f"No datetime column found. Columns: {list(df.columns)}")

    # Parse datetime (handles timezone like +0530)
    df["dt"] = pd.to_datetime(df[dt_col], errors='coerce')
    df = df.dropna(subset=["dt"])
    df = df.sort_values("dt").reset_index(drop=True)

    # Ensure OHLC columns exist and are numeric
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=list(required))

    # Add derived columns
    df["date"] = df["dt"].dt.date
    df["hour"] = df["dt"].dt.hour
    df["minute"] = df["dt"].dt.minute

    # Determine month (for summary, but we already have a single month file)
    month_period = df["dt"].dt.to_period("M").iloc[0] if not df.empty else None
    print(f"[DATA] Loaded {len(df):,} rows | "
          f"{df['date'].nunique()} trading days | "
          f"Period: {df['dt'].min().date()} → {df['dt'].max().date()} "
          f"(month {month_period})")
    return df

def sharpe(daily_pnl: list, capital: float, rf_annual: float = 0.065) -> float:
    if len(daily_pnl) < 2:
        return float("nan")
    arr = np.array(daily_pnl, dtype=float)
    rf_daily = rf_annual / 252
    excess = arr - rf_daily * capital
    if np.std(excess, ddof=1) == 0:
        return float("nan")
    return float(np.mean(excess) / np.std(excess, ddof=1) * math.sqrt(252))

# --- Single-day simulation ----------------------------------------------------
def simulate_day(day_df: pd.DataFrame, prev_close: float,
                 consecutive_up: bool, config: Config) -> list:
    """
    Simulates one trading day.
    consecutive_up: True if the last two days' closes were both higher than the previous day.
    Returns list of trade dicts.
    """
    trades = []

    def bar_at(h, m):
        return day_df[(day_df["hour"] == h) & (day_df["minute"] == m)]

    def bars_in(h1, m1, h2, m2):
        t = day_df["hour"] * 60 + day_df["minute"]
        t1 = h1 * 60 + m1
        t2 = h2 * 60 + m2
        return day_df[(t >= t1) & (t < t2)]

    # --- 9:15 open ------------------------------------------------------------
    b915 = bar_at(9, 15)
    if b915.empty:
        return trades
    day_open = b915.iloc[0]["open"]

    # --- 9:16 bias ------------------------------------------------------------
    first = b915.iloc[0]
    bias = -1 if first["close"] < first["open"] else 1

    # --- 9:16–9:29 ORB + H2 tracking ------------------------------------------
    orb_bars = bars_in(9, 15, 9, 30)
    if orb_bars.empty:
        return trades

    orb_high_run = orb_bars["high"].max()
    orb_low_run  = orb_bars["low"].min()
    ever_above   = orb_bars["high"].max() > day_open
    ever_below   = orb_bars["low"].min()  < day_open

    # --- 9:29 lock ------------------------------------------------------------
    b929 = bar_at(9, 29)
    if b929.empty:
        fallback = bars_in(9, 15, 9, 30)
        if fallback.empty:
            return trades
        b929_row = fallback.iloc[-1]
    else:
        b929_row = b929.iloc[0]

    orb_high = orb_high_run
    orb_low  = orb_low_run
    f15 = 1 if b929_row["close"] >= day_open else -1
    h2_short = not ever_above
    h2_long  = not ever_below

    # --- Sizing helper --------------------------------------------------------
    def calc_size(direction, all_agree):
        base = config.full_size if all_agree else config.half_size
        # Halve if long trade and last two days were up (consecutive_up)
        if direction == 1 and consecutive_up:
            base = max(0.5, base / 2)
        return base

    # --- 9:30 decision --------------------------------------------------------
    b930 = bar_at(9, 30)
    if b930.empty:
        return trades
    b930_row = b930.iloc[0]

    morning_dir = 0
    morning_on = False
    entry_price = None
    entry_stop = None
    entry_size = None
    entry_time = None
    trade_step = None
    wait_d = False
    no_trade = False

    # Step A: H2
    if h2_short:
        agree = bias == -1
        sz = calc_size(-1, agree)
        morning_dir = -1
        morning_on = True
        entry_price = b930_row["open"]
        entry_stop = day_open
        entry_size = sz
        entry_time = b930_row["dt"]
        trade_step = "A-SHORT"

    elif h2_long:
        agree = bias == 1
        sz = calc_size(1, agree)
        morning_dir = 1
        morning_on = True
        entry_price = b930_row["open"]
        entry_stop = day_open
        entry_size = sz
        entry_time = b930_row["dt"]
        trade_step = "A-LONG"

    else:
        # Step B: ORB breakout
        broke_up = b930_row["close"] > orb_high
        broke_dn = b930_row["close"] < orb_low

        if broke_up and not broke_dn and not ever_below:
            all_agree = (f15 == 1 and bias == 1)
            sz = calc_size(1, all_agree)
            morning_dir = 1
            morning_on = True
            entry_price = b930_row["open"]
            entry_stop = orb_low
            entry_size = sz
            entry_time = b930_row["dt"]
            trade_step = "B-LONG"

        elif broke_dn and not broke_up and not ever_above:
            all_agree = (f15 == -1 and bias == -1)
            sz = calc_size(-1, all_agree)
            morning_dir = -1
            morning_on = True
            entry_price = b930_row["open"]
            entry_stop = orb_high
            entry_size = sz
            entry_time = b930_row["dt"]
            trade_step = "B-SHORT"

        elif broke_up and broke_dn:
            no_trade = True
            trade_step = "C-BOTH"

        else:
            # Step D: wait till 9:45
            wait_d = True
            trade_step = "D-WAIT"

    # Step D: retry at 9:45
    if wait_d and not morning_on and not no_trade:
        b945 = bar_at(9, 45)
        if not b945.empty:
            b945_row = b945.iloc[0]
            broke_up2 = b945_row["close"] > orb_high
            broke_dn2 = b945_row["close"] < orb_low

            if broke_up2 and not broke_dn2:
                all_agree = (f15 == 1 and bias == 1)
                sz = calc_size(1, all_agree)
                morning_dir = 1
                morning_on = True
                entry_price = b945_row["open"]
                entry_stop = orb_low
                entry_size = sz
                entry_time = b945_row["dt"]
                trade_step = "D-LONG"

            elif broke_dn2 and not broke_up2:
                all_agree = (f15 == -1 and bias == -1)
                sz = calc_size(-1, all_agree)
                morning_dir = -1
                morning_on = True
                entry_price = b945_row["open"]
                entry_stop = orb_high
                entry_size = sz
                entry_time = b945_row["dt"]
                trade_step = "D-SHORT"

            else:
                no_trade = True
                trade_step = "D-NO TRADE"

    # --- Morning trade management ----------------------------------------------
    if morning_on:
        mgmt_bars = bars_in(9, 30, 15, 16)   # includes 15:15
        exit_price = None
        exit_time = None
        exit_reason = None

        if mgmt_bars.empty:
            # No data after entry – cannot exit, skip trade
            pass
        else:
            for _, bar in mgmt_bars.iterrows():
                bh = bar["hour"]
                bm = bar["minute"]

                # Stop hit
                if morning_dir == 1 and bar["low"] <= entry_stop:
                    exit_price = entry_stop
                    exit_time = bar["dt"]
                    exit_reason = "STOP"
                    break
                if morning_dir == -1 and bar["high"] >= entry_stop:
                    exit_price = entry_stop
                    exit_time = bar["dt"]
                    exit_reason = "STOP"
                    break

                # Signal failure (re‑enter ORB range) — exit at the close of the violating bar
                if morning_dir == 1 and bar["low"] < orb_high:
                    exit_price = bar["close"]
                    exit_time = bar["dt"]
                    exit_reason = "SIG_FAIL"
                    break
                if morning_dir == -1 and bar["high"] > orb_low:
                    exit_price = bar["close"]
                    exit_time = bar["dt"]
                    exit_reason = "SIG_FAIL"
                    break

                # 15:15 hard close
                if bh == 15 and bm == 15:
                    exit_price = bar["close"]
                    exit_time = bar["dt"]
                    exit_reason = "15:15"
                    break

        if exit_price is None:
            # Fallback: close at last bar of day
            last = day_df.iloc[-1] if not day_df.empty else None
            if last is not None:
                exit_price = last["close"]
                exit_time = last["dt"]
                exit_reason = "EOD"
            else:
                return trades

        if exit_price is not None:
            raw_pnl = (exit_price - entry_price) * morning_dir * entry_size
            trades.append({
                "type": "MORNING",
                "step": trade_step,
                "entry_time": entry_time,
                "entry_price": round(entry_price, 2),
                "exit_time": exit_time,
                "exit_price": round(exit_price, 2),
                "direction": "LONG" if morning_dir == 1 else "SHORT",
                "size": entry_size,
                "pnl": round(raw_pnl, 2),
                "exit_reason": exit_reason,
            })

    # --- H13: 15:28 BUY / 15:29 SELL ------------------------------------------
    b1528 = bar_at(15, 28)
    b1529 = bar_at(15, 29)
    if not b1528.empty and not b1529.empty:
        h13_entry = b1528.iloc[0]["close"]
        h13_exit  = b1529.iloc[0]["close"]
        h13_pnl   = (h13_exit - h13_entry) * config.full_size
        trades.append({
            "type": "H13",
            "step": "H13",
            "entry_time": b1528.iloc[0]["dt"],
            "entry_price": round(h13_entry, 2),
            "exit_time": b1529.iloc[0]["dt"],
            "exit_price": round(h13_exit, 2),
            "direction": "LONG",
            "size": config.full_size,
            "pnl": round(h13_pnl, 2),
            "exit_reason": "H13",
        })

    return trades

# --- Main backtest for a single month -----------------------------------------
def run_backtest(df: pd.DataFrame, config: Config):
    dates = sorted(df["date"].unique())
    day_groups = {d: g.reset_index(drop=True) for d, g in df.groupby("date")}

    # Variables to track close-to-close consecutive up days
    prev_close = None
    prev_prev_close = None

    all_trades = []        # list of dicts (with date)
    daily_pnl = {}         # date -> total pnl

    for i, date in enumerate(dates):
        day_df = day_groups[date]
        last_bar = day_df.iloc[-1]
        current_close = last_bar["close"]

        # Determine if the last two days were both up (close > previous close)
        consecutive_up = False
        if i >= 2 and prev_prev_close is not None and prev_close is not None:
            # Yesterday's close > day-before's close AND today's close > yesterday's close?
            # Wait, we are evaluating before the day's trades, so we use previous days' data.
            # Actually consecutive_up should be based on the two days *before* today.
            # i=0: first day, no history; i=1: one previous day only.
            if i >= 2:
                # The two days prior to today are the ones at indices i-2 and i-1
                # We have prev_prev_close = close of day i-2, prev_close = close of day i-1
                consecutive_up = (prev_close > prev_prev_close) and (current_close > prev_close)
        # For i=0 and i=1, consecutive_up remains False

        day_trades = simulate_day(day_df, prev_close or 0, consecutive_up, config)

        # Tag trades with date and add to all_trades
        for t in day_trades:
            t["date"] = str(date)
            all_trades.append(t)

        # Calculate total P&L for the day (sum of morning + H13)
        total_day_pnl = sum(t["pnl"] for t in day_trades)
        daily_pnl[str(date)] = total_day_pnl

        # Update prev_prev_close and prev_close for next iteration
        prev_prev_close = prev_close
        prev_close = current_close

    # Build daily results dataframe
    daily_records = []
    for date in dates:
        date_str = str(date)
        total_pnl = daily_pnl.get(date_str, 0.0)
        # Extract morning and H13 pnl for this day
        day_trades = [t for t in all_trades if t["date"] == date_str]
        morn_pnl = sum(t["pnl"] for t in day_trades if t["type"] == "MORNING")
        h13_pnl  = sum(t["pnl"] for t in day_trades if t["type"] == "H13")
        daily_records.append({
            "date": date_str,
            "morning_pnl": morn_pnl,
            "h13_pnl": h13_pnl,
            "total_pnl": total_pnl,
            "trades_count": len(day_trades)
        })

    daily_df = pd.DataFrame(daily_records)

    # Monthly summary
    month_period = str(df["dt"].dt.to_period("M").iloc[0]) if not df.empty else "Unknown"
    total_month_pnl = sum(daily_pnl.values())
    total_morn_pnl  = sum(t["pnl"] for t in all_trades if t["type"] == "MORNING")
    total_h13_pnl   = sum(t["pnl"] for t in all_trades if t["type"] == "H13")
    trading_days = len(dates)

    morn_trades = [t for t in all_trades if t["type"] == "MORNING"]
    h13_trades  = [t for t in all_trades if t["type"] == "H13"]
    morn_win_rate = (sum(1 for t in morn_trades if t["pnl"] > 0) / len(morn_trades) * 100) if morn_trades else float("nan")
    h13_win_rate  = (sum(1 for t in h13_trades if t["pnl"] > 0) / len(h13_trades) * 100) if h13_trades else float("nan")

    # Sharpe (using daily P&L)
    daily_pnl_list = [daily_pnl.get(str(d), 0.0) for d in dates]
    monthly_sharpe = sharpe(daily_pnl_list, config.monthly_capital, config.risk_free_annual)

    monthly_return = (total_month_pnl / config.monthly_capital) * 100

    summary = {
        "month": month_period,
        "trading_days": trading_days,
        "total_pnl": round(total_month_pnl, 2),
        "morning_pnl": round(total_morn_pnl, 2),
        "h13_pnl": round(total_h13_pnl, 2),
        "monthly_return": round(monthly_return, 2),
        "sharpe": round(monthly_sharpe, 3) if not math.isnan(monthly_sharpe) else "NaN",
        "n_morning_trades": len(morn_trades),
        "win_rate_morning": round(morn_win_rate, 1) if not math.isnan(morn_win_rate) else "NaN",
        "n_h13_trades": len(h13_trades),
        "win_rate_h13": round(h13_win_rate, 1) if not math.isnan(h13_win_rate) else "NaN",
    }

    return daily_df, summary, all_trades

# --- Output ------------------------------------------------------------------
def print_daily(daily_df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("DAILY P&L")
    print("=" * 80)
    print(daily_df.to_string(index=False, float_format="%.2f"))
    print("=" * 80)

def print_summary(summary: dict, config: Config):
    print("\n" + "=" * 80)
    print("MONTHLY SUMMARY")
    print("=" * 80)
    print(f"Month                 : {summary['month']}")
    print(f"Trading days          : {summary['trading_days']}")
    print(f"Total P&L (points)    : {summary['total_pnl']:,.2f}")
    print(f"Morning P&L           : {summary['morning_pnl']:,.2f}")
    print(f"H13 P&L               : {summary['h13_pnl']:,.2f}")
    print(f"Return (on ₹{config.monthly_capital:,.0f} capital) : {summary['monthly_return']:.2f}%")
    print(f"Sharpe (annualised)   : {summary['sharpe']}")
    print(f"Morning trades        : {summary['n_morning_trades']} (win rate {summary['win_rate_morning']}%)")
    print(f"H13 trades            : {summary['n_h13_trades']} (win rate {summary['win_rate_h13']}%)")
    print("=" * 80)

def save_outputs(daily_df: pd.DataFrame, all_trades: list, summary: dict, out_dir: str = "."):
    out = Path(out_dir)
    out.mkdir(exist_ok=True)

    daily_path = out / "daily_results.csv"
    daily_df.to_csv(daily_path, index=False)
    print(f"\n[SAVED] Daily results → {daily_path}")

    trades_path = out / "all_trades.csv"
    trades_df = pd.DataFrame(all_trades)
    trades_df.to_csv(trades_path, index=False)
    print(f"[SAVED] All trades → {trades_path}")

    # Also save summary as a CSV (one row)
    summary_df = pd.DataFrame([summary])
    summary_path = out / "monthly_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[SAVED] Monthly summary → {summary_path}")

# --- Main --------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="BRT Intraday Strategy v2 – Single Month Backtester")
    parser.add_argument("--csv", required=True, help="Path to monthly 1‑min OHLC CSV")
    parser.add_argument("--out", default=".", help="Output directory (default: current)")
    parser.add_argument("--full", type=float, default=75.0, help="Full position size (default: 75)")
    parser.add_argument("--half", type=float, default=1.0, help="Half position size (default: 1)")
    parser.add_argument("--capital", type=float, default=100000, help="Monthly capital in ₹ (default: 100000)")
    args = parser.parse_args()

    config = Config(
        monthly_capital=args.capital,
        full_size=args.full,
        half_size=args.half,
        risk_free_annual=0.065
    )

    print(f"\n[CONFIG] Capital: ₹{config.monthly_capital:,.0f} | "
          f"Full: {config.full_size} | Half: {config.half_size}")

    df = load_csv(args.csv)
    print("[RUN] Simulating strategy day by day...")
    daily_df, summary, all_trades = run_backtest(df, config)

    print_daily(daily_df)
    print_summary(summary, config)
    save_outputs(daily_df, all_trades, summary, out_dir=args.out)

if __name__ == "__main__":
    main()