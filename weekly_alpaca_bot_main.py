from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from zoneinfo import ZoneInfo

from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from strategy_weekly_bot_ready import StrategyConfig, PositionState, WeeklyTrendStrategy


NY = ZoneInfo("America/New_York")
UTC = timezone.utc
STATE_PATH = Path(os.getenv("STATE_PATH", "data/weekly_strategy_state.json"))
SYMBOLS_PATH = Path(os.getenv("SYMBOLS_PATH", "symbols.txt"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
BENCHMARK = os.getenv("BENCHMARK", "SPY")
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "900"))
SLIPPAGE_BUFFER = float(os.getenv("SLIPPAGE_BUFFER", "0.0"))
MIN_PRICE = float(os.getenv("MIN_PRICE", "5"))
RUN_WEEKDAY_ONLY = os.getenv("RUN_WEEKDAY_ONLY", "0") == "1"
ALLOW_NEW_ENTRIES = os.getenv("ALLOW_NEW_ENTRIES", "1") == "1"
ALLOW_PARTIAL_EXITS = os.getenv("ALLOW_PARTIAL_EXITS", "1") == "1"
DRY_RUN = os.getenv("DRY_RUN", "1") == "1"


def setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def build_config() -> StrategyConfig:
    return StrategyConfig(
        sma_weeks=int(os.getenv("SMA_WEEKS", "30")),
        ema_exit_weeks=int(os.getenv("EMA_EXIT_WEEKS", "10")),
        vol_avg_weeks=int(os.getenv("VOL_AVG_WEEKS", "10")),
        box_weeks=int(os.getenv("BOX_WEEKS", "3")),
        min_volume_mult=float(os.getenv("MIN_VOLUME_MULT", "1.0")),
        break_even_r=float(os.getenv("BREAK_EVEN_R", "1.0")),
        partial_r=float(os.getenv("PARTIAL_R", "2.5")),
        partial_exit_fraction=float(os.getenv("PARTIAL_EXIT_FRACTION", "0.33")),
        trail_mode=os.getenv("TRAIL_MODE", "prior_2w_low"),
        risk_pct_per_trade=float(os.getenv("RISK_PCT_PER_TRADE", "0.02")),
        max_positions=int(os.getenv("MAX_POSITIONS", "10")),
        min_history_weeks=int(os.getenv("MIN_HISTORY_WEEKS", "40")),
    )


def load_symbols() -> List[str]:
    env_symbols = os.getenv("SYMBOLS")
    if env_symbols:
        return sorted({s.strip().upper() for s in env_symbols.split(",") if s.strip()})
    if SYMBOLS_PATH.exists():
        return sorted({line.strip().upper() for line in SYMBOLS_PATH.read_text().splitlines() if line.strip()})
    raise FileNotFoundError("No encontré símbolos. Usa env SYMBOLS o crea symbols.txt")


def get_clients() -> Tuple[TradingClient, StockHistoricalDataClient]:
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_API_SECRET")
    paper = env_bool("ALPACA_PAPER", True)
    if not key or not secret:
        raise RuntimeError("Faltan ALPACA_API_KEY y/o ALPACA_API_SECRET")
    trading = TradingClient(api_key=key, secret_key=secret, paper=paper)
    data = StockHistoricalDataClient(api_key=key, secret_key=secret)
    return trading, data


def now_ny() -> datetime:
    return datetime.now(tz=NY)


def latest_completed_week_end(reference: Optional[datetime] = None) -> pd.Timestamp:
    ref = reference or now_ny()
    ref_date = ref.date()
    weekday = ref.weekday()  # Mon=0 ... Sun=6
    days_since_friday = (weekday - 4) % 7
    if weekday == 4 and ref.hour < 16:
        days_since_friday = 7
    completed_friday = ref_date - timedelta(days=days_since_friday)
    return pd.Timestamp(completed_friday)


def should_run_weekly(reference: Optional[datetime] = None) -> bool:
    ref = reference or now_ny()
    if RUN_WEEKDAY_ONLY and ref.weekday() >= 5:
        return False
    completed = latest_completed_week_end(ref)
    state = load_state_raw()
    last_processed = state.get("meta", {}).get("last_processed_week")
    return last_processed != str(completed.date())


def bars_to_df(bars) -> pd.DataFrame:
    rows = []
    for bar in bars:
        rows.append({
            "timestamp": pd.Timestamp(bar.timestamp).tz_convert(NY).tz_localize(None),
            "Open": float(bar.open),
            "High": float(bar.high),
            "Low": float(bar.low),
            "Close": float(bar.close),
            "Adj Close": float(bar.close),
            "Volume": float(bar.volume),
        })
    if not rows:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"])
    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    return df


def fetch_daily_bars(data_client: StockHistoricalDataClient, symbols: List[str]) -> Dict[str, pd.DataFrame]:
    start = datetime.now(tz=UTC) - timedelta(days=LOOKBACK_DAYS)
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=start,
        adjustment="all",
        feed=os.getenv("ALPACA_DATA_FEED", "iex"),
    )
    raw = data_client.get_stock_bars(req)
    out: Dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        symbol_bars = raw.data.get(symbol, [])
        out[symbol] = bars_to_df(symbol_bars)
    return out


def position_to_dict(pos: PositionState) -> Dict:
    d = asdict(pos)
    d["entry_date"] = str(pd.Timestamp(pos.entry_date))
    return d


def position_from_dict(d: Dict) -> PositionState:
    d = dict(d)
    d["entry_date"] = pd.Timestamp(d["entry_date"])
    return PositionState(**d)


def load_state_raw() -> Dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"positions": {}, "meta": {}}


def load_state() -> Tuple[Dict[str, PositionState], Dict]:
    raw = load_state_raw()
    positions = {sym: position_from_dict(v) for sym, v in raw.get("positions", {}).items()}
    meta = raw.get("meta", {})
    return positions, meta


def save_state(positions: Dict[str, PositionState], meta: Dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "positions": {sym: position_to_dict(pos) for sym, pos in positions.items()},
        "meta": meta,
    }
    STATE_PATH.write_text(json.dumps(payload, indent=2, default=str))


def get_account_snapshot(trading: TradingClient) -> Tuple[float, float]:
    account = trading.get_account()
    equity = float(account.equity)
    cash = float(account.cash)
    return equity, cash


def get_latest_open_position_qty(trading: TradingClient, symbol: str) -> Optional[float]:
    try:
        p = trading.get_open_position(symbol)
        return float(p.qty)
    except Exception:
        return None


def submit_market_order(trading: TradingClient, symbol: str, qty: int, side: OrderSide) -> None:
    if qty <= 0:
        return
    if DRY_RUN:
        logging.info("[DRY_RUN] %s %s qty=%s", side.value.upper(), symbol, qty)
        return
    order = MarketOrderRequest(symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.DAY)
    trading.submit_order(order)
    logging.info("[ORDER] %s %s qty=%s", side.value.upper(), symbol, qty)


def get_next_session_entry_price(df_daily: pd.DataFrame, week_end: pd.Timestamp) -> Optional[Tuple[pd.Timestamp, float]]:
    fut = df_daily[df_daily.index > week_end]
    if fut.empty:
        return None
    entry_date = pd.Timestamp(fut.index[0])
    entry_price = float(fut.iloc[0]["Open"]) * (1.0 + SLIPPAGE_BUFFER)
    return entry_date, entry_price


def score_candidates(strategy: WeeklyTrendStrategy, weekly_map: Dict[str, pd.DataFrame], week_end: pd.Timestamp,
                     open_symbols: set[str]) -> List[Tuple[str, float, pd.Series]]:
    candidates = []
    for symbol, weekly in weekly_map.items():
        if symbol in open_symbols or week_end not in weekly.index:
            continue
        row = weekly.loc[week_end]
        if not bool(row.get("entry_signal", False)):
            continue
        price = float(row["Adj Close"])
        if price < MIN_PRICE:
            continue
        score = strategy.score_candidate(row)
        candidates.append((symbol, score, row))
    return sorted(candidates, key=lambda x: x[1], reverse=True)


def manage_open_positions(
    trading: TradingClient,
    strategy: WeeklyTrendStrategy,
    weekly_map: Dict[str, pd.DataFrame],
    positions: Dict[str, PositionState],
    week_end: pd.Timestamp,
) -> Dict[str, PositionState]:
    updated = dict(positions)

    for symbol in list(updated.keys()):
        if symbol not in weekly_map or week_end not in weekly_map[symbol].index:
            logging.warning("[MANAGE] %s sin datos semanales para %s", symbol, week_end.date())
            continue

        pos = updated[symbol]
        pos = strategy.activate_pending_stop(pos)
        row = weekly_map[symbol].loc[week_end]
        decision = strategy.evaluate_position_week(pos, row)

        logging.info(
            "[MANAGE] %s action=%s reason=%s qty=%s stop=%.4f pending=%s",
            symbol, decision["action"], decision.get("reason"), decision.get("qty"), pos.stop_price,
            decision.get("new_pending_stop"),
        )

        if decision["action"] == "exit_all":
            live_qty = get_latest_open_position_qty(trading, symbol)
            qty = int(math.floor(live_qty if live_qty is not None else pos.shares))
            submit_market_order(trading, symbol, qty, OrderSide.SELL)
            del updated[symbol]
            continue

        if decision["action"] == "partial_exit" and ALLOW_PARTIAL_EXITS:
            live_qty = get_latest_open_position_qty(trading, symbol)
            base_qty = int(math.floor(live_qty if live_qty is not None else pos.shares))
            qty = max(1, min(int(decision["qty"]), max(0, base_qty - 1))) if base_qty > 1 else 0
            if qty > 0:
                submit_market_order(trading, symbol, qty, OrderSide.SELL)
                pos.shares = max(0, pos.shares - qty)
                logging.info("[PARTIAL] %s qty=%s remaining_model_qty=%s", symbol, qty, pos.shares)

        updated[symbol] = strategy.apply_week_transition(pos, decision)

    return updated


def open_new_positions(
    trading: TradingClient,
    strategy: WeeklyTrendStrategy,
    daily_map: Dict[str, pd.DataFrame],
    weekly_map: Dict[str, pd.DataFrame],
    positions: Dict[str, PositionState],
    week_end: pd.Timestamp,
) -> Dict[str, PositionState]:
    if not ALLOW_NEW_ENTRIES:
        return positions

    updated = dict(positions)
    equity, cash = get_account_snapshot(trading)
    slots = max(0, strategy.cfg.max_positions - len(updated))
    if slots <= 0:
        logging.info("[ENTRY] Sin slots disponibles")
        return updated

    candidates = score_candidates(strategy, weekly_map, week_end, set(updated.keys()))[:slots]
    logging.info("[ENTRY] Candidatos=%s", [c[0] for c in candidates])

    for symbol, score, row in candidates:
        nxt = get_next_session_entry_price(daily_map[symbol], week_end)
        if nxt is None:
            logging.warning("[ENTRY] %s sin próxima apertura después de %s", symbol, week_end.date())
            continue
        entry_date, entry_price = nxt
        model_pos = strategy.build_position(symbol, entry_date, entry_price, row, equity, cash)
        if model_pos is None:
            logging.info("[ENTRY] %s descartado por sizing/stop inválido", symbol)
            continue

        submit_market_order(trading, symbol, int(model_pos.shares), OrderSide.BUY)
        updated[symbol] = model_pos
        cash -= model_pos.shares * entry_price
        logging.info(
            "[ENTRY] %s score=%.4f qty=%s entry_est=%.4f stop=%.4f",
            symbol, score, model_pos.shares, entry_price, model_pos.stop_price,
        )

    return updated


def build_weekly_maps(strategy: WeeklyTrendStrategy, daily_map: Dict[str, pd.DataFrame], benchmark: str) -> Dict[str, pd.DataFrame]:
    if benchmark not in daily_map or daily_map[benchmark].empty:
        raise RuntimeError(f"No hay datos del benchmark {benchmark}")
    bench_weekly = strategy.to_weekly(daily_map[benchmark])
    out = {}
    for symbol, df in daily_map.items():
        if symbol == benchmark or df.empty:
            continue
        weekly = strategy.to_weekly(df)
        if len(weekly) < strategy.cfg.min_history_weeks:
            continue
        out[symbol] = strategy.add_indicators(weekly, bench_weekly)
    return out


def mark_processed(meta: Dict, week_end: pd.Timestamp) -> Dict:
    meta = dict(meta)
    meta["last_processed_week"] = str(week_end.date())
    meta["updated_at"] = str(now_ny())
    return meta


def reconcile_state_with_broker(trading: TradingClient, positions: Dict[str, PositionState]) -> Dict[str, PositionState]:
    reconciled = {}
    for symbol, pos in positions.items():
        live_qty = get_latest_open_position_qty(trading, symbol)
        if live_qty is None or live_qty <= 0:
            logging.warning("[RECON] %s no existe en broker. Se elimina del state.", symbol)
            continue
        pos.shares = int(math.floor(live_qty))
        reconciled[symbol] = pos
    return reconciled


def main() -> None:
    setup_logging()
    strategy = WeeklyTrendStrategy(build_config())
    logging.info("[BOOT] config=%s", strategy.cfg)

    if not should_run_weekly():
        logging.info("[SKIP] Ya se procesó la última semana completada o no corresponde correr ahora.")
        return

    trading, data_client = get_clients()
    symbols = load_symbols()
    all_symbols = sorted(set(symbols + [BENCHMARK]))
    logging.info("[DATA] símbolos=%s benchmark=%s", len(symbols), BENCHMARK)

    daily_map = fetch_daily_bars(data_client, all_symbols)
    weekly_map = build_weekly_maps(strategy, daily_map, BENCHMARK)
    week_end = latest_completed_week_end()
    logging.info("[WEEK] última semana completada=%s", week_end.date())

    positions, meta = load_state()
    positions = reconcile_state_with_broker(trading, positions)
    logging.info("[STATE] posiciones actuales=%s", list(positions.keys()))

    positions = manage_open_positions(trading, strategy, weekly_map, positions, week_end)
    positions = open_new_positions(trading, strategy, daily_map, weekly_map, positions, week_end)

    meta = mark_processed(meta, week_end)
    save_state(positions, meta)
    logging.info("[DONE] posiciones finales=%s", list(positions.keys()))


if __name__ == "__main__":
    main()
