import os 
import json 
import math 
import time 
import logging 
from dataclasses import dataclass 
from typing import Dict, List, Optional

import numpy as np 
import pandas as pd 
from zoneinfo import ZoneInfo 
from datetime import datetime

from alpaca.trading.client import TradingClient 
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest 
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus 
from alpaca.data.historical.stock import StockHistoricalDataClient 
from alpaca.data.requests import StockBarsRequest 
from alpaca.data.timeframe import TimeFrame

=========================

CONFIG

=========================

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "") ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "") PAPER = True

TOP_UNIVERSE = [ "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "AVGO", "META", "TSLA", "BRK-B", "WMT", "LLY", "JPM", "XOM", "JNJ", "V", "MA", "COST", "MU", "ORCL", "NFLX", "CVX", "ABBV", "PLTR", "BAC", "KO", "UNH", "HD", "CSCO", "PM", "IBM", "AMD", "LIN", "GE", "CRM", "MCD", "WFC", "ABT", "INTU", "MS", "PEP", "T", "AXP", "ADBE", "DIS", "QCOM", "GS", "MRK", "NOW", "CAT" ] EXCLUDED = {"INTU", "CRM", "AVGO"} ACTIVE_UNIVERSE = [t for t in TOP_UNIVERSE if t not in EXCLUDED] BENCHMARK = "SPY"

RISK_PCT_PER_TRADE = 0.02 MAX_POSITIONS = 10 BOX_WEEKS = 3 SMA_WEEKS = 30 EMA_EXIT_WEEKS = 10 VOL_AVG_WEEKS = 10 MIN_VOLUME_MULT = 1.0 SLIPPAGE_PCT = 0.0005 MIN_HISTORY_WEEKS = 40 LOOKBACK_DAYS = 450 STATE_FILE = "alpaca_weekly_state.json" LOG_FILE = "alpaca_weekly_bot.log" TIMEZONE = ZoneInfo("America/New_York") CHECK_INTERVAL_SECONDS = int(os.getenv("CHECK_INTERVAL_SECONDS", "1800"))  # 30 min ONLY_OPEN_ON_WEEK_START = True

=========================

LOGGING

=========================

def setup_logging() -> None: logging.basicConfig( level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", handlers=[ logging.StreamHandler(), logging.FileHandler(LOG_FILE, encoding="utf-8"), ], force=True, )

=========================

MODEL

=========================

@dataclass class SignalRow: symbol: str score: float entry_ref_close: float stop_price: float

=========================

ALPACA CLIENTS

=========================

def get_clients(): if not ALPACA_API_KEY or not ALPACA_SECRET_KEY: raise RuntimeError("Faltan ALPACA_API_KEY / ALPACA_SECRET_KEY en variables de entorno.")

trading = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=PAPER)
data = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
return trading, data

=========================

STATE

=========================

def load_state() -> dict: if not os.path.exists(STATE_FILE): return {"stops": {}, "last_entry_week": None} with open(STATE_FILE, "r", encoding="utf-8") as f: return json.load(f)

def save_state(state: dict) -> None: with open(STATE_FILE, "w", encoding="utf-8") as f: json.dump(state, f, indent=2)

=========================

DATA

=========================

def fetch_daily_bars(data_client: StockHistoricalDataClient, symbols: List[str]) -> Dict[str, pd.DataFrame]: end = pd.Timestamp.now(tz=TIMEZONE) start = end - pd.Timedelta(days=LOOKBACK_DAYS)

request = StockBarsRequest(
    symbol_or_symbols=symbols,
    timeframe=TimeFrame.Day,
    start=start.to_pydatetime(),
    end=end.to_pydatetime(),
    adjustment="raw",
    feed="iex",
)

bars = data_client.get_stock_bars(request)
out: Dict[str, pd.DataFrame] = {}

for symbol in symbols:
    df = bars.df
    if df.empty:
        continue
    try:
        sdf = df.xs(symbol, level=0).copy()
    except Exception:
        continue
    if sdf.empty:
        continue

    sdf = sdf.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    })
    sdf.index = pd.to_datetime(sdf.index).tz_convert(TIMEZONE).tz_localize(None)
    sdf["Adj Close"] = sdf["Close"]
    out[symbol] = sdf[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].dropna()

return out

def to_weekly(df: pd.DataFrame) -> pd.DataFrame: weekly = pd.DataFrame() weekly["Open"] = df["Open"].resample("W-FRI").first() weekly["High"] = df["High"].resample("W-FRI").max() weekly["Low"] = df["Low"].resample("W-FRI").min() weekly["Close"] = df["Close"].resample("W-FRI").last() weekly["Adj Close"] = df["Adj Close"].resample("W-FRI").last() weekly["Volume"] = df["Volume"].resample("W-FRI").sum() return weekly.dropna()

def add_indicators(weekly: pd.DataFrame, spy_weekly: pd.DataFrame) -> pd.DataFrame: w = weekly.copy() w["sma30"] = w["Adj Close"].rolling(SMA_WEEKS).mean() w["ema10"] = w["Adj Close"].ewm(span=EMA_EXIT_WEEKS, adjust=False).mean() w["vol_avg"] = w["Volume"].rolling(VOL_AVG_WEEKS).mean() w["box_high_prev"] = w["High"].shift(1).rolling(BOX_WEEKS).max() w["box_low_prev"] = w["Low"].shift(1).rolling(BOX_WEEKS).min() w["sma30_up"] = w["sma30"] > w["sma30"].shift(1)

aligned_spy = spy_weekly["Adj Close"].reindex(w.index).ffill()
rel = w["Adj Close"] / aligned_spy
w["rs_ratio"] = rel
w["rs_ma"] = rel.rolling(13).mean()
w["rs_ok"] = w["rs_ratio"] > w["rs_ma"]

w["trend_ok"] = (w["Adj Close"] > w["sma30"]) & w["sma30_up"]
w["vol_ok"] = w["Volume"] >= (w["vol_avg"] * MIN_VOLUME_MULT)
w["breakout"] = w["Adj Close"] > w["box_high_prev"]
w["entry_signal"] = w["trend_ok"] & w["vol_ok"] & w["breakout"] & w["rs_ok"]
w["exit_signal"] = w["Adj Close"] < w["ema10"]
return w

=========================

ACCOUNT / POSITIONS

=========================

def get_equity(trading: TradingClient) -> float: account = trading.get_account() return float(account.equity)

def get_positions_map(trading: TradingClient) -> Dict[str, dict]: positions = trading.get_all_positions() out = {} for p in positions: out[p.symbol] = { "qty": float(p.qty), "avg_entry_price": float(p.avg_entry_price), "market_value": float(p.market_value), "side": p.side, } return out

def get_open_order_symbols(trading: TradingClient) -> set: try: request = GetOrdersRequest(status=QueryOrderStatus.OPEN) orders = trading.get_orders(filter=request) except TypeError: # Compatibilidad con versiones distintas del SDK try: orders = trading.list_orders(status="open") except Exception: orders = trading.get_orders()

open_symbols = set()
for o in orders:
    status = str(getattr(o, "status", "")).lower()
    if status == "open":
        open_symbols.add(o.symbol)
return open_symbols

=========================

SIGNALS

=========================

def build_signals(data_map: Dict[str, pd.DataFrame]) -> tuple[Dict[str, pd.DataFrame], pd.Timestamp]: if BENCHMARK not in data_map: raise RuntimeError("No se pudo descargar SPY para fuerza relativa.")

spy_weekly = to_weekly(data_map[BENCHMARK])
weekly_map = {}

for symbol in ACTIVE_UNIVERSE:
    if symbol not in data_map:
        continue
    wk = to_weekly(data_map[symbol])
    if len(wk) < MIN_HISTORY_WEEKS:
        continue
    weekly_map[symbol] = add_indicators(wk, spy_weekly)

if spy_weekly.empty:
    raise RuntimeError("SPY semanal está vacío.")

last_completed_week = spy_weekly.index.max()
return weekly_map, last_completed_week

def get_candidates(weekly_map: Dict[str, pd.DataFrame], positions_map: Dict[str, dict], open_order_symbols: set) -> List[SignalRow]: candidates: List[SignalRow] = []

for symbol, weekly in weekly_map.items():
    if symbol in positions_map or symbol in open_order_symbols:
        continue
    row = weekly.iloc[-1]
    if not bool(row["entry_signal"]):
        continue
    stop_price = float(row["box_low_prev"])
    entry_ref_close = float(row["Adj Close"])
    if math.isnan(stop_price) or stop_price >= entry_ref_close:
        continue
    score = float(row["rs_ratio"] / row["rs_ma"]) + float(row["Volume"] / max(row["vol_avg"], 1))
    candidates.append(SignalRow(symbol, score, entry_ref_close, stop_price))

return sorted(candidates, key=lambda x: x.score, reverse=True)

=========================

ORDERS

=========================

def submit_buy(trading: TradingClient, symbol: str, qty: int) -> None: order = MarketOrderRequest( symbol=symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY, ) trading.submit_order(order_data=order)

def submit_sell(trading: TradingClient, symbol: str, qty: float) -> None: order = MarketOrderRequest( symbol=symbol, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY, ) trading.submit_order(order_data=order)

=========================

BOT LOGIC

=========================

def is_week_start_trading_day(now_ny: datetime) -> bool: return now_ny.weekday() == 0

def manage_exits(trading: TradingClient, weekly_map: Dict[str, pd.DataFrame], positions_map: Dict[str, dict], state: dict) -> None: open_order_symbols = get_open_order_symbols(trading) stops = state.setdefault("stops", {})

for symbol, pos in positions_map.items():
    if symbol not in weekly_map:
        continue
    if symbol in open_order_symbols:
        continue

    weekly = weekly_map[symbol]
    row = weekly.iloc[-1]
    qty = float(pos["qty"])
    current_stop = float(stops.get(symbol, 0))

    latest_box_stop = float(row["box_low_prev"]) if not math.isnan(float(row["box_low_prev"])) else current_stop
    if latest_box_stop > current_stop:
        current_stop = latest_box_stop
        stops[symbol] = current_stop

    if bool(row["exit_signal"]):
        logging.info(f"[EXIT EMA10] {symbol} qty={qty}")
        submit_sell(trading, symbol, qty)
        continue

    daily_close = float(weekly.iloc[-1]["Close"])
    if current_stop > 0 and daily_close <= current_stop:
        logging.info(f"[EXIT STOP] {symbol} qty={qty} stop={current_stop:.2f} close={daily_close:.2f}")
        submit_sell(trading, symbol, qty)
        continue

    if symbol not in stops and current_stop > 0:
        stops[symbol] = current_stop

def open_new_positions(trading: TradingClient, weekly_map: Dict[str, pd.DataFrame], positions_map: Dict[str, dict], state: dict) -> None: now_ny = datetime.now(TIMEZONE) if ONLY_OPEN_ON_WEEK_START and not is_week_start_trading_day(now_ny): logging.info("[INFO] Hoy no es inicio de semana. No se abren nuevas posiciones.") return

open_order_symbols = get_open_order_symbols(trading)
slots = MAX_POSITIONS - len(positions_map) - len(open_order_symbols)
if slots <= 0:
    logging.info("[INFO] Sin cupos para nuevas posiciones.")
    return

candidates = get_candidates(weekly_map, positions_map, open_order_symbols)[:slots]
if not candidates:
    logging.info("[INFO] No hay candidatos nuevos.")
    return

equity = get_equity(trading)
stops = state.setdefault("stops", {})

for c in candidates:
    risk_per_share = c.entry_ref_close - c.stop_price
    if risk_per_share <= 0:
        continue

    risk_budget = equity * RISK_PCT_PER_TRADE
    shares = int(risk_budget // risk_per_share)
    if shares <= 0:
        logging.info(f"[SKIP] {c.symbol} shares<=0 con riesgo {risk_per_share:.2f}")
        continue

    logging.info(f"[BUY] {c.symbol} shares={shares} ref_close={c.entry_ref_close:.2f} stop={c.stop_price:.2f} score={c.score:.2f}")
    submit_buy(trading, c.symbol, shares)
    stops[c.symbol] = c.stop_price

def run_once(): trading, data_client = get_clients() state = load_state()

account = trading.get_account()
logging.info(f"[ACCOUNT] equity={account.equity} cash={account.cash} buying_power={account.buying_power}")

symbols = ACTIVE_UNIVERSE + [BENCHMARK]
data_map = fetch_daily_bars(data_client, symbols)
weekly_map, last_completed_week = build_signals(data_map)
logging.info(f"[WEEK] última semana completada: {last_completed_week.date()}")

positions_map = get_positions_map(trading)
logging.info(f"[POSITIONS] abiertas={list(positions_map.keys())}")

manage_exits(trading, weekly_map, positions_map, state)
open_new_positions(trading, weekly_map, positions_map, state)

state["last_entry_week"] = str(last_completed_week.date())
save_state(state)
logging.info("[DONE] ciclo completado.")

def main_loop(): setup_logging() logging.info("[BOOT] Bot iniciado.") while True: try: run_once() except Exception as e: logging.exception(f"[FATAL CYCLE ERROR] {e}") logging.info(f"[SLEEP] esperando {CHECK_INTERVAL_SECONDS} segundos...") time.sleep(CHECK_INTERVAL_SECONDS)

if name == "main": main_loop()
