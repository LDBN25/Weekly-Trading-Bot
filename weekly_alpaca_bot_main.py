from __future__ import annotations

import json
import logging
import math
import os
import time
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
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest

from strategy_weekly_bot_ready import StrategyConfig, PositionState, WeeklyTrendStrategy
from trade_tracker import record_trade, send_weekly_summary, send_failure_alert


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
DRY_RUN = os.getenv("DRY_RUN", "0") == "1"

# La cuenta tiene entitlement SIP; iex solo cubre ~2-3% del volumen consolidado
# y los stops se calculan sobre esos minimos.
DATA_FEED = os.getenv("ALPACA_DATA_FEED", "sip")
# Revisa los stops contra el cierre diario en cada corrida, no solo una vez por
# semana. Reduce la exposicion desprotegida de 5 sesiones a 1.
DAILY_STOP_CHECK = os.getenv("DAILY_STOP_CHECK", "1") == "1"
# Adopta posiciones que estan en el broker pero no en el state, en vez de
# dejarlas sin gestionar para siempre.
ADOPT_ORPHANS = os.getenv("ADOPT_ORPHANS", "1") == "1"
FILL_POLL_SECONDS = float(os.getenv("FILL_POLL_SECONDS", "2.0"))
# En la apertura Alpaca puede tardar bastante en confirmar. Con 12s el bot daba
# la venta por no ejecutada y volvia a venderla, dejando la cuenta en corto.
FILL_POLL_ATTEMPTS = int(os.getenv("FILL_POLL_ATTEMPTS", "20"))

# Simbolos con una orden ya enviada en esta corrida. Reintentar una venta es
# mucho peor que asumirla enviada: duplicarla abre un corto.
_ORDENES_ENVIADAS: set = set()


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
        # Apagados por defecto: cambian el tamaño de las posiciones y conviene
        # medirlos contra el histórico antes de activarlos.
        max_stop_pct=float(os.getenv("MAX_STOP_PCT", "0")),
        atr_stop_mult=float(os.getenv("ATR_STOP_MULT", "0")),
        atr_weeks=int(os.getenv("ATR_WEEKS", "10")),
        require_bull_regime=env_bool("REQUIRE_BULL_REGIME", True),
        max_per_sector=int(os.getenv("MAX_PER_SECTOR", "0")),
        score_rs_weight=float(os.getenv("SCORE_RS_WEIGHT", "1.0")),
        score_vol_weight=float(os.getenv("SCORE_VOL_WEIGHT", "1.0")),
        min_position_fraction=float(os.getenv("MIN_POSITION_FRACTION", "0")),
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
        feed=DATA_FEED,
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
    known = {f for f in PositionState.__dataclass_fields__}
    return PositionState(**{k: v for k, v in d.items() if k in known})


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


def get_pending_order_symbols(trading: TradingClient) -> set:
    try:
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        orders = trading.get_orders(filter=req)
        symbols = {o.symbol for o in orders}
        if symbols:
            logging.info("[ENTRY] Órdenes pendientes en broker: %s", symbols)
        return symbols
    except Exception as exc:
        logging.warning("[ENTRY] No se pudieron obtener órdenes pendientes: %s", exc)
        return set()


def get_broker_positions(trading: TradingClient) -> Dict[str, Dict]:
    """Una sola llamada en vez de N get_open_position."""
    try:
        return {
            p.symbol: {"qty": float(p.qty), "avg_entry_price": float(p.avg_entry_price)}
            for p in trading.get_all_positions()
        }
    except Exception as exc:
        logging.warning("[RECON] No se pudieron obtener posiciones del broker: %s", exc)
        return {}


def make_client_order_id(symbol: str, reason: str) -> str:
    """Etiqueta la orden con su motivo para poder auditar los fills después."""
    stamp = datetime.now(tz=UTC).strftime("%Y%m%d%H%M%S")
    return f"{reason}_{symbol}_{stamp}"[:48]


def submit_market_order(
    trading: TradingClient,
    symbol: str,
    qty: int,
    side: OrderSide,
    reason: str = "manual",
) -> Optional[Tuple[float, float]]:
    """Envía la orden y espera el fill real.

    Devuelve (precio_promedio, cantidad_ejecutada) o None si no se ejecutó.
    Registrar el precio teórico en vez del fill real era la causa de que el
    historial de trades sobrestimara sistemáticamente el P&L.
    """
    if qty <= 0:
        return None
    if DRY_RUN:
        logging.info("[DRY_RUN] %s %s qty=%s (%s)", side.value.upper(), symbol, qty, reason)
        _ORDENES_ENVIADAS.add(symbol)
        return None

    # Salvaguarda dura: una sola orden por simbolo por corrida.
    if symbol in _ORDENES_ENVIADAS:
        logging.error(
            "[ORDER] %s YA tuvo una orden en esta corrida; se omite %s qty=%s (%s). "
            "Duplicarla abriria un corto.",
            symbol, side.value.upper(), qty, reason,
        )
        return None
    _ORDENES_ENVIADAS.add(symbol)

    order = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=side,
        time_in_force=TimeInForce.DAY,
        client_order_id=make_client_order_id(symbol, reason),
    )
    submitted = trading.submit_order(order)
    logging.info("[ORDER] %s %s qty=%s (%s)", side.value.upper(), symbol, qty, reason)

    # Hay que esperar el estado terminal, no el primer llenado: una orden grande
    # se ejecuta en tramos y cortar en el primero registraba 33 de 58 acciones.
    parcial: Optional[Tuple[float, float]] = None
    for _ in range(FILL_POLL_ATTEMPTS):
        try:
            live = trading.get_order_by_id(submitted.id)
        except Exception as exc:
            logging.warning("[ORDER] %s no se pudo consultar: %s", symbol, exc)
            break
        filled_qty = float(live.filled_qty or 0)
        estado = str(live.status.value)
        if filled_qty > 0 and live.filled_avg_price is not None:
            parcial = (float(live.filled_avg_price), filled_qty)
        if estado == "filled" and parcial:
            logging.info("[FILL] %s qty=%s precio=%.4f", symbol, parcial[1], parcial[0])
            return parcial
        if estado in {"canceled", "expired", "rejected"}:
            if parcial:
                logging.warning(
                    "[ORDER] %s terminó en %s con ejecución parcial %s de %s",
                    symbol, estado, parcial[1], qty,
                )
                return parcial
            logging.warning("[ORDER] %s terminó en %s sin ejecutarse", symbol, estado)
            return None
        time.sleep(FILL_POLL_SECONDS)

    if parcial:
        logging.warning(
            "[ORDER] %s sin estado terminal tras %.0fs; se registra lo ejecutado: %s de %s",
            symbol, FILL_POLL_SECONDS * FILL_POLL_ATTEMPTS, parcial[1], qty,
        )
        return parcial

    logging.warning(
        "[ORDER] %s enviada pero sin confirmación de fill tras %.0fs. Se asume ENVIADA "
        "y se saca del state; la reconciliación de la próxima corrida ajusta si no llenó.",
        symbol, FILL_POLL_SECONDS * FILL_POLL_ATTEMPTS,
    )
    return None


def get_next_session_entry_price(df_daily: pd.DataFrame, week_end: pd.Timestamp) -> Optional[Tuple[pd.Timestamp, float]]:
    """Precio de referencia para dimensionar.

    Antes exigía una barra posterior al viernes, así que el bot nunca entraba si
    corría pre-market del lunes. Ahora cae al último cierre disponible.
    """
    fut = df_daily[df_daily.index > week_end]
    if not fut.empty:
        return pd.Timestamp(fut.index[0]), float(fut.iloc[0]["Open"]) * (1.0 + SLIPPAGE_BUFFER)
    if df_daily.empty:
        return None
    return pd.Timestamp(df_daily.index[-1]), float(df_daily.iloc[-1]["Close"]) * (1.0 + SLIPPAGE_BUFFER)


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


def _log_exit(
    symbol: str,
    pos: PositionState,
    week_end: pd.Timestamp,
    qty: int,
    reason: str,
    modeled_price: float,
    fill: Optional[Tuple[float, float]],
    is_partial: bool,
    exits_log: List[Dict],
) -> None:
    """Registra la salida usando el fill real cuando existe."""
    exit_price = fill[0] if fill else modeled_price
    exit_qty = int(fill[1]) if fill else qty
    if DRY_RUN:
        # Una simulación no puede ensuciar el historial real de operaciones.
        logging.info(
            "[DRY_RUN] no se registra el trade %s %s @ %.4f", symbol, reason, exit_price
        )
        return
    record_trade(
        symbol=symbol,
        entry_date=str(pos.entry_date)[:10],
        entry_price=pos.entry_price,
        initial_shares=pos.initial_shares,
        exit_date=str(week_end.date()),
        exit_price=exit_price,
        exit_shares=exit_qty,
        exit_reason=reason,
        risk_per_share=pos.risk_per_share,
        is_partial=is_partial,
        modeled_exit_price=modeled_price,
    )
    exits_log.append({
        "symbol": symbol,
        "entry_price": pos.entry_price,
        "exit_price": exit_price,
        "exit_reason": reason,
        "gross_pnl": round((exit_price - pos.entry_price) * exit_qty, 2),
        "r_multiple": round((exit_price - pos.entry_price) / pos.risk_per_share, 3)
        if pos.risk_per_share > 0 else 0.0,
        "is_partial": is_partial,
    })


def check_daily_stops(
    trading: TradingClient,
    daily_map: Dict[str, pd.DataFrame],
    positions: Dict[str, PositionState],
    broker: Dict[str, Dict],
) -> Tuple[Dict[str, PositionState], List[Dict]]:
    """Cierra posiciones cuyo cierre diario perforó el stop.

    La estrategia define el stop sobre cierres, no sobre mínimos intradía: una
    orden stop en reposo en el broker se dispararía con cualquier mecha y
    sacaría de los ganadores. Revisando a diario se acota la exposición sin
    cambiar la semántica de la regla.
    """
    updated = dict(positions)
    exits_log: List[Dict] = []
    if not DAILY_STOP_CHECK:
        return updated, exits_log

    for symbol in list(updated.keys()):
        df = daily_map.get(symbol)
        if df is None or df.empty:
            continue
        pos = updated[symbol]
        last_close = float(df["Close"].iloc[-1])
        last_date = pd.Timestamp(df.index[-1])
        if last_close > pos.stop_price:
            continue

        live_qty = float(broker.get(symbol, {}).get("qty", pos.shares))
        qty = int(math.floor(live_qty))
        if qty <= 0:
            logging.warning("[STOP_DIARIO] %s qty=%s no vendible; se conserva en state", symbol, live_qty)
            continue

        logging.warning(
            "[STOP_DIARIO] %s cierre=%.4f <= stop=%.4f → salida", symbol, last_close, pos.stop_price
        )
        fill = submit_market_order(trading, symbol, qty, OrderSide.SELL, reason="stopdia")
        # La orden se envió: sale del state aunque el fill no se haya confirmado
        # a tiempo. Conservarla llevaba a que la lógica semanal la vendiera otra
        # vez y la cuenta quedara en corto.
        _log_exit(symbol, pos, last_date, qty, "stop_diario", last_close, fill, False, exits_log)
        del updated[symbol]

    return updated, exits_log


def manage_open_positions(
    trading: TradingClient,
    strategy: WeeklyTrendStrategy,
    weekly_map: Dict[str, pd.DataFrame],
    positions: Dict[str, PositionState],
    week_end: pd.Timestamp,
    broker: Dict[str, Dict],
) -> Tuple[Dict[str, PositionState], List[Dict]]:
    updated = dict(positions)
    exits_log: List[Dict] = []

    for symbol in list(updated.keys()):
        if symbol in _ORDENES_ENVIADAS:
            # Ya lo tocó la revisión diaria de stops en esta misma corrida.
            logging.info("[MANAGE] %s omitido: ya tuvo orden esta corrida", symbol)
            continue
        if symbol not in weekly_map or week_end not in weekly_map[symbol].index:
            logging.warning("[MANAGE] %s sin datos semanales para %s", symbol, week_end.date())
            continue

        pos = updated[symbol]
        pos = strategy.activate_pending_stop(pos)
        row = weekly_map[symbol].loc[week_end]
        decision = strategy.evaluate_position_week(pos, row)

        # Sin este guardia, apagar ALLOW_PARTIAL_EXITS marcaba partial_taken sin
        # vender y la posición no volvía a tomar parcial nunca más.
        if decision["action"] == "partial_exit" and not ALLOW_PARTIAL_EXITS:
            logging.info("[MANAGE] %s parcial omitido (ALLOW_PARTIAL_EXITS=0)", symbol)
            decision["action"] = "hold"
            decision["qty"] = 0
            decision["partial_taken"] = pos.partial_taken

        logging.info(
            "[MANAGE] %s action=%s reason=%s qty=%s stop=%.4f pending=%s",
            symbol, decision["action"], decision.get("reason"), decision.get("qty"), pos.stop_price,
            decision.get("new_pending_stop"),
        )

        if decision["action"] == "exit_all":
            live_qty = float(broker.get(symbol, {}).get("qty", pos.shares))
            qty = int(math.floor(live_qty))
            if qty <= 0:
                # Fracciones de acción: antes se borraba del state una posición
                # que seguía abierta en el broker y quedaba huérfana.
                logging.warning("[MANAGE] %s qty=%s no vendible; se conserva en state", symbol, live_qty)
                updated[symbol] = strategy.apply_week_transition(pos, decision)
                continue

            reason = decision.get("reason", "exit_all")
            modeled = pos.stop_price if reason == "stop" else float(row["Adj Close"])
            fill = submit_market_order(trading, symbol, qty, OrderSide.SELL, reason=reason)
            _log_exit(symbol, pos, week_end, qty, reason, modeled, fill, False, exits_log)
            del updated[symbol]
            continue

        if decision["action"] == "partial_exit":
            live_qty = float(broker.get(symbol, {}).get("qty", pos.shares))
            base_qty = int(math.floor(live_qty))
            qty = max(1, min(int(decision["qty"]), max(0, base_qty - 1))) if base_qty > 1 else 0
            if qty > 0:
                modeled = float(row["Adj Close"])
                fill = submit_market_order(trading, symbol, qty, OrderSide.SELL, reason="partial")
                if fill is None and not DRY_RUN:
                    logging.warning("[PARTIAL] %s sin fill confirmado; se reintentará", symbol)
                    decision["partial_taken"] = pos.partial_taken
                else:
                    sold = int(fill[1]) if fill else qty
                    pos.shares = max(0, pos.shares - sold)
                    logging.info("[PARTIAL] %s qty=%s restante=%s", symbol, sold, pos.shares)
                    _log_exit(symbol, pos, week_end, qty, decision.get("reason", "partial"),
                              modeled, fill, True, exits_log)
            else:
                decision["partial_taken"] = pos.partial_taken

        updated[symbol] = strategy.apply_week_transition(pos, decision)

    return updated, exits_log


def open_new_positions(
    trading: TradingClient,
    strategy: WeeklyTrendStrategy,
    daily_map: Dict[str, pd.DataFrame],
    weekly_map: Dict[str, pd.DataFrame],
    positions: Dict[str, PositionState],
    week_end: pd.Timestamp,
    bench_weekly: Optional[pd.DataFrame] = None,
    broker: Optional[Dict[str, Dict]] = None,
) -> Tuple[Dict[str, PositionState], List[Dict]]:
    if not ALLOW_NEW_ENTRIES:
        return positions, []

    updated = dict(positions)
    entries_log: List[Dict] = []

    if not strategy.regime_ok(bench_weekly, week_end):
        logging.info("[ENTRY] Régimen bajista en %s: no se abren posiciones nuevas", BENCHMARK)
        return updated, entries_log

    equity, cash = get_account_snapshot(trading)
    slots = max(0, strategy.cfg.max_positions - len(updated))
    if slots <= 0:
        logging.info("[ENTRY] Sin slots disponibles")
        return updated, entries_log

    pending_symbols = get_pending_order_symbols(trading)
    # Lo que el broker ya tiene queda excluido aunque no esté en el state. Sin
    # esto, con ADOPT_ORPHANS=0 las posiciones huérfanas no bloqueaban la
    # entrada y el bot compraba encima de lo que ya tenía.
    held_at_broker = {s for s, v in (broker or {}).items() if v.get("qty", 0) > 0}
    excluded = set(updated.keys()) | pending_symbols | held_at_broker
    if held_at_broker - set(updated.keys()):
        logging.info(
            "[ENTRY] Excluidos por tenencia en broker fuera del state: %s",
            sorted(held_at_broker - set(updated.keys())),
        )
    candidates = score_candidates(strategy, weekly_map, week_end, excluded)
    logging.info("[ENTRY] Candidatos=%s", [c[0] for c in candidates[:slots * 2]])

    for symbol, score, row in candidates:
        if slots <= 0:
            break
        if not strategy.sector_slot_available(symbol, updated):
            logging.info("[ENTRY] %s descartado por tope de sector", symbol)
            continue

        nxt = get_next_session_entry_price(daily_map[symbol], week_end)
        if nxt is None:
            logging.warning("[ENTRY] %s sin precio de referencia", symbol)
            continue
        entry_date, est_price = nxt
        model_pos = strategy.build_position(symbol, entry_date, est_price, row, equity, cash)
        if model_pos is None:
            logging.info("[ENTRY] %s descartado por sizing/stop inválido", symbol)
            continue

        fill = submit_market_order(trading, symbol, int(model_pos.shares), OrderSide.BUY, reason="entry")
        if fill is None and not DRY_RUN:
            logging.warning("[ENTRY] %s sin fill confirmado; no se agrega al state", symbol)
            continue

        # Realinear la posición con lo realmente ejecutado: si se usa el precio
        # modelado, el riesgo por acción y los objetivos R quedan corridos desde
        # el primer día.
        if fill is not None:
            real_price, real_qty = fill
            model_pos.entry_price = real_price
            model_pos.shares = int(real_qty)
            model_pos.initial_shares = int(real_qty)
            model_pos.risk_per_share = max(real_price - model_pos.stop_price, 1e-9)
            model_pos.entry_date = pd.Timestamp(now_ny().date())

        updated[symbol] = model_pos
        slots -= 1
        cash -= model_pos.shares * model_pos.entry_price
        entries_log.append({
            "symbol": symbol,
            "entry_price": model_pos.entry_price,
            "shares": model_pos.shares,
            "stop_price": model_pos.stop_price,
        })
        logging.info(
            "[ENTRY] %s score=%.4f qty=%s entrada=%.4f stop=%.4f",
            symbol, score, model_pos.shares, model_pos.entry_price, model_pos.stop_price,
        )

    return updated, entries_log


def build_weekly_maps(strategy: WeeklyTrendStrategy, daily_map: Dict[str, pd.DataFrame],
                      benchmark: str) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
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
    return out, bench_weekly


def mark_processed(meta: Dict, week_end: pd.Timestamp) -> Dict:
    meta = dict(meta)
    meta["last_processed_week"] = str(week_end.date())
    meta["updated_at"] = str(now_ny())
    return meta


def week_data_available(weekly_map: Dict[str, pd.DataFrame], week_end: pd.Timestamp) -> bool:
    """¿Llegaron ya las barras de la semana que decimos procesar?

    Sin este control, correr un sábado antes de que el feed publicara el viernes
    marcaba la semana como procesada sin haber hecho nada, y se perdía para
    siempre.
    """
    return any(week_end in weekly.index for weekly in weekly_map.values())


def has_meaningful_changes(
    positions_before: Dict[str, PositionState],
    positions_after: Dict[str, PositionState],
) -> bool:
    before_keys = set(positions_before.keys())
    after_keys = set(positions_after.keys())

    if before_keys != after_keys:
        return True

    for symbol in before_keys:
        b = positions_before[symbol]
        a = positions_after[symbol]

        if int(b.shares) != int(a.shares):
            return True

        if float(b.stop_price) != float(a.stop_price):
            return True

        b_pending = None if b.pending_stop_price is None else float(b.pending_stop_price)
        a_pending = None if a.pending_stop_price is None else float(a.pending_stop_price)
        if b_pending != a_pending:
            return True

        if bool(b.break_even_armed) != bool(a.break_even_armed):
            return True

        if bool(b.partial_taken) != bool(a.partial_taken):
            return True

    return False


def reconcile_state_with_broker(
    positions: Dict[str, PositionState],
    broker: Dict[str, Dict],
    weekly_map: Optional[Dict[str, pd.DataFrame]] = None,
    week_end: Optional[pd.Timestamp] = None,
) -> Dict[str, PositionState]:
    reconciled: Dict[str, PositionState] = {}
    removed = []

    for symbol, pos in positions.items():
        live = broker.get(symbol)
        if live is None or live["qty"] <= 0:
            logging.warning("[RECON] %s no existe en broker → eliminado del state.", symbol)
            removed.append(symbol)
            continue
        live_shares = int(math.floor(live["qty"]))
        if live_shares != pos.shares:
            logging.warning(
                "[RECON] %s qty_broker=%s != qty_state=%s → state actualizado.",
                symbol, live_shares, pos.shares,
            )
        pos.shares = live_shares
        reconciled[symbol] = pos

    # Posiciones en el broker que no están en el state. Antes solo se logueaban:
    # quedaban sin stop, sin trailing y sin registro para siempre.
    for symbol, live in broker.items():
        if symbol in reconciled or symbol == BENCHMARK or live["qty"] <= 0:
            continue
        if not ADOPT_ORPHANS or weekly_map is None or week_end is None:
            logging.warning(
                "[RECON] HUÉRFANO: %s en broker (qty=%s) pero no en state. Revisión manual requerida.",
                symbol, live["qty"],
            )
            continue
        weekly = weekly_map.get(symbol)
        if weekly is None or week_end not in weekly.index:
            logging.warning("[RECON] HUÉRFANO %s sin datos semanales; no se adopta.", symbol)
            continue
        row = weekly.loc[week_end]
        stop = float(row.get("box_low_prev", float("nan")))
        entry = float(live["avg_entry_price"])
        if math.isnan(stop) or stop >= entry:
            stop = entry * 0.9
        reconciled[symbol] = PositionState(
            symbol=symbol,
            entry_date=pd.Timestamp(week_end),
            entry_price=entry,
            shares=int(math.floor(live["qty"])),
            initial_shares=int(math.floor(live["qty"])),
            stop_price=stop,
            initial_stop_price=stop,
            risk_per_share=max(entry - stop, 1e-9),
            notes={"adoptado": True},
        )
        logging.warning(
            "[RECON] ADOPTADO %s qty=%s entrada=%.4f stop=%.4f",
            symbol, reconciled[symbol].shares, entry, stop,
        )

    if removed:
        logging.warning("[RECON] Posiciones eliminadas del state: %s", removed)
    logging.info("[RECON] Reconciliación completa. State válido: %s", list(reconciled.keys()))
    return reconciled


def main() -> None:
    setup_logging()
    strategy = WeeklyTrendStrategy(build_config())
    logging.info("[BOOT] config=%s", strategy.cfg)

    run_weekly = should_run_weekly()
    if not run_weekly and not DAILY_STOP_CHECK:
        logging.info("[SKIP] Semana ya procesada y la revisión diaria está apagada.")
        return

    trading, data_client = get_clients()
    symbols = load_symbols()
    all_symbols = sorted(set(symbols + [BENCHMARK]))
    logging.info("[DATA] símbolos=%s benchmark=%s feed=%s", len(symbols), BENCHMARK, DATA_FEED)

    daily_map = fetch_daily_bars(data_client, all_symbols)
    weekly_map, bench_weekly = build_weekly_maps(strategy, daily_map, BENCHMARK)
    week_end = latest_completed_week_end()
    logging.info("[WEEK] última semana completada=%s procesar_semana=%s", week_end.date(), run_weekly)

    positions, meta = load_state()
    broker = get_broker_positions(trading)
    positions = reconcile_state_with_broker(positions, broker, weekly_map, week_end)
    logging.info("[STATE] posiciones actuales=%s", list(positions.keys()))

    positions_before = {sym: PositionState(**asdict(pos)) for sym, pos in positions.items()}

    positions, exits_log = check_daily_stops(trading, daily_map, positions, broker)

    entries_log: List[Dict] = []
    if run_weekly:
        if not week_data_available(weekly_map, week_end):
            logging.warning(
                "[WEEK] Sin barras para %s todavía; no se marca como procesada.", week_end.date()
            )
        else:
            broker = get_broker_positions(trading)
            positions, weekly_exits = manage_open_positions(
                trading, strategy, weekly_map, positions, week_end, broker
            )
            exits_log.extend(weekly_exits)
            positions, entries_log = open_new_positions(
                trading, strategy, daily_map, weekly_map, positions, week_end,
                bench_weekly, broker,
            )
            meta = mark_processed(meta, week_end)

    changed = has_meaningful_changes(positions_before, positions)
    if DRY_RUN:
        # Una simulación no puede dejar rastro: si persiste el state, borra
        # posiciones que nunca se vendieron y marca la semana como procesada.
        logging.info(
            "[DRY_RUN] state NO guardado. Simulado: %s | actual en disco: %s",
            list(positions.keys()), list(load_state()[0].keys()),
        )
    else:
        save_state(positions, meta)
    logging.info("[DONE] cambios=%s posiciones finales=%s", changed, list(positions.keys()))

    try:
        equity, cash = get_account_snapshot(trading)
        open_positions_summary = [
            {"symbol": sym, "shares": pos.shares, "entry_price": pos.entry_price, "stop_price": pos.stop_price}
            for sym, pos in positions.items()
        ]
        send_weekly_summary(
            week_end=str(week_end.date()),
            equity=equity,
            cash=cash,
            exits=exits_log,
            entries=entries_log,
            open_positions=open_positions_summary,
        )
    except Exception as exc:
        logging.warning("[NOTIFY] Error al enviar resumen semanal: %s", exc)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        # Antes un crash dejaba al bot en silencio: 42 días sin correr sin que
        # nadie se enterara.
        logging.exception("[FATAL] %s", exc)
        send_failure_alert(exc)
        raise
