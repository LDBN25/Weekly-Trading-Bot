"""
Reconstruye el state.json desde cero usando posiciones reales del broker
y datos históricos de Alpaca para calcular stops y trailing correctos.
Ejecutar UNA SOLA VEZ antes del próximo ciclo del bot.
"""
from __future__ import annotations

import json
import os
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from strategy_weekly_bot_ready import WeeklyTrendStrategy, StrategyConfig, PositionState

NY = ZoneInfo("America/New_York")
UTC = timezone.utc
STATE_PATH = Path(os.getenv("STATE_PATH", "data/weekly_strategy_state.json"))
LOOKBACK_DAYS = 900


def get_clients():
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_API_SECRET")
    if not key or not secret:
        raise RuntimeError("Faltan ALPACA_API_KEY y/o ALPACA_API_SECRET")
    trading = TradingClient(api_key=key, secret_key=secret, paper=True)
    data = StockHistoricalDataClient(api_key=key, secret_key=secret)
    return trading, data


def fetch_weekly(data_client, symbol: str) -> pd.DataFrame:
    start = datetime.now(UTC) - timedelta(days=LOOKBACK_DAYS)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start,
        adjustment="all",
        feed=os.getenv("ALPACA_DATA_FEED", "iex"),
    )
    raw = data_client.get_stock_bars(req)
    bars = raw.data.get(symbol, [])
    if not bars:
        return pd.DataFrame()

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
    df = pd.DataFrame(rows).set_index("timestamp").sort_index()

    strategy = WeeklyTrendStrategy(StrategyConfig())
    return strategy.to_weekly(df)


def calcular_stop_inicial(weekly: pd.DataFrame, entry_date: pd.Timestamp, box_weeks: int = 3) -> float:
    """Recalcula el stop inicial: mínimo de las 3 semanas previas a la entrada."""
    semanas_previas = weekly[weekly.index < entry_date]
    if len(semanas_previas) < box_weeks:
        return float("nan")
    return float(semanas_previas["Low"].iloc[-box_weeks:].min())


def calcular_trailing_stop(weekly: pd.DataFrame, entry_date: pd.Timestamp) -> float:
    """Calcula el trailing stop actual: mínimo de las 2 últimas semanas completadas."""
    semanas_desde_entrada = weekly[weekly.index >= entry_date]
    if len(semanas_desde_entrada) < 2:
        return float("nan")
    # Las 2 semanas más recientes (excluye la semana en curso si no está cerrada)
    ultimas_2 = semanas_desde_entrada["Low"].iloc[-2:]
    return float(ultimas_2.min())


def now_ny() -> datetime:
    return datetime.now(tz=NY)


def latest_completed_week_end() -> pd.Timestamp:
    ref = now_ny()
    weekday = ref.weekday()
    days_since_friday = (weekday - 4) % 7
    if weekday == 4 and ref.hour < 16:
        days_since_friday = 7
    completed_friday = ref.date() - timedelta(days=days_since_friday)
    return pd.Timestamp(completed_friday)


def main():
    print("[+] Conectando a Alpaca...")
    trading, data_client = get_clients()

    account = trading.get_account()
    equity = float(account.equity)
    print(f"    Equity: ${equity:,.2f}")

    print("[+] Obteniendo posiciones abiertas del broker...")
    broker_positions = trading.get_all_positions()

    if not broker_positions:
        print("[!] No hay posiciones abiertas en el broker. No se genera state.")
        return

    print(f"    Posiciones encontradas: {[p.symbol for p in broker_positions]}")

    strategy = WeeklyTrendStrategy(StrategyConfig())
    week_end = latest_completed_week_end()
    print(f"[+] Última semana completada: {week_end.date()}")

    positions = {}
    problemas = []

    for bp in broker_positions:
        symbol = bp.symbol
        qty = int(math.floor(float(bp.qty)))
        avg_entry = float(bp.avg_entry_price)
        print(f"\n[>] Procesando {symbol}: qty={qty} @ ${avg_entry:.4f}")

        print(f"    Descargando datos históricos...")
        weekly = fetch_weekly(data_client, symbol)

        if weekly.empty or len(weekly) < 10:
            print(f"    [!] Sin datos suficientes para {symbol} — stop manual requerido")
            problemas.append(symbol)
            # Crear posición con stop conservador del 8%
            stop_fallback = round(avg_entry * 0.92, 4)
            pos = PositionState(
                symbol=symbol,
                entry_date=pd.Timestamp("2026-04-20"),
                entry_price=avg_entry,
                shares=qty,
                initial_shares=qty,
                stop_price=stop_fallback,
                initial_stop_price=stop_fallback,
                risk_per_share=round(avg_entry - stop_fallback, 4),
                notes={"reconstruido": True, "stop_fallback": True},
            )
            positions[symbol] = pos
            continue

        # Estimar fecha de entrada desde avg_entry_price
        # Buscamos la semana más cercana al precio de entrada en el histórico
        diffs = (weekly["Close"] - avg_entry).abs()
        semana_entrada_aprox = weekly.index[diffs.argmin()]

        stop_inicial = calcular_stop_inicial(weekly, semana_entrada_aprox)
        trailing = calcular_trailing_stop(weekly, semana_entrada_aprox)

        if math.isnan(stop_inicial) or stop_inicial >= avg_entry:
            stop_inicial = round(avg_entry * 0.92, 4)
            print(f"    [!] Stop inicial inválido → usando 8% fallback: ${stop_inicial:.4f}")
            problemas.append(f"{symbol}(stop_fallback)")

        risk_per_share = round(avg_entry - stop_inicial, 4)

        # Stop activo: el mayor entre stop inicial y trailing
        stop_activo = stop_inicial
        if not math.isnan(trailing) and trailing > stop_inicial:
            stop_activo = round(trailing, 4)

        # Break-even: se arma cuando el precio alcanzó +1R
        max_price = float(weekly[weekly.index >= semana_entrada_aprox]["High"].max()) if len(weekly[weekly.index >= semana_entrada_aprox]) > 0 else avg_entry
        one_r_price = avg_entry + risk_per_share
        break_even_armed = max_price >= one_r_price

        # Partial: se tomó si alcanzó +2.5R
        partial_r_price = avg_entry + 2.5 * risk_per_share
        partial_taken = max_price >= partial_r_price

        pos = PositionState(
            symbol=symbol,
            entry_date=pd.Timestamp(semana_entrada_aprox),
            entry_price=avg_entry,
            shares=qty,
            initial_shares=qty,
            stop_price=stop_activo,
            initial_stop_price=stop_inicial,
            risk_per_share=risk_per_share,
            break_even_armed=break_even_armed,
            partial_taken=partial_taken,
            notes={"reconstruido": True, "trailing_calculado": round(trailing, 4) if not math.isnan(trailing) else None},
        )
        positions[symbol] = pos

        print(f"    Stop inicial:  ${stop_inicial:.4f}")
        print(f"    Trailing stop: ${trailing:.4f}" if not math.isnan(trailing) else "    Trailing stop: N/A")
        print(f"    Stop activo:   ${stop_activo:.4f}")
        print(f"    Risk/share:    ${risk_per_share:.4f}")
        print(f"    Break-even:    {'✅ armado' if break_even_armed else '❌ no armado'}")
        print(f"    Parcial:       {'✅ tomado' if partial_taken else '❌ no tomado'}")

    # Serializar y guardar
    from dataclasses import asdict
    meta = {
        "last_processed_week": str(week_end.date()),
        "updated_at": str(now_ny()),
        "reconstruido_manualmente": True,
    }

    payload = {
        "positions": {
            sym: {**asdict(pos), "entry_date": str(pos.entry_date)}
            for sym, pos in positions.items()
        },
        "meta": meta,
    }

    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(payload, indent=2, default=str))

    print(f"\n{'=' * 55}")
    print(f"[✓] State guardado en: {STATE_PATH}")
    print(f"    Posiciones registradas: {list(positions.keys())}")
    print(f"    Semana marcada como procesada: {week_end.date()}")
    if problemas:
        print(f"\n[!] Revisar stops manualmente: {problemas}")
        print(f"    Estos usan stop fallback del 8% — ajustar si es necesario")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
