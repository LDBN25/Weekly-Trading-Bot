"""Informe diario del estado de la estrategia, al cierre del mercado.

Corre en un entorno donde paper-api.alpaca.markets esta bloqueado, de modo que
no lee la cuenta: parte de la ultima foto confirmada (cartera_conocida.json) y
reconstruye los stops replicando la estrategia semana a semana desde la entrada
de cada posicion. Todo lo demas sale del feed de datos, que si esta permitido.

    python informe_diario.py
    python informe_diario.py --cartera cartera_conocida.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import weekly_alpaca_bot_main as bot
from strategy_weekly_bot_ready import PositionState, WeeklyTrendStrategy

# Umbral para marcar una posicion como pegada al stop.
CERCA_PCT = 3.0


def replay_stop(strat, daily, weekly, pos: PositionState, hasta: pd.Timestamp):
    """Reproduce el trailing semanal desde la entrada hasta la ultima semana.

    Devuelve (posicion_al_dia, salida) donde salida es None si sigue abierta.
    Es la misma secuencia que corre el bot: activar el stop pendiente, mirar si
    la semana lo perforo, y recalcular el trailing para la siguiente.
    """
    entrada = pd.Timestamp(pos.entry_date).normalize()
    for lab in [x for x in weekly.index if entrada < x <= hasta]:
        pos = strat.activate_pending_stop(pos)
        tramo = daily[(daily.index > lab - pd.Timedelta(days=7)) & (daily.index <= lab)]
        golpe = tramo[tramo["Close"] <= pos.stop_price]
        if not golpe.empty:
            return pos, (golpe.index[0].date(), "stop intrasemanal",
                         float(golpe.iloc[0]["Close"]))
        d = strat.evaluate_position_week(pos, weekly.loc[lab])
        if d["action"] == "exit_all":
            return pos, (lab.date(), d.get("reason") or "salida",
                         float(weekly.loc[lab]["Adj Close"]))
        pos = strat.apply_week_transition(pos, d)
    return strat.activate_pending_stop(pos), None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cartera", default="cartera_conocida.json")
    args = ap.parse_args()

    foto = json.loads(Path(args.cartera).read_text(encoding="utf-8"))
    if not os.getenv("ALPACA_API_KEY") or not os.getenv("ALPACA_API_SECRET"):
        sys.exit("Faltan ALPACA_API_KEY / ALPACA_API_SECRET")

    from alpaca.data.historical.stock import StockHistoricalDataClient
    dc = StockHistoricalDataClient(os.getenv("ALPACA_API_KEY"),
                                   os.getenv("ALPACA_API_SECRET"))
    strat = WeeklyTrendStrategy(bot.build_config())
    tenidas = [p["symbol"] for p in foto["posiciones"]]
    syms = sorted(set(bot.load_symbols() + tenidas + ["SPY"]))
    daily = bot.fetch_daily_bars(dc, syms)
    wmap, bench_w = bot.build_weekly_maps(strat, daily, "SPY")
    semana = bot.latest_completed_week_end()

    spy = daily["SPY"]["Close"]
    ultimo = spy.index[-1]
    print("=" * 78)
    print(f"INFORME DIARIO  |  cierre {ultimo.date()}  |  "
          f"generado {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"foto de cartera del {foto['fecha']} ({len(tenidas)} posiciones)")
    print("=" * 78)
    # El informe corre de lunes a viernes, pero hay feriados. Si la ultima barra
    # no es de hoy en Nueva York, no hubo rueda y todo lo que sigue es de ayer.
    hoy_ny = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    if ultimo.normalize() < hoy_ny:
        print(f"\n*** SIN RUEDA HOY ({hoy_ny.date()}): mercado cerrado o datos aun "
              f"no publicados. Las cifras son del {ultimo.date()}. ***")

    # ── 1. mercado ───────────────────────────────────────────────────────────
    print("\n[1] MERCADO")
    dia = (spy.iloc[-1] / spy.iloc[-2] - 1) * 100
    sem = (spy.iloc[-1] / spy.iloc[-6] - 1) * 100 if len(spy) > 6 else float("nan")
    desde_abr = spy[spy.index >= pd.Timestamp("2026-04-01")]
    print(f"  SPY {spy.iloc[-1]:,.2f}   dia {dia:+.2f}%   5 ruedas {sem:+.2f}%   "
          f"desde 1-abr {(desde_abr.iloc[-1] / desde_abr.iloc[0] - 1) * 100:+.2f}%")
    r = strat.add_indicators(bench_w.copy(), bench_w).iloc[-1]
    alcista = float(r["Adj Close"]) > float(r["sma30"])
    margen = (float(r["Adj Close"]) / float(r["sma30"]) - 1) * 100
    print(f"  cierre semanal {float(r['Adj Close']):,.2f} vs SMA30 {float(r['sma30']):,.2f} "
          f"({margen:+.2f}%)  ->  regimen {'ALCISTA' if alcista else 'BAJISTA'}")
    if alcista and margen < 2.0:
        print("  AVISO: el regimen esta a menos de 2% de darse vuelta. "
              "Si cae, se cierran las entradas nuevas.")

    # ── 2. posiciones ────────────────────────────────────────────────────────
    print("\n[2] POSICIONES")
    print(f"  {'SYM':6}{'qty':>5}{'entrada':>10}{'hoy':>10}{'ret%':>8}"
          f"{'valor':>11}{'P&L':>10}{'stop':>10}{'dist%':>8}  estado")
    alertas, valor_t, costo_t = [], 0.0, 0.0
    for p in foto["posiciones"]:
        s = p["symbol"]
        if s not in daily or daily[s].empty or s not in wmap:
            print(f"  {s:6}  sin datos")
            continue
        hoy = float(daily[s]["Close"].iloc[-1])
        valor, costo = hoy * p["qty"], p["avg"] * p["qty"]
        valor_t += valor
        costo_t += costo

        wk = wmap[s]
        ent = pd.Timestamp(p["entrada"])
        fila = wk[wk.index >= ent]
        stop0 = float(fila.iloc[0]["box_low_prev"]) if not fila.empty else p["avg"] * 0.9
        if pd.isna(stop0) or stop0 >= p["avg"]:
            stop0 = p["avg"] * 0.9
        pos = PositionState(
            symbol=s, entry_date=ent, entry_price=p["avg"], shares=p["qty"],
            initial_shares=p["qty"], stop_price=stop0, initial_stop_price=stop0,
            risk_per_share=max(p["avg"] - stop0, 1e-9),
            # Una adoptada no tiene R conocida: no le corresponde parcial.
            partial_taken=bool(p.get("adoptada")))
        pos, salida = replay_stop(strat, daily[s], wk, pos, semana)

        dist = (hoy / pos.stop_price - 1) * 100
        if salida:
            estado = f"SALIDA {salida[0]} por {salida[1]} @ {salida[2]:.2f}"
            alertas.append(f"{s}: la estrategia ya la daba cerrada el {salida[0]} ({salida[1]})")
        elif dist <= 0:
            estado = "STOP PERFORADO"
            alertas.append(f"{s}: cotiza {abs(dist):.2f}% POR DEBAJO del stop {pos.stop_price:.2f}")
        elif dist <= CERCA_PCT:
            estado = "pegada al stop"
            alertas.append(f"{s}: a {dist:.2f}% del stop {pos.stop_price:.2f}")
        else:
            estado = "abierta"
        print(f"  {s:6}{p['qty']:>5}{p['avg']:>10.2f}{hoy:>10.2f}"
              f"{(hoy / p['avg'] - 1) * 100:>7.1f}%{valor:>11,.0f}{valor - costo:>+10,.0f}"
              f"{pos.stop_price:>10.2f}{dist:>7.1f}%  {estado}")
    print(f"  {'TOTAL':6}{'':25}{'':8}{valor_t:>11,.0f}{valor_t - costo_t:>+10,.0f}"
          f"   ({(valor_t / costo_t - 1) * 100:+.2f}% sobre costo)" if costo_t else "")

    # ── 3. candidatos ────────────────────────────────────────────────────────
    print(f"\n[3] CANDIDATOS PARA LA PROXIMA EVALUACION (semana {semana.date()})")
    if not alcista:
        print("  regimen bajista: no se abren posiciones nuevas")
    else:
        cands = bot.score_candidates(strat, wmap, semana, set(tenidas))
        if not cands:
            print("  ninguno pasa los filtros")
        for s, sc, row in cands[:8]:
            print(f"    {s:6} score={sc:+.3f}  rs_z={float(row['rs_z']):+.2f}  "
                  f"vol_z={float(row['vol_z']):+.2f}  stop={float(row['box_low_prev']):.2f}")

    # ── 4. alertas ───────────────────────────────────────────────────────────
    print("\n[4] ALERTAS")
    if alertas:
        for a in alertas:
            print(f"  - {a}")
    else:
        print("  ninguna: nada pegado al stop ni pendiente de cierre")
    print()


if __name__ == "__main__":
    main()
