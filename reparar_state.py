"""Reconstruye el state del bot desde el broker y lo compara con el actual.

MA y V quedaron semanas con stops que la estrategia ya habia superado: con las
reglas del propio bot, partiendo del state del 3 de agosto, las dos tenian que
salir en la corrida del 21 de septiembre y no salieron. Este script rehace la
gestion semana a semana exactamente como la corre el bot y muestra, posicion por
posicion, en que difiere del archivo que el bot esta usando.

    python reparar_state.py              # solo muestra la comparacion
    python reparar_state.py --escribir   # respalda el state actual y lo reemplaza

No envia ordenes. Si una posicion ya deberia estar cerrada, la deja en el state
con el stop que perforo: la proxima corrida del bot la vende por la via normal.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd

import weekly_alpaca_bot_main as bot
from strategy_weekly_bot_ready import PositionState, WeeklyTrendStrategy

# Posiciones cuyo state inicial se conoce por el log de la corrida del 3 de
# agosto, la ultima en que el state se verifico a mano. Se parte de ahi en vez
# de recalcular el stop de entrada: MA entro en dos tramos y el stop real no es
# el que daria la formula sobre el precio promedio.
ANCLAS = {
    "MA": {"entrada": "2026-08-03", "stop": 515.11, "pendiente": 523.6075, "riesgo": 67.44},
    "V":  {"entrada": "2026-08-03", "stop": 344.42, "pendiente": 348.50,   "riesgo": 26.73},
    "KO": {"entrada": "2026-08-03", "stop": 80.83,  "pendiente": 80.83,    "riesgo": 6.97},
}
# Posiciones que el bot no abrio sino que adopto del broker, con la fecha de la
# corrida en que las adopto.
ADOPTADAS = {"JPM": "2026-08-10"}


def lotes_abiertos(fills: pd.DataFrame) -> Dict[str, List[dict]]:
    """Lotes de compra que siguen abiertos por simbolo, emparejando FIFO."""
    lotes: Dict[str, deque] = defaultdict(deque)
    for _, f in fills.sort_values("fecha").iterrows():
        s = f["symbol"]
        if f["side"] == "buy":
            lotes[s].append({"q": float(f["qty"]), "p": float(f["price"]),
                             "d": pd.Timestamp(f["fecha"])})
            continue
        resto = float(f["qty"])
        while resto > 1e-9 and lotes[s]:
            tomo = min(resto, lotes[s][0]["q"])
            lotes[s][0]["q"] -= tomo
            resto -= tomo
            if lotes[s][0]["q"] <= 1e-9:
                lotes[s].popleft()
    return {s: list(v) for s, v in lotes.items() if v}


def semana_anterior(weekly: pd.DataFrame, fecha: pd.Timestamp) -> Optional[pd.Timestamp]:
    previas = [x for x in weekly.index if x < fecha.normalize()]
    return previas[-1] if previas else None


def posicion_inicial(strat: WeeklyTrendStrategy, sym: str, qty: int, avg: float,
                     weekly: pd.DataFrame, lotes: List[dict]) -> Tuple[PositionState, str]:
    """El state que el bot le habria dado a la posicion el dia que la tomo."""
    if sym in ANCLAS:
        a = ANCLAS[sym]
        pos = PositionState(
            symbol=sym, entry_date=pd.Timestamp(a["entrada"]), entry_price=avg,
            shares=qty, initial_shares=qty, stop_price=a["stop"],
            initial_stop_price=a["stop"], risk_per_share=a["riesgo"],
            pending_stop_price=a["pendiente"])
        return pos, "ancla 3-ago"

    if sym in ADOPTADAS or not lotes:
        # Misma regla que reconcile_state_with_broker.
        corrida = pd.Timestamp(ADOPTADAS.get(sym, str(weekly.index[-1].date())))
        wk = semana_anterior(weekly, corrida) or weekly.index[-1]
        stop = float(weekly.loc[wk].get("box_low_prev", float("nan")))
        if math.isnan(stop) or stop >= avg:
            stop = avg * 0.9
        pos = PositionState(
            symbol=sym, entry_date=wk, entry_price=avg, shares=qty,
            initial_shares=qty, stop_price=stop, initial_stop_price=stop,
            risk_per_share=max(avg - stop, 1e-9), partial_taken=True,
            notes={"adoptado": True})
        return pos, f"adoptada {wk.date()}"

    entrada = min(l["d"] for l in lotes)
    wk = semana_anterior(weekly, entrada)
    stop = strat.resolve_stop(avg, weekly.loc[wk]) if wk is not None else float("nan")
    if math.isnan(stop) or stop >= avg:
        stop = avg * 0.9
    pos = PositionState(
        symbol=sym, entry_date=entrada.normalize(), entry_price=avg, shares=qty,
        initial_shares=int(round(sum(l["q"] for l in lotes))), stop_price=stop,
        initial_stop_price=stop, risk_per_share=max(avg - stop, 1e-9))
    return pos, f"entrada {entrada.date()}"


def reproducir(strat: WeeklyTrendStrategy, pos: PositionState, weekly: pd.DataFrame,
               hasta: pd.Timestamp) -> Tuple[PositionState, Optional[Tuple]]:
    """Gestion semanal tal como la corre el bot, semana por semana.

    Devuelve el state que el bot deberia tener guardado tras procesar `hasta`,
    y la salida si la estrategia la cerro por el camino.
    """
    entrada = pd.Timestamp(pos.entry_date).normalize()
    for lab in [x for x in weekly.index if entrada < x <= hasta]:
        pos = strat.activate_pending_stop(pos)
        d = strat.evaluate_position_week(pos, weekly.loc[lab])
        if d["action"] == "exit_all":
            return pos, (lab.date(), d.get("reason") or "salida", float(weekly.loc[lab]["Low"]))
        pos = strat.apply_week_transition(pos, d)
    return pos, None


def reconstruir(strat, wmap, tenencias: Dict[str, dict], fills: pd.DataFrame,
                hasta: pd.Timestamp) -> Dict[str, dict]:
    abiertos = lotes_abiertos(fills) if not fills.empty else {}
    out = {}
    for sym, t in sorted(tenencias.items()):
        weekly = wmap.get(sym)
        if weekly is None or weekly.empty:
            out[sym] = {"error": "sin datos semanales"}
            continue
        pos, origen = posicion_inicial(strat, sym, int(t["qty"]), float(t["avg"]),
                                       weekly, abiertos.get(sym, []))
        pos, salida = reproducir(strat, pos, weekly, hasta)
        out[sym] = {"pos": pos, "origen": origen, "salida": salida}
    return out


def combinar(viejo: dict, pos: PositionState) -> PositionState:
    """El reconstruido, pero sin bajar nunca lo que el bot ya tenia.

    El trailing de la estrategia solo sube. Si la reconstruccion calcula un stop
    menor que el guardado, lo mas probable es que la reconstruccion sea la que se
    equivoca (un stop inicial estimado, una entrada en tramos), asi que manda el
    mayor. Con esto escribir nunca deja una posicion menos protegida que antes.
    """
    if not viejo:
        return pos
    pos.stop_price = max(pos.stop_price, float(viejo.get("stop_price") or 0))
    pend = [x for x in (pos.pending_stop_price, viejo.get("pending_stop_price")) if x is not None]
    pos.pending_stop_price = max(map(float, pend)) if pend else None
    pos.partial_taken = bool(pos.partial_taken or viejo.get("partial_taken"))
    pos.break_even_armed = bool(pos.break_even_armed or viejo.get("break_even_armed"))
    return pos


def semana_de_la_ultima_corrida(ahora: Optional[datetime] = None) -> pd.Timestamp:
    """La semana que proceso la corrida semanal mas reciente del bot.

    El bot corre los lunes cerca de las 9:35 de Nueva York. No se toma la del
    state porque justamente el state puede ser lo que esta mal: si quedo viejo,
    reconstruir hasta esa fecha dejaria los stops tan atrasados como estan.
    """
    ahora = ahora or bot.now_ny()
    lunes = (ahora - pd.Timedelta(days=ahora.weekday())).replace(
        hour=10, minute=0, second=0, microsecond=0)
    if ahora < lunes:
        lunes -= pd.Timedelta(days=7)
    return bot.latest_completed_week_end(lunes)


def _fmt(x) -> str:
    return "-" if x is None else f"{float(x):.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--escribir", action="store_true",
                    help="respalda el state actual y lo reemplaza por el reconstruido")
    ap.add_argument("--hasta", help="ultima semana procesada (YYYY-MM-DD); por defecto "
                                    "la que figura en el state actual")
    args = ap.parse_args()

    from informe_pdf import clientes, traer_ordenes
    trading, data = clientes()

    actual = bot.load_state_raw()
    meta = actual.get("meta", {})
    print(f"state: {bot.STATE_PATH}")
    if bot.STATE_PATH.exists():
        mod = datetime.fromtimestamp(bot.STATE_PATH.stat().st_mtime, tz=timezone.utc)
        print(f"       modificado {mod:%Y-%m-%d %H:%M} UTC | "
              f"ultima semana procesada {meta.get('last_processed_week')}")
    else:
        print("       NO EXISTE en esta ruta")

    hasta = pd.Timestamp(args.hasta) if args.hasta else semana_de_la_ultima_corrida()
    registrada = meta.get("last_processed_week")
    if registrada and pd.Timestamp(registrada) != hasta:
        print(f"AVISO: el state dice que la ultima semana procesada es {registrada}, pero la "
              f"corrida del lunes debio procesar la del {hasta.date()}. El state que lee el bot "
              f"no es el que escribio.")

    tenencias = {p.symbol: {"qty": float(p.qty), "avg": float(p.avg_entry_price)}
                 for p in trading.get_all_positions() if float(p.qty) > 0}
    fills = traer_ordenes(trading)
    strat = WeeklyTrendStrategy(bot.build_config())
    daily = bot.fetch_daily_bars(data, sorted(set(tenencias) | {bot.BENCHMARK}))
    wmap, _ = bot.build_weekly_maps(strat, daily, bot.BENCHMARK)
    rec = reconstruir(strat, wmap, tenencias, fills, hasta)

    print(f"\nreconstruido hasta la semana del {hasta.date()}\n")
    print("(el stop nuevo nunca es menor que el actual: el trailing solo sube)\n")
    print(f"{'SYM':6}{'stop actual':>13}{'pend actual':>13}{'stop nuevo':>12}"
          f"{'pend nueva':>12}  veredicto")
    cambios = 0
    for sym, r in rec.items():
        viejo = actual.get("positions", {}).get(sym, {})
        if "error" in r:
            print(f"{sym:6}  {r['error']}")
            continue
        pos, sal = combinar(viejo, r["pos"]), r["salida"]
        r["pos"] = pos
        if not viejo:
            ver = "no estaba en el state"
        elif sal:
            ver = f"DEBIO SALIR semana {sal[0]} (min {sal[2]:.2f} <= stop {pos.stop_price:.2f})"
        elif abs(float(viejo.get("stop_price", 0)) - pos.stop_price) > 0.01:
            ver = "stop desactualizado"
        else:
            ver = "coincide"
        cambios += ver != "coincide"
        print(f"{sym:6}{_fmt(viejo.get('stop_price')):>13}{_fmt(viejo.get('pending_stop_price')):>13}"
              f"{pos.stop_price:>12.2f}{_fmt(pos.pending_stop_price):>12}  {ver}  [{r['origen']}]")
    fuera = sorted(set(actual.get("positions", {})) - set(rec))
    if fuera:
        print(f"\nen el state pero no en el broker (se descartan): {fuera}")

    if not args.escribir:
        print(f"\n{cambios} diferencias. Nada escrito: agrega --escribir para aplicar.")
        return
    if bot.STATE_PATH.exists():
        copia = bot.STATE_PATH.with_name(
            f"{bot.STATE_PATH.stem}.backup-{datetime.now():%Y%m%d-%H%M%S}.json")
        shutil.copy2(bot.STATE_PATH, copia)
        print(f"\nrespaldo: {copia}")
    posiciones = {s: r["pos"] for s, r in rec.items() if "pos" in r}
    meta = dict(meta, last_processed_week=str(hasta.date()),
                reparado=str(datetime.now(timezone.utc)))
    bot.save_state(posiciones, meta)
    print(f"escrito: {bot.STATE_PATH}")
    # El bot no vende por salidas pasadas: vende si el ultimo cierre sigue debajo
    # del stop (chequeo diario) o si el minimo de la semana que procesa lo toca.
    # Una posicion que ya se recupero puede seguir abierta, y hay que decirlo.
    for s, r in rec.items():
        if not r.get("salida") or s not in daily:
            continue
        cierre = float(daily[s]["Close"].iloc[-1])
        stop = r["pos"].stop_price
        if cierre <= stop:
            print(f"  {s}: ultimo cierre {cierre:.2f} <= stop {stop:.2f} -> la vende la proxima corrida")
        else:
            print(f"  {s}: ya se recupero ({cierre:.2f} > stop {stop:.2f}) -> solo sale si "
                  f"vuelve a perforarlo; si queres respetar la salida original, vendela a mano")


if __name__ == "__main__":
    main()
