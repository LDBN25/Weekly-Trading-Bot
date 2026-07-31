"""Backtest de la estrategia semanal sobre datos reales de Alpaca.

Reutiliza WeeklyTrendStrategy para que lo que se mide sea el codigo que
realmente corre en produccion, no una reimplementacion paralela.

Modelo de ejecucion (deliberadamente pesimista, para no repetir el error del
historial que registraba precios teoricos):

  - Las senales de la semana W se calculan con datos hasta el cierre de W.
  - Toda orden se ejecuta a la APERTURA de la siguiente sesion, que es lo que
    hace el bot real. Nunca se asume el fill al precio del stop.
  - El deslizamiento se aplica en contra en ambos lados.

Uso:
    ALPACA_API_KEY=... ALPACA_API_SECRET=... python3 backtest.py
    python3 backtest.py --start 2019-01-01 --scenarios base,regimen,atr
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategy_weekly_bot_ready import (
    SECTOR_MAP, PositionState, StrategyConfig, WeeklyTrendStrategy,
)

CACHE = "backtest_bars.pkl"


# --------------------------------------------------------------------- datos
def load_bars(symbols: List[str], start: str, feed: str = "sip") -> Dict[str, pd.DataFrame]:
    """Descarga barras diarias, con cache en disco para iterar rapido."""
    if os.path.exists(CACHE):
        cached = pd.read_pickle(CACHE)
        if set(symbols).issubset(cached.keys()):
            print(f"[DATOS] cache {CACHE}")
            return cached

    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    key, sec = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET")
    if not key or not sec:
        sys.exit("Faltan ALPACA_API_KEY / ALPACA_API_SECRET")
    client = StockHistoricalDataClient(key, sec)

    out: Dict[str, pd.DataFrame] = {}
    chunk = 20
    for i in range(0, len(symbols), chunk):
        part = symbols[i:i + chunk]
        print(f"[DATOS] descargando {i + 1}-{i + len(part)} de {len(symbols)}...", flush=True)
        bars = client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=part, timeframe=TimeFrame.Day,
            start=pd.Timestamp(start).to_pydatetime().replace(tzinfo=timezone.utc),
            adjustment="all", feed=feed))
        for s in part:
            rows = [{"timestamp": pd.Timestamp(b.timestamp).tz_convert(None).normalize(),
                     "Open": float(b.open), "High": float(b.high), "Low": float(b.low),
                     "Close": float(b.close), "Adj Close": float(b.close),
                     "Volume": float(b.volume)} for b in bars.data.get(s, [])]
            if rows:
                out[s] = pd.DataFrame(rows).set_index("timestamp").sort_index()
    pd.to_pickle(out, CACHE)
    return out


# ---------------------------------------------------------------- resultados
@dataclass
class Trade:
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    shares: int
    reason: str
    risk_per_share: float
    is_partial: bool = False

    @property
    def pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.shares

    @property
    def r_multiple(self) -> float:
        return (self.exit_price - self.entry_price) / self.risk_per_share if self.risk_per_share > 0 else 0.0

    @property
    def hold_days(self) -> int:
        return (self.exit_date - self.entry_date).days


@dataclass
class Result:
    name: str
    equity: pd.Series
    trades: List[Trade] = field(default_factory=list)

    def stats(self, bench: Optional[pd.Series] = None) -> Dict[str, float]:
        eq = self.equity.dropna()
        if len(eq) < 2:
            return {}
        years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
        total = eq.iloc[-1] / eq.iloc[0] - 1
        cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1
        dd = (eq / eq.cummax() - 1)
        daily = eq.pct_change().dropna()
        sharpe = daily.mean() / daily.std() * math.sqrt(252) if daily.std() > 0 else 0.0
        downside = daily[daily < 0].std()
        sortino = daily.mean() / downside * math.sqrt(252) if downside and downside > 0 else 0.0

        closed = [t for t in self.trades if not t.is_partial]
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        gp = sum(t.pnl for t in wins)
        gl = abs(sum(t.pnl for t in losses))

        s = {
            "retorno_total": total * 100,
            "cagr": cagr * 100,
            "max_dd": dd.min() * 100,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": (cagr * 100) / abs(dd.min() * 100) if dd.min() < 0 else 0.0,
            "trades": len(self.trades),
            "cerrados": len(closed),
            "win_rate": (len(wins) / len(self.trades) * 100) if self.trades else 0.0,
            "profit_factor": (gp / gl) if gl > 0 else float("inf"),
            "expectativa_r": float(np.mean([t.r_multiple for t in self.trades])) if self.trades else 0.0,
            "hold_ganadores": float(np.mean([t.hold_days for t in wins])) if wins else 0.0,
            "hold_perdedores": float(np.mean([t.hold_days for t in losses])) if losses else 0.0,
        }
        if bench is not None:
            b = bench.reindex(eq.index).ffill().dropna()
            if len(b) > 1:
                s["spy_retorno"] = (b.iloc[-1] / b.iloc[0] - 1) * 100
                s["alfa"] = s["retorno_total"] - s["spy_retorno"]
                s["spy_max_dd"] = (b / b.cummax() - 1).min() * 100
        return s


# ----------------------------------------------------------------- simulador
class Backtest:
    def __init__(self, cfg: StrategyConfig, daily: Dict[str, pd.DataFrame], benchmark: str,
                 initial_equity: float = 100_000, slippage_bps: float = 5.0,
                 daily_stop_check: bool = True, allow_partials: bool = True):
        self.strategy = WeeklyTrendStrategy(cfg)
        self.daily = daily
        self.benchmark = benchmark
        self.initial_equity = initial_equity
        self.slip = slippage_bps / 10_000.0
        self.daily_stop_check = daily_stop_check
        self.allow_partials = allow_partials

    def prepare(self) -> None:
        s = self.strategy
        self.bench_weekly = s.to_weekly(self.daily[self.benchmark])
        self.weekly: Dict[str, pd.DataFrame] = {}
        for sym, df in self.daily.items():
            if sym == self.benchmark or df.empty:
                continue
            wk = s.to_weekly(df)
            if len(wk) < s.cfg.min_history_weeks:
                continue
            self.weekly[sym] = s.add_indicators(wk, self.bench_weekly)

        # calendario comun de sesiones
        idx = sorted(set().union(*[set(df.index) for df in self.daily.values()]))
        self.sessions = pd.DatetimeIndex(idx)
        # ultima sesion de cada semana -> etiqueta W-FRI
        lab = pd.Series(self.sessions, index=self.sessions).resample("W-FRI").last().dropna()
        self.week_last_session = {pd.Timestamp(v): k for k, v in lab.items()}

    # ---- helpers de precio
    def _px(self, sym: str, day: pd.Timestamp, field: str) -> Optional[float]:
        df = self.daily.get(sym)
        if df is None or day not in df.index:
            return None
        return float(df.at[day, field])

    def _next_session(self, day: pd.Timestamp) -> Optional[pd.Timestamp]:
        i = self.sessions.searchsorted(day, side="right")
        return self.sessions[i] if i < len(self.sessions) else None

    def run(self, name: str) -> Result:
        self.prepare()
        s = self.strategy
        cash = self.initial_equity
        positions: Dict[str, PositionState] = {}
        trades: List[Trade] = []
        curve: Dict[pd.Timestamp, float] = {}
        pending_orders: List[Tuple[str, str, int, str]] = []  # (accion, sym, qty, motivo)

        for day in self.sessions:
            # ---------- 1. ejecutar ordenes decididas la sesion anterior
            for action, sym, qty, reason in pending_orders:
                if action == "sell":
                    px = self._px(sym, day, "Open")
                    if px is None or sym not in positions:
                        continue
                    px *= (1 - self.slip)
                    pos = positions[sym]
                    qty = min(qty, pos.shares)
                    if qty <= 0:
                        continue
                    cash += px * qty
                    trades.append(Trade(sym, pos.entry_date, pos.entry_price, day, px, qty,
                                        reason, pos.risk_per_share,
                                        is_partial=(reason.startswith("partial"))))
                    pos.shares -= qty
                    if pos.shares <= 0:
                        del positions[sym]
                elif action == "buy":
                    px = self._px(sym, day, "Open")
                    if px is None or sym in positions:
                        continue
                    px *= (1 + self.slip)
                    wk = self.weekly.get(sym)
                    label = self.week_last_session.get(self._prev_session(day)) if wk is not None else None
                    if wk is None or label is None or label not in wk.index:
                        continue
                    row = wk.loc[label]
                    equity = cash + sum(
                        (self._px(p, day, "Open") or pos.entry_price) * pos.shares
                        for p, pos in positions.items())
                    newpos = s.build_position(sym, day, px, row, equity, cash)
                    if newpos is None or newpos.shares <= 0:
                        continue
                    cost = newpos.shares * px
                    if cost > cash:
                        newpos.shares = int(cash // px)
                        newpos.initial_shares = newpos.shares
                        if newpos.shares <= 0:
                            continue
                        cost = newpos.shares * px
                    cash -= cost
                    positions[sym] = newpos
            pending_orders = []

            # ---------- 2. revision diaria de stops (sobre cierres)
            if self.daily_stop_check:
                for sym, pos in list(positions.items()):
                    close = self._px(sym, day, "Close")
                    if close is not None and close <= pos.stop_price:
                        pending_orders.append(("sell", sym, pos.shares, "stop_diario"))

            # ---------- 3. logica semanal en la ultima sesion de la semana
            label = self.week_last_session.get(day)
            if label is not None:
                ya_vendidos = {o[1] for o in pending_orders}
                for sym, pos in list(positions.items()):
                    if sym in ya_vendidos:
                        continue
                    wk = self.weekly.get(sym)
                    if wk is None or label not in wk.index:
                        continue
                    pos = s.activate_pending_stop(pos)
                    d = s.evaluate_position_week(pos, wk.loc[label])
                    if d["action"] == "partial_exit" and not self.allow_partials:
                        d["action"], d["partial_taken"] = "hold", pos.partial_taken
                    if d["action"] == "exit_all":
                        pending_orders.append(("sell", sym, pos.shares, d.get("reason", "exit")))
                    elif d["action"] == "partial_exit":
                        q = max(1, min(int(d["qty"]), pos.shares - 1)) if pos.shares > 1 else 0
                        if q > 0:
                            pending_orders.append(("sell", sym, q, "partial"))
                    positions[sym] = s.apply_week_transition(pos, d)

                # entradas nuevas
                if s.regime_ok(self.bench_weekly, label):
                    slots = s.cfg.max_positions - len(positions)
                    saliendo = {o[1] for o in pending_orders if o[3] != "partial"}
                    slots += len(saliendo)
                    if slots > 0:
                        cands = []
                        for sym, wk in self.weekly.items():
                            if sym in positions or label not in wk.index:
                                continue
                            row = wk.loc[label]
                            if not bool(row.get("entry_signal", False)):
                                continue
                            cands.append((sym, s.score_candidate(row)))
                        cands.sort(key=lambda x: x[1], reverse=True)
                        proyectado = {k: v for k, v in positions.items() if k not in saliendo}
                        for sym, _ in cands:
                            if slots <= 0:
                                break
                            if not s.sector_slot_available(sym, proyectado):
                                continue
                            pending_orders.append(("buy", sym, 0, "entry"))
                            proyectado[sym] = None
                            slots -= 1

            # ---------- 4. marca a mercado
            mv = sum((self._px(p, day, "Close") or pos.entry_price) * pos.shares
                     for p, pos in positions.items())
            curve[day] = cash + mv

        return Result(name, pd.Series(curve).sort_index(), trades)

    def _prev_session(self, day: pd.Timestamp) -> Optional[pd.Timestamp]:
        i = self.sessions.searchsorted(day, side="left") - 1
        return self.sessions[i] if i >= 0 else None


# ------------------------------------------------------------------ escenarios
def scenarios(base: StrategyConfig) -> Dict[str, Dict]:
    sin_regimen = replace(base, require_bull_regime=False, max_per_sector=0)
    return {
        "original": dict(cfg=replace(sin_regimen), daily_stop_check=False),
        "+stop_diario": dict(cfg=replace(sin_regimen), daily_stop_check=True),
        "+regimen": dict(cfg=replace(sin_regimen, require_bull_regime=True), daily_stop_check=True),
        "+sector": dict(cfg=replace(sin_regimen, require_bull_regime=True, max_per_sector=3),
                        daily_stop_check=True),
        "+stop_15%": dict(cfg=replace(sin_regimen, require_bull_regime=True, max_per_sector=3,
                                      max_stop_pct=0.15), daily_stop_check=True),
        "+atr_3x": dict(cfg=replace(sin_regimen, require_bull_regime=True, max_per_sector=3,
                                    atr_stop_mult=3.0), daily_stop_check=True),
        "sin_break_even": dict(cfg=replace(sin_regimen, require_bull_regime=True, max_per_sector=3,
                                          atr_stop_mult=3.0, break_even_r=999.0),
                               daily_stop_check=True),
        "sin_parciales": dict(cfg=replace(sin_regimen, require_bull_regime=True, max_per_sector=3,
                                         atr_stop_mult=3.0, break_even_r=999.0),
                              daily_stop_check=True, allow_partials=False),
        # Aislados sobre la misma base (regimen + stop diario, sin tope de
        # sector) para poder atribuir cada efecto por separado.
        "iso_regimen": dict(cfg=replace(sin_regimen, require_bull_regime=True),
                            daily_stop_check=True),
        "iso_stop15": dict(cfg=replace(sin_regimen, require_bull_regime=True, max_stop_pct=0.15),
                           daily_stop_check=True),
        "iso_stop10": dict(cfg=replace(sin_regimen, require_bull_regime=True, max_stop_pct=0.10),
                           daily_stop_check=True),
        "iso_stop20": dict(cfg=replace(sin_regimen, require_bull_regime=True, max_stop_pct=0.20),
                           daily_stop_check=True),
        "iso_atr3": dict(cfg=replace(sin_regimen, require_bull_regime=True, atr_stop_mult=3.0),
                         daily_stop_check=True),
        "iso_sin_be": dict(cfg=replace(sin_regimen, require_bull_regime=True, max_stop_pct=0.15,
                                       break_even_r=999.0), daily_stop_check=True),
        "iso_sin_parc": dict(cfg=replace(sin_regimen, require_bull_regime=True, max_stop_pct=0.15),
                             daily_stop_check=True, allow_partials=False),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--equity", type=float, default=100_000)
    ap.add_argument("--slippage-bps", type=float, default=5.0)
    ap.add_argument("--scenarios", default="all")
    args = ap.parse_args()

    syms = sorted({l.strip().upper() for l in open("symbols.txt") if l.strip()})
    daily = load_bars(sorted(set(syms + ["SPY"])), args.start)
    print(f"[DATOS] {len(daily)} simbolos  "
          f"{min(d.index[0] for d in daily.values()).date()} -> "
          f"{max(d.index[-1] for d in daily.values()).date()}\n")

    bench_close = daily["SPY"]["Close"]
    todos = scenarios(StrategyConfig())
    elegidos = list(todos) if args.scenarios == "all" else args.scenarios.split(",")

    results: List[Result] = []
    for name in elegidos:
        if name not in todos:
            print(f"  escenario desconocido: {name}")
            continue
        spec = dict(todos[name])
        cfg = spec.pop("cfg")
        print(f"[SIM] {name}...", flush=True)
        bt = Backtest(cfg, daily, "SPY", initial_equity=args.equity,
                      slippage_bps=args.slippage_bps, **spec)
        results.append(bt.run(name))

    print("\n" + "=" * 108)
    print("RESULTADOS")
    print("=" * 108)
    hdr = (f"{'escenario':16}{'retorno':>10}{'CAGR':>8}{'maxDD':>9}{'Sharpe':>8}"
           f"{'Calmar':>8}{'trades':>8}{'win%':>7}{'PF':>7}{'E[R]':>7}{'holdG/P':>11}")
    print(hdr)
    print("-" * 108)
    for r in results:
        s = r.stats(bench_close)
        if not s:
            continue
        hp = f"{s['hold_ganadores']:.0f}/{s['hold_perdedores']:.0f}d"
        print(f"{r.name:16}{s['retorno_total']:>9.1f}%{s['cagr']:>7.1f}%{s['max_dd']:>8.1f}%"
              f"{s['sharpe']:>8.2f}{s['calmar']:>8.2f}{s['trades']:>8}{s['win_rate']:>6.0f}%"
              f"{s['profit_factor']:>7.2f}{s['expectativa_r']:>7.2f}{hp:>11}")

    if results:
        s = results[0].stats(bench_close)
        if "spy_retorno" in s:
            print("-" * 108)
            print(f"{'SPY (buy&hold)':16}{s['spy_retorno']:>9.1f}%{'':>7}{s['spy_max_dd']:>8.1f}%")

    print("\nNota: el universo son los mega-caps de HOY, asi que hay sesgo de "
          "supervivencia.\nLos retornos absolutos estan inflados; lo comparable "
          "es la diferencia ENTRE escenarios.")


if __name__ == "__main__":
    main()
