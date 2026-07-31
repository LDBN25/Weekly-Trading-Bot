from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, Any

import math
import pandas as pd
import numpy as np

TrailMode = Literal["prior_week_low", "prior_2w_low"]

# Los semiconductores van aparte del resto de tecnología: se mueven juntos y
# tratarlos como riesgos independientes subestima la concentración.
SECTOR_MAP: Dict[str, str] = {
    "NVDA": "semis", "MU": "semis", "AMD": "semis", "QCOM": "semis",
    "AAPL": "tech", "MSFT": "tech", "ORCL": "tech", "PLTR": "tech",
    "CSCO": "tech", "IBM": "tech", "ADBE": "tech", "NOW": "tech",
    "GOOGL": "comm", "GOOG": "comm", "META": "comm", "NFLX": "comm",
    "DIS": "comm", "T": "comm",
    "AMZN": "cons_disc", "TSLA": "cons_disc", "HD": "cons_disc", "MCD": "cons_disc",
    "WMT": "cons_staples", "COST": "cons_staples", "KO": "cons_staples",
    "PM": "cons_staples", "PEP": "cons_staples",
    "BRK.B": "financials", "JPM": "financials", "V": "financials", "MA": "financials",
    "BAC": "financials", "WFC": "financials", "MS": "financials", "GS": "financials",
    "AXP": "financials",
    "LLY": "health", "JNJ": "health", "ABBV": "health", "UNH": "health",
    "ABT": "health", "MRK": "health",
    "XOM": "energy", "CVX": "energy",
    "LIN": "materials",
    "GE": "industrials", "CAT": "industrials",
}


@dataclass
class StrategyConfig:
    sma_weeks: int = 30
    ema_exit_weeks: int = 10
    vol_avg_weeks: int = 10
    box_weeks: int = 3
    min_volume_mult: float = 1.0
    break_even_r: float = 1.0
    partial_r: float = 2.5
    partial_exit_fraction: float = 0.33
    trail_mode: TrailMode = "prior_2w_low"
    risk_pct_per_trade: float = 0.02
    max_positions: int = 10
    min_history_weeks: int = 40
    # Normalización del stop. Ambas apagadas por defecto: cambian el tamaño de
    # las posiciones y conviene medirlas antes de activarlas.
    max_stop_pct: float = 0.0      # tope duro de distancia al stop (0.15 = 15%)
    atr_stop_mult: float = 0.0     # stop = entrada - mult * ATR semanal
    atr_weeks: int = 10
    # Sin filtro de régimen el bot compra rupturas dentro de un mercado bajista.
    require_bull_regime: bool = True
    max_per_sector: int = 3
    score_rs_weight: float = 1.0
    score_vol_weight: float = 1.0
    score_z_weeks: int = 52


@dataclass
class PositionState:
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    initial_shares: int
    stop_price: float
    initial_stop_price: float
    risk_per_share: float
    break_even_armed: bool = False
    partial_taken: bool = False
    pending_stop_price: Optional[float] = None
    notes: Dict[str, Any] = field(default_factory=dict)

    @property
    def one_r_price(self) -> float:
        return self.entry_price + self.risk_per_share

    def target_price(self, multiple_r: float) -> float:
        return self.entry_price + multiple_r * self.risk_per_share


class WeeklyTrendStrategy:
    def __init__(self, config: Optional[StrategyConfig] = None):
        self.cfg = config or StrategyConfig()

    def to_weekly(self, df: pd.DataFrame) -> pd.DataFrame:
        weekly = pd.DataFrame(index=pd.Index([], name=df.index.name))
        weekly["Open"] = df["Open"].resample("W-FRI").first()
        weekly["High"] = df["High"].resample("W-FRI").max()
        weekly["Low"] = df["Low"].resample("W-FRI").min()
        weekly["Close"] = df["Close"].resample("W-FRI").last()
        adj_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        weekly["Adj Close"] = df[adj_col].resample("W-FRI").last()
        weekly["Volume"] = df["Volume"].resample("W-FRI").sum()
        return weekly.dropna()

    def add_indicators(self, weekly: pd.DataFrame, benchmark_weekly: pd.DataFrame) -> pd.DataFrame:
        w = weekly.copy()
        c = self.cfg

        w["sma30"] = w["Adj Close"].rolling(c.sma_weeks).mean()
        w["ema_exit"] = w["Adj Close"].ewm(span=c.ema_exit_weeks, adjust=False).mean()
        w["vol_avg"] = w["Volume"].rolling(c.vol_avg_weeks).mean()
        w["box_high_prev"] = w["High"].shift(1).rolling(c.box_weeks).max()
        w["box_low_prev"] = w["Low"].shift(1).rolling(c.box_weeks).min()
        w["sma_up"] = w["sma30"] > w["sma30"].shift(1)

        aligned_bench = benchmark_weekly["Adj Close"].reindex(w.index).ffill()
        rs_ratio = w["Adj Close"] / aligned_bench
        w["rs_ratio"] = rs_ratio
        w["rs_ma"] = rs_ratio.rolling(13).mean()
        w["rs_ok"] = w["rs_ratio"] > w["rs_ma"]

        w["trend_ok"] = (w["Adj Close"] > w["sma30"]) & w["sma_up"]
        w["vol_ok"] = w["Volume"] >= (w["vol_avg"] * c.min_volume_mult)
        w["breakout"] = w["Adj Close"] > w["box_high_prev"]
        w["entry_signal"] = w["trend_ok"] & w["vol_ok"] & w["breakout"] & w["rs_ok"]
        w["exit_signal"] = w["Adj Close"] < w["ema_exit"]
        w["prior_week_low"] = w["Low"].shift(1)
        w["prior_2w_low"] = w["Low"].shift(1).rolling(2).min()

        # ATR semanal para dimensionar el stop en unidades de volatilidad.
        prev_close = w["Adj Close"].shift(1)
        true_range = pd.concat([
            w["High"] - w["Low"],
            (w["High"] - prev_close).abs(),
            (w["Low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        w["atr"] = true_range.rolling(c.atr_weeks).mean()

        # Los dos términos del score viven en escalas distintas: sumarlos crudos
        # hacía que el volumen dominara el ranking y la fuerza relativa fuera
        # decorativa. Se estandarizan contra su propia historia.
        w["vol_ratio"] = w["Volume"] / w["vol_avg"].replace(0, np.nan)
        w["rs_mom"] = w["rs_ratio"] / w["rs_ma"].replace(0, np.nan)
        for src, dst in (("vol_ratio", "vol_z"), ("rs_mom", "rs_z")):
            mean = w[src].rolling(c.score_z_weeks, min_periods=13).mean()
            std = w[src].rolling(c.score_z_weeks, min_periods=13).std()
            w[dst] = ((w[src] - mean) / std.replace(0, np.nan)).fillna(0.0)
        return w

    def score_candidate(self, row: pd.Series) -> float:
        c = self.cfg
        rs_z = float(row.get("rs_z", 0.0) or 0.0)
        vol_z = float(row.get("vol_z", 0.0) or 0.0)
        return c.score_rs_weight * rs_z + c.score_vol_weight * vol_z

    def regime_ok(self, benchmark_weekly: Optional[pd.DataFrame], week_end: pd.Timestamp) -> bool:
        """¿El mercado está por encima de su media de largo plazo?"""
        if not self.cfg.require_bull_regime:
            return True
        if benchmark_weekly is None or benchmark_weekly.empty:
            return True
        bench = benchmark_weekly[benchmark_weekly.index <= week_end]
        if len(bench) < self.cfg.sma_weeks:
            return True
        sma = bench["Adj Close"].rolling(self.cfg.sma_weeks).mean().iloc[-1]
        if pd.isna(sma):
            return True
        return float(bench["Adj Close"].iloc[-1]) > float(sma)

    def sector_slot_available(self, symbol: str, positions: Dict[str, Any]) -> bool:
        """Evita concentrar la cartera en un puñado de nombres correlacionados."""
        if self.cfg.max_per_sector <= 0:
            return True
        sector = SECTOR_MAP.get(symbol)
        if sector is None:
            return True
        held = sum(1 for s in positions if SECTOR_MAP.get(s) == sector)
        return held < self.cfg.max_per_sector

    def resolve_stop(self, entry_price: float, row: pd.Series) -> float:
        """Combina el mínimo de la caja con los topes de volatilidad."""
        stop_price = float(row["box_low_prev"])
        atr = float(row.get("atr", float("nan")) or float("nan"))
        if self.cfg.atr_stop_mult > 0 and not math.isnan(atr) and atr > 0:
            stop_price = max(stop_price, entry_price - self.cfg.atr_stop_mult * atr)
        if self.cfg.max_stop_pct > 0:
            stop_price = max(stop_price, entry_price * (1.0 - self.cfg.max_stop_pct))
        return stop_price

    def build_position(self, symbol: str, entry_date: pd.Timestamp, entry_price: float,
                       row: pd.Series, equity: float, cash: float) -> Optional[PositionState]:
        stop_price = self.resolve_stop(entry_price, row)
        if math.isnan(stop_price) or stop_price >= entry_price:
            return None

        risk_per_share = entry_price - stop_price
        risk_budget = equity * self.cfg.risk_pct_per_trade
        shares = int(risk_budget // risk_per_share)
        if shares <= 0:
            return None

        max_affordable = int(cash // entry_price)
        shares = min(shares, max_affordable)
        if shares <= 0:
            return None

        return PositionState(
            symbol=symbol,
            entry_date=entry_date,
            entry_price=entry_price,
            shares=shares,
            initial_shares=shares,
            stop_price=stop_price,
            initial_stop_price=stop_price,
            risk_per_share=risk_per_share,
        )

    def evaluate_position_week(self, pos: PositionState, row: pd.Series) -> Dict[str, Any]:
        """
        Evalúa una posición usando SOLO información disponible al cierre de la semana.
        La actualización de stop queda pendiente para la semana siguiente.
        Devuelve una acción estructurada para que el bot/ejecutor la procese.
        """
        high = float(row["High"])
        low = float(row["Low"])
        close = float(row["Adj Close"])

        result: Dict[str, Any] = {
            "action": "hold",
            "qty": 0,
            "reason": None,
            "new_pending_stop": pos.pending_stop_price,
            "new_stop_now": pos.stop_price,
            "break_even_armed": pos.break_even_armed,
            "partial_taken": pos.partial_taken,
        }

        # 1) Stop vigente desde semanas anteriores
        if low <= pos.stop_price:
            result.update({
                "action": "exit_all",
                "qty": pos.shares,
                "reason": "stop",
            })
            return result

        # 2) Salida por debilidad al cierre
        if bool(row.get("exit_signal", False)):
            result.update({
                "action": "exit_all",
                "qty": pos.shares,
                "reason": "ema_exit",
            })
            return result

        # 3) Break-even al alcanzar +1R
        if (not pos.break_even_armed) and high >= pos.target_price(self.cfg.break_even_r):
            result["break_even_armed"] = True
            result["new_pending_stop"] = max(pos.stop_price, pos.entry_price)

        # 4) Parcial al alcanzar +2.5R
        if (not pos.partial_taken) and self.cfg.partial_exit_fraction > 0 and high >= pos.target_price(self.cfg.partial_r):
            qty = int(round(pos.initial_shares * self.cfg.partial_exit_fraction))
            qty = max(1, min(qty, pos.shares - 1)) if pos.shares > 1 else 0
            if qty > 0:
                result.update({
                    "action": "partial_exit",
                    "qty": qty,
                    "reason": f"partial_{self.cfg.partial_r}R",
                    "partial_taken": True,
                })

        # 5) Trailing stop para la semana siguiente
        trail_key = self.cfg.trail_mode
        trail_candidate = row.get(trail_key, np.nan)
        if pd.notna(trail_candidate):
            pending = max(
                pos.stop_price,
                pos.entry_price if result["break_even_armed"] else pos.stop_price,
                float(trail_candidate),
            )
            result["new_pending_stop"] = pending

        return result

    def apply_week_transition(self, pos: PositionState, decision: Dict[str, Any]) -> PositionState:
        pos.break_even_armed = bool(decision.get("break_even_armed", pos.break_even_armed))
        pos.partial_taken = bool(decision.get("partial_taken", pos.partial_taken))
        pending = decision.get("new_pending_stop")
        if pending is not None:
            pos.pending_stop_price = float(pending)
        return pos

    def activate_pending_stop(self, pos: PositionState) -> PositionState:
        if pos.pending_stop_price is not None and pos.pending_stop_price > pos.stop_price:
            pos.stop_price = pos.pending_stop_price
        pos.pending_stop_price = None
        return pos


__all__ = [
    "StrategyConfig",
    "PositionState",
    "WeeklyTrendStrategy",
]
