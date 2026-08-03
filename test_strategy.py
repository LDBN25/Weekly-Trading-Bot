"""Tests de la estrategia y de los bugs corregidos.

Ejecutar con:  python3 -m pytest test_strategy.py -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy_weekly_bot_ready import (
    SECTOR_MAP, PositionState, StrategyConfig, WeeklyTrendStrategy,
)
import weekly_alpaca_bot_main as bot


def make_daily(periods: int = 400, drift: float = 0.0012, seed: int = 7) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", periods=periods)
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(drift, 0.012, len(idx)))
    return pd.DataFrame({
        "Open": close, "High": close * 1.02, "Low": close * 0.98,
        "Close": close, "Adj Close": close,
        "Volume": rng.uniform(1e6, 3e6, len(idx)),
    }, index=idx)


def make_position(**kw) -> PositionState:
    base = dict(
        symbol="X", entry_date=pd.Timestamp("2025-01-03"), entry_price=100.0,
        shares=100, initial_shares=100, stop_price=90.0,
        initial_stop_price=90.0, risk_per_share=10.0,
    )
    base.update(kw)
    return PositionState(**base)


@pytest.fixture
def strat() -> WeeklyTrendStrategy:
    return WeeklyTrendStrategy(StrategyConfig())


@pytest.fixture
def weekly(strat) -> pd.DataFrame:
    df = make_daily()
    bench = make_daily(drift=0.0006, seed=11)
    return strat.add_indicators(strat.to_weekly(df), strat.to_weekly(bench))


# --------------------------------------------------------------- indicadores
def test_indicadores_no_miran_al_futuro(weekly):
    """box_high_prev en la semana N solo puede usar datos hasta N-1."""
    highs = weekly["High"]
    for i in range(5, len(weekly)):
        expected = highs.iloc[i - 3:i].max()
        actual = weekly["box_high_prev"].iloc[i]
        if not pd.isna(actual):
            assert actual == pytest.approx(expected)


def test_atr_es_positivo_y_finito(weekly):
    atr = weekly["atr"].dropna()
    assert len(atr) > 0
    assert (atr > 0).all()


def test_score_pondera_ambos_terminos(weekly):
    """El score estandarizado ya no está dominado por el volumen."""
    ok = weekly.dropna(subset=["rs_z", "vol_z"])
    assert len(ok) > 20
    # Tras estandarizar, ambos términos tienen dispersión comparable.
    ratio = ok["vol_z"].std() / max(ok["rs_z"].std(), 1e-9)
    assert 0.4 < ratio < 2.5, f"escalas aun desbalanceadas: {ratio:.2f}"


def test_score_responde_a_fuerza_relativa(strat):
    fuerte = pd.Series({"rs_z": 2.0, "vol_z": 0.0})
    debil = pd.Series({"rs_z": -2.0, "vol_z": 0.0})
    assert strat.score_candidate(fuerte) > strat.score_candidate(debil)


# ------------------------------------------------------------------- régimen
def test_regime_ok_bloquea_mercado_bajista(strat):
    bajista = strat.to_weekly(make_daily(drift=-0.004, seed=3))
    assert strat.regime_ok(bajista, bajista.index[-1]) is False


def test_regime_ok_permite_mercado_alcista(strat):
    alcista = strat.to_weekly(make_daily(drift=0.004, seed=3))
    assert strat.regime_ok(alcista, alcista.index[-1]) is True


def test_regime_desactivable():
    s = WeeklyTrendStrategy(StrategyConfig(require_bull_regime=False))
    bajista = s.to_weekly(make_daily(drift=-0.004, seed=3))
    assert s.regime_ok(bajista, bajista.index[-1]) is True


# -------------------------------------------------------------------- sector
def test_tope_por_sector_bloquea_cuarto_banco():
    s = WeeklyTrendStrategy(StrategyConfig(max_per_sector=3))
    bancos = {sym: make_position(symbol=sym) for sym in ("BAC", "GS", "MS")}
    assert s.sector_slot_available("WFC", bancos) is False
    assert s.sector_slot_available("NVDA", bancos) is True


def test_tope_por_sector_apagado_por_defecto(strat):
    """El backtest mostró que el tope costaba retorno sin bajar el drawdown."""
    bancos = {sym: make_position(symbol=sym) for sym in ("BAC", "GS", "MS")}
    assert strat.sector_slot_available("WFC", bancos) is True


def test_semis_separados_de_tech():
    assert SECTOR_MAP["NVDA"] == SECTOR_MAP["AMD"] == "semis"
    assert SECTOR_MAP["AAPL"] == "tech"


# ---------------------------------------------------------------------- stop
def test_tope_de_stop_limita_distancia():
    s = WeeklyTrendStrategy(StrategyConfig(max_stop_pct=0.15))
    row = pd.Series({"box_low_prev": 60.0, "atr": 5.0})   # stop crudo a -40%
    assert s.resolve_stop(100.0, row) == pytest.approx(85.0)


def test_stop_por_atr():
    s = WeeklyTrendStrategy(StrategyConfig(atr_stop_mult=3.0))
    row = pd.Series({"box_low_prev": 50.0, "atr": 5.0})
    assert s.resolve_stop(100.0, row) == pytest.approx(85.0)


def test_stop_sin_topes_usa_la_caja():
    s = WeeklyTrendStrategy(StrategyConfig())
    row = pd.Series({"box_low_prev": 60.0, "atr": 5.0})
    assert s.resolve_stop(100.0, row) == pytest.approx(60.0)


def test_build_position_rechaza_stop_invalido(strat):
    row = pd.Series({"box_low_prev": 120.0, "atr": 5.0})
    assert strat.build_position("X", pd.Timestamp("2025-01-06"), 100.0, row, 1e5, 1e5) is None


def test_piso_de_tamano_rechaza_posicion_testimonial():
    """Con el piso activo, no abre 1 accion de un objetivo de 200."""
    s = WeeklyTrendStrategy(StrategyConfig(min_position_fraction=0.5))
    row = pd.Series({"box_low_prev": 90.0, "atr": 5.0})
    assert s.build_position("X", pd.Timestamp("2025-01-06"), 100.0, row, 100_000, 150.0) is None
    # 120 de 200 objetivo = 60%, por encima del piso
    pos = s.build_position("X", pd.Timestamp("2025-01-06"), 100.0, row, 100_000, 12_000.0)
    assert pos is not None and pos.shares == 120


def test_sin_piso_por_defecto_acepta_posicion_chica(strat):
    """Default 0: el backtest mostro que saltear entradas chicas cuesta mas."""
    row = pd.Series({"box_low_prev": 90.0, "atr": 5.0})
    pos = strat.build_position("X", pd.Timestamp("2025-01-06"), 100.0, row, 100_000, 150.0)
    assert pos is not None and pos.shares == 1


def test_build_position_respeta_riesgo(strat):
    row = pd.Series({"box_low_prev": 90.0, "atr": 5.0})
    pos = strat.build_position("X", pd.Timestamp("2025-01-06"), 100.0, row, 100_000, 1e9)
    assert pos is not None
    # 2% de 100k = 2000 de riesgo, 10 por acción -> 200 acciones
    assert pos.shares == 200
    assert pos.risk_per_share == pytest.approx(10.0)


# ------------------------------------------------------- gestión de posición
def test_stop_tiene_prioridad_sobre_todo(strat):
    pos = make_position()
    row = pd.Series({"High": 130.0, "Low": 85.0, "Adj Close": 88.0, "exit_signal": True})
    d = strat.evaluate_position_week(pos, row)
    assert d["action"] == "exit_all" and d["reason"] == "stop"


def test_break_even_se_arma_en_1r(strat):
    pos = make_position()
    row = pd.Series({"High": 112.0, "Low": 99.0, "Adj Close": 108.0,
                     "exit_signal": False, "prior_2w_low": 95.0})
    d = strat.evaluate_position_week(pos, row)
    assert d["break_even_armed"] is True
    assert d["new_pending_stop"] >= pos.entry_price


def test_trailing_nunca_baja_el_stop(strat):
    pos = make_position(stop_price=105.0)
    row = pd.Series({"High": 120.0, "Low": 108.0, "Adj Close": 115.0,
                     "exit_signal": False, "prior_2w_low": 95.0})
    d = strat.evaluate_position_week(pos, row)
    assert d["new_pending_stop"] >= 105.0


def test_stop_pendiente_se_activa_la_semana_siguiente(strat):
    pos = make_position(pending_stop_price=97.0)
    pos = strat.activate_pending_stop(pos)
    assert pos.stop_price == 97.0
    assert pos.pending_stop_price is None


def test_parcial_deja_al_menos_una_accion(strat):
    pos = make_position(shares=1, initial_shares=1)
    row = pd.Series({"High": 200.0, "Low": 99.0, "Adj Close": 190.0,
                     "exit_signal": False, "prior_2w_low": 95.0})
    d = strat.evaluate_position_week(pos, row)
    assert d["action"] != "partial_exit"


# ------------------------------------------------------- regresiones del bot
def test_semana_completada_viernes_antes_del_cierre():
    from datetime import datetime
    ref = datetime(2026, 7, 31, 10, 0, tzinfo=bot.NY)   # viernes 10:00
    assert bot.latest_completed_week_end(ref) == pd.Timestamp("2026-07-24")


def test_semana_completada_viernes_tras_el_cierre():
    from datetime import datetime
    ref = datetime(2026, 7, 31, 17, 0, tzinfo=bot.NY)
    assert bot.latest_completed_week_end(ref) == pd.Timestamp("2026-07-31")


def test_semana_completada_lunes():
    from datetime import datetime
    ref = datetime(2026, 7, 27, 9, 0, tzinfo=bot.NY)
    assert bot.latest_completed_week_end(ref) == pd.Timestamp("2026-07-24")


def test_week_data_available_detecta_barras_faltantes(weekly):
    presente = weekly.index[-1]
    assert bot.week_data_available({"X": weekly}, presente) is True
    assert bot.week_data_available({"X": weekly}, pd.Timestamp("2030-01-04")) is False


def test_precio_de_referencia_cae_al_ultimo_cierre():
    """Antes devolvía None pre-market del lunes y el bot no entraba nunca."""
    df = make_daily(periods=50)
    futuro = pd.Timestamp(df.index[-1]) + pd.Timedelta(days=7)
    resultado = bot.get_next_session_entry_price(df, futuro)
    assert resultado is not None
    fecha, precio = resultado
    assert precio == pytest.approx(float(df["Close"].iloc[-1]))


def test_position_from_dict_ignora_campos_extra():
    """El state reconstruido a mano traía claves que no son del dataclass."""
    d = {
        "symbol": "X", "entry_date": "2026-05-01 00:00:00", "entry_price": 100.0,
        "shares": 10, "initial_shares": 10, "stop_price": 90.0,
        "initial_stop_price": 90.0, "risk_per_share": 10.0,
        "notes": {"reconstruido": True}, "campo_desconocido": 123,
    }
    pos = bot.position_from_dict(d)
    assert pos.symbol == "X" and pos.shares == 10


def test_has_meaningful_changes():
    a = {"X": make_position()}
    b = {"X": make_position(shares=50)}
    assert bot.has_meaningful_changes(a, b) is True
    assert bot.has_meaningful_changes(a, {"X": make_position()}) is False


def test_client_order_id_incluye_motivo_y_es_valido():
    cid = bot.make_client_order_id("AAPL", "stopdia")
    assert cid.startswith("stopdia_AAPL_")
    assert len(cid) <= 48


# ------------------------------------------------------------------ backtest
def test_backtest_corre_y_conserva_capital():
    """Humo del simulador: sin señales el capital no puede cambiar."""
    import backtest as bt

    plano = make_daily(periods=300, drift=0.0, seed=5)
    plano[["High", "Low"]] = plano[["Close", "Close"]].values  # sin rango, sin rupturas
    daily = {"AAA": plano.copy(), "SPY": plano.copy()}
    sim = bt.Backtest(StrategyConfig(), daily, "SPY", initial_equity=50_000)
    res = sim.run("plano")
    assert len(res.equity) > 0
    assert res.equity.iloc[-1] == pytest.approx(50_000, rel=1e-6)


def test_backtest_no_gasta_mas_efectivo_del_disponible():
    import backtest as bt

    daily = {s: make_daily(periods=400, seed=i) for i, s in enumerate(["AAA", "BBB", "CCC"])}
    daily["SPY"] = make_daily(periods=400, drift=0.0005, seed=99)
    sim = bt.Backtest(StrategyConfig(), daily, "SPY", initial_equity=25_000)
    res = sim.run("cash")
    assert (res.equity > 0).all(), "el equity nunca puede volverse negativo"
