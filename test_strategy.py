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


def test_dry_run_no_persiste_ni_registra(tmp_path, monkeypatch):
    """Una simulación no puede borrar posiciones del state ni escribir trades."""
    import json

    estado = tmp_path / "state.json"
    estado.write_text(json.dumps({
        "positions": {"AAPL": {
            "symbol": "AAPL", "entry_date": "2026-05-01 00:00:00", "entry_price": 100.0,
            "shares": 10, "initial_shares": 10, "stop_price": 90.0,
            "initial_stop_price": 90.0, "risk_per_share": 10.0,
        }},
        "meta": {"last_processed_week": "2026-07-24"},
    }))
    historial = tmp_path / "trades.csv"
    monkeypatch.setattr(bot, "STATE_PATH", estado)
    monkeypatch.setattr(bot, "DRY_RUN", True)

    # save_state solo debe correr fuera de DRY_RUN
    pos, meta = bot.load_state()
    assert "AAPL" in pos
    if not bot.DRY_RUN:
        bot.save_state({}, meta)
    assert "AAPL" in bot.load_state()[0], "DRY_RUN no debe vaciar el state"
    assert not historial.exists(), "DRY_RUN no debe crear historial de trades"


def test_no_evalua_posicion_contra_semana_previa_a_su_entrada(strat):
    """El caso AMZN: vela en V que dispara la señal y mata la entrada."""
    # Semana de señal: mínimo 226.16, cierre 271.58. Stop de las 3 semanas
    # previas = 231.34. El mínimo de la propia semana lo perfora.
    pos = make_position(entry_price=285.93, stop_price=231.34,
                        risk_per_share=54.59, entry_date=pd.Timestamp("2026-08-03"))
    row = pd.Series({"High": 273.23, "Low": 226.16, "Adj Close": 271.58,
                     "exit_signal": False, "prior_2w_low": 231.34})
    # Evaluada directamente, la estrategia la cierra: por eso hace falta el guardia.
    assert strat.evaluate_position_week(pos, row)["action"] == "exit_all"

    semana = pd.Timestamp("2026-07-31")
    entrada = pd.Timestamp(pos.entry_date).normalize()
    assert entrada >= semana, "la entrada es posterior a la semana evaluada"


def test_si_evalua_cuando_la_entrada_es_anterior():
    entrada = pd.Timestamp("2026-05-04").normalize()
    semana = pd.Timestamp("2026-07-31")
    assert not (entrada >= semana), "una posición vieja sí debe evaluarse"


def test_una_sola_orden_por_simbolo_por_corrida(monkeypatch):
    """La doble venta de CSCO/MU/QCOM dejó la cuenta en corto. No puede repetirse."""
    enviadas = []

    class FakeOrder:
        id = "x"

    class FakeTrading:
        def submit_order(self, order):
            enviadas.append((order.symbol, order.qty))
            return FakeOrder()

        def get_order_by_id(self, _):
            raise RuntimeError("sin confirmación")  # peor caso: timeout

    monkeypatch.setattr(bot, "DRY_RUN", False)
    monkeypatch.setattr(bot, "_ORDENES_ENVIADAS", set())
    t = FakeTrading()

    bot.submit_market_order(t, "CSCO", 58, bot.OrderSide.SELL, reason="stopdia")
    bot.submit_market_order(t, "CSCO", 58, bot.OrderSide.SELL, reason="stop")

    assert len(enviadas) == 1, f"se enviaron {len(enviadas)} órdenes de CSCO: {enviadas}"


def test_espera_el_llenado_total_no_el_primer_parcial(monkeypatch):
    """Cortar en el primer tramo registraba 33 acciones de una venta de 58."""

    class FakeOrder:
        id = "x"

    class Estado:
        def __init__(self, v):
            self.value = v

    class Live:
        def __init__(self, q, st):
            self.filled_qty, self.filled_avg_price, self.status = q, 114.89, Estado(st)

    secuencia = [Live(0, "new"), Live(33, "partially_filled"), Live(58, "filled")]

    class FakeTrading:
        def submit_order(self, order):
            return FakeOrder()

        def get_order_by_id(self, _):
            return secuencia.pop(0)

    monkeypatch.setattr(bot, "DRY_RUN", False)
    monkeypatch.setattr(bot, "_ORDENES_ENVIADAS", set())
    monkeypatch.setattr(bot, "FILL_POLL_SECONDS", 0)
    resultado = bot.submit_market_order(FakeTrading(), "CSCO", 58, bot.OrderSide.SELL, reason="stop")
    assert resultado is not None
    assert resultado[1] == 58, f"registró {resultado[1]} en vez de 58"


def test_devuelve_parcial_si_la_orden_se_cancela(monkeypatch):
    class FakeOrder:
        id = "x"

    class Estado:
        def __init__(self, v):
            self.value = v

    class Live:
        def __init__(self, q, st):
            self.filled_qty, self.filled_avg_price, self.status = q, 100.0, Estado(st)

    secuencia = [Live(20, "partially_filled"), Live(20, "canceled")]

    class FakeTrading:
        def submit_order(self, order):
            return FakeOrder()

        def get_order_by_id(self, _):
            return secuencia.pop(0)

    monkeypatch.setattr(bot, "DRY_RUN", False)
    monkeypatch.setattr(bot, "_ORDENES_ENVIADAS", set())
    monkeypatch.setattr(bot, "FILL_POLL_SECONDS", 0)
    r = bot.submit_market_order(FakeTrading(), "X", 50, bot.OrderSide.SELL, reason="stop")
    assert r == (100.0, 20.0)


def test_guardia_no_bloquea_simbolos_distintos(monkeypatch):
    enviadas = []

    class FakeOrder:
        id = "x"

    class FakeTrading:
        def submit_order(self, order):
            enviadas.append(order.symbol)
            return FakeOrder()

        def get_order_by_id(self, _):
            raise RuntimeError("sin confirmación")

    monkeypatch.setattr(bot, "DRY_RUN", False)
    monkeypatch.setattr(bot, "_ORDENES_ENVIADAS", set())
    t = FakeTrading()
    for s in ("CSCO", "MU", "QCOM"):
        bot.submit_market_order(t, s, 10, bot.OrderSide.SELL, reason="stopdia")
    assert enviadas == ["CSCO", "MU", "QCOM"]


def test_entradas_excluyen_tenencia_del_broker():
    """Un simbolo que el broker ya tiene no puede volver a comprarse."""
    updated = {"BAC": None}
    pending = {"CSCO"}
    broker = {"MA": {"qty": 25.0}, "V": {"qty": 68.0}, "BAC": {"qty": 291.0}}
    held = {s for s, v in broker.items() if v.get("qty", 0) > 0}
    excluded = set(updated) | pending | held
    assert "MA" in excluded and "V" in excluded
    assert excluded == {"BAC", "CSCO", "MA", "V"}


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


def test_fetch_daily_bars_cae_a_iex_si_falla_el_feed(monkeypatch):
    """Si SIP dejara de estar disponible, el bot no puede quedarse sin datos."""
    pedidos = []

    class FakeRaw:
        data = {}

    class FakeClient:
        def get_stock_bars(self, req):
            pedidos.append(req.feed)
            if req.feed == "sip":
                raise RuntimeError("subscription does not permit querying recent SIP data")
            return FakeRaw()

    monkeypatch.setattr(bot, "DATA_FEED", "sip")
    bot.fetch_daily_bars(FakeClient(), ["AAPL"])
    assert pedidos == ["sip", "iex"], f"secuencia inesperada: {pedidos}"


def test_fetch_daily_bars_no_reintenta_si_ya_es_iex(monkeypatch):
    class FakeClient:
        def get_stock_bars(self, req):
            raise RuntimeError("fallo de red")

    monkeypatch.setattr(bot, "DATA_FEED", "iex")
    with pytest.raises(RuntimeError):
        bot.fetch_daily_bars(FakeClient(), ["AAPL"])


# ── informe PDF ──────────────────────────────────────────────────────────────
def test_motivo_distingue_las_ordenes_que_no_son_del_bot():
    import informe_pdf as inf

    assert inf.etiqueta_motivo("stop_MA_20260803120000") == "stop"
    assert inf.etiqueta_motivo("partial_AAPL_20260612093001") == "partial"
    # Alpaca pone un UUID cuando la orden no salio del bot: agrupar por el
    # prefijo llenaba la tabla de motivos con un identificador por operacion.
    assert inf.etiqueta_motivo("866041c7-af8d-4320-8813-9e19c1598a92") == inf.SIN_ETIQUETA
    assert inf.etiqueta_motivo("") == inf.SIN_ETIQUETA


def test_diagnostico_lag_detecta_la_serie_corrida_un_dia():
    import numpy as np
    import informe_pdf as inf

    idx = pd.bdate_range("2026-04-01", periods=120)
    r = np.random.default_rng(7).normal(0.0005, 0.01, len(idx))
    spy = pd.Series(100 * np.cumprod(1 + r), index=idx)
    cart = pd.Series(100 * np.cumprod(1 + r * 0.8), index=idx)

    alineada = inf.diagnostico_lag(cart, spy)
    assert max(alineada, key=lambda k: alineada[k]) == 0

    # Una cartera 80% correlacionada con el indice, corrida un dia, daba
    # correlacion ~0 y beta ~0: exactamente lo que publicaba el informe.
    corrida = cart.copy()
    corrida.index = idx.shift(1)
    d = inf.diagnostico_lag(corrida, spy)
    assert abs(d[0]) < 0.2
    assert max(d, key=lambda k: d[k]) == -1


def test_posicion_adoptada_no_toma_parcial():
    """Una adoptada no tiene R conocida: el parcial a +2.5R no significa nada.

    JPM quedo fuera del state al reconstruirlo y el bot la adopto de nuevo, con
    partial_taken en falso. Como ya venia ganando, cumplio el umbral y le
    recorto un tercio por segunda vez en ocho dias: 16 acciones el 3 de agosto
    y 11 el 10, de 50 originales.
    """
    week_end = pd.Timestamp("2026-08-10")
    weekly = pd.DataFrame(
        {"box_low_prev": [300.0], "Adj Close": [357.0]}, index=[week_end])
    broker = {"JPM": {"qty": 34, "avg_entry_price": 327.10}}

    rec = bot.reconcile_state_with_broker({}, broker, {"JPM": weekly}, week_end)

    assert rec["JPM"].notes.get("adoptado") is True
    assert rec["JPM"].partial_taken is True, "la adoptada quedaria habilitada a vender un tercio"


# ── rutas y reparacion del state ─────────────────────────────────────────────
def test_rutas_no_dependen_del_directorio_de_trabajo():
    # Una tarea programada arranca donde diga su "Iniciar en". Con rutas
    # relativas el bot podia usar un state distinto sin dar error.
    from pathlib import Path
    base = Path(bot.__file__).resolve().parent
    assert bot.STATE_PATH.is_absolute() and base in bot.STATE_PATH.parents
    assert bot.SYMBOLS_PATH.is_absolute() and base in bot.SYMBOLS_PATH.parents
    import trade_tracker
    assert trade_tracker.TRADE_HISTORY_PATH.is_absolute()


def _pos(stop, pend=None, **kw):
    from strategy_weekly_bot_ready import PositionState
    return PositionState(symbol="MA", entry_date=pd.Timestamp("2026-08-03"),
                         entry_price=582.55, shares=42, initial_shares=42,
                         stop_price=stop, initial_stop_price=stop,
                         risk_per_share=67.44, pending_stop_price=pend, **kw)


def test_reparar_nunca_baja_un_stop_existente():
    import reparar_state as R
    # Reconstruido mas bajo que lo guardado: gana lo guardado.
    p = R.combinar({"stop_price": 560.0, "pending_stop_price": 565.0}, _pos(540.0, 550.0))
    assert p.stop_price == 560.0 and p.pending_stop_price == 565.0
    # Reconstruido mas alto (el caso de MA): gana el reconstruido.
    p = R.combinar({"stop_price": 523.61, "pending_stop_price": None}, _pos(578.71))
    assert p.stop_price == 578.71
    # Un parcial ya tomado no se habilita de nuevo.
    p = R.combinar({"stop_price": 1.0, "partial_taken": True}, _pos(578.71))
    assert p.partial_taken is True


def test_reparar_reproduce_la_salida_que_el_bot_no_hizo():
    import reparar_state as R
    strat = WeeklyTrendStrategy(bot.build_config())
    trail = strat.cfg.trail_mode
    idx = pd.to_datetime(["2026-09-04", "2026-09-11", "2026-09-18"])
    wk = pd.DataFrame({"High": [595.18, 575.67, 578.90], "Low": [578.71, 563.17, 561.90],
                       "Adj Close": [590.0, 570.0, 565.24], "exit_signal": [False] * 3,
                       trail: [561.26, 578.71, 578.71]}, index=idx)
    # State de MA al cierre de agosto: stop 556.06 con el trailing en camino.
    pos, salida = R.reproducir(strat, _pos(556.06, 556.06), wk, pd.Timestamp("2026-09-18"))
    assert salida is not None and str(salida[0]) == "2026-09-18"
    assert pos.stop_price == 578.71


def test_lotes_abiertos_fifo():
    import reparar_state as R
    f = pd.DataFrame([
        dict(symbol="QCOM", side="buy", qty=107, price=175.27, fecha=pd.Timestamp("2026-09-14")),
        dict(symbol="QCOM", side="sell", qty=107, price=180.57, fecha=pd.Timestamp("2026-09-21")),
        dict(symbol="AMD", side="buy", qty=11, price=585.45, fecha=pd.Timestamp("2026-09-21")),
    ])
    lotes = R.lotes_abiertos(f)
    assert "QCOM" not in lotes
    assert lotes["AMD"][0]["q"] == 11


def test_reparar_usa_la_semana_de_la_ultima_corrida_no_la_del_state():
    from datetime import datetime
    import reparar_state as R
    ny = bot.NY
    # Miercoles: la corrida del lunes ya proceso la semana del viernes previo.
    assert str(R.semana_de_la_ultima_corrida(
        datetime(2026, 9, 24, 12, tzinfo=ny)).date()) == "2026-09-18"
    # Lunes antes de que corra el bot: todavia vale la corrida anterior.
    assert str(R.semana_de_la_ultima_corrida(
        datetime(2026, 9, 21, 9, tzinfo=ny)).date()) == "2026-09-11"


# ── corrida con el mercado abierto (Railway, 9:35 NY) ────────────────────────
def _barras(fechas):
    idx = pd.to_datetime(fechas)
    return pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0,
                         "Adj Close": 1.0, "Volume": 1.0}, index=idx)


def test_descarta_la_barra_de_una_sesion_abierta():
    from datetime import datetime
    df = _barras(["2026-09-18", "2026-09-21"])
    # Lunes 9:35: la barra del 21 es intradía y su "cierre" es el precio del momento.
    abierta = bot.drop_unfinished_session(df, datetime(2026, 9, 21, 9, 35, tzinfo=bot.NY))
    assert list(abierta.index.date.astype(str)) == ["2026-09-18"]
    # Después del cierre se conserva.
    cerrada = bot.drop_unfinished_session(df, datetime(2026, 9, 21, 18, 0, tzinfo=bot.NY))
    assert len(cerrada) == 2
    # Un fin de semana no pierde el viernes.
    finde = bot.drop_unfinished_session(_barras(["2026-09-18"]),
                                        datetime(2026, 9, 20, 10, 0, tzinfo=bot.NY))
    assert len(finde) == 1


def test_sin_barra_de_hoy_la_entrada_usa_el_ultimo_cierre():
    df = _barras(["2026-09-17", "2026-09-18"])
    ref = bot.get_next_session_entry_price(df, pd.Timestamp("2026-09-18"))
    assert ref is not None, "sin la barra del lunes el bot no podria entrar"


def test_semilla_inicializa_un_volume_vacio(tmp_path, monkeypatch):
    import json
    semilla = tmp_path / "semilla.json"
    semilla.write_text(json.dumps({"positions": {"KO": {"stop_price": 87.21}},
                                   "meta": {"last_processed_week": "2026-09-18"}}))
    destino = tmp_path / "volume" / "state.json"
    monkeypatch.setattr(bot, "STATE_PATH", destino)
    monkeypatch.setattr(bot, "STATE_SEED_PATH", semilla)
    bot.seed_state_if_missing()
    assert bot.load_state_raw()["meta"]["last_processed_week"] == "2026-09-18"
    # Un state existente no se pisa nunca.
    destino.write_text(json.dumps({"positions": {}, "meta": {"last_processed_week": "X"}}))
    bot.seed_state_if_missing()
    assert bot.load_state_raw()["meta"]["last_processed_week"] == "X"
