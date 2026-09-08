"""Genera un informe PDF del desempeno de la estrategia contra el SPY.

Toma todo de la fuente: ordenes ejecutadas, historial de cartera y posiciones
desde Alpaca, y la serie del SPY desde el feed de datos. No depende de ningun
CSV previo.

    python informe_pdf.py
    python informe_pdf.py --desde 2026-04-01 --salida informe.pdf
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.graphics.shapes import Drawing, Line, PolyLine, String, Rect
from reportlab.platypus import (
    KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

# ── identidad visual, la misma del reporte HTML ──────────────────────────────
TINTA   = colors.HexColor("#16181C")
TINTA2  = colors.HexColor("#454A53")
GRIS    = colors.HexColor("#7C8189")
REGLA   = colors.HexColor("#DCDAD3")
ACENTO  = colors.HexColor("#2D4A8A")
POS     = colors.HexColor("#0E8A63")
NEG     = colors.HexColor("#CE4A33")
FONDO   = colors.HexColor("#F5F4F1")

SANS, SANS_B, MONO = "Helvetica", "Helvetica-Bold", "Courier"


# ── datos ────────────────────────────────────────────────────────────────────
def clientes():
    key = os.getenv("ALPACA_API_KEY")
    sec = os.getenv("ALPACA_API_SECRET") or os.getenv("ALPACA_SECRET_KEY")
    if not key or not sec:
        sys.exit("Faltan ALPACA_API_KEY / ALPACA_API_SECRET")
    from alpaca.trading.client import TradingClient
    from alpaca.data.historical.stock import StockHistoricalDataClient
    paper = os.getenv("ALPACA_PAPER", "1") != "0"
    return TradingClient(key, sec, paper=paper), StockHistoricalDataClient(key, sec)


def traer_ordenes(t) -> pd.DataFrame:
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    filas, vistos = [], set()
    desde = datetime(2020, 1, 1, tzinfo=timezone.utc)
    while True:
        lote = t.get_orders(filter=GetOrdersRequest(
            status=QueryOrderStatus.CLOSED, limit=500, after=desde, direction="asc"))
        nuevas = [o for o in lote if str(o.id) not in vistos]
        if not nuevas:
            break
        for o in nuevas:
            vistos.add(str(o.id))
            q = float(o.filled_qty or 0)
            if q <= 0 or o.filled_avg_price is None:
                continue
            filas.append({"symbol": o.symbol, "side": o.side.value, "qty": q,
                          "price": float(o.filled_avg_price),
                          "fecha": pd.Timestamp(o.filled_at).tz_convert(None),
                          "cid": o.client_order_id or ""})
        desde = max(pd.Timestamp(o.submitted_at) for o in nuevas).to_pydatetime() \
            + timedelta(microseconds=1)
        if len(lote) < 500:
            break
    return pd.DataFrame(filas).sort_values("fecha").reset_index(drop=True)


def round_trips(fills: pd.DataFrame) -> pd.DataFrame:
    lotes: dict[str, deque] = defaultdict(deque)
    ops = []
    for _, f in fills.iterrows():
        s = f["symbol"]
        if f["side"] == "buy":
            lotes[s].append({"q": f["qty"], "p": f["price"], "d": f["fecha"]})
            continue
        resto = f["qty"]
        while resto > 1e-9 and lotes[s]:
            l = lotes[s][0]
            tomo = min(resto, l["q"])
            ops.append({"symbol": s, "entrada": l["d"], "p_ent": l["p"],
                        "salida": f["fecha"], "p_sal": f["price"], "qty": tomo,
                        "pnl": (f["price"] - l["p"]) * tomo,
                        "pct": (f["price"] / l["p"] - 1) * 100,
                        "dias": (f["fecha"] - l["d"]).total_seconds() / 86400,
                        "motivo": f["cid"].split("_")[0] if f["cid"] else "?"})
            l["q"] -= tomo
            resto -= tomo
            if l["q"] <= 1e-9:
                lotes[s].popleft()
    return pd.DataFrame(ops)


def serie_cartera(t, desde: pd.Timestamp) -> pd.Series:
    from alpaca.trading.requests import GetPortfolioHistoryRequest
    h = t.get_portfolio_history(GetPortfolioHistoryRequest(
        start=desde.to_pydatetime().replace(tzinfo=timezone.utc),
        end=datetime.now(timezone.utc), timeframe="1D"))
    s = pd.Series(h.equity, index=pd.to_datetime(h.timestamp, unit="s")).dropna()
    s.index = s.index.normalize()
    return s[s > 0]


def serie_spy(d, desde: pd.Timestamp) -> pd.Series:
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    b = d.get_stock_bars(StockBarsRequest(
        symbol_or_symbols=["SPY"], timeframe=TimeFrame.Day,
        start=(desde - pd.Timedelta(days=7)).to_pydatetime().replace(tzinfo=timezone.utc),
        adjustment="all", feed=os.getenv("ALPACA_DATA_FEED", "sip")))
    s = pd.Series({pd.Timestamp(x.timestamp).tz_convert(None).normalize(): float(x.close)
                   for x in b.data.get("SPY", [])}).sort_index()
    return s[s.index >= desde]


# ── metricas ─────────────────────────────────────────────────────────────────
def metricas(s: pd.Series) -> dict:
    r = s.pct_change().dropna()
    dd = s / s.cummax() - 1
    dias = max((s.index[-1] - s.index[0]).days, 1)
    tot = s.iloc[-1] / s.iloc[0] - 1
    neg = r[r < 0].std()
    return {
        "retorno": tot * 100,
        "anualizado": ((1 + tot) ** (365 / dias) - 1) * 100,
        "maxdd": dd.min() * 100,
        "dd_fecha": dd.idxmin(),
        "vol": r.std() * math.sqrt(252) * 100 if r.std() else 0.0,
        "sharpe": r.mean() / r.std() * math.sqrt(252) if r.std() else 0.0,
        "sortino": r.mean() / neg * math.sqrt(252) if neg and neg > 0 else 0.0,
        "mejor": r.max() * 100 if len(r) else 0.0,
        "peor": r.min() * 100 if len(r) else 0.0,
        "positivos": (r > 0).mean() * 100 if len(r) else 0.0,
        "n": len(s),
    }


def relativas(cart: pd.Series, bench: pd.Series) -> dict:
    a = cart.pct_change().dropna()
    b = bench.reindex(cart.index).ffill().pct_change().dropna()
    j = a.index.intersection(b.index)
    a, b = a.loc[j], b.loc[j]
    if len(j) < 3 or b.std() == 0:
        return {}
    beta = float(np.cov(a, b)[0, 1] / np.var(b))
    activo = a - b
    return {
        "beta": beta,
        "correlacion": float(np.corrcoef(a, b)[0, 1]),
        "alfa_anual": float((a.mean() - beta * b.mean()) * 252 * 100),
        "tracking": float(activo.std() * math.sqrt(252) * 100),
        "info_ratio": float(activo.mean() / activo.std() * math.sqrt(252))
        if activo.std() else 0.0,
    }


# ── piezas visuales ──────────────────────────────────────────────────────────
def grafico_curvas(cart: pd.Series, spy: pd.Series, ancho=165 * mm, alto=72 * mm) -> Drawing:
    """Ambas series en base 100. Un solo eje: son la misma magnitud."""
    d = Drawing(ancho, alto)
    ml, mr, mb, mt = 16 * mm, 4 * mm, 11 * mm, 6 * mm
    pw, ph = ancho - ml - mr, alto - mb - mt

    b = spy.reindex(cart.index).ffill().bfill()
    ic = cart / cart.iloc[0] * 100
    ib = b / b.iloc[0] * 100
    lo = min(ic.min(), ib.min()) * 0.995
    hi = max(ic.max(), ib.max()) * 1.005
    rng = max(hi - lo, 1e-9)

    def xy(i, v):
        return (ml + pw * i / max(len(ic) - 1, 1), mb + ph * (v - lo) / rng)

    for k in range(5):
        v = lo + rng * k / 4
        y = mb + ph * k / 4
        d.add(Line(ml, y, ml + pw, y, strokeColor=REGLA, strokeWidth=0.4))
        d.add(String(ml - 2, y - 1.6, f"{v:.0f}", fontName=MONO, fontSize=6,
                     fillColor=GRIS, textAnchor="end"))

    for serie, color, grosor in ((ib, GRIS, 1.2), (ic, ACENTO, 1.8)):
        pts = []
        for i, v in enumerate(serie.values):
            pts.extend(xy(i, v))
        d.add(PolyLine(pts, strokeColor=color, strokeWidth=grosor,
                       strokeLineJoin=1, strokeLineCap=1))

    for i in (0, len(ic) // 2, len(ic) - 1):
        d.add(String(ml + pw * i / max(len(ic) - 1, 1),
                     mb - 7, ic.index[i].strftime("%d %b"), fontName=SANS,
                     fontSize=6, fillColor=GRIS, textAnchor="middle"))

    lx, ly = ml + 3 * mm, mb + ph - 3 * mm
    d.add(Line(lx, ly, lx + 6 * mm, ly, strokeColor=ACENTO, strokeWidth=1.8))
    d.add(String(lx + 7 * mm, ly - 2, "Estrategia", fontName=SANS_B, fontSize=7,
                 fillColor=TINTA))
    d.add(Line(lx, ly - 5 * mm, lx + 6 * mm, ly - 5 * mm, strokeColor=GRIS, strokeWidth=1.2))
    d.add(String(lx + 7 * mm, ly - 5 * mm - 2, "SPY", fontName=SANS, fontSize=7,
                 fillColor=TINTA2))
    return d


def grafico_pnl(ops: pd.DataFrame, ancho=165 * mm) -> Drawing:
    """Barras divergentes: el signo se lee por direccion, no solo por color."""
    por = ops.groupby("symbol")["pnl"].sum().sort_values(ascending=False)
    fila, alto = 6.2 * mm, 0
    alto = fila * len(por) + 6 * mm
    d = Drawing(ancho, alto)
    tope = max(abs(por.min()), abs(por.max()), 1.0)
    cx, semi = ancho * 0.46, ancho * 0.34

    d.add(Line(cx, 2 * mm, cx, alto - 2 * mm, strokeColor=REGLA, strokeWidth=0.6))
    for i, (sym, v) in enumerate(por.items()):
        y = alto - 4 * mm - i * fila
        d.add(String(cx - semi - 2 * mm, y - 1.6, sym, fontName=MONO, fontSize=7,
                     fillColor=TINTA, textAnchor="end"))
        w = semi * abs(v) / tope
        if v >= 0:
            d.add(Rect(cx, y - 2.2, w, 4.4, fillColor=POS, strokeColor=None))
        else:
            d.add(Rect(cx - w, y - 2.2, w, 4.4, fillColor=NEG, strokeColor=None))
        d.add(String(ancho - 1 * mm, y - 1.6, f"{v:+,.0f}", fontName=MONO, fontSize=7,
                     fillColor=POS if v >= 0 else NEG, textAnchor="end"))
    return d


def tabla(datos, anchos, cols_num=None, cabecera=True):
    """cols_num: indices de columnas numericas (mono, a la derecha).

    Por defecto todas menos la primera. Las columnas de prosa se dejan fuera:
    alinearlas a la derecha en monoespaciada las vuelve ilegibles.
    """
    ncol = len(datos[0])
    if cols_num is None:
        cols_num = list(range(1, ncol))
    est = [
        ("FONTNAME", (0, 0), (-1, -1), SANS),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TEXTCOLOR", (0, 0), (-1, -1), TINTA),
        ("LINEBELOW", (0, 0), (-1, -2), 0.35, REGLA),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
    for c in cols_num:
        est += [("ALIGN", (c, 0), (c, -1), "RIGHT"),
                ("FONTNAME", (c, 1), (c, -1), MONO)]
    if cabecera:
        est += [("FONTNAME", (0, 0), (-1, 0), SANS_B),
                ("FONTSIZE", (0, 0), (-1, 0), 7),
                ("TEXTCOLOR", (0, 0), (-1, 0), GRIS),
                ("LINEBELOW", (0, 0), (-1, 0), 0.8, TINTA2)]
    t = Table(datos, colWidths=anchos, hAlign="LEFT")
    t.setStyle(TableStyle(est))
    return t


# ── documento ────────────────────────────────────────────────────────────────
def construir(ruta, cart, spy, ops, mc, ms, rel, posiciones, cuenta, desde):
    doc = SimpleDocTemplate(str(ruta), pagesize=A4,
                            leftMargin=22 * mm, rightMargin=22 * mm,
                            topMargin=20 * mm, bottomMargin=18 * mm,
                            title="Desempeno de la estrategia semanal",
                            author="Weekly Trading Bot")
    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=ss["Title"], fontName=SANS_B, fontSize=21,
                        leading=24, textColor=TINTA, alignment=TA_LEFT, spaceAfter=2)
    SUB = ParagraphStyle("SUB", fontName=SANS, fontSize=10, leading=14,
                         textColor=TINTA2, spaceAfter=10)
    H2 = ParagraphStyle("H2", fontName=SANS_B, fontSize=13, leading=16,
                        textColor=TINTA, spaceBefore=13, spaceAfter=5)
    H3 = ParagraphStyle("H3", fontName=SANS_B, fontSize=9.5, leading=12,
                        textColor=ACENTO, spaceBefore=9, spaceAfter=3)
    P = ParagraphStyle("P", fontName=SANS, fontSize=9, leading=13.2,
                       textColor=TINTA, spaceAfter=6)
    CAP = ParagraphStyle("CAP", fontName=SANS, fontSize=7.5, leading=10,
                         textColor=GRIS, spaceAfter=8)
    NOTA = ParagraphStyle("NOTA", parent=P, leftIndent=7, borderPadding=6,
                          backColor=FONDO, borderColor=ACENTO, borderWidth=0,
                          spaceBefore=4, spaceAfter=8)

    e = []
    ini, fin = cart.index[0].date(), cart.index[-1].date()
    e.append(Paragraph("Desempeno de la estrategia semanal", H1))
    e.append(Paragraph(
        f"Analisis de las operaciones ejecutadas entre el {ini} y el {fin}, "
        f"contrastadas con el S&amp;P 500 (SPY) en el mismo periodo.", SUB))

    # KPIs
    dif = mc["retorno"] - ms["retorno"]
    kpi = [["Retorno estrategia", "Retorno SPY", "Diferencia", "Operaciones"],
           [f"{mc['retorno']:+.2f}%", f"{ms['retorno']:+.2f}%",
            f"{dif:+.2f} pts", f"{len(ops)}"]]
    t = Table(kpi, colWidths=[41 * mm] * 4, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), SANS), ("FONTSIZE", (0, 0), (-1, 0), 7.5),
        ("TEXTCOLOR", (0, 0), (-1, 0), GRIS),
        ("FONTNAME", (0, 1), (-1, 1), SANS_B), ("FONTSIZE", (0, 1), (-1, 1), 16),
        ("TEXTCOLOR", (0, 1), (0, 1), POS if mc["retorno"] >= 0 else NEG),
        ("TEXTCOLOR", (1, 1), (1, 1), TINTA2),
        ("TEXTCOLOR", (2, 1), (2, 1), POS if dif >= 0 else NEG),
        ("TEXTCOLOR", (3, 1), (3, 1), TINTA),
        ("LINEABOVE", (0, 0), (-1, 0), 1.2, TINTA),
        ("LINEBELOW", (0, 1), (-1, 1), 0.4, REGLA),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
    ]))
    e.append(t)
    e.append(Spacer(1, 12))

    # curva
    e.append(Paragraph("Evolucion comparada", H2))
    e.append(grafico_curvas(cart, spy))
    e.append(Paragraph(
        "Ambas series en base 100 al inicio del periodo. Un solo eje: son la misma "
        "magnitud, de modo que la distancia vertical es directamente la diferencia "
        "de rendimiento acumulado.", CAP))

    # riesgo y retorno
    e.append(Paragraph("Riesgo y retorno", H2))
    filas = [["Metrica", "Estrategia", "SPY", "Diferencia"]]
    for et, k, u in (("Retorno del periodo", "retorno", "%"),
                     ("Retorno anualizado", "anualizado", "%"),
                     ("Maximo drawdown", "maxdd", "%"),
                     ("Volatilidad anualizada", "vol", "%"),
                     ("Ratio de Sharpe", "sharpe", ""),
                     ("Ratio de Sortino", "sortino", ""),
                     ("Mejor dia", "mejor", "%"),
                     ("Peor dia", "peor", "%"),
                     ("Dias positivos", "positivos", "%")):
        a, b = mc[k], ms[k]
        filas.append([et, f"{a:+.2f}{u}" if u else f"{a:.2f}",
                      f"{b:+.2f}{u}" if u else f"{b:.2f}",
                      f"{a - b:+.2f}"])
    e.append(tabla(filas, [58 * mm, 34 * mm, 34 * mm, 34 * mm]))
    e.append(Spacer(1, 4))

    if rel:
        f2 = [["Metrica", "Valor", "Lectura"],
              ["Beta", f"{rel['beta']:.2f}",
               "Sensibilidad al mercado; 1.00 replicaria al indice"],
              ["Correlacion", f"{rel['correlacion']:.2f}",
               "Cuanto se mueven juntos dia a dia"],
              ["Alfa anualizada", f"{rel['alfa_anual']:+.2f}%",
               "Retorno no explicado por la exposicion al mercado"],
              ["Tracking error", f"{rel['tracking']:.2f}%",
               "Dispersion tipica frente al indice"],
              ["Information ratio", f"{rel['info_ratio']:.2f}",
               "Retorno activo por unidad de desvio activo"]]
        # Junto con su titulo: partir esta tabla dejaba filas huerfanas.
        e.append(KeepTogether([
            Paragraph("Relacion con el indice", H3),
            tabla(f2, [38 * mm, 24 * mm, 98 * mm], cols_num=[1]),
        ]))

    # operaciones — sin salto forzado: el flujo natural evita paginas semivacias
    e.append(Paragraph("Analisis de las operaciones", H2))
    if ops.empty:
        e.append(Paragraph("No hay operaciones cerradas en el periodo.", P))
    else:
        g = ops[ops.pnl > 0]
        p = ops[ops.pnl <= 0]
        gb, pb = g.pnl.sum(), abs(p.pnl.sum())
        pf = gb / pb if pb else float("inf")
        ratio = (g.dias.mean() / p.dias.mean()) if len(p) and p.dias.mean() else 0

        f3 = [["Metrica", "Valor"],
              ["Operaciones cerradas", f"{len(ops)}"],
              ["Resultado neto", f"${ops.pnl.sum():+,.2f}"],
              ["Tasa de acierto", f"{len(g) / len(ops) * 100:.1f}%  ({len(g)}G / {len(p)}P)"],
              ["Ganancia bruta", f"${gb:,.2f}"],
              ["Perdida bruta", f"${-pb:,.2f}"],
              ["Factor de beneficio", f"{pf:.2f}"],
              ["Esperanza por operacion", f"${ops.pnl.mean():+,.2f}"],
              ["Ganancia media", f"${g.pnl.mean():+,.2f}  ({g.pct.mean():+.2f}%)" if len(g) else "-"],
              ["Perdida media", f"${p.pnl.mean():+,.2f}  ({p.pct.mean():+.2f}%)" if len(p) else "-"],
              ["Ratio ganancia/perdida", f"{abs(g.pnl.mean() / p.pnl.mean()):.2f}" if len(g) and len(p) and p.pnl.mean() else "-"],
              ["Mejor operacion", f"${ops.pnl.max():+,.2f}"],
              ["Peor operacion", f"${ops.pnl.min():+,.2f}"]]
        e.append(tabla(f3, [58 * mm, 102 * mm]))

        f4 = [["", "Ganadoras", "Perdedoras", "Cociente"],
              ["Duracion media (dias)",
               f"{g.dias.mean():.1f}" if len(g) else "-",
               f"{p.dias.mean():.1f}" if len(p) else "-",
               f"{ratio:.2f}x"],
              ["Mediana (dias)",
               f"{g.dias.median():.1f}" if len(g) else "-",
               f"{p.dias.median():.1f}" if len(p) else "-", ""]]
        e.append(KeepTogether([
            Paragraph("Duracion de las posiciones", H3),
            Paragraph(
                "En seguimiento de tendencia, las ganadoras deben sostenerse bastante mas "
                "que las perdedoras. Un cociente por debajo de 1.5 indica que las salidas "
                "estan cortando la cola derecha, que es de donde vive el sistema.", P),
            tabla(f4, [58 * mm, 34 * mm, 34 * mm, 34 * mm]),
        ]))

        e.append(KeepTogether([
            Paragraph("Resultado por instrumento", H3),
            grafico_pnl(ops),
        ]))

        if "motivo" in ops.columns and ops.motivo.nunique() > 1:
            m = ops.groupby("motivo").agg(n=("pnl", "size"), total=("pnl", "sum"),
                                          media=("pnl", "mean"))
            f5 = [["Motivo de salida", "N", "Resultado", "Media"]]
            for k, r in m.sort_values("total", ascending=False).iterrows():
                f5.append([str(k), f"{int(r.n)}", f"${r.total:+,.2f}", f"${r.media:+,.2f}"])
            e.append(KeepTogether([
                Paragraph("Resultado por motivo de salida", H3),
                tabla(f5, [58 * mm, 24 * mm, 44 * mm, 34 * mm]),
            ]))

    # cartera actual
    if posiciones:
        f6 = [["Simbolo", "Cantidad", "Costo medio", "Valor", "No realizado"]]
        for p_ in posiciones:
            f6.append([p_["symbol"], f"{p_['qty']:g}", f"{p_['avg']:,.2f}",
                       f"{p_['mv']:,.2f}", f"{p_['pl']:+,.2f}"])
        e.append(KeepTogether([
            Paragraph("Cartera al cierre del periodo", H2),
            tabla(f6, [30 * mm, 30 * mm, 33 * mm, 33 * mm, 34 * mm]),
        ]))
    else:
        e.append(Paragraph("Cartera al cierre del periodo", H2))
        e.append(Paragraph("Sin posiciones abiertas.", P))
    e.append(Paragraph(
        f"Patrimonio ${cuenta['equity']:,.2f} &nbsp;|&nbsp; efectivo ${cuenta['cash']:,.2f} "
        f"&nbsp;|&nbsp; valor de mercado ${cuenta['mv']:,.2f}", CAP))

    # contexto
    e.append(Paragraph("Como leer estos numeros", H2))
    e.append(Paragraph(
        "<b>La ventaja de esta estrategia no es el retorno, es el drawdown.</b> "
        "Sobre 2018-2026 el sistema rinde 17.6% anualizado contra 14.5% del indice, "
        "una diferencia modesta, pero lo hace con una caida maxima de 13.1% frente al "
        "33.8% del SPY. Su valor aparece en mercados bajistas, no en rallies.", NOTA))
    e.append(Paragraph(
        "En consecuencia, quedar por detras del indice durante un tramo alcista es el "
        "comportamiento esperado, no una senal para modificar parametros. Un seguidor de "
        "tendencia entra tarde por diseno, porque exige confirmacion de ruptura, y "
        "sostiene una decena de nombres en lugar del indice completo.", P))
    e.append(Paragraph(
        "El indicador mas informativo a esta altura no es el retorno sino el cociente de "
        "duracion entre ganadoras y perdedoras. Mientras se mantenga por encima de 1.5, "
        "el mecanismo funciona como debe aunque el resultado acumulado sea negativo.", P))

    e.append(Paragraph("Limitaciones de este analisis", H2))
    for tit, txt in (
        ("Muestra insuficiente",
         "Estimar la esperanza de un sistema de tendencia requiere entre 30 y 50 "
         "operaciones cerradas. Por debajo de eso, una sola operacion distinta da vuelta "
         "cualquier estadistico."),
        ("Periodo contaminado por interrupciones",
         "El bot estuvo 42 dias sin ejecutarse y acumulo salidas vencidas que se cerraron "
         "todas juntas. Esas operaciones miden el costo de la interrupcion, no el "
         "comportamiento del sistema."),
        ("Sesgo de supervivencia en la referencia historica",
         "Las cifras del backtest usan el universo de mega-caps actual, de modo que los "
         "retornos absolutos estan inflados. Solo son comparables las diferencias entre "
         "configuraciones."),
        ("Capa de ejecucion joven",
         "La logica de estrategia esta validada contra 8.5 anos. El manejo de ordenes "
         "acumula pocas semanas de operacion real y produjo seis fallos en su primer dia."),
    ):
        e.append(Paragraph(tit, H3))
        e.append(Paragraph(txt, P))

    def pie(canv, _doc):
        canv.saveState()
        canv.setStrokeColor(REGLA)
        canv.setLineWidth(0.4)
        canv.line(22 * mm, 13 * mm, A4[0] - 22 * mm, 13 * mm)
        canv.setFont(SANS, 7)
        canv.setFillColor(GRIS)
        canv.drawString(22 * mm, 9 * mm,
                        f"Weekly Trading Bot  |  cuenta paper  |  {ini} a {fin}")
        canv.drawRightString(A4[0] - 22 * mm, 9 * mm, f"{canv.getPageNumber()}")
        canv.restoreState()

    doc.build(e, onFirstPage=pie, onLaterPages=pie)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde", default="2026-04-01")
    ap.add_argument("--salida", default="informe_desempeno.pdf")
    ap.add_argument("--universo", default="symbols.txt")
    args = ap.parse_args()

    desde = pd.Timestamp(args.desde)
    t, d = clientes()

    print("[1/5] descargando ordenes ejecutadas...")
    fills = traer_ordenes(t)
    if fills.empty:
        sys.exit("La cuenta no tiene ordenes ejecutadas.")
    print(f"      {len(fills)} fills ({fills.fecha.min().date()} a {fills.fecha.max().date()})")

    print("[2/5] emparejando operaciones (FIFO)...")
    ops = round_trips(fills)
    if Path(args.universo).exists():
        uni = {l.strip().upper() for l in open(args.universo) if l.strip()}
        ops = ops[ops.symbol.isin(uni)]
    ops = ops[ops.entrada >= desde].reset_index(drop=True)
    print(f"      {len(ops)} operaciones cerradas desde {desde.date()}")

    print("[3/5] historial de cartera y serie del indice...")
    cart = serie_cartera(t, desde)
    spy = serie_spy(d, desde)
    if len(cart) < 3:
        sys.exit("Historial de cartera insuficiente.")

    # Recortar el indice a la ventana exacta de la cartera. Sin esto se compara
    # el rendimiento de la cartera hasta su ultimo dato contra un SPY que llega
    # hasta hoy, y la diferencia sale inflada por los dias que sobran.
    spy = spy[(spy.index >= cart.index[0]) & (spy.index <= cart.index[-1])]
    if len(spy) < 3:
        sys.exit("Serie del indice insuficiente en la ventana de la cartera.")
    print(f"      cartera {cart.index[0].date()} a {cart.index[-1].date()} "
          f"({len(cart)} puntos) | indice {len(spy)} sesiones")

    print("[4/5] calculando metricas...")
    mc, ms = metricas(cart), metricas(spy)
    rel = relativas(cart, spy)
    a = t.get_account()
    cuenta = {"equity": float(a.equity), "cash": float(a.cash),
              "mv": float(a.long_market_value)}
    posiciones = sorted(
        [{"symbol": p.symbol, "qty": float(p.qty), "avg": float(p.avg_entry_price),
          "mv": float(p.market_value), "pl": float(p.unrealized_pl)}
         for p in t.get_all_positions()],
        key=lambda x: -x["mv"])

    print("[5/5] generando PDF...")
    construir(Path(args.salida), cart, spy, ops, mc, ms, rel, posiciones, cuenta, desde)
    print(f"\n[OK] {args.salida}")
    print(f"     estrategia {mc['retorno']:+.2f}%  |  SPY {ms['retorno']:+.2f}%  |  "
          f"diferencia {mc['retorno'] - ms['retorno']:+.2f} pts")


if __name__ == "__main__":
    main()
