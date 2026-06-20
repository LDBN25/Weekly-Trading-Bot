"""
Extrae y analiza el desempeño del bot desde Alpaca paper account.
Genera reporte en consola y archivos CSV en data/analisis/.
"""
from __future__ import annotations

import os
import csv
from datetime import datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus, OrderSide
from alpaca.trading.requests import GetOrdersRequest, GetPortfolioHistoryRequest

NY = ZoneInfo("America/New_York")
UTC = timezone.utc
OUTPUT_DIR = Path("data/analisis")

INICIO = datetime(2026, 4, 1, tzinfo=UTC)


def get_client() -> TradingClient:
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_API_SECRET")
    if not key or not secret:
        raise RuntimeError("Faltan ALPACA_API_KEY y/o ALPACA_API_SECRET")
    return TradingClient(api_key=key, secret_key=secret, paper=True)


def fetch_all_orders(trading: TradingClient) -> list:
    """Obtiene todas las órdenes desde abril (completas + canceladas + pendientes)."""
    ordenes = []
    for status in [QueryOrderStatus.CLOSED, QueryOrderStatus.OPEN]:
        req = GetOrdersRequest(
            status=status,
            after=INICIO,
            limit=500,
        )
        result = trading.get_orders(filter=req)
        ordenes.extend(result)
    # Ordenar por fecha de creación
    ordenes.sort(key=lambda o: o.created_at)
    return ordenes


def clasificar_ordenes(ordenes: list) -> list[dict]:
    """Convierte las órdenes de Alpaca a dicts legibles."""
    rows = []
    for o in ordenes:
        filled_price = float(o.filled_avg_price) if o.filled_avg_price else None
        filled_qty = float(o.filled_qty) if o.filled_qty else 0.0
        created_ny = o.created_at.astimezone(NY)
        filled_at_ny = o.filled_at.astimezone(NY) if o.filled_at else None
        rows.append({
            "order_id": str(o.id),
            "symbol": o.symbol,
            "side": o.side.value,
            "status": o.status.value,
            "qty_ordered": float(o.qty) if o.qty else 0.0,
            "qty_filled": filled_qty,
            "filled_avg_price": filled_price,
            "created_at": created_ny.strftime("%Y-%m-%d %H:%M"),
            "filled_at": filled_at_ny.strftime("%Y-%m-%d %H:%M") if filled_at_ny else "",
            "order_type": o.order_type.value if o.order_type else "",
            "time_in_force": o.time_in_force.value if o.time_in_force else "",
        })
    return rows


def calcular_trades(ordenes_rows: list[dict]) -> list[dict]:
    """
    Empareja BUYs con SELLs por símbolo para calcular trades completos.
    Un 'trade' = una entrada + su(s) salida(s).
    """
    # Agrupar órdenes ejecutadas por símbolo
    buys: dict[str, list] = {}
    sells: dict[str, list] = {}

    for o in ordenes_rows:
        if o["status"] not in ("filled", "partially_filled"):
            continue
        sym = o["symbol"]
        if o["side"] == "buy":
            buys.setdefault(sym, []).append(o)
        else:
            sells.setdefault(sym, []).append(o)

    trades = []
    for sym in set(list(buys.keys()) + list(sells.keys())):
        sym_buys = buys.get(sym, [])
        sym_sells = sells.get(sym, [])

        for buy in sym_buys:
            entry_price = buy["filled_avg_price"]
            entry_qty = buy["qty_filled"]
            entry_date = buy["filled_at"] or buy["created_at"]

            # Buscar sells posteriores a este buy
            related_sells = [
                s for s in sym_sells
                if (s["filled_at"] or s["created_at"]) >= entry_date
            ]

            if not related_sells:
                # Posición aún abierta
                trades.append({
                    "symbol": sym,
                    "entry_date": entry_date,
                    "entry_price": entry_price,
                    "entry_qty": entry_qty,
                    "exit_date": "ABIERTA",
                    "exit_price": None,
                    "exit_qty": 0,
                    "gross_pnl": None,
                    "pnl_pct": None,
                    "status": "abierta",
                })
            else:
                total_exit_qty = sum(s["qty_filled"] for s in related_sells)
                avg_exit_price = (
                    sum(s["filled_avg_price"] * s["qty_filled"] for s in related_sells)
                    / total_exit_qty
                    if total_exit_qty > 0 else None
                )
                exit_date = max(s["filled_at"] or s["created_at"] for s in related_sells)
                gross_pnl = None
                pnl_pct = None
                if avg_exit_price and entry_price:
                    gross_pnl = round((avg_exit_price - entry_price) * total_exit_qty, 2)
                    pnl_pct = round((avg_exit_price - entry_price) / entry_price * 100, 2)

                trades.append({
                    "symbol": sym,
                    "entry_date": entry_date,
                    "entry_price": entry_price,
                    "entry_qty": entry_qty,
                    "exit_date": exit_date,
                    "exit_price": avg_exit_price,
                    "exit_qty": total_exit_qty,
                    "gross_pnl": gross_pnl,
                    "pnl_pct": pnl_pct,
                    "status": "cerrada",
                })

    trades.sort(key=lambda t: t["entry_date"])
    return trades


def fetch_portfolio_history(trading: TradingClient) -> list[dict]:
    """Obtiene el historial del valor del portafolio desde abril."""
    try:
        req = GetPortfolioHistoryRequest(
            period="3M",
            timeframe="1D",
            intraday_reporting="market_hours",
        )
        hist = trading.get_portfolio_history(filter=req)
        rows = []
        for i, ts in enumerate(hist.timestamp):
            equity = hist.equity[i]
            profit_loss = hist.profit_loss[i]
            profit_loss_pct = hist.profit_loss_pct[i]
            if equity is None:
                continue
            dt = datetime.fromtimestamp(ts, tz=UTC).astimezone(NY)
            rows.append({
                "date": dt.strftime("%Y-%m-%d"),
                "equity": round(equity, 2),
                "profit_loss": round(profit_loss, 2) if profit_loss else 0.0,
                "profit_loss_pct": round(profit_loss_pct * 100, 4) if profit_loss_pct else 0.0,
            })
        return rows
    except Exception as exc:
        print(f"[WARN] No se pudo obtener portfolio history: {exc}")
        return []


def metricas_globales(trades: list[dict], portfolio_hist: list[dict], account) -> dict:
    """Calcula métricas de desempeño globales."""
    cerradas = [t for t in trades if t["status"] == "cerrada" and t["gross_pnl"] is not None]
    abiertas = [t for t in trades if t["status"] == "abierta"]

    total_trades = len(cerradas)
    ganadoras = [t for t in cerradas if t["gross_pnl"] > 0]
    perdedoras = [t for t in cerradas if t["gross_pnl"] <= 0]

    win_rate = len(ganadoras) / total_trades * 100 if total_trades > 0 else 0
    pnl_total = sum(t["gross_pnl"] for t in cerradas)
    pnl_promedio = pnl_total / total_trades if total_trades > 0 else 0

    avg_ganadora = sum(t["gross_pnl"] for t in ganadoras) / len(ganadoras) if ganadoras else 0
    avg_perdedora = sum(t["gross_pnl"] for t in perdedoras) / len(perdedoras) if perdedoras else 0
    profit_factor = abs(avg_ganadora / avg_perdedora) if avg_perdedora != 0 else float("inf")

    best = max(cerradas, key=lambda t: t["gross_pnl"]) if cerradas else None
    worst = min(cerradas, key=lambda t: t["gross_pnl"]) if cerradas else None

    equity_actual = float(account.equity)
    cash_actual = float(account.cash)

    # Calcular retorno total desde el primer dato de portfolio
    retorno_total_pct = None
    if portfolio_hist:
        primer_equity = next((r["equity"] for r in portfolio_hist if r["equity"] > 0), None)
        if primer_equity:
            retorno_total_pct = round((equity_actual - primer_equity) / primer_equity * 100, 2)

    return {
        "equity_actual": equity_actual,
        "cash_actual": cash_actual,
        "trades_cerrados": total_trades,
        "trades_abiertos": len(abiertas),
        "ganadoras": len(ganadoras),
        "perdedoras": len(perdedoras),
        "win_rate_pct": round(win_rate, 1),
        "pnl_total": round(pnl_total, 2),
        "pnl_promedio_por_trade": round(pnl_promedio, 2),
        "avg_ganadora": round(avg_ganadora, 2),
        "avg_perdedora": round(avg_perdedora, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "∞",
        "mejor_trade": f"{best['symbol']} +${best['gross_pnl']:.2f}" if best else "N/A",
        "peor_trade": f"{worst['symbol']} ${worst['gross_pnl']:.2f}" if worst else "N/A",
        "retorno_total_pct": retorno_total_pct,
    }


def guardar_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  → Guardado: {path}")


def imprimir_reporte(metricas: dict, trades: list[dict], ordenes_rows: list[dict]) -> None:
    print("\n" + "=" * 65)
    print("  REPORTE DE DESEMPEÑO — Weekly Trading Bot (Demo / Paper)")
    print(f"  Periodo: Abril 2026 → {datetime.now(NY).strftime('%d %b %Y')}")
    print("=" * 65)

    print(f"\n{'CUENTA':}")
    print(f"  Equity actual:          ${metricas['equity_actual']:>12,.2f}")
    print(f"  Cash disponible:        ${metricas['cash_actual']:>12,.2f}")
    if metricas["retorno_total_pct"] is not None:
        signo = "+" if metricas["retorno_total_pct"] >= 0 else ""
        print(f"  Retorno total (est.):   {signo}{metricas['retorno_total_pct']}%")

    print(f"\n{'ACTIVIDAD':}")
    total_ordenes = len([o for o in ordenes_rows if o["status"] in ("filled", "partially_filled")])
    print(f"  Órdenes ejecutadas:     {total_ordenes}")
    print(f"  Trades cerrados:        {metricas['trades_cerrados']}")
    print(f"  Posiciones abiertas:    {metricas['trades_abiertos']}")

    print(f"\n{'MÉTRICAS DE RENDIMIENTO':}")
    print(f"  Win rate:               {metricas['win_rate_pct']}%")
    print(f"  Trades ganadores:       {metricas['ganadoras']}")
    print(f"  Trades perdedores:      {metricas['perdedoras']}")
    print(f"  PnL total:              ${metricas['pnl_total']:>+,.2f}")
    print(f"  PnL promedio/trade:     ${metricas['pnl_promedio_por_trade']:>+,.2f}")
    print(f"  Promedio ganadora:      ${metricas['avg_ganadora']:>+,.2f}")
    print(f"  Promedio perdedora:     ${metricas['avg_perdedora']:>+,.2f}")
    print(f"  Profit factor:          {metricas['profit_factor']}")
    print(f"  Mejor trade:            {metricas['mejor_trade']}")
    print(f"  Peor trade:             {metricas['peor_trade']}")

    cerradas = [t for t in trades if t["status"] == "cerrada"]
    abiertas = [t for t in trades if t["status"] == "abierta"]

    if cerradas:
        print(f"\n{'TRADES CERRADOS':}")
        print(f"  {'SÍMBOLO':<7} {'ENTRADA':>10} {'SALIDA':>10} {'QTY':>5} {'P. ENTRADA':>10} {'P. SALIDA':>10} {'PnL':>10} {'%':>7}")
        print("  " + "-" * 73)
        for t in cerradas:
            pnl_str = f"${t['gross_pnl']:>+,.2f}" if t["gross_pnl"] is not None else "  N/A"
            pct_str = f"{t['pnl_pct']:>+.2f}%" if t["pnl_pct"] is not None else "N/A"
            ep = f"${t['entry_price']:.2f}" if t["entry_price"] else "N/A"
            xp = f"${t['exit_price']:.2f}" if t["exit_price"] else "N/A"
            print(f"  {t['symbol']:<7} {str(t['entry_date'])[:10]:>10} {str(t['exit_date'])[:10]:>10} "
                  f"{int(t['entry_qty']):>5} {ep:>10} {xp:>10} {pnl_str:>10} {pct_str:>7}")

    if abiertas:
        print(f"\n{'POSICIONES ABIERTAS':}")
        print(f"  {'SÍMBOLO':<7} {'ENTRADA':>10} {'QTY':>5} {'P. ENTRADA':>10}")
        print("  " + "-" * 37)
        for t in abiertas:
            ep = f"${t['entry_price']:.2f}" if t["entry_price"] else "N/A"
            print(f"  {t['symbol']:<7} {str(t['entry_date'])[:10]:>10} {int(t['entry_qty']):>5} {ep:>10}")

    print("\n" + "=" * 65)


def main():
    print("[+] Conectando a Alpaca paper account...")
    trading = get_client()
    account = trading.get_account()
    print(f"[+] Cuenta: {account.account_number} | Equity: ${float(account.equity):,.2f}")

    print("[+] Descargando órdenes desde abril 2026...")
    ordenes = fetch_all_orders(trading)
    print(f"    Total órdenes encontradas: {len(ordenes)}")

    ordenes_rows = clasificar_ordenes(ordenes)
    trades = calcular_trades(ordenes_rows)

    print("[+] Descargando historial del portafolio...")
    portfolio_hist = fetch_portfolio_history(trading)
    print(f"    Días de historial: {len(portfolio_hist)}")

    metricas = metricas_globales(trades, portfolio_hist, account)

    imprimir_reporte(metricas, trades, ordenes_rows)

    print("\n[+] Guardando archivos CSV...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    guardar_csv(
        OUTPUT_DIR / "ordenes.csv",
        ordenes_rows,
        ["order_id", "symbol", "side", "status", "qty_ordered", "qty_filled",
         "filled_avg_price", "created_at", "filled_at", "order_type", "time_in_force"],
    )

    guardar_csv(
        OUTPUT_DIR / "trades.csv",
        trades,
        ["symbol", "entry_date", "entry_price", "entry_qty",
         "exit_date", "exit_price", "exit_qty", "gross_pnl", "pnl_pct", "status"],
    )

    if portfolio_hist:
        guardar_csv(
            OUTPUT_DIR / "portfolio_history.csv",
            portfolio_hist,
            ["date", "equity", "profit_loss", "profit_loss_pct"],
        )

    print("\n[LISTO] Análisis completado.")


if __name__ == "__main__":
    main()
