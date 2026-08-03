"""Cierra cualquier posicion en corto de la cuenta.

La estrategia es solo-largo: un corto solo puede venir de un error de
ejecucion, asi que se cubre entero.

    python cerrar_cortos.py            # previsualiza, no manda ordenes
    python cerrar_cortos.py --ejecutar
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ejecutar", action="store_true", help="Sin esto solo previsualiza")
    args = ap.parse_args()

    key = os.getenv("ALPACA_API_KEY")
    sec = os.getenv("ALPACA_API_SECRET") or os.getenv("ALPACA_SECRET_KEY")
    if not key or not sec:
        sys.exit("Faltan ALPACA_API_KEY / ALPACA_API_SECRET")
    t = TradingClient(key, sec, paper=os.getenv("ALPACA_PAPER", "1") != "0")

    cortos = [p for p in t.get_all_positions() if float(p.qty) < 0]
    if not cortos:
        print("[OK] No hay posiciones en corto.")
        return

    print(f"{'SYM':8}{'qty':>8}{'a cubrir':>10}{'precio':>11}{'costo aprox':>14}")
    total = 0.0
    plan = []
    for p in sorted(cortos, key=lambda x: x.symbol):
        qty = int(math.ceil(abs(float(p.qty))))
        px = abs(float(p.market_value)) / abs(float(p.qty))
        costo = qty * px
        total += costo
        plan.append((p.symbol, qty))
        print(f"{p.symbol:8}{p.qty:>8}{qty:>10}{px:>11.2f}{costo:>14,.2f}")
    print(f"\n{'TOTAL':8}{'':18}{'':11}{total:>14,.2f}")

    if not args.ejecutar:
        print("\n[PREVIEW] Nada enviado. Volvé a correr con --ejecutar.")
        return

    a = t.get_account()
    if total > float(a.buying_power):
        sys.exit(f"Poder de compra insuficiente: {float(a.buying_power):,.2f} < {total:,.2f}")

    print()
    for sym, qty in plan:
        orden = MarketOrderRequest(
            symbol=sym, qty=qty, side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            client_order_id=f"cubrir_{sym}_{int(time.time())}",
        )
        enviada = t.submit_order(orden)
        print(f"[COVER] {sym} qty={qty} enviada")
        for _ in range(20):
            live = t.get_order_by_id(enviada.id)
            if float(live.filled_qty or 0) > 0 and live.filled_avg_price:
                print(f"[FILL]  {sym} qty={live.filled_qty} @ {float(live.filled_avg_price):.4f}")
                break
            if str(live.status.value) in {"canceled", "expired", "rejected"}:
                print(f"[!]     {sym} terminó en {live.status.value}")
                break
            time.sleep(2)

    restantes = [p.symbol for p in t.get_all_positions() if float(p.qty) < 0]
    print(f"\n[OK] Cortos restantes: {restantes or 'ninguno'}")


if __name__ == "__main__":
    main()
