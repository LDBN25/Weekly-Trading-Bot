"""Vuelca el estado real de la cuenta para diagnostico.

Solo lectura: no manda ordenes ni escribe nada.

    python dump_estado.py            # desde el 1 de agosto
    python dump_estado.py --desde 2026-07-01
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde", default="2026-08-01")
    args = ap.parse_args()

    key = os.getenv("ALPACA_API_KEY")
    sec = os.getenv("ALPACA_API_SECRET") or os.getenv("ALPACA_SECRET_KEY")
    if not key or not sec:
        sys.exit("Faltan ALPACA_API_KEY / ALPACA_API_SECRET")
    t = TradingClient(key, sec, paper=os.getenv("ALPACA_PAPER", "1") != "0")

    a = t.get_account()
    print("=== CUENTA ===")
    print(f"equity={float(a.equity):,.2f} cash={float(a.cash):,.2f} "
          f"long_mv={float(a.long_market_value):,.2f} bp={float(a.buying_power):,.2f}")

    print("\n=== POSICIONES ===")
    print(f"{'SYM':8}{'qty':>10}{'avg_entry':>12}{'market_value':>14}{'unreal_pl':>12}")
    for p in sorted(t.get_all_positions(), key=lambda x: x.symbol):
        print(f"{p.symbol:8}{p.qty:>10}{float(p.avg_entry_price):>12.4f}"
              f"{float(p.market_value):>14,.2f}{float(p.unrealized_pl):>12,.2f}")

    print(f"\n=== ORDENES DESDE {args.desde} ===")
    desde = datetime.fromisoformat(args.desde).replace(tzinfo=timezone.utc)
    req = GetOrdersRequest(status=QueryOrderStatus.ALL, after=desde,
                           limit=500, direction="asc")
    ordenes = t.get_orders(filter=req)
    if not ordenes:
        print("  (ninguna)")
    print(f"{'enviada':20}{'lado':6}{'SYM':8}{'qty':>8}{'precio':>11}  {'estado':11} client_order_id")
    for o in ordenes:
        px = f"{float(o.filled_avg_price):.4f}" if o.filled_avg_price else "-"
        print(f"{str(o.submitted_at)[:19]:20}{o.side.value:6}{o.symbol:8}"
              f"{str(o.filled_qty):>8}{px:>11}  {o.status.value:11} {o.client_order_id}")


if __name__ == "__main__":
    main()
