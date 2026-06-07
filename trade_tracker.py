from __future__ import annotations

import csv
import json
import logging
import os
import smtplib
import urllib.request
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List

TRADE_HISTORY_PATH = Path(os.getenv("TRADE_HISTORY_PATH", "data/trade_history.csv"))

_CSV_FIELDS = [
    "symbol", "entry_date", "entry_price", "initial_shares",
    "exit_date", "exit_price", "exit_shares", "exit_reason",
    "gross_pnl", "r_multiple", "is_partial",
]


def record_trade(
    symbol: str,
    entry_date: str,
    entry_price: float,
    initial_shares: int,
    exit_date: str,
    exit_price: float,
    exit_shares: int,
    exit_reason: str,
    risk_per_share: float,
    is_partial: bool = False,
) -> None:
    gross_pnl = round((exit_price - entry_price) * exit_shares, 2)
    r_multiple = round((exit_price - entry_price) / risk_per_share, 3) if risk_per_share > 0 else 0.0

    TRADE_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not TRADE_HISTORY_PATH.exists()

    with TRADE_HISTORY_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "symbol": symbol,
            "entry_date": str(entry_date)[:10],
            "entry_price": round(entry_price, 4),
            "initial_shares": initial_shares,
            "exit_date": str(exit_date)[:10],
            "exit_price": round(exit_price, 4),
            "exit_shares": exit_shares,
            "exit_reason": exit_reason,
            "gross_pnl": gross_pnl,
            "r_multiple": r_multiple,
            "is_partial": is_partial,
        })
    logging.info(
        "[TRADE_LOG] %s exit=%s precio=%.4f pnl=%.2f R=%.3f",
        symbol, exit_reason, exit_price, gross_pnl, r_multiple,
    )


def _build_summary_text(
    week_end: str,
    equity: float,
    cash: float,
    exits: List[Dict],
    entries: List[Dict],
    open_positions: List[Dict],
) -> str:
    lines = [
        f"=== Resumen Semanal: {week_end} ===",
        f"Equity: ${equity:,.2f}   Cash: ${cash:,.2f}",
        "",
    ]

    if exits:
        lines.append(f"-- Cerradas ({len(exits)}) --")
        total_pnl = sum(float(e["gross_pnl"]) for e in exits)
        for e in exits:
            sign = "+" if float(e["gross_pnl"]) >= 0 else ""
            partial_tag = " [parcial]" if str(e.get("is_partial", "")).lower() == "true" else ""
            lines.append(
                f"  {e['symbol']:6s}  {e['exit_reason']:12s}  "
                f"${float(e['entry_price']):.2f} → ${float(e['exit_price']):.2f}  "
                f"PnL: {sign}{float(e['gross_pnl']):.2f}  ({e['r_multiple']}R){partial_tag}"
            )
        sign = "+" if total_pnl >= 0 else ""
        lines.append(f"  Total semana: {sign}{total_pnl:.2f}")
    else:
        lines.append("-- Sin cierres esta semana --")

    lines.append("")

    if entries:
        lines.append(f"-- Nuevas Entradas ({len(entries)}) --")
        for e in entries:
            lines.append(
                f"  {e['symbol']:6s}  qty={e['shares']}  "
                f"entrada est. ${float(e['entry_price']):.2f}  stop=${float(e['stop_price']):.2f}"
            )
    else:
        lines.append("-- Sin nuevas entradas --")

    lines.append("")

    if open_positions:
        lines.append(f"-- Portafolio Abierto ({len(open_positions)}) --")
        for p in open_positions:
            lines.append(
                f"  {p['symbol']:6s}  qty={p['shares']}  "
                f"entrada ${float(p['entry_price']):.2f}  stop=${float(p['stop_price']):.2f}"
            )
    else:
        lines.append("-- Sin posiciones abiertas --")

    return "\n".join(lines)


def _send_email(subject: str, body: str) -> None:
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASSWORD")
    to_addr = os.getenv("NOTIFY_EMAIL")

    if not all([host, user, password, to_addr]):
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to_addr
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(user, password)
            server.sendmail(user, to_addr, msg.as_string())
        logging.info("[NOTIFY] Email enviado a %s", to_addr)
    except Exception as exc:
        logging.warning("[NOTIFY] Error enviando email: %s", exc)


def _send_telegram(text: str) -> None:
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps({"chat_id": chat_id, "text": text, "parse_mode": "HTML"}).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=10)
        logging.info("[NOTIFY] Telegram enviado a chat_id=%s", chat_id)
    except Exception as exc:
        logging.warning("[NOTIFY] Error enviando Telegram: %s", exc)


def send_weekly_summary(
    week_end: str,
    equity: float,
    cash: float,
    exits: List[Dict],
    entries: List[Dict],
    open_positions: List[Dict],
) -> None:
    if not (os.getenv("NOTIFY_EMAIL") or os.getenv("TELEGRAM_TOKEN")):
        return
    body = _build_summary_text(week_end, equity, cash, exits, entries, open_positions)
    subject = f"[Trading Bot] Semana {week_end}"
    _send_email(subject, body)
    _send_telegram(body)
