#   GET  /position_size?stop_on_spy=...&max_loss_usd=...
#   GET  /positions
#   POST /buy   {stop_on_spy,max_loss_usd}  (auto-size)
#               or {qty,stop_on_spy,max_loss_usd} (manual)
#   POST /sell  {qty} or {mode:"half"|"all"}
#   POST /set_right {"right":"C"|"P"}  → reselect option (requires flat)

from ib_insync import IB, Stock, Option, MarketOrder
from datetime import datetime, timezone
import math
import threading
import queue
import calendar
from typing import Any, Callable
from flask import Flask, jsonify, request

# ---------------- Config ----------------
TWS_HOSTNAME: str      = '127.0.0.1'
TWS_PORT: int          = 7490           # your working port
TWS_CLIENT_ID: int     = 101

UNDERLYING_SYMBOL: str = 'SPY'
OPTION_RIGHT: str      = 'C'            # default; UI can change via /set_right
OPTION_EXCHANGE: str   = 'SMART'
REFRESH_SECONDS: float = 0.25
ACCOUNT_REFRESH_SECONDS: float = 1.0   # cadence for account summary refresh

BACKEND_HOST = '127.0.0.1'
BACKEND_PORT = 8001

# --- NEW: strict OTM enforcement (auto-roll when flat) -----------------------
ENFORCE_STRICT_OTM = True           # always stream the nearest strictly OTM when flat
MIN_SWITCH_INTERVAL = 0.5           # seconds, tiny debounce to avoid flapping

# ------------- Helpers ------------------
def pick_nearest_by_key(items, key_func):
    return min(items, key=key_func)

def pick_nearest_expiration(expiration_strings):
    today_date = datetime.now(timezone.utc).date()
    future_expirations = [
        exp for exp in expiration_strings
        if datetime.strptime(exp, '%Y%m%d').date() >= today_date
    ]
    pool = future_expirations if future_expirations else list(expiration_strings)
    return pick_nearest_by_key(pool, key_func=lambda e: datetime.strptime(e, '%Y%m%d').date())

def pick_nearest_otm_strike(strike_values, underlying_price, option_right):
    if not strike_values:
        raise RuntimeError("No strikes available to select from.")
    unique_sorted_strikes = sorted({float(s) for s in strike_values})
@@ -87,50 +88,56 @@ def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def readable_name(symbol: str, exp: str, strike: float, right: str) -> str:
    """Return e.g. 'SPY Sep19 662.5C' for display."""
    mon = calendar.month_abbr[int(exp[4:6])]
    day = exp[6:8]
    s = float(strike)
    strike_str = f"{s:.2f}".rstrip('0').rstrip('.') if not s.is_integer() else str(int(s))
    return f"{symbol} {mon}{day} {strike_str}{right}"

# ----------- Shared snapshot ------------
_snapshot = {
    "time": "--:--:--",
    "status": "starting",
    "underlying": UNDERLYING_SYMBOL,
    "right": OPTION_RIGHT,      # <-- expose current right
    "spy_price": "n/a",
    "contract": "n/a",          # human-readable name sent to UI
    "bid": "n/a",
    "ask": "n/a",
    "delta": "n/a",
    "iv": "n/a",
    "portfolio": {
        "net_liquidity": None,
        "daily_pnl": None,
        "daily_pnl_pct": None,
        "buying_power": None,
    },
}
_snapshot_lock = threading.Lock()

# -------- Position management -----------
position_lock = threading.Lock()

class OrderExecutionError(RuntimeError):
    """Raised when the brokerage rejects or cancels an order before filling."""

class Position:
    def __init__(self, local_symbol, con_id):
        self.local_symbol = local_symbol   # IB machine label (keep)
        self.con_id = con_id
        self.qty = 0
        self.avg_price = 0.0
        self.stop_on_spy = 0.0
        self.max_loss_usd = 0.0
        self.active = False

        # --- fixed-on-entry risk anchors (NEW) ---
        self.entry_delta_abs = None         # |delta| captured at first activation
        self.fixed_stop_price = None        # computed once, remains fixed while open
        self.fixed_protect_price = None     # computed once, remains fixed while open

        self.last_stop_price = None         # cached for UI
@@ -765,50 +772,121 @@ def _ib_set_right(right: str) -> dict:
    # refresh cached strikes and current selection state
    _cached_strikes = sorted({float(s) for s in row.strikes})
    _current_strike = float(strike)
    _current_exp = exp
    _last_switch_ts = datetime.now().timestamp()

    # Update snapshot header bits immediately
    with _snapshot_lock:
        _snapshot["contract"] = readable
        _snapshot["right"] = right

    return {"right": right, "contract": readable}

# ------------- Streaming main ------------
def start_backend():
    # Start HTTP server in a background thread
    threading.Thread(
        target=lambda: app.run(host=BACKEND_HOST, port=BACKEND_PORT, debug=False, use_reloader=False, threaded=True),
        daemon=True
    ).start()

    ib = IB()
    ib.connect(TWS_HOSTNAME, TWS_PORT, clientId=TWS_CLIENT_ID)
    ib.reqMarketDataType(1)  # 1=LIVE

    # --- Track account-level daily PnL via streaming subscription -----------------
    managed_accounts: list[str] = []
    try:
        managed_accounts = ib.managedAccounts()
    except Exception:
        managed_accounts = []
    account_id: str | None = managed_accounts[0] if managed_accounts else None

    pnl_state_lock = threading.Lock()
    pnl_state: dict[str, float | None] = {"daily": None, "account_value": None}

    pnl_stream = None

    account_summary_subscription = None

    def _ensure_pnl_stream(account_name: str | None):
        nonlocal account_id, pnl_stream
        if not account_name or pnl_stream is not None:
            return
        try:
            pnl_stream = ib.reqPnL(account=account_name, modelCode='')
            pnl_stream.updateEvent += _capture_account_pnl
            account_id = account_name
        except Exception:
            pnl_stream = None

    def _ensure_account_summary_subscription():
        nonlocal account_summary_subscription
        if account_summary_subscription is not None:
            return
        try:
            # Request only the account summary tags we surface in the UI.
            account_summary_subscription = ib.reqAccountSummary(
                'All',
                'NetLiquidation,BuyingPower,AvailableFunds,DailyPnL,PnL'
            )
        except Exception:
            account_summary_subscription = None

    def _capture_account_pnl(pnl_obj):
        """Persist latest daily PnL from reqPnL stream and push to snapshot quickly."""

        daily = getattr(pnl_obj, 'dailyPnL', None)
        if daily is None:
            return
        try:
            daily_float = float(daily)
        except Exception:
            return
        if not math.isfinite(daily_float):
            return

        with pnl_state_lock:
            pnl_state["daily"] = daily_float

        with _snapshot_lock:
            portfolio = _snapshot.get('portfolio')
            if not isinstance(portfolio, dict):
                portfolio = {}
                _snapshot['portfolio'] = portfolio
            portfolio["daily_pnl"] = round(daily_float, 2)

            net_liq_existing = portfolio.get("net_liquidity")
            if isinstance(net_liq_existing, (int, float)) and math.isfinite(net_liq_existing):
                prior_close = net_liq_existing - daily_float
                if prior_close:
                    portfolio["daily_pnl_pct"] = round((daily_float / prior_close) * 100.0, 2)

    _ensure_pnl_stream(account_id)
    _ensure_account_summary_subscription()

    # Underlying contract + RTVolume (233) for better "last"
    stk = Stock(UNDERLYING_SYMBOL, 'SMART', 'USD', primaryExchange='ARCA')
    ib.qualifyContracts(stk)
    t_und = ib.reqMktData(stk, '233', False, False)

    # Wait for a valid underlying price
    for _ in range(200):
        ib.sleep(0.1)
        px = t_und.marketPrice()
        if math.isfinite(px) and px > 0:
            break
    und_px = t_und.marketPrice()
    if not (math.isfinite(und_px) and und_px > 0):
        with _snapshot_lock:
            _snapshot["status"] = "error: no underlying price"
        raise RuntimeError("No live underlying price available.")

    # Option selection: nearest expiry + nearest OTM (default right)
    params = ib.reqSecDefOptParams(stk.symbol, '', stk.secType, stk.conId)
    rows = [p for p in params if p.exchange in ('SMART', OPTION_EXCHANGE)
            and p.tradingClass in (UNDERLYING_SYMBOL, stk.symbol)]
    if not rows:
        with _snapshot_lock:
            _snapshot["status"] = "error: no option params"
        raise RuntimeError("No option parameters available.")
@@ -867,87 +945,204 @@ def start_backend():
        # cancel previous stream and subscribe new (with greeks)
        try:
            if t_opt is not None:
                ib.cancelMktData(t_opt.contract)
        except Exception:
            pass
        new_ticker = ib.reqMktData(new_opt, '106', False, False)

        # swap live references
        t_opt = new_ticker
        _engine["t_opt"] = new_ticker
        pos_manager.contract = new_opt
        local_symbol = new_opt.localSymbol or f"{UNDERLYING_SYMBOL} {_current_exp} {new_strike} {right}"
        readable2 = readable_name(UNDERLYING_SYMBOL, _current_exp, float(new_strike), right)
        _engine["local_symbol"] = local_symbol
        _engine["display_name"] = readable2

        # update current markers
        _current_strike = float(new_strike)
        _last_switch_ts = datetime.now().timestamp()

        with _snapshot_lock:
            _snapshot["contract"] = readable2

    # Main loop
    last_account_refresh = 0.0
    try:
        while True:
            ib.sleep(REFRESH_SECONDS)

            # ---- Process any queued IB jobs (buy/sell/right switch) ----
            while True:
                try:
                    fn, args, kwargs, reply_q = ib_jobs.get_nowait()
                except queue.Empty:
                    break
                try:
                    res = fn(*args, **kwargs)
                    reply_q.put((True, res))
                except Exception as e:
                    reply_q.put((False, e))

            # ---- Auto-roll to strictly OTM when flat (fast & deterministic) ----
            if ENFORCE_STRICT_OTM and not pos_manager.has_open():
                und_px_live = t_und.marketPrice()
                if math.isfinite(und_px_live) and und_px_live > 0:
                    now_ts = datetime.now().timestamp()
                    if (now_ts - _last_switch_ts) >= MIN_SWITCH_INTERVAL and _cached_strikes:
                        desired = pick_nearest_otm_cached(_cached_strikes, und_px_live, _engine["right"])
                        if _current_strike is None or float(desired) != _current_strike:
                            reseat_option(desired)

            # ---- Periodically refresh account summary / portfolio snapshot ----
            now_ts = datetime.now().timestamp()
            if (now_ts - last_account_refresh) >= ACCOUNT_REFRESH_SECONDS:
                last_account_refresh = now_ts
                _ensure_account_summary_subscription()
                summary_rows = None
                if account_summary_subscription is not None:
                    try:
                        summary_rows = list(account_summary_subscription)
                    except Exception:
                        summary_rows = None
                if summary_rows is None:
                    try:
                        summary_rows = ib.accountSummary()
                    except Exception:
                        summary_rows = None
                if summary_rows:
                    tags_of_interest = {
                        "NetLiquidation",
                        "DailyPnL",
                        "PnL",
                        "BuyingPower",
                        "AvailableFunds",
                    }
                    per_account: dict[str, dict[str, str]] = {}
                    for row in summary_rows:
                        tag = getattr(row, 'tag', None)
                        if tag not in tags_of_interest:
                            continue
                        # Keep rows even if the currency is reported as 'BASE' – no filtering here.
                        acct = getattr(row, 'account', None) or ''
                        if account_id and acct and acct != account_id:
                            continue
                        acct_map = per_account.setdefault(acct, {})
                        acct_map[tag] = getattr(row, 'value', None)

                    summary_map: dict[str, str] = {}
                    if account_id and account_id in per_account:
                        summary_map = per_account[account_id]
                    elif per_account:
                        first_account = next(iter(per_account))
                        summary_map = per_account[first_account]
                        if first_account and not account_id:
                            account_id = first_account
                        if first_account:
                            _ensure_pnl_stream(first_account)

                    net_liq = _to_float(summary_map.get('NetLiquidation'))
                    with pnl_state_lock:
                        fallback_daily = pnl_state.get('daily')
                        fallback_value = pnl_state.get('account_value')
                    if net_liq is None and fallback_value is not None:
                        net_liq = float(fallback_value)

                    daily_pnl = _to_float(summary_map.get('DailyPnL'))
                    if daily_pnl is None:
                        daily_pnl = _to_float(summary_map.get('PnL'))
                    if daily_pnl is None and fallback_daily is not None:
                        daily_pnl = float(fallback_daily)

                    buying_power_val = _to_float(summary_map.get('BuyingPower'))
                    if buying_power_val is None:
                        buying_power_val = _to_float(summary_map.get('AvailableFunds'))

                    daily_pct = None
                    if (
                        net_liq is not None and math.isfinite(net_liq)
                        and daily_pnl is not None and math.isfinite(daily_pnl)
                    ):
                        prior_close_value = net_liq - daily_pnl
                        if prior_close_value:
                            daily_pct = (daily_pnl / prior_close_value) * 100.0

                    net_liq_val = (
                        round(net_liq, 2)
                        if (net_liq is not None and math.isfinite(net_liq))
                        else None
                    )
                    daily_pnl_val = (
                        round(daily_pnl, 2)
                        if (daily_pnl is not None and math.isfinite(daily_pnl))
                        else None
                    )
                    daily_pct_val = (
                        round(daily_pct, 2)
                        if (daily_pct is not None and math.isfinite(daily_pct))
                        else None
                    )
                    buying_power_num = (
                        round(buying_power_val, 2)
                        if (buying_power_val is not None and math.isfinite(buying_power_val))
                        else None
                    )

                    with _snapshot_lock:
                        portfolio = _snapshot.get('portfolio')
                        if not isinstance(portfolio, dict):
                            portfolio = {}
                            _snapshot['portfolio'] = portfolio
                        portfolio.update({
                            "net_liquidity": net_liq_val,
                            "daily_pnl": daily_pnl_val,
                            "daily_pnl_pct": daily_pct_val,
                            "buying_power": buying_power_num,
                        })

            # ---- Update snapshot ----
            now_str = datetime.now().strftime('%H:%M:%S')
            spy_price_str = format_number_safe(t_und.marketPrice(), 2)
            t_opt_cur = _engine["t_opt"]
            bid_str = format_number_safe(t_opt_cur.bid, 2) if t_opt_cur else "n/a"
            ask_str = format_number_safe(t_opt_cur.ask, 2) if t_opt_cur else "n/a"
            g = t_opt_cur.modelGreeks if t_opt_cur else None
            delta_str = format_number_safe(getattr(g, 'delta', None), 4) if g else "n/a"
            iv_str = format_number_safe(getattr(g, 'impliedVol', None), 4) if g else "n/a"

            with _snapshot_lock:
                _snapshot.update({
                    "time": now_str,
                    "spy_price": spy_price_str,
                    "bid": bid_str,
                    "ask": ask_str,
                    "delta": delta_str,
                    "iv": iv_str,
                })

            # ---- Enforce server-side stop/protect if needed (uses fixed anchors) ----
            pos_manager.monitor_auto_stop()

            # Optional console ticker (prefer readable name)
            name_for_print = _engine.get('display_name') or _engine.get('local_symbol') or 'n/a'
            line = f"{now_str} | SPY {spy_price_str} | {name_for_print} | Bid {bid_str} Ask {ask_str} | Δ {delta_str} | IV {iv_str}"
            print(line.ljust(150), end='\r', flush=True)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        try:
            if pnl_stream is not None:
                ib.cancelPnL(pnl_stream)
        except Exception:
            pass
        try:
            if account_summary_subscription is not None:
                ib.cancelAccountSummary(account_summary_subscription)
        except Exception:
            pass
        ib.disconnect()

if __name__ == "__main__":
    start_backend()
#