# backend_ib.py
# Run: python backend_ib.py  → http://127.0.0.1:8001
#
# Endpoints:
#   GET  /snapshot
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
    r = option_right.upper()
    if r == 'C':
        otm = [s for s in unique_sorted_strikes if s > underlying_price]
        if otm: return otm[0]
    elif r == 'P':
        otm = [s for s in unique_sorted_strikes if s < underlying_price]
        if otm: return otm[-1]
    return pick_nearest_by_key(unique_sorted_strikes, key_func=lambda k: abs(k - underlying_price))

# NEW: fast strict-OTM pick from a cached sorted list (no abs fallback)
def pick_nearest_otm_cached(sorted_strikes, und_px, right):
    if right == 'C':
        for s in sorted_strikes:
            if s > und_px:
                return s
        return sorted_strikes[-1]
    else:
        for s in reversed(sorted_strikes):
            if s < und_px:
                return s
        return sorted_strikes[0]

def format_number_safe(value, decimals=4):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    try:
        return f"{float(value):.{decimals}f}"
    except Exception:
        return str(value)

def _to_float(x):
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
        self.auto_protect_enabled = False
        self.protect_ratio = 1.0
        self.protect_triggered = False
        self.last_protect_price = None

class PositionManager:
    def __init__(self, ib: IB):
        self.ib = ib
        self.pos: Position | None = None
        self.contract = None          # Option contract (set by main loop)
        self._liquidating = False     # prevent double fire

    def has_open(self) -> bool:
        with position_lock:
            return bool(self.pos and self.pos.active and self.pos.qty > 0)

    def positions_json(self) -> dict:
        with position_lock, _snapshot_lock:
            if not self.pos or not self.pos.active:
                return {"status": "flat"}

            bid   = _to_float(_snapshot.get('bid'))
            ask   = _to_float(_snapshot.get('ask'))

            mid = None
            if bid and ask and bid > 0 and ask > 0:
                mid = round((bid + ask) / 2, 4)

            pnl = None
            if bid is not None:
                pnl = round((bid - self.pos.avg_price) * self.pos.qty * 100, 2)  # options multiplier 100

            # keep UI mirrors updated from fixed anchors
            self.pos.last_stop_price = self.pos.fixed_stop_price
            self.pos.last_protect_price = self.pos.fixed_protect_price

            loss_per_contract = None
            if self.pos.entry_delta_abs is not None and self.pos.stop_on_spy > 0:
                loss_per_contract = round(self.pos.entry_delta_abs * self.pos.stop_on_spy * 100, 2)

            return {
                "status": "open",
                "contract": self.pos.local_symbol,  # machine label
                "contract_readable": readable_name(
                    self.contract.symbol,
                    self.contract.lastTradeDateOrContractMonth,
                    float(self.contract.strike),
                    self.contract.right
                ) if self.contract else None,
                "qty": int(self.pos.qty),
                "avg_price": round(self.pos.avg_price, 4),
                "calc_stop_price": self.pos.fixed_stop_price,             # fixed
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "pnl": pnl,
                "strike": float(self.contract.strike) if self.contract else None,
                "right": self.contract.right if self.contract else None,

                # include inputs & derived risk numbers so UI can display them
                "stop_on_spy": round(self.pos.stop_on_spy, 4),
                "max_loss_usd": round(self.pos.max_loss_usd, 2),
                "loss_per_contract": loss_per_contract,
                "auto_protect": bool(self.pos.auto_protect_enabled),
                "protect_ratio": round(self.pos.protect_ratio, 4) if self.pos.auto_protect_enabled else None,
                "protect_triggered": bool(self.pos.protect_triggered),
                "calc_protect_price": self.pos.fixed_protect_price,       # fixed
            }

    def _await_fill(self, trade) -> tuple[int, float]:
        """Wait for an order to finish while surfacing broker rejections early."""
        error_states = {'Rejected', 'Cancelled', 'ApiCancelled', 'Error', 'Inactive'}
        last_log_idx = 0

        while not trade.isDone():
            status = (trade.orderStatus.status or '').strip()
            if status in error_states:
                raise OrderExecutionError(self._format_trade_failure(trade, status))

            log_entries = getattr(trade, 'log', [])
            if len(log_entries) > last_log_idx:
                new_entries = log_entries[last_log_idx:]
                last_log_idx = len(log_entries)
                for entry in new_entries:
                    entry_status = (getattr(entry, 'status', '') or '').strip()
                    if entry_status in error_states or entry_status == 'Error':
                        raise OrderExecutionError(self._format_trade_failure(trade, entry_status or status))

            self.ib.sleep(0.1)

        shares = 0
        notional = 0.0
        for f in trade.fills:
            px = getattr(f.execution, 'price', None)
            qty = getattr(f.execution, 'shares', None)
            if px is not None and qty is not None:
                shares += int(qty)
                notional += float(px) * int(qty)
        avg = (notional / shares) if shares > 0 else 0.0

        final_status = (trade.orderStatus.status or '').strip()
        if shares <= 0 and final_status in error_states:
            raise OrderExecutionError(self._format_trade_failure(trade, final_status))

        return shares, avg

    def _format_trade_failure(self, trade, status_hint: str | None = None) -> str:
        """Create a readable error message from IB trade logs."""
        error_statuses = {'Error', 'Rejected', 'Cancelled', 'ApiCancelled', 'Inactive'}
        messages: list[str] = []
        for entry in getattr(trade, 'log', []) or []:
            entry_status = (getattr(entry, 'status', '') or '').strip()
            entry_message = (getattr(entry, 'message', '') or '').strip()
            if entry_status in error_statuses and entry_message:
                if entry_message not in messages:
                    messages.append(entry_message)

        if not messages and getattr(trade, 'log', None):
            last_message = (getattr(trade.log[-1], 'message', '') or '').strip()
            if last_message:
                messages.append(last_message)

        detail = '; '.join(messages)
        status_text = str(status_hint or trade.orderStatus.status or 'failed').strip()
        status_text = status_text if status_text else 'failed'
        status_text_lower = status_text.lower()

        if detail:
            return f"Order {status_text_lower}: {detail}"
        return f"Order {status_text_lower}."

    def open_position(
        self,
        qty: int,
        stop_on_spy: float,
        max_loss_usd: float,
        auto_protect: bool = False,
        protect_ratio: float | None = None,
        entry_delta_abs: float | None = None,   # NEW: capture |delta| once
    ) -> dict:
        if qty <= 0:
            return {"error": "qty must be > 0"}
        if not self.contract:
            return {"error": "No option contract selected"}

        trade = self.ib.placeOrder(self.contract, MarketOrder('BUY', int(qty)))
        filled, avg = self._await_fill(trade)

        with position_lock:
            if not self.pos:
                self.pos = Position(self.contract.localSymbol, self.contract.conId)
            self.pos.local_symbol = self.contract.localSymbol
            self.pos.con_id = self.contract.conId

            new_qty = self.pos.qty + filled
            if new_qty > 0:
                self.pos.avg_price = ((self.pos.avg_price * self.pos.qty) + (avg * filled)) / new_qty
            self.pos.qty = new_qty
            self.pos.stop_on_spy = float(stop_on_spy)
            self.pos.max_loss_usd = float(max_loss_usd)

            was_inactive = not self.pos.active
            self.pos.active = self.pos.qty > 0

            # auto-protect toggles
            enable_protect = bool(auto_protect)
            ratio_value = 0.0
            if enable_protect:
                try:
                    ratio_value = float(protect_ratio if protect_ratio is not None else 0.0)
                except Exception:
                    ratio_value = 0.0
                if ratio_value <= 0:
                    enable_protect = False
                    ratio_value = 0.0
                else:
                    ratio_value = max(1.0, ratio_value)
            self.pos.auto_protect_enabled = enable_protect
            self.pos.protect_ratio = ratio_value if enable_protect else 0.0

            # --- set fixed anchors ONLY on first activation (opening trade) ---
            if was_inactive and self.pos.active:
                if entry_delta_abs is None or not math.isfinite(entry_delta_abs):
                    with _snapshot_lock:
                        entry_delta_abs = abs(_to_float(_snapshot.get('delta')) or 0.0)
                self.pos.entry_delta_abs = float(max(0.0, entry_delta_abs))

                # compute fixed stop once
                if self.pos.entry_delta_abs > 0 and self.pos.stop_on_spy > 0:
                    self.pos.fixed_stop_price = round(
                        self.pos.avg_price - (self.pos.entry_delta_abs * self.pos.stop_on_spy), 4
                    )
                else:
                    self.pos.fixed_stop_price = None

                # compute fixed protect once (only if enabled)
                if self.pos.auto_protect_enabled and self.pos.protect_ratio > 0 and self.pos.entry_delta_abs > 0:
                    self.pos.fixed_protect_price = round(
                        self.pos.avg_price + (self.pos.entry_delta_abs * self.pos.stop_on_spy * self.pos.protect_ratio), 4
                    )
                else:
                    self.pos.fixed_protect_price = None

            # refresh UI mirrors
            self.pos.last_stop_price = self.pos.fixed_stop_price
            self.pos.last_protect_price = self.pos.fixed_protect_price
            self.pos.protect_triggered = False

        return {"filled": filled, "avg_price": round(avg, 4), "total_qty": int(self.pos.qty)}

    def sell(self, qty: int) -> dict:
        if qty <= 0:
            return {"error": "qty must be > 0"}
        if not self.contract:
            return {"error": "No option contract selected"}

        with position_lock:
            if not self.pos or not self.pos.active or self.pos.qty <= 0:
                return {"error": "No open position"}
            sell_qty = min(int(qty), int(self.pos.qty))

        trade = self.ib.placeOrder(self.contract, MarketOrder('SELL', sell_qty))
        filled, avg = self._await_fill(trade)

        with position_lock:
            self.pos.qty -= filled
            if self.pos.qty <= 0:
                self.pos.qty = 0
                self.pos.active = False
                self.pos.auto_protect_enabled = False
                self.pos.protect_triggered = False
                self.pos.protect_ratio = 0.0
                self.pos.last_protect_price = None
                # clear anchors (NEW)
                self.pos.entry_delta_abs = None
                self.pos.fixed_stop_price = None
                self.pos.fixed_protect_price = None

        return {"sold": filled, "avg_price": round(avg, 4), "remaining": int(self.pos.qty)}

    def monitor_auto_stop(self):
        """Server-side stop: if bid <= fixed_stop, liquidate remaining qty at market.
           Auto-protect: if bid >= fixed_protect (first time), sell half.
        """
        action = None
        sell_qty = 0
        mark_protected = False
        with position_lock, _snapshot_lock:
            if not self.pos or not self.pos.active or self._liquidating:
                return

            bid = _to_float(_snapshot.get('bid'))
            if bid is None or not math.isfinite(bid):
                return

            # Use fixed anchors (no recompute)
            stop_price = self.pos.fixed_stop_price
            protect_price = self.pos.fixed_protect_price

            # keep UI mirrors updated
            self.pos.last_stop_price = stop_price
            self.pos.last_protect_price = protect_price

            remaining = int(self.pos.qty)
            if remaining <= 0:
                self.pos.active = False
                return

            if stop_price is not None and bid <= stop_price:
                action = 'stop'
                sell_qty = remaining
            elif (
                protect_price is not None
                and not self.pos.protect_triggered
                and bid >= protect_price
            ):
                action = 'protect'
                sell_qty = max(1, math.ceil(remaining / 2))
                mark_protected = True

            if not action:
                return

            self._liquidating = True
        try:
            trade = self.ib.placeOrder(self.contract, MarketOrder('SELL', sell_qty))
            filled, _ = self._await_fill(trade)
            with position_lock:
                self.pos.qty -= filled
                if self.pos.qty <= 0:
                    self.pos.qty = 0
                    self.pos.active = False
                    self.pos.auto_protect_enabled = False
                    self.pos.protect_triggered = False
                    self.pos.protect_ratio = 0.0
                    self.pos.last_protect_price = None
                    # clear anchors (NEW)
                    self.pos.entry_delta_abs = None
                    self.pos.fixed_stop_price = None
                    self.pos.fixed_protect_price = None
                elif mark_protected and filled > 0:
                    self.pos.protect_triggered = True
        except OrderExecutionError as oe:
            print(f"\nAuto-stop order failed: {oe}", flush=True)
            if mark_protected:
                with position_lock:
                    self.pos.protect_triggered = False
        finally:
            self._liquidating = False

# --------------- Engine state ------------
_engine = {
    "ib": None,
    "stk": None,
    "t_und": None,
    "t_opt": None,
    "right": OPTION_RIGHT,
    "local_symbol": None,   # machine label
    "display_name": None,   # human-readable label
}

# NEW: cached strikes / current selection / debounce
_cached_strikes = []
_current_strike = None
_current_exp = None
_last_switch_ts = 0.0

# --------------- Job queue ---------------
ib_jobs: "queue.Queue[tuple[Callable, tuple, dict, queue.Queue]]" = queue.Queue()

def submit_ib_job(fn: Callable, *args, **kwargs) -> Any:
    """Enqueue a function to run on the IB thread; wait for result (or exception)."""
    reply_q: "queue.Queue[Any]" = queue.Queue(maxsize=1)
    ib_jobs.put((fn, args, kwargs, reply_q))
    ok, payload = reply_q.get()
    if ok:
        return payload
    else:
        raise payload

# --------------- Flask app ---------------
app = Flask(__name__)
pos_manager: PositionManager | None = None  # set in start_backend()

@app.get('/snapshot')
def get_snapshot():
    with _snapshot_lock:
        return jsonify(dict(_snapshot))

@app.get('/position_size')
def position_size():
    """Suggested contracts from stop_on_spy & max_loss_usd using live delta."""
    try:
        stop_on_spy = float(request.args.get('stop_on_spy', 0))
        max_loss_usd = float(request.args.get('max_loss_usd', 0))
    except Exception:
        return jsonify({"error": "Invalid input"}), 400

    with _snapshot_lock:
        delta = _to_float(_snapshot.get('delta'))

    if delta is None or stop_on_spy <= 0 or max_loss_usd <= 0:
        return jsonify({"error": "Need live delta and positive inputs"}), 400

    delta_abs = abs(delta)  # use magnitude for risk sizing
    loss_per_contract = delta_abs * stop_on_spy * 100
    suggested = int(max_loss_usd // loss_per_contract)  # floor

    return jsonify({
        "delta": round(delta, 4),  # keep original sign for display
        "stop_on_spy": stop_on_spy,
        "max_loss_usd": max_loss_usd,
        "loss_per_contract": round(loss_per_contract, 2),
        "suggested_contracts": suggested if suggested > 0 else 0
    })

@app.get('/positions')
def positions():
    return jsonify(pos_manager.positions_json() if pos_manager else {"status": "flat"})

@app.post('/buy')
def buy():
    """
    Body JSON:
      Auto size: { "stop_on_spy": 0.25, "max_loss_usd": 500 }
      Manual   : { "qty": 3, "stop_on_spy": 0.25 }
                 (max_loss_usd will be computed from live delta)
    """
    if not pos_manager:
        return jsonify({"error": "Engine not ready"}), 503

    data = request.get_json(force=True) or {}
    try:
        stop_on_spy = float(data.get('stop_on_spy', 0))
    except Exception:
        return jsonify({"error": "Invalid stop_on_spy"}), 400
    if stop_on_spy <= 0:
        return jsonify({"error": "stop_on_spy must be > 0"}), 400

    qty_override = data.get('qty', data.get('override_qty'))
    max_loss_usd = data.get('max_loss_usd')  # only used in auto-size
    auto_protect_flag = bool(data.get('auto_protect'))
    protect_ratio_raw = data.get('protect_ratio')
    protect_ratio_value = None
    if auto_protect_flag:
        try:
            protect_ratio_value = float(protect_ratio_raw if protect_ratio_raw is not None else 0)
        except Exception:
            return jsonify({"error": "Invalid protect_ratio"}), 400
        if protect_ratio_value < 1:
            return jsonify({"error": "protect_ratio must be ≥ 1"}), 400

    # Need live delta for both auto and manual (for projected loss & stop calc sanity)
    with _snapshot_lock:
        delta = _to_float(_snapshot.get('delta'))

    if delta is None or not math.isfinite(delta):
        return jsonify({"error": "No live delta yet"}), 503

    delta_abs = abs(delta)  # use magnitude for risk sizing

    if qty_override is None:
        try:
            max_loss_usd = float(max_loss_usd or 0)
        except Exception:
            return jsonify({"error": "Invalid max_loss_usd"}), 400
        if max_loss_usd <= 0:
            return jsonify({"error": "max_loss_usd must be > 0 for auto size"}), 400

        loss_per_contract = delta_abs * stop_on_spy * 100.0
        if loss_per_contract <= 0:
            return jsonify({"error": "Sizing failed: zero loss_per_contract"}), 400

        qty = int(max_loss_usd // loss_per_contract)  # floor
        if qty <= 0:
            return jsonify({"error": "Computed qty <= 0 (increase max_loss or stop)"}), 400
        projected_max_loss_usd = max_loss_usd
    else:
        try:
            qty = int(qty_override)
        except Exception:
            return jsonify({"error": "qty must be integer"}), 400
        if qty <= 0:
            return jsonify({"error": "qty must be > 0"}), 400

        loss_per_contract = delta_abs * stop_on_spy * 100.0
        if loss_per_contract <= 0:
            return jsonify({"error": "Sizing failed: zero loss_per_contract"}), 400

        projected_max_loss_usd = round(qty * loss_per_contract, 2)

    # Place order on IB thread (pass entry |delta| to anchor fixed stops/protects)
    try:
        result = submit_ib_job(
            pos_manager.open_position,
            qty,
            stop_on_spy,
            projected_max_loss_usd,
            auto_protect_flag,
            protect_ratio_value,
            delta_abs,  # NEW
        )
    except OrderExecutionError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    if "error" in result:
        return jsonify(result), 400

    result.update({
        "delta": round(delta, 4),
        "loss_per_contract": round(loss_per_contract, 2),
        "projected_max_loss_usd": projected_max_loss_usd,
    })
    return jsonify(result)

@app.post('/sell')
def sell():
    """
    Body JSON:
      { "qty": 3 }            → sell 3
      { "mode": "half" }      → sell floor(current_qty/2)
      { "mode": "all" }       → sell all
    """
    if not pos_manager:
        return jsonify({"error": "Engine not ready"}), 503

    data = request.get_json(force=True) or {}
    mode = data.get('mode')
    qty  = data.get('qty')

    with position_lock:
        if not pos_manager.pos or not pos_manager.pos.active:
            return jsonify({"error": "No open position"}), 400
        current_qty = int(pos_manager.pos.qty)

    sell_qty = None
    if qty is not None:
        try:
            sell_qty = max(0, int(qty))
        except Exception:
            return jsonify({"error": "qty must be integer"}), 400
    elif mode == 'half':
        sell_qty = max(1, current_qty // 2)
    elif mode == 'all':
        sell_qty = current_qty

    if not sell_qty or sell_qty <= 0:
        return jsonify({"error": "No qty to sell"}), 400

    try:
        result = submit_ib_job(pos_manager.sell, sell_qty)
    except OrderExecutionError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)

@app.post('/set_right')
def set_right():
    """
    Body JSON: { "right": "C" | "P" }
    Re-selects the option (nearest expiry + nearest OTM) for the chosen right.
    Requires no open position.
    """
    if not pos_manager:
        return jsonify({"error": "Engine not ready"}), 503

    data = request.get_json(force=True) or {}
    right = str(data.get('right', '')).upper()
    if right not in ('C', 'P'):
        return jsonify({"error": "right must be 'C' or 'P'"}), 400

    if pos_manager.has_open():
        return jsonify({"error": "Close position before switching Call/Put"}), 400

    try:
        res = submit_ib_job(_ib_set_right, right)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify(res)

@app.get('/projected_loss')
def projected_loss():
    """
    Manual sizing projection (no position required).
    Query params:
      qty (int)             : number of contracts you intend to buy
      stop_on_spy (float)   : stop distance on the underlying (e.g. 0.25)
    Returns:
      loss_per_contract ($) and projected_max_loss_usd ($) using live delta.
    """
    try:
        qty = int(request.args.get('qty', 0))
        stop_on_spy = float(request.args.get('stop_on_spy', 0))
    except Exception:
        return jsonify({"error": "Invalid inputs"}), 400

    if qty <= 0 or stop_on_spy <= 0:
        return jsonify({"error": "qty and stop_on_spy must be > 0"}), 400

    with _snapshot_lock:
        delta = _to_float(_snapshot.get('delta'))

    if delta is None or not math.isfinite(delta):
        return jsonify({"error": "No live delta yet"}), 503

    delta_abs = abs(delta)
    loss_per_contract = delta_abs * stop_on_spy * 100.0
    projected = round(qty * loss_per_contract, 2)

    return jsonify({
        "qty": qty,
        "stop_on_spy": stop_on_spy,
        "delta": round(delta, 4),             # signed for display
        "loss_per_contract": round(loss_per_contract, 2),
        "projected_max_loss_usd": projected,
    })


# ------------- IB-thread helpers ----------
def _ib_set_right(right: str) -> dict:
    """Run on IB thread: cancel old option stream, select new right, stream it, update state."""
    global _cached_strikes, _current_strike, _current_exp, _last_switch_ts

    ib: IB = _engine["ib"]
    stk: Stock = _engine["stk"]
    t_und = _engine["t_und"]
    t_opt_old = _engine["t_opt"]

    und_px = t_und.marketPrice()
    if not (math.isfinite(und_px) and und_px > 0):
        raise RuntimeError("No live underlying price available to reselect option.")

    params = ib.reqSecDefOptParams(stk.symbol, '', stk.secType, stk.conId)
    rows = [p for p in params if p.exchange in ('SMART', OPTION_EXCHANGE)
            and p.tradingClass in (UNDERLYING_SYMBOL, stk.symbol)]
    if not rows:
        raise RuntimeError("No option parameters available for right switch.")

    row = rows[0]
    exp = pick_nearest_expiration(row.expirations)
    strike = pick_nearest_otm_strike(row.strikes, und_px, right)

    opt = Option(UNDERLYING_SYMBOL, exp, float(strike), right, OPTION_EXCHANGE, currency='USD')
    ib.qualifyContracts(opt)

    # Cancel previous option stream if any
    if t_opt_old is not None:
        try:
            ib.cancelMktData(t_opt_old.contract)
        except Exception:
            pass

    # Start new stream -- request greeks via '106'
    t_opt_new = ib.reqMktData(opt, '106', False, False)
    local_symbol = opt.localSymbol or f"{UNDERLYING_SYMBOL} {exp} {strike} {right}"
    readable = readable_name(UNDERLYING_SYMBOL, exp, float(strike), right)

    # Update engine & position manager contract
    _engine["t_opt"] = t_opt_new
    _engine["right"] = right
    _engine["local_symbol"] = local_symbol
    _engine["display_name"] = readable
    pos_manager.contract = opt

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

    row = rows[0]
    exp = pick_nearest_expiration(row.expirations)
    strike = pick_nearest_otm_strike(row.strikes, und_px, OPTION_RIGHT)

    opt = Option(UNDERLYING_SYMBOL, exp, float(strike), OPTION_RIGHT, OPTION_EXCHANGE, currency='USD')
    ib.qualifyContracts(opt)
    # request greeks/implied vol reliably
    t_opt = ib.reqMktData(opt, '106', False, False)

    local_symbol = opt.localSymbol or f"{UNDERLYING_SYMBOL} {exp} {strike} {OPTION_RIGHT}"
    readable = readable_name(UNDERLYING_SYMBOL, exp, float(strike), OPTION_RIGHT)

    # Position manager & engine state
    global pos_manager, _cached_strikes, _current_strike, _current_exp, _last_switch_ts
    pos_manager = PositionManager(ib)
    pos_manager.contract = opt

    _engine.update({
        "ib": ib,
        "stk": stk,
        "t_und": t_und,
        "t_opt": t_opt,
        "right": OPTION_RIGHT,
        "local_symbol": local_symbol,   # machine label kept
        "display_name": readable,       # human label for UI
    })

    # cache strikes & current selection (for fast strict-OTM reseat)
    _cached_strikes = sorted({float(s) for s in row.strikes})
    _current_strike = float(strike)
    _current_exp = exp
    _last_switch_ts = datetime.now().timestamp()

    with _snapshot_lock:
        _snapshot.update({
            "status": f"connected (clientId={TWS_CLIENT_ID})",
            "contract": readable,        # send readable to UI
            "right": OPTION_RIGHT
        })

    print("Backend streaming... endpoints: /snapshot /position_size /positions /buy /sell /set_right")

    # Helper to reseat the option to a new strictly-OTM strike (when flat)
    def reseat_option(new_strike: float):
        nonlocal t_opt, local_symbol, opt
        global _current_strike, _last_switch_ts

        right = _engine["right"]
        new_opt = Option(UNDERLYING_SYMBOL, _current_exp, float(new_strike), right, OPTION_EXCHANGE, currency='USD')
        ib.qualifyContracts(new_opt)

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
        ib.disconnect()

if __name__ == "__main__":
    start_backend()
