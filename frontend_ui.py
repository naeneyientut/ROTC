# frontend_ui.py — pretty panel edition (clean header + footer status bar + unified SELL)
# Run: python frontend_ui.py  → http://127.0.0.1:8080

from nicegui import ui, app
import httpx
import math

# ---------------- Config / client ----------------
BACKEND_URL = 'http://127.0.0.1:8001'
client = httpx.AsyncClient(timeout=3.0)
REFRESH_SEC = 0.1

latest_delta = None           # cache latest delta for projections
current_side = None           # last side confirmed by backend: 'C' or 'P'
_is_switching_side = False    # debounce side switching

# Some Globals track qty syncing so we don't overwrite user edits
last_backend_qty = None
user_qty_dirty = False
_setting_qty_programmatically = False
_setting_auto_protect_programmatically = False


# Fixed window (optional when running as native window)
try:
    app.native.window_args['resizable'] = True
except Exception:
    pass

# ----------------- Theme & CSS -------------------
ui.dark_mode().enable()
ui.colors(primary='#3b82f6')
ui.add_head_html("""
<style>
  html, body, #app { height: 100%; margin: 0; background: #0e1014; display:flex; align-items:center; justify-content:center; overflow:hidden; }
  .panel { width: 420px; background: #14171c; border-radius: 0px; box-shadow: 0 10px 26px rgba(0,0,0,.5); color: #e8eaf0; }
  .muted { color: #9aa3af; }
  .divider { border-bottom: 1px solid #2a2e35; margin: 10px 0; }
  .value { color:#e8eaf0; }
  .green { color:#5bd17a; }
  .red { color:#ff6466; }
  .btn {width:50%; border-radius:0px;font-weight:500;}
  .chip { font-size:10px; color:#9aa3af; }
  .row { display:flex; align-items:center; justify-content:space-between; }
  .grid2 { display:grid; grid-template-columns: 1fr 1fr; gap:8px 20px; }
  .wfull { width:100%; }
  .tight { margin-top: 2px; margin-bottom: 2px; }
  .mono { font-variant-numeric: tabular-nums; }
  .pill { padding: 2px 6px; border: 1px solid #2a2e35; border-radius: 999px; font-size: 10px; color:#9aa3af; }
  .toggle .q-btn { border-radius:0 !important; }

  /* Footer status bar */
  .footerbar { border-top: 1px solid #2a2e35; padding-top: 8px; margin-top: 12px; }
  .rowc { display:flex; align-items:center; justify-content:space-between; }
  .rowl { display:flex; align-items:center; gap:8px; }
  .dot { width:10px; height:10px; border-radius:50%; background:#ef4444; } /* red by default */
  .dot.ok { background:#22c55e; }  /* green when connected */
</style>
""")

# ----------------- Small helpers -----------------
def fmt_money(x):
    if x in (None, 'n/a'):
        return 'n/a'
    try:
        value = float(x)
    except Exception:
        return str(x)
    if not math.isfinite(value):
        return 'n/a'
    return f'{value:,.2f}'

def fmt_plain(x):
    return '--' if x in (None, 'n/a') else str(x)


def _attach_floor_guard(input_element, *, minimum: float, field_name: str, message: str = None):
    """Ensure the given input never drops below ``minimum``."""

    def _validate(_):
        value = input_element.value
        if value in (None, ''):
            return
        try:
            numeric_value = float(value)
        except Exception:
            input_element.value = ''
            ui.notify(f'{field_name} must be a number', color='warning')
            return
        if numeric_value < minimum:
            minimum_text = (
                str(int(minimum)) if isinstance(minimum, (int, float)) and float(minimum).is_integer()
                else str(minimum)
            )
            input_element.value = minimum_text
            ui.notify(message or f'{field_name} must be ≥ {minimum_text}', color='warning')

    input_element.on('blur', _validate)
    input_element.on('keydown.enter', _validate)

# ----------------- UI: Outer container -----------
with ui.element('div').classes('panel p-4'):

    # ---------- Top: Calls / Puts toggle ----------
    with ui.element('div').classes('row'):
        ui.label('Option Side').classes('text-sm muted')
        side = ui.toggle({'C': 'Calls', 'P': 'Puts'}, value='C').props('spread no-caps unelevated').classes('toggle')

    ui.element('div').classes('divider')

    # ---------- Header (SPY price only) ----------
    with ui.element('div').classes('row mt-1'):
        with ui.column().classes('items-start'):
            ui.label('SPY').classes('text-[10px] muted tight')  # tiny label above price
            lbl_header_spy = ui.label('—').classes('text-xl font-semibold tracking-wide mono')
            ui.label('REALTIME CONSOLIDATED').classes('text-[10px] green tight')  # subtitle

    ui.element('div').classes('divider')

    # ---------- Contract (now where SPY used to be) ----------
    with ui.element('div').classes('row'):
        with ui.column().classes('items-start'):
            ui.label('Contract').classes('text-[10px] muted tight')
            lbl_contract_main = ui.label('n/a').classes('text-lg font-semibold mono')
        with ui.column().classes('items-end'):
            ui.label('Ask').classes('text-[10px] muted tight')
            lbl_ask = ui.label('n/a').classes('text-md value mono')
            ui.label('Bid').classes('text-[10px] muted tight')
            lbl_bid = ui.label('n/a').classes('text-md value mono')

    with ui.element('div').classes('row mt-1'):
        ui.label('Δ / IV').classes('text-[10px] muted tight')
        lbl_greeks = ui.label('Δ n/a | IV n/a').classes('text-sm value mono')

    ui.element('div').classes('divider')
    # ---------- Buy / Sell buttons ----------
    with ui.row().classes('w-full no-wrap gap-2 mt-2'):
        btn_buy  = ui.button('BUY').props('unelevated color=primary').classes('w-full')
        btn_sell = ui.button('SELL').props('unelevated color=negative').classes('w-full')  # red SELL
    # ---------- Position Size Calculator ----------
    #ui.label('Position Size').classes('text-sm muted')
    with ui.element('div').classes('row mt-2'):
        auto_sz  = ui.switch('Auto Size', value=True).props('size="xs"')
        qty_in   = ui.input('Quantity').props('type=number dense min=0').classes('w-28')
        _attach_floor_guard(qty_in, minimum=0, field_name='Quantity', message='Quantity cannot be negative')
        qty_in.disable()
    with ui.element('div').classes('grid2 mt-1'):
        stop_in  = ui.input('Stop on SPY ($)').props('type=number step=0.05 dense min=0').classes('wfull')
        loss_in  = ui.input('Max loss ($)').props('type=number step=10 dense min=0').classes('wfull')
        _attach_floor_guard(stop_in, minimum=0, field_name='Stop on SPY', message='Stop on SPY cannot be negative')
        _attach_floor_guard(loss_in, minimum=0, field_name='Max loss', message='Max loss cannot be negative')
    with ui.element('div').classes('row mt-1').style('margin-top:15px'):
        lbl_calc = ui.label('Suggested: n/a').classes('text-[11px] green mono')
        lbl_proj = ui.label('Projected Loss: n/a').classes('text-[11px] muted mono')
    
    ui.element('div').classes('divider')

    with ui.element('div').classes('row mt-2'):
        auto_protect  = ui.switch('Auto Protect', value=True).props('size="xs"')
        protect_ratio   = ui.input('R:R Ratio').props('type=number dense step=0.5 min=1').classes('w-28')
        _attach_floor_guard(protect_ratio, minimum=1, field_name='R:R Ratio', message='R:R Ratio must be at least 1')
        protect_ratio.value = '1'
        #qty_in.disable()

    ui.element('div').classes('divider')

    # ---------- Open position ----------
    ui.label('Open Position').classes('text-sm muted')
    lbl_pos_status = ui.label('flat').classes('text-[12px] muted').style('display:none')
    # Disable SELL button when there's no open position
    btn_sell.bind_enabled_from(lbl_pos_status, 'text', lambda s: s == 'open')
    # 4-column grid: label | value | label | value
    with ui.element('div')\
            .classes('mt-1 text-[12px] mono')\
            .style('display:grid; grid-template-columns: auto 1fr auto 1fr; column-gap:16px; row-gap:6px; align-items:center;'):

        ui.label('Contracts').classes('muted'); lbl_pos_contracts = ui.label('-').classes('value')
        ui.label('Avg. Price').classes('muted'); lbl_pos_avg       = ui.label('-').classes('value')

        ui.label('Strike').classes('muted');    lbl_pos_strike    = ui.label('-').classes('value')
        ui.label('Bid').classes('muted');       lbl_pos_bid       = ui.label('-').classes('value')

        ui.label('Side').classes('muted');      lbl_pos_side      = ui.label('-').classes('value')
        ui.label('Stop Price').classes('muted'); lbl_pos_stop     = ui.label('-').classes('value')

        ui.label('PnL').classes('muted');       lbl_pos_pnl       = ui.label('-').classes('value')
        ui.label('Ask').classes('muted');       lbl_pos_ask       = ui.label('-').classes('value')

        ui.label('Protect Price').classes('muted');  lbl_pos_protect = ui.label('-').classes('value')
        ui.label('Protect Status').classes('muted'); lbl_pos_protect_status = ui.label('-').classes('value')

    ui.element('div').classes('divider')

    ui.label('Portfolio').classes('text-sm muted')
    with ui.element('div')\
            .classes('mt-1 text-[12px] mono')\
            .style('display:grid; grid-template-columns: auto 1fr; column-gap:16px; row-gap:6px; align-items:center;'):

        ui.label('Net Liquidity').classes('muted'); lbl_port_net = ui.label('-').classes('value mono')
        ui.label('Daily PnL').classes('muted');      lbl_port_daily = ui.label('-').classes('value mono')
        ui.label('Buying Power').classes('muted');   lbl_port_buying = ui.label('-').classes('value mono')

    # Disable side toggle whenever a position is open
    side.bind_enabled_from(lbl_pos_status, 'text', lambda s: s == 'flat')

    # ---------- Footer status bar ----------
    with ui.element('div').classes('footerbar rowc'):
        with ui.element('div').classes('rowl'):
            status_dot = ui.element('div').classes('dot')  # red by default (disconnected)
            lbl_status = ui.label('connecting…').classes('text-[12px] muted')
        lbl_time = ui.label('--:--:--').classes('text-[12px] mono muted')

# Optional native window sizing (ignored in browser)
try:
    from nicegui import native as _native
    _native.window_args = {'resizable': False, 'min_size': (460, 900), 'max_size': (460, 900)}
except Exception:
    pass

def _on_qty_change(_):
    global user_qty_dirty, _setting_qty_programmatically
    # ignore changes we make in code
    if _setting_qty_programmatically:
        return
    user_qty_dirty = True

qty_in.on('update:model-value', _on_qty_change)

# ---------------- Pollers ----------------
async def poll_snapshot():
    global latest_delta, current_side
    try:
        r = await client.get(f'{BACKEND_URL}/snapshot')
        d = r.json()

        # footer status (connected)
        lbl_status.text = 'Connected'
        status_dot.classes(replace='dot ok')  # green
        lbl_time.text   = f"{d.get('time','--:--:--')}"

        # HEADER: price only
        spy  = d.get('spy_price', 'n/a')
        lbl_header_spy.text = f"{spy}"

        # CONTRACT block
        contract = d.get('contract', 'n/a')
        lbl_contract_main.text = contract

        bid = d.get('bid', 'n/a')
        ask = d.get('ask', 'n/a')
        lbl_bid.text = fmt_plain(bid)
        lbl_ask.text = fmt_plain(ask)

        delta = d.get('delta', 'n/a')
        iv    = d.get('iv', 'n/a')
        lbl_greeks.text = f"Δ {delta} | IV {iv}"

        # one-time side sync
        srv_right = d.get('right')
        if current_side is None and srv_right in ('C','P'):
            current_side = srv_right
            side.value = srv_right

        # keep latest_delta cache
        try:
            latest_delta = float(delta)
        except:
            latest_delta = None

        portfolio = d.get('portfolio') if isinstance(d.get('portfolio'), dict) else {}

        net_liq_val = portfolio.get('net_liquidity')
        try:
            net_liq_num = float(net_liq_val)
            if not math.isfinite(net_liq_num):
                raise ValueError
        except Exception:
            net_liq_num = None
        if net_liq_num is None:
            lbl_port_net.text = 'n/a'
            lbl_port_net.classes(replace='value mono muted')
        else:
            lbl_port_net.text = fmt_money(net_liq_num)
            lbl_port_net.classes(replace='value mono')

        buying_power_val = portfolio.get('buying_power')
        try:
            buying_power_num = float(buying_power_val)
            if not math.isfinite(buying_power_num):
                raise ValueError
        except Exception:
            buying_power_num = None
        if buying_power_num is None:
            lbl_port_buying.text = 'n/a'
            lbl_port_buying.classes(replace='value mono muted')
        else:
            lbl_port_buying.text = fmt_money(buying_power_num)
            lbl_port_buying.classes(replace='value mono')

        daily_pnl_val = portfolio.get('daily_pnl')
        try:
            daily_pnl_num = float(daily_pnl_val)
            if not math.isfinite(daily_pnl_num):
                raise ValueError
        except Exception:
            daily_pnl_num = None

        daily_pct_val = portfolio.get('daily_pnl_pct')
        pct_display = ''
        pct_num = None
        try:
            pct_candidate = float(daily_pct_val)
            if math.isfinite(pct_candidate):
                pct_num = pct_candidate
        except Exception:
            pct_num = None
        if pct_num is not None:
            sign = '+' if pct_num > 0 else ''
            pct_display = f" ({sign}{pct_num:.2f}%)"
        elif daily_pct_val not in (None, 'n/a'):
            pct_display = f" ({daily_pct_val})"

        if daily_pnl_num is None:
            lbl_port_daily.text = 'n/a'
            lbl_port_daily.classes(replace='value mono muted')
        else:
            pnl_text = fmt_money(daily_pnl_num)
            lbl_port_daily.text = f"{pnl_text}{pct_display}"
            if daily_pnl_num > 0:
                lbl_port_daily.classes(replace='value mono green')
            elif daily_pnl_num < 0:
                lbl_port_daily.classes(replace='value mono red')
            else:
                lbl_port_daily.classes(replace='value mono')
    except Exception:
        # footer status (disconnected)
        lbl_status.text = 'Disconnected'
        status_dot.classes(replace='dot')  # red
        lbl_time.text   = '--:--:--'

async def poll_position_size():
    """Auto ON: ask backend for suggestion. Auto OFF: live projection mirrored into max-loss field."""
    try:
        # toggle enable states based on auto size & position status
        is_open = (lbl_pos_status.text == 'open')
        if is_open:
            # enable quantity for SELL usage regardless of auto size
            qty_in.enable()
            loss_in.disable()  # don't edit max loss while position is open
        else:
            # flat: classic behavior
            if auto_sz.value:
                qty_in.disable();  loss_in.enable()
            else:
                qty_in.enable();   loss_in.disable()

        if auto_sz.value and not is_open:
            # only do backend suggestion while flat
            if not stop_in.value or not loss_in.value:
                lbl_calc.text = 'Suggested: n/a'
                lbl_proj.text = 'Projected Loss: n/a'
                return
            params = {"stop_on_spy": stop_in.value, "max_loss_usd": loss_in.value}
            r = await client.get(f'{BACKEND_URL}/position_size', params=params)
            if r.status_code != 200:
                lbl_calc.text = 'Suggested: n/a'; lbl_proj.text = 'Projected Loss: n/a'; return
            d = r.json()
            sug = d.get('suggested_contracts'); lpc = d.get('loss_per_contract')
            lbl_calc.text = f"Suggested: {sug if sug is not None else 'n/a'} | Loss/contract ${lpc if lpc is not None else 'n/a'}"
            if sug is not None:
                qty_in.value = str(sug)
            lbl_proj.text = 'Projected Loss: n/a'
        else:
            # manual qty OR open position → compute projected loss locally
            if not stop_in.value or not qty_in.value or latest_delta is None:
                lbl_proj.text = 'Projected Loss: n/a'
                if not is_open:
                    loss_in.value = ''
                lbl_calc.text = 'Loss/contract: n/a'
                return
            try:
                stop_val = float(stop_in.value)
                qty_val  = int(float(qty_in.value))
            except:
                lbl_proj.text = 'Projected Loss: n/a'
                if not is_open:
                    loss_in.value = ''
                lbl_calc.text = 'Loss/contract: n/a'
                return
            if stop_val <= 0 or qty_val <= 0:
                lbl_proj.text = 'Projected Loss: n/a'
                if not is_open:
                    loss_in.value = ''
                lbl_calc.text = 'Loss/contract: n/a'
                return

            # use magnitude so it works for puts and calls
            delta_mag = abs(latest_delta)
            loss_per_contract = delta_mag * stop_val * 100.0
            projected = loss_per_contract * qty_val

            lbl_proj.text = f'Projected Loss: ${projected:.2f}'
            lbl_calc.text = f'Loss/contract: ${loss_per_contract:.2f}'
            if not is_open:
                loss_in.value = f'{projected:.2f}'
    except Exception as e:
        lbl_calc.text = f'Error: {e}'

async def poll_positions():
    """Show open position and sync qty field only when backend qty changes."""
    global last_backend_qty, _setting_auto_protect_programmatically
    try:
        r = await client.get(f'{BACKEND_URL}/positions')
        r.raise_for_status()
        p = r.json()
    except Exception:
        # transient errors: keep whatever is on screen
        return

    status = p.get('status', 'flat')
    lbl_pos_status.text = status

    if status != 'open':
        # flat → show dashes
        lbl_pos_contracts.text = '-'
        lbl_pos_strike.text    = '-'
        lbl_pos_side.text      = '-'
        lbl_pos_avg.text       = '-'
        lbl_pos_stop.text      = '-'
        lbl_pos_bid.text       = '-'
        lbl_pos_ask.text       = '-'
        lbl_pos_pnl.text       = '-'
        lbl_pos_protect.text   = '-'
        lbl_pos_protect_status.text = '-'

        # qty input behavior when flat: follow auto size toggle
        if auto_sz.value:
            qty_in.disable()
        else:
            qty_in.enable()

        # reset sync baseline so next open can set once
        last_backend_qty = None
        return

    # ----- status == 'open': fill fields -----
    qty     = p.get('qty')
    strike  = p.get('strike')
    right   = p.get('right')
    avg     = p.get('avg_price')
    stop_px = p.get('calc_stop_price')
    bid     = p.get('bid')
    ask     = p.get('ask')
    pnl     = p.get('pnl')
    protect_px = p.get('calc_protect_price')
    auto_protect_enabled = p.get('auto_protect')
    protect_ratio = p.get('protect_ratio')
    protect_triggered = p.get('protect_triggered')

    desired_toggle = bool(auto_protect_enabled)
    if bool(auto_protect.value) != desired_toggle:
        _setting_auto_protect_programmatically = True
        auto_protect.value = desired_toggle
        _setting_auto_protect_programmatically = False

    lbl_pos_contracts.text = fmt_plain(qty)
    lbl_pos_strike.text    = fmt_plain(strike)
    lbl_pos_side.text      = ('Call' if right == 'C' else 'Put' if right == 'P' else '-')
    lbl_pos_avg.text       = fmt_plain(avg)
    lbl_pos_stop.text      = fmt_plain(stop_px)
    lbl_pos_bid.text       = fmt_plain(bid)
    lbl_pos_ask.text       = fmt_plain(ask)
    lbl_pos_pnl.text       = (f"{float(pnl):,.2f}" if isinstance(pnl, (int, float)) else fmt_plain(pnl))
    if auto_protect_enabled:
        lbl_pos_protect.text = fmt_plain(protect_px)
        ratio_text = '--'
        if isinstance(protect_ratio, (int, float)):
            ratio_text = f"{float(protect_ratio):g}"
        elif protect_ratio not in (None, 'n/a'):
            ratio_text = str(protect_ratio)
        lbl_pos_protect_status.text = (
            f"Triggered (R:R {ratio_text})" if protect_triggered else f"Active (R:R {ratio_text})"
        )
    else:
        lbl_pos_protect.text = '-'
        lbl_pos_protect_status.text = 'Off'

    # When a position is open we always enable Quantity for SELL usage
    qty_in.enable()

    # ---- set qty input ONLY when backend qty changed ----
    backend_qty = None
    try:
        backend_qty = int(qty) if qty is not None else None
    except Exception:
        backend_qty = None

    if backend_qty is not None and backend_qty != last_backend_qty:
        qty_in.value = str(backend_qty)  # one-time set per change
        last_backend_qty = backend_qty

ui.timer(REFRESH_SEC, poll_snapshot)
ui.timer(REFRESH_SEC, poll_position_size)
ui.timer(REFRESH_SEC, poll_positions)

# ---------------- Actions ----------------
def on_toggle_auto(_):
    # Only apply when flat; if open, qty is enabled for SELL
    if lbl_pos_status.text == 'open':
        return
    if auto_sz.value:
        qty_in.disable(); loss_in.enable(); lbl_proj.text = 'Projected Loss: n/a'
    else:
        qty_in.enable();  loss_in.disable()
auto_sz.on('update:model-value', on_toggle_auto)

async def on_auto_protect_toggle(e):
    global _setting_auto_protect_programmatically
    if _setting_auto_protect_programmatically or lbl_pos_status.text != 'open':
        return

    desired_state = bool(auto_protect.value)
    previous_state = not desired_state
    payload = {"enabled": desired_state}

    if desired_state:
        if not protect_ratio.value:
            ui.notify('Enter R:R Ratio', color='warning')
            _setting_auto_protect_programmatically = True
            auto_protect.value = previous_state
            _setting_auto_protect_programmatically = False
            return
        try:
            ratio_val = float(protect_ratio.value)
        except Exception:
            ui.notify('R:R Ratio must be a number', color='warning')
            _setting_auto_protect_programmatically = True
            auto_protect.value = previous_state
            _setting_auto_protect_programmatically = False
            return
        if ratio_val < 1:
            ui.notify('R:R Ratio must be at least 1', color='warning')
            _setting_auto_protect_programmatically = True
            auto_protect.value = previous_state
            _setting_auto_protect_programmatically = False
            return
        payload["protect_ratio"] = ratio_val

    try:
        r = await client.post(f'{BACKEND_URL}/auto_protect', json=payload)
        try:
            data = r.json()
        except Exception:
            data = {}
        if r.status_code != 200:
            message = data.get('error') if isinstance(data, dict) else None
            if not message:
                message = f'HTTP {r.status_code}'
            ui.notify(message, color='negative')
            _setting_auto_protect_programmatically = True
            auto_protect.value = previous_state
            _setting_auto_protect_programmatically = False
            return

        if desired_state:
            ui.notify('Auto Protect enabled', color='positive')
            if isinstance(data, dict):
                new_ratio = data.get('protect_ratio')
                if isinstance(new_ratio, (int, float)):
                    protect_ratio.value = f'{float(new_ratio):g}'
        else:
            ui.notify('Auto Protect disabled', color='positive')
    except Exception as ex:
        ui.notify(f'Auto Protect update failed: {ex}', color='negative')
        _setting_auto_protect_programmatically = True
        auto_protect.value = previous_state
        _setting_auto_protect_programmatically = False

auto_protect.on('update:model-value', on_auto_protect_toggle)

async def on_side_change(e):
    global current_side, _is_switching_side
    if _is_switching_side or side.value == current_side:
        return
    _is_switching_side = True
    new_right = side.value
    try:
        r = await client.post(f'{BACKEND_URL}/set_right', json={"right": new_right})
        try:
            d = r.json()
        except Exception:
            d = None
        if r.status_code != 200:
            msg = (d.get('error') if isinstance(d, dict) and 'error' in d
                   else f'HTTP {r.status_code}: {r.text[:200]}')
            ui.notify(msg, color='negative')
            side.value = current_side if current_side in ('C','P') else 'C'
            return
        current_side = new_right
        ui.notify(f"Switched to {'Call' if new_right=='C' else 'Put'}", color='positive')
    except Exception as ex:
        ui.notify(f'Switch failed: {ex}', color='negative')
        side.value = current_side if current_side in ('C','P') else 'C'
    finally:
        _is_switching_side = False
side.on('update:model-value', on_side_change)

async def do_buy():
    try:
        if not stop_in.value:
            ui.notify('Enter Stop on SPY', color='warning'); return
        try:
            stop_value = float(stop_in.value)
        except Exception:
            ui.notify('Stop on SPY must be a number', color='warning'); return
        if stop_value < 0:
            ui.notify('Stop on SPY cannot be negative', color='warning'); return
        payload = {"stop_on_spy": stop_value}
        if lbl_pos_status.text != 'open' and auto_sz.value:
            # auto-size only applies when flat
            if not loss_in.value:
                ui.notify('Enter Max loss (USD)', color='warning'); return
            try:
                max_loss_value = float(loss_in.value)
            except Exception:
                ui.notify('Max loss must be a number', color='warning'); return
            if max_loss_value < 0:
                ui.notify('Max loss cannot be negative', color='warning'); return
            payload["max_loss_usd"] = max_loss_value
        else:
            if not qty_in.value:
                ui.notify('Enter Quantity', color='warning'); return
            try:
                qty_value = int(float(qty_in.value))
            except Exception:
                ui.notify('Quantity must be a number', color='warning'); return
            if qty_value < 0:
                ui.notify('Quantity cannot be negative', color='warning'); return
            payload["qty"] = qty_value
        payload["auto_protect"] = bool(auto_protect.value)
        if auto_protect.value:
            if not protect_ratio.value:
                ui.notify('Enter R:R Ratio', color='warning'); return
            try:
                ratio_val = float(protect_ratio.value)
            except Exception:
                ui.notify('R:R Ratio must be a number', color='warning'); return
            if ratio_val < 1:
                ui.notify('R:R Ratio must be at least 1', color='warning'); return
            payload["protect_ratio"] = ratio_val
        r = await client.post(f'{BACKEND_URL}/buy', json=payload)
        d = r.json()
        if r.status_code != 200:
            ui.notify(d.get('error','buy failed'), color='negative'); return
        msg = f"Bought {d.get('filled',0)} @ {d.get('avg_price','?')}"
        ploss = d.get('projected_max_loss_usd')
        if ploss is not None:
            msg += f" | Projected Loss ${ploss}"
        ui.notify(msg, color='positive')
    except Exception as e:
        ui.notify(f'Buy failed: {e}', color='negative')
btn_buy.on('click', do_buy)

async def do_sell():
    try:
        if not qty_in.value:
            ui.notify('Enter Quantity to sell', color='warning'); return
        try:
            qty_value = int(float(qty_in.value))
        except Exception:
            ui.notify('Quantity must be a number', color='warning'); return
        if qty_value < 0:
            ui.notify('Quantity cannot be negative', color='warning'); return
        payload = {"qty": qty_value}
        r = await client.post(f'{BACKEND_URL}/sell', json=payload)
        d = r.json()
        if r.status_code != 200:
            ui.notify(d.get('error','sell failed'), color='negative'); return
        ui.notify(f"Sold {d.get('sold',0)} @ {d.get('avg_price','?')}", color='positive')
    except Exception as e:
        ui.notify(f'Sell failed: {e}', color='negative')
btn_sell.on('click', do_sell)

def start_ui():
    # ui elements already defined above
    ui.run(title='Rapid Options Trader Cockpit for IBKR', window_size=(460, 970), fullscreen=False, reload=False, port=8080)

if __name__ == '__main__':
    start_ui()