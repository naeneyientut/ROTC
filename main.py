# main.py
import sys, asyncio, threading, time, contextlib, urllib.request

# 1) Safer event-loop policy on Windows (important for frozen builds)
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

BACKEND_URLS = [
    "http://127.0.0.1:8001/health",
    "http://127.0.0.1:8001/snapshot",
]

def run_backend():
    # Each thread needs its own loop; set it BEFORE importing the backend.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    import backend_ib  # import AFTER loop exists (ib_insync/websockets rely on this)
    for name in ("main", "start_backend", "run", "start"):
        fn = getattr(backend_ib, name, None)
        if callable(fn):
            fn()  # no args; you hard-coded host/port
            return
    app = getattr(backend_ib, "app", None)
    if app is not None and hasattr(app, "run"):
        app.run()
        return
    print("[Launcher] No backend entry found.", file=sys.stderr)

def wait_until_ready(timeout=15.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        for url in BACKEND_URLS:
            try:
                with contextlib.closing(urllib.request.urlopen(url, timeout=1.5)) as r:
                    if 200 <= r.status < 300:
                        return True
            except Exception:
                pass
        time.sleep(0.25)
    return False

def run_frontend():
    # Import late so NiceGUI builds UI after backend is up.
    import frontend_ui
    for name in ("main", "start_ui", "run", "start"):
        fn = getattr(frontend_ui, name, None)
        if callable(fn):
            fn()
            return
    ui = getattr(frontend_ui, "ui", None)
    if ui is not None and hasattr(ui, "run"):
        ui.run(reload=False)
        return
    print("[Launcher] No frontend entry found.", file=sys.stderr)

if __name__ == "__main__":
    t = threading.Thread(target=run_backend, name="backend-thread", daemon=True)
    t.start()
    if not wait_until_ready():
        print("[Launcher] Backend not responding yet; starting UI anyway.")
    run_frontend()
