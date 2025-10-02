# app_shell.py
import webview

webview.create_window(
    title='ROTC',
    url='https://rotc.yientut.com',   # your Cloudflare URL
    width=460, height=970, resizable=True
)
webview.start()