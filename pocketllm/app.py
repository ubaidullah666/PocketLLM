"""Loopback-only desktop backend, with an unguessable per-launch session token."""
import argparse
import json
import secrets
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from .engine import Engine
from .gguf import inspect_model
from .hardware import snapshot, engine_devices
from .planner import plan
from .streaming_model import inspect_streaming, plan_streaming, public_model

UI = Path(__file__).parent/'ui'
TOKEN = secrets.token_urlsafe(32)
engine = None
window = None

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):
        pass
    def reply(self,data,status=200):
        raw = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(raw)))
        self.send_header('Cache-Control','no-store')
        self.end_headers()
        self.wfile.write(raw)
    def trusted(self):
        host = f'127.0.0.1:{self.server.server_port}'
        return self.headers.get('Host') == host and self.headers.get('X-Session') == TOKEN and self.headers.get('Origin',f'http://{host}') == f'http://{host}'
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == '/api/state':
            if not self.trusted():
                return self.reply({'error':'Unauthorized'},403)
            return self.reply({'hardware':snapshot(engine.gpu_id,devices=engine.devices,mode=engine.mode),'engine':engine.state()})
        path = {'/':'index.html','/app.js':'app.js','/style.css':'style.css'}.get(parsed.path)
        if not path or self.headers.get('Host') != f'127.0.0.1:{self.server.server_port}':
            return self.reply({'error':'Not found'},404)
        # Bootstrap token is delivered only to the launch URL, never arbitrary requests.
        if path == 'index.html' and urllib.parse.parse_qs(parsed.query).get('session') != [TOKEN]:
            return self.reply({'error':'Open the application using its launch URL.'},403)
        raw = (UI/path).read_bytes()
        if path == 'index.html':
            raw = raw.replace(b'__SESSION__',TOKEN.encode())
        self.send_response(200)
        self.send_header('Content-Type',{'index.html':'text/html; charset=utf-8','app.js':'text/javascript','style.css':'text/css'}[path])
        self.send_header('Content-Length',str(len(raw)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(raw)
    def do_POST(self):
        if not self.trusted():
            return self.reply({'error':'Unauthorized'},403)
        try:
            length = int(self.headers.get('Content-Length','0'))
            if not 0 < length <= 1_000_000:
                raise ValueError('Request too large or empty.')
            data = json.loads(self.rfile.read(length))
            if self.path == '/api/devices':
                devices=engine_devices(data.get('executable','')) if data.get('mode','resident')=='resident' else []
                return self.reply({'hardware':snapshot(data.get('gpu'),devices=devices,mode=data.get('mode','resident'))})
            if self.path == '/api/inspect':
                if engine.proc:
                    raise ValueError('Unload the current model before changing the plan.')
                streaming=data.get('mode','resident')=='streaming'
                model = (inspect_streaming if streaming else inspect_model)(data['path'])
                p = (plan_streaming if streaming else plan)(model,snapshot(data.get('gpu'),devices=engine_devices(data.get('executable','')) if not streaming else [],mode=data.get('mode','resident')),float(data['ram']),float(data['vram']),int(data['context']),**({'profile':data.get('profile','smart')} if streaming else {}))
                return self.reply({'model':public_model(model),'plan':p})
            if self.path == '/api/load':
                return self.reply({'plan':engine.launch(data['path'],data.get('executable',''),float(data['ram']),float(data['vram']),int(data['context']),mode=data.get('mode','resident'),profile=data.get('profile','smart'),gpu_id=data.get('gpu'))})
            if self.path == '/api/unload':
                engine.stop()
                return self.reply({'ok':True})
            if self.path == '/api/stop':
                engine.cancel_generation()
                return self.reply({'ok':True})
            if self.path == '/api/pick':
                if window is None:
                    raise ValueError('Native file picker is available in desktop mode. Paste a full local path here.')
                import webview
                types = ('GGUF models (*.gguf)',) if data.get('kind') == 'model' else ('All files (*.*)',)
                if data.get('kind')=='folder':
                    result=window.create_file_dialog(webview.FileDialog.FOLDER,allow_multiple=False)
                else:
                    result = window.create_file_dialog(webview.FileDialog.OPEN,allow_multiple=False,file_types=types)
                return self.reply({'path':result[0] if result else ''})
            if self.path == '/api/chat':
                messages = data.get('messages')
                if not isinstance(messages,list) or not 1 <= len(messages) <= 200:
                    raise ValueError('Invalid conversation.')
                if any(not isinstance(m,dict) or m.get('role') not in ('user','assistant') or not isinstance(m.get('content'),str) for m in messages):
                    raise ValueError('Invalid message format.')
                self.send_response(200)
                self.send_header('Content-Type','text/event-stream')
                self.send_header('Cache-Control','no-cache')
                self.end_headers()
                try:
                    for item in engine.stream(messages):
                        self.wfile.write(('data: '+json.dumps(item)+'\n\n').encode())
                        self.wfile.flush()
                except (BrokenPipeError,ConnectionResetError):
                    engine.cancel.set()
                except Exception as exc:
                    try:
                        self.wfile.write(('data: '+json.dumps({'type':'error','error':str(exc)})+'\n\n').encode())
                    except OSError:
                        pass
                return
            self.reply({'error':'Not found'},404)
        except Exception as exc:
            self.reply({'error':str(exc)},400)

class Server(ThreadingHTTPServer):
    daemon_threads = True
    def get_request(self):
        sock,addr = super().get_request()
        sock.settimeout(240)
        return sock,addr

def main():
    global engine,window
    parser = argparse.ArgumentParser()
    parser.add_argument('--browser',action='store_true',help='Use a browser instead of a native window')
    args = parser.parse_args()
    engine = Engine()
    server = Server(('127.0.0.1',0),Handler)
    thread = threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}/?session={TOKEN}'
    try:
        if args.browser:
            print('PocketLLM running. Press Ctrl+C to exit. Open:',url)
            webbrowser.open(url)
            threading.Event().wait()
        else:
            import webview
            window = webview.create_window('PocketLLM — Local intelligence',url,width=1360,height=900,min_size=(920,680),background_color='#0b0e12')
            webview.start()
    except KeyboardInterrupt:
        pass
    finally:
        engine.closed = True
        engine.stop()
        server.shutdown()

if __name__ == '__main__':
    main()
