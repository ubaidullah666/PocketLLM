#!/usr/bin/env python3
"""Protocol test fixture only. Does not perform inference."""
import json,sys,time
from http.server import BaseHTTPRequestHandler,HTTPServer
if '--help' in sys.argv:
    print('--fit --load-mode --no-kv-offload --no-repack --no-op-offload');sys.exit()
port=int(sys.argv[sys.argv.index('--port')+1])
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_GET(self):
        self.send_response(200);self.end_headers();self.wfile.write(b'{"status":"ok"}')
    def do_POST(self):
        data=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.send_response(200);self.end_headers()
        if self.path=='/apply-template':self.wfile.write(b'{"prompt":"hello"}')
        elif self.path=='/tokenize':self.wfile.write(b'{"tokens":[1,2,3]}')
        else:
            for token in ['Hello',' from',' fixture']:
                self.wfile.write(('data: '+json.dumps({'choices':[{'delta':{'content':token}}]})+'\n\n').encode());self.wfile.flush()
            self.wfile.write(b'data: [DONE]\n\n')
print('offloaded 0/3 layers to GPU',flush=True)
HTTPServer(('127.0.0.1',port),Handler).serve_forever()
