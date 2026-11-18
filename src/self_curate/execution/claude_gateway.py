"""Per-service authenticated Anthropic gateway with episode accounting.

Bind only to the approved private Docker-facing address. vLLM itself stays on
host loopback. No caller can choose a different model or sampling policy.
"""
from __future__ import annotations
import json
import uuid
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.error
import urllib.request
from urllib.parse import urlsplit, parse_qs
from self_curate.training.recovery import atomic_json as save


def http_json(url, data=None, timeout=60, key=None, generation=None, budget=None):
    request = urllib.request.Request(url, data=None if data is None else json.dumps(data).encode(),
                                    headers={'Content-Type':'application/json', **({'Authorization':'Bearer '+key} if key else {}),
                                             **({'X-Self-Curate-Generation':json.dumps(generation,separators=(',',':'))} if generation is not None else {})})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    from .supervised_http import upstream_response
    context=(upstream_response(url,request.data,dict(request.header_items()),
                cancel=budget.cancel, deadline=budget.started+budget.cfg['agent']['wall_seconds'])
             if budget is not None else opener.open(request,timeout=timeout))
    with context as response:
        content=response.read()
        return json.loads(content) if content else {}


class Budget:
    def __init__(self, model, cfg, path):
        self.model, self.cfg, self.path = model, cfg, path
        self.started = time.monotonic()
        self.input_tokens = self.output_tokens = self.requests = 0
        self.stop_reason = None
        self.lock = threading.Lock()
        self.cancel = threading.Event()
        self.incomplete_usage = False
        self.records = []
        self.session_id = uuid.uuid4().hex

    def prepare(self, body, input_tokens):
        if body.get('model') != self.model:
            raise ValueError('Unexpected model route')
        if self.stop_reason:
            raise ValueError(self.stop_reason)
        if time.monotonic()-self.started >= self.cfg['agent']['wall_seconds']:
            self.stop_reason = 'WALL_LIMIT'
            raise ValueError(self.stop_reason)
        remaining = self.cfg['agent']['total_output_tokens']-self.output_tokens
        context = self.cfg['generation']['context_length']-input_tokens
        requested=body.get('max_tokens',self.cfg['generation']['max_new_tokens'])
        if type(requested) is not int or requested<=0:
            raise ValueError('Positive integer max_tokens required')
        maximum = min(requested,self.cfg['generation']['max_new_tokens'], remaining, context)
        if maximum <= 0:
            self.stop_reason = 'TOKEN_LIMIT' if remaining <= 0 else 'CONTEXT_LIMIT'
            raise ValueError(self.stop_reason)
        body = dict(body)
        # This is the actual generation budget, including thinking tokens.
        body['max_tokens'] = maximum
        g = self.cfg['generation']
        body.update(temperature=g['temperature'], top_p=g['top_p'], top_k=g['top_k'])
        body['chat_template_kwargs'] = {'enable_thinking':g['thinking'],
            'preserve_thinking':g['preserve_thinking'], 'reasoning_effort':g['reasoning_effort']}
        body.update(min_p=g.get('min_p',0.0),presence_penalty=g.get('presence_penalty',0.0),repetition_penalty=g.get('repetition_penalty',1.0))
        return body

    def finish(self, input_tokens, output_tokens, body):
        if type(output_tokens) is not int or output_tokens < 0 or output_tokens > body['max_tokens']:
            self.stop_reason = 'INFRASTRUCTURE_ERROR'
            raise ValueError('Missing/invalid actual generation usage')
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.requests += 1
        if self.output_tokens >= self.cfg['agent']['total_output_tokens']:
            self.stop_reason = 'TOKEN_LIMIT'
        self.records.append({'session_id':self.session_id,'request_sequence':self.requests,
             'started_monotonic':getattr(self,'current_started',None), 'finished_monotonic':time.monotonic(), 'input_tokens':input_tokens, 'output_tokens':output_tokens,
             'max_tokens':body['max_tokens'], 'model':body['model'],
             'temperature':body['temperature'], 'top_p':body['top_p'], 'top_k':body['top_k'],
             'chat_template_kwargs':body['chat_template_kwargs'],
             'history_thinking_blocks':sum(b.get('type')=='thinking' for m in body.get('messages',[])
                  if isinstance(m.get('content'),list) for b in m['content']),
             'messages':len(body.get('messages',[])), 'capture_id': self.current_capture_id})
        self.persist()

    def snapshot(self, timeout=30):
        """Wait until the last HTTP handler commits usage before SDK finalization."""
        if not self.lock.acquire(timeout=timeout):
            raise TimeoutError('Inference handler has not settled; reconcile before retry')
        try:
            return {'requests':self.requests,'records':list(self.records),
                    'incomplete_usage':self.incomplete_usage,'stop_reason':self.stop_reason}
        finally:
            self.lock.release()

    def persist(self):
        save(self.path, {'session_id':self.session_id,'model':self.model, 'input_tokens':self.input_tokens,
             'output_tokens':self.output_tokens, 'requests':self.requests,
             'stop_reason':self.stop_reason, 'records':self.records,
             'incomplete_usage':self.incomplete_usage})


class Gateway:
    def __init__(self, address, upstream, cfg, upstream_key=None):
        self.upstream, self.cfg = upstream.rstrip('/'), cfg
        self.upstream_key = upstream_key
        self.sessions = {}
        owner = self
        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.0'
            def log_message(self, *args):
                pass  # Never log authorization or task content to shared HTTP logs.
            def do_GET(self):
                if self.path != '/health':
                    self.send_error(404)
                    return
                try:
                    http_json(owner.upstream+'/health')
                    self.send_response(200); self.end_headers(); self.wfile.write(b'{}')
                except Exception:
                    self.send_error(503)
            def do_POST(self):
                route=urlsplit(self.path)
                if route.query and parse_qs(route.query,keep_blank_values=True)!={'beta':['true']}:
                    self.send_error(400); return
                self.path=route.path
                key = self.headers.get('x-api-key') or self.headers.get('Authorization','').removeprefix('Bearer ')
                budget = owner.sessions.get(key)
                if not budget:
                    self.send_error(403); return
                if self.path not in {'/v1/messages','/v1/messages/count_tokens'}:
                    self.send_error(404); return
                started = False
                with budget.lock:
                    try:
                        length = int(self.headers.get('Content-Length','0'))
                        if not 0 < length <= 64*1024*1024:
                            raise ValueError('Invalid request size')
                        body = json.loads(self.rfile.read(length))
                        if body.get('model') != budget.model:
                            raise ValueError('Model route mismatch')
                        body['chat_template_kwargs'] = {'enable_thinking':owner.cfg['generation']['thinking'],
                            'preserve_thinking':owner.cfg['generation']['preserve_thinking'],
                            'reasoning_effort':owner.cfg['generation']['reasoning_effort']}
                        count_body = {k:v for k,v in body.items() if k in {'model','messages','system','tools','tool_choice','chat_template_kwargs'}}
                        count = http_json(owner.upstream+'/v1/messages/count_tokens', count_body, key=owner.upstream_key,generation=owner.cfg['generation'],budget=budget)['input_tokens']
                        if type(count) is not int or count < 0:
                            raise ValueError('Invalid input token count')
                        if self.path.endswith('/count_tokens'):
                            self.send_response(200); self.end_headers()
                            self.wfile.write(json.dumps({'input_tokens':count}).encode()); return
                        body = budget.prepare(body, count)
                        budget.current_capture_id = uuid.uuid4().hex
                        budget.current_started = time.monotonic()
                        budget.persist()
                        save(budget.path.with_name('request-%06d.json' % budget.requests), body)
                        request = urllib.request.Request(owner.upstream+'/v1/messages',
                            data=json.dumps(body).encode(), headers={'Content-Type':'application/json', 'X-Request-Id':budget.current_capture_id, 'X-Self-Curate-Seed':str(owner.cfg['generation'].get('seed',0)+budget.requests), 'X-Self-Curate-Generation':json.dumps(owner.cfg['generation'],separators=(',',':')), **({'Authorization':'Bearer '+owner.upstream_key} if owner.upstream_key else {})})
                        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                        budget.incomplete_usage=True
                        from .supervised_http import upstream_response
                        with upstream_response(request.full_url,request.data,dict(request.header_items()),
                                cancel=budget.cancel, deadline=budget.started+owner.cfg['agent']['wall_seconds']) as response:
                            self.send_response(200)
                            raw_path=budget.path.with_name('response-%06d.raw' % budget.requests)
                            raw_log=raw_path.open('xb')
                            streaming = bool(body.get('stream'))
                            self.send_header('Content-Type','text/event-stream' if streaming else 'application/json')
                            self.end_headers(); started = True
                            output = None
                            if streaming:
                                complete = False
                                for line in response:
                                    if budget.cancel.is_set():
                                        budget.incomplete_usage=True
                                        raise ValueError('Supervised episode cancellation')
                                    if line.startswith(b'data: '):
                                        event = json.loads(line[6:])
                                        if event.get('type') == 'error':
                                            raise ValueError('Upstream stream error')
                                        if event.get('type') == 'message_delta':
                                            output = event.get('usage',{}).get('output_tokens',output)
                                        if event.get('type') == 'message_stop':
                                            complete = True
                                    raw_log.write(line); raw_log.flush()
                                    self.wfile.write(line); self.wfile.flush()
                                if not complete:
                                    raise ValueError('Incomplete upstream stream')
                            else:
                                content = response.read()
                                output = json.loads(content).get('usage',{}).get('output_tokens')
                                raw_log.write(content); raw_log.flush()
                                self.wfile.write(content)
                            raw_log.close()
                            budget.finish(count, output, body)
                            budget.incomplete_usage=False
                            budget.persist()
                    except Exception as exc:
                        if 'raw_log' in locals() and not raw_log.closed:
                            raw_log.close()
                        if started:
                            budget.incomplete_usage=True
                        if not budget.stop_reason:
                            budget.stop_reason = 'INFRASTRUCTURE_ERROR'
                        budget.persist()
                        if not started:
                            self.send_error(429 if budget.stop_reason.endswith('LIMIT') else 502,
                                            'Local inference gate rejected request')
                        self.close_connection = True
        self.server = ThreadingHTTPServer(address, Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def register(self, model, path):
        token = secrets.token_urlsafe(32)
        budget = Budget(model, self.cfg, path)
        self.sessions[token] = budget
        return token, budget

    def close(self):
        for budget in self.sessions.values():
            budget.cancel.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
