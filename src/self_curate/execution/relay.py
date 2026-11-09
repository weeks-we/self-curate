"""Docker-bridge inference relay: scoped model grants, no adapter administration."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from contextlib import contextmanager
import argparse, hashlib, json, os, secrets, time, urllib.request, urllib.error
from self_curate.contracts.types import require
from self_curate.training.recovery import atomic_json, read_json


@contextmanager
def trial_grant(root, model_name, *, timeout):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        root.chmod(0o700)
    token = secrets.token_urlsafe(32)
    path = root / (hashlib.sha256(token.encode()).hexdigest() + ".json")
    atomic_json(
        path, {"model_name": model_name, "expires_at": time.time() + timeout + 300}
    )
    try:
        yield token
    finally:
        atomic_json(path, {"model_name": model_name, "expires_at": 0, "closed": True})


def handler(target, grant_root, key_env):
    class Relay(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *args):
            pass

        def grant(self):
            authorization = self.headers.get("Authorization", "")
            if not authorization.startswith("Bearer "):
                return None
            token = authorization[len("Bearer ") :]
            path = Path(grant_root) / (
                hashlib.sha256(token.encode()).hexdigest() + ".json"
            )
            if not path.exists():
                return None
            value = read_json(path)
            return (
                value
                if value["expires_at"] > time.time() and not value.get("closed")
                else None
            )

        def respond(self, status, value):
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            grant = self.grant()
            if grant is None:
                self.respond(401, {"error": "invalid or expired trial grant"})
                return
            if self.path != "/v1/models":
                self.respond(403, {"error": "inference-only endpoint"})
                return
            self.respond(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": grant["model_name"],
                            "object": "model",
                            "owned_by": "self-curate",
                        }
                    ],
                },
            )

        def do_POST(self):
            grant = self.grant()
            if grant is None:
                self.respond(401, {"error": "invalid or expired trial grant"})
                return
            if self.path not in ("/v1/chat/completions", "/v1/completions"):
                self.respond(403, {"error": "inference-only endpoint"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.respond(400, {"error": "invalid content length"})
                return
            if not 0 < length <= 64 * 1024**2:
                self.respond(413, {"error": "request body exceeds relay budget"})
                return
            raw = self.rfile.read(length)
            try:
                value = json.loads(raw)
            except (ValueError, TypeError):
                self.respond(400, {"error": "invalid JSON body"})
                return
            if type(value) is not dict or value.get("model") != grant["model_name"]:
                self.respond(403, {"error": "trial grant is bound to another model"})
                return
            key = os.environ.get(key_env)
            if not key:
                self.respond(503, {"error": "host serving credential unavailable"})
                return
            request = urllib.request.Request(
                target.rstrip("/") + self.path,
                data=raw,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + key,
                },
            )
            try:
                response = urllib.request.urlopen(request, timeout=7200)
            except urllib.error.HTTPError as exc:
                response = exc
            except OSError:
                self.respond(502, {"error": "upstream inference unavailable"})
                return
            with response:
                self.send_response(response.status)
                self.send_header(
                    "Content-Type",
                    response.headers.get("Content-Type", "application/json"),
                )
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                try:
                    while True:
                        block = response.read1(65536)
                        if not block:
                            break
                        self.wfile.write(block)
                        self.wfile.flush()
                except (OSError, TimeoutError):
                    self.close_connection = True

    return Relay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--target-port", type=int, required=True)
    parser.add_argument("--grants", required=True)
    parser.add_argument("--key-env", required=True)
    parser.add_argument("--execution", choices=["server"], required=True)
    args = parser.parse_args()
    import ipaddress

    address = ipaddress.ip_address(args.bind)
    require(
        address.is_private and not address.is_unspecified,
        "Relay must bind a private Docker bridge address",
    )
    server = ThreadingHTTPServer(
        (args.bind, args.port),
        handler("http://127.0.0.1:" + str(args.target_port), args.grants, args.key_env),
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
