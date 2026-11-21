"""Small artifact helpers; immutable identities and explicit synthetic/real domains."""
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_replace(source,target):
    """Preserve atomic publication while tolerating short Windows reader/indexer locks."""
    import time
    for attempt in range(6):
        try:
            os.replace(source,target)
            return
        except PermissionError as exc:
            if os.name!='nt' or getattr(exc,'winerror',None) not in (5,32,33) or attempt==5:
                raise
            time.sleep(.025*(2**attempt))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
    with temporary.open('w',encoding='utf-8') as stream:
        stream.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
        stream.flush();os.fsync(stream.fileno())
    atomic_replace(temporary, path)


def rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def jsonl(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush();os.fsync(stream.fileno())
    atomic_replace(temp, path)


def run_id():
    return datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S') + '-' + uuid4().hex[:10]


def safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', value) or value in ('.', '..'):
        raise ValueError('Unsafe artifact identifier')
    return value


def event(root, name, **values):
    path = Path(root) / 'events.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    # Each job owns its own log; distributed ranks log separately.
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'time_utc': datetime.now(timezone.utc).isoformat(),
                                 'event': name, **values}, ensure_ascii=False) + '\n')
