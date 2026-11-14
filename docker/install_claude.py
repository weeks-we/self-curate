"""Workstation build only; require independently obtained official checksum."""
from pathlib import Path
import os,re,urllib.request,hashlib,subprocess
expected=os.environ.get('CLAUDE_SHA256','')
assert re.fullmatch('[0-9a-f]{64}',expected),'Official linux-x64 SHA256 required'
data=urllib.request.urlopen('https://downloads.claude.ai/claude-code-releases/2.1.153/linux-x64/claude',timeout=120).read()
assert hashlib.sha256(data).hexdigest()==expected,'Claude checksum mismatch'
path=Path('/opt/self-curate-sdk/bin/claude-pinned');path.write_bytes(data);path.chmod(0o755)
assert subprocess.check_output([str(path),'--version'],text=True).split()[0]=='2.1.153'
Path('/opt/self-curate-sdk/claude-sha256.txt').write_text(expected+'\n')
