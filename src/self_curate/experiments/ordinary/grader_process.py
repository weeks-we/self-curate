"""Owned CPU-container lifecycle; cancel the named grader without touching GPU lanes."""
import subprocess,time,uuid
from pathlib import Path


def run(command,log,*,cancel=None):
    name='sc-grade-'+uuid.uuid4().hex[:20]
    command=list(command);command[2:2]=['--name',name]
    with Path(log).open('a',encoding='utf-8') as stream:
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                if cancel is not None and cancel.is_set():
                    subprocess.run(['docker','kill',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=20)
                    raise RuntimeError('CPU grader cancelled: '+name)
                time.sleep(.25)
            if process.returncode:raise RuntimeError('CPU grader failed; inspect '+str(log))
        finally:
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=20)
                except subprocess.TimeoutExpired:process.kill();process.wait()
