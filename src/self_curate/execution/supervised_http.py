"""Socket-level cancellation, including silent upstream headers and bodies."""
import http.client, socket, threading, time, io, select
from contextlib import contextmanager
from urllib.parse import urlsplit

@contextmanager
def upstream_response(url, data, headers, *, cancel, deadline):
    parsed=urlsplit(url)
    cls=http.client.HTTPSConnection if parsed.scheme=='https' else http.client.HTTPConnection
    connection=cls(parsed.hostname,parsed.port,timeout=max(.01,deadline-time.monotonic()))
    done=threading.Event(); aborted=[]; sockets=[]
    def watch():
        while not done.wait(.05):
            if cancel.is_set() or time.monotonic()>=deadline:
                aborted.append('cancelled' if cancel.is_set() else 'deadline')
                sock=sockets[0] if sockets else connection.sock
                if sock is not None:
                    try: sock.shutdown(socket.SHUT_RDWR); sock.close()
                    except OSError: pass
                return
    watcher=threading.Thread(target=watch,daemon=True)
    try:
        # DNS/connect is bounded by the remaining deadline; once connected the
        # watcher owns interruption even before response headers arrive.
        connection.connect()
        sockets.append(connection.sock)
        class Reader(io.RawIOBase):
            def readable(self): return True
            def readinto(self, buffer):
                while True:
                    if cancel.is_set(): raise TimeoutError('cancelled')
                    remaining=deadline-time.monotonic()
                    if remaining<=0: raise TimeoutError('deadline')
                    if getattr(sockets[0], 'pending', lambda: 0)() or select.select([sockets[0]],[],[],min(.1,remaining))[0]:
                        return sockets[0].recv_into(buffer)
        class SocketView:
            def close(self): pass  # Context owns the socket while response is readable.
            def makefile(self, *args, **kwargs): return io.BufferedReader(Reader())
            def __getattr__(self,name): return getattr(sockets[0],name)
        connection.sock=SocketView()
        watcher.start()
        connection.request('POST',parsed.path or '/',body=data,headers=headers)
        response=connection.getresponse()
        if not 200<=response.status<300:
            raise RuntimeError('Upstream HTTP '+str(response.status))
        yield response
        if aborted: raise TimeoutError('Upstream '+aborted[0])
    finally:
        if 'response' in locals(): response.close()
        done.set()
        connection.close()
        for sock in sockets: sock.close()
        if watcher.ident is not None: watcher.join(timeout=1)
