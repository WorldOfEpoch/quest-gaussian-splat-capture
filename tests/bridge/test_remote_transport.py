import base64
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi.testclient import TestClient
from bridge.remote_transport import create_relay,allowed,CHUNK


def test_relay_reserves_capacity_before_slow_authenticated_bodies():
    import httpx
    async def scenario():
        ph='phone-fixture-'+'p'*40;home='home-fixture-'+'h'*40
        app=create_relay(ph,home);bodies=[];four_started=asyncio.Event();release=asyncio.Event()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://fixture') as c:
            poll=asyncio.create_task(c.get('/relay/next',headers={'Authorization':'Bearer '+home}))
            await asyncio.sleep(.01)
            async def slow_body():
                bodies.append(True)
                if len(bodies)==4:four_started.set()
                yield b'{'
                await release.wait()
                yield b'}'
            uploads=[asyncio.create_task(c.post('/api/v1/projects',headers={'Authorization':'Bearer '+ph},content=slow_body())) for _ in range(4)]
            try:
                await asyncio.wait_for(four_started.wait(),2)
                fifth=await asyncio.wait_for(c.post('/api/v1/projects',headers={'Authorization':'Bearer '+ph},content=b'{}'),2)
                assert fifth.status_code==429
            finally:
                for task in [poll,*uploads]:task.cancel()
                await asyncio.gather(poll,*uploads,return_exceptions=True)
    asyncio.run(scenario())


def test_home_transport_never_follows_authorized_redirect():
    from bridge.remote_transport import _open
    from urllib.request import Request
    from urllib.error import HTTPError
    from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
    import threading
    import pytest
    reached=[]
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            reached.append(self.path)
            self.send_response(302);self.send_header('Location','/redirect-target');self.end_headers()
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        request=Request(f'http://127.0.0.1:{server.server_port}/origin',headers={'Authorization':'Bearer synthetic-fixture-token'})
        with pytest.raises(HTTPError) as error:_open(request,2)
        assert error.value.code==302 and reached==['/origin']
    finally:server.shutdown();server.server_close();thread.join(2)


def test_relay_auth_allowlist_outbound_response_and_offline():
    ph='phone-fixture-'+'p'*40;home='home-fixture-'+'h'*40
    with TestClient(create_relay(ph,home)) as c:
        assert c.get('/api/v1/projects').status_code==401
        assert c.get('/api/v1/projects',headers={'Authorization':'Bearer '+ph}).status_code==503
        assert c.get('/relay/next',headers={'Authorization':'Bearer '+ph}).status_code==401
        assert c.get('/api/v1/projects/abc/captures',headers={'Authorization':'Bearer '+ph}).status_code==404
        with ThreadPoolExecutor(2) as pool:
            poll=pool.submit(c.get,'/relay/next',headers={'Authorization':'Bearer '+home})
            import time;time.sleep(.1)
            response=pool.submit(c.get,'/api/v1/projects',headers={'Authorization':'Bearer '+ph})
            packet=poll.result(timeout=3).json();rid=packet['id'];assert packet['method']=='GET'
            assert 'authorization' not in packet['headers']
            for seq,frame in enumerate([{'kind':'start','status':200,'headers':{'Content-Type':'application/json'}},
                                       {'kind':'data','body':base64.b64encode(b'{"projects":[]}').decode()}, {'kind':'done'}]):
                r=c.post('/relay/'+rid+'/response',headers={'Authorization':'Bearer '+home},json={**frame,'sequence':seq});assert r.status_code==200
            assert response.result(timeout=3).json()=={'projects':[]}
        assert c.get('/api/v1/ui',headers={'Authorization':'Bearer '+ph}).status_code==404
        assert not allowed('GET','/ui') and not allowed('GET','/api/v1/sessions/abc/captures')
