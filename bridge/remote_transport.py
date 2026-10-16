"""Staged outbound-only transport to the existing homebase, disabled by default.

No filesystem or scan processing exists at the relay. Tokens are distinct,
explicitly supplied at deployment; the homebase bearer stays on the home PC.
"""
from __future__ import annotations
import asyncio
import base64
import hmac
import json
import os
import re
import time
import uuid
from urllib.request import Request, build_opener, HTTPRedirectHandler
from urllib.error import HTTPError
from urllib.parse import urlparse
from fastapi import FastAPI, Request as WebRequest, HTTPException
from fastapi.responses import StreamingResponse

CHUNK=1024*1024
ID=r'[0-9a-fA-F-]{36}'


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_args):
        return None


def _open(request, timeout):
    # Never carry scanner credentials to a redirected origin.
    return build_opener(_NoRedirect).open(request,timeout=timeout)


def allowed(method, path):
    routes = [
        ('GET',r'/api/v1/projects'),('POST',r'/api/v1/projects'),('POST',r'/api/v1/sessions'),
        ('GET',rf'/api/v1/projects/{ID}/sessions'),('GET',rf'/api/v1/sessions/{ID}'),
        ('GET',rf'/api/v1/mobile/sessions/{ID}/status'),('POST',rf'/api/v1/mobile/sessions/{ID}/update'),
        ('POST',rf'/api/v1/mobile/sessions/{ID}/uploads'),
        ('GET',rf'/api/v1/mobile/sessions/{ID}/uploads/{ID}'),
        ('PUT',rf'/api/v1/mobile/sessions/{ID}/uploads/{ID}'),
        ('POST',rf'/api/v1/mobile/sessions/{ID}/uploads/{ID}/commit'),
        ('GET',rf'/api/v1/projects/{ID}/reconstructions/{ID}/(splat|quest-splat)'),
        ('GET',rf'/api/v1/projects/{ID}/reconstructions/{ID}/meshes(/{ID})?')]
    return any(method==m and re.fullmatch(p,path) for m,p in routes)


def create_relay(phone_token: str, home_token: str):
    if min(len(phone_token),len(home_token))<32 or phone_token==home_token:
        raise ValueError('Distinct explicitly approved private phone and relay credentials are required')
    app=FastAPI(docs_url=None,redoc_url=None,openapi_url=None)
    work=asyncio.Queue(maxsize=4); pending={}; last_poll=[0.0]

    def auth(request,token):
        if not hmac.compare_digest(request.headers.get('authorization',''),'Bearer '+token):
            raise HTTPException(401,'Unauthorized')

    async def body(request,limit):
        result=bytearray()
        async for chunk in request.stream():
            result.extend(chunk)
            if len(result)>limit:raise HTTPException(413,'Transport budget exceeded')
        return bytes(result)

    @app.get('/relay/next')
    async def next_work(request:WebRequest):
        auth(request,home_token);last_poll[0]=time.monotonic()
        try:
            while True:
                packet=await asyncio.wait_for(work.get(),25)
                if packet['id'] in pending:return packet
        except asyncio.TimeoutError:return {'idle':True}

    @app.post('/relay/{request_id}/response')
    async def home_response(request_id:uuid.UUID,request:WebRequest):
        auth(request,home_token)
        entry=pending.get(str(request_id))
        if not entry:raise HTTPException(404,'Request closed')
        packet=json.loads(await body(request,CHUNK*2))
        if packet.get('sequence')!=entry['sequence']:raise HTTPException(409,'Response sequence mismatch')
        if packet.get('kind') not in {'start','data','done'}:raise HTTPException(422,'Invalid response frame')
        if packet['kind']=='data':
            try:payload=base64.b64decode(packet['body'],validate=True)
            except (ValueError,KeyError):raise HTTPException(422,'Invalid response data')
            if len(payload)>CHUNK:raise HTTPException(413,'Response chunk too large')
        await asyncio.wait_for(entry['queue'].put(packet),60)
        entry['sequence']+=1
        return {'sequence':entry['sequence']}

    @app.api_route('/api/v1/{path:path}',methods=['GET','POST','PUT','PATCH','DELETE'])
    async def forward(path:str,request:WebRequest):
        auth(request,phone_token);path='/api/v1/'+path
        if request.url.query or not allowed(request.method,path):raise HTTPException(404,'Unavailable route')
        if time.monotonic()-last_poll[0]>60:raise HTTPException(503,'Homebase offline; retry durable phone queue')
        if len(pending)>=4:raise HTTPException(429,'Relay busy')
        # Reserve without yielding: simultaneous slow request bodies must also
        # consume the four-request budget, rather than all passing its check.
        rid=str(uuid.uuid4());q=asyncio.Queue(maxsize=2);pending[rid]={'queue':q,'sequence':0}
        try:
            payload=await asyncio.wait_for(body(request,CHUNK),60)
            headers={k:v for k,v in request.headers.items() if k.lower() in {'content-type','upload-offset','upload-chunk-sha256'}}
            work.put_nowait({'id':rid,'method':request.method,'path':path,'headers':headers,'body':base64.b64encode(payload).decode()})
            start=await asyncio.wait_for(q.get(),60)
            if start['kind']!='start' or not 200<=int(start['status'])<=599:raise HTTPException(502,'Invalid upstream response')
        except BaseException:
            pending.pop(rid,None);raise
        async def stream():
            total=0
            try:
                while True:
                    packet=await asyncio.wait_for(q.get(),90)
                    if packet['kind']=='done':break
                    if packet['kind']!='data':raise RuntimeError('Unexpected relay frame')
                    chunk=base64.b64decode(packet['body'],validate=True);total+=len(chunk)
                    if total>128*CHUNK:raise RuntimeError('Relay response budget exceeded')
                    yield chunk
            finally:pending.pop(rid,None)
        safe={k:v for k,v in start.get('headers',{}).items() if k.lower() in {'content-type','content-length','etag','cache-control','x-content-type-options'}}
        safe['Cache-Control']='private, no-store'
        return StreamingResponse(stream(),status_code=start['status'],headers=safe)
    return app


def relay_request(origin,token,method,path,body=None):
    data=json.dumps(body).encode() if body is not None else None
    request=Request(origin+path,data=data,method=method,headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
    with _open(request,timeout=90) as response:return json.loads(response.read(2*CHUNK))


def run_home(relay_origin,relay_token,bridge_token,home_origin='http://127.0.0.1:8765'):
    remote=urlparse(relay_origin);home=urlparse(home_origin)
    if remote.scheme!='https' or not remote.hostname or remote.username or remote.path not in {'','/'}:
        raise ValueError('A dedicated approved public HTTPS origin is required')
    if home.scheme!='http' or home.hostname!='127.0.0.1' or home.port!=8765:
        raise ValueError('Home relay must target this existing loopback bridge')
    while True:
        try:
            packet=relay_request(relay_origin,relay_token,'GET','/relay/next')
            if packet.get('idle'):continue
            if not allowed(packet['method'],packet['path']):continue
            payload=base64.b64decode(packet['body'],validate=True)
            if len(payload)>CHUNK:continue
            headers={k:v for k,v in packet['headers'].items() if k.lower() in {'content-type','upload-offset','upload-chunk-sha256'}}
            headers['Authorization']='Bearer '+bridge_token
            request=Request(home_origin+packet['path'],data=payload if packet['method']!='GET' else None,method=packet['method'],headers=headers)
            try:response=_open(request,timeout=60)
            except HTTPError as e:response=e
            rid=str(uuid.UUID(packet['id']));seq=0
            def send(frame):
                nonlocal seq
                frame['sequence']=seq
                ack=relay_request(relay_origin,relay_token,'POST',f'/relay/{rid}/response',frame)
                if ack['sequence']!=seq+1:raise ValueError('Relay acknowledgement mismatch')
                seq+=1
            with response:
                send({'kind':'start','status':response.status,'headers':dict(response.headers)})
                while chunk:=response.read(CHUNK):send({'kind':'data','body':base64.b64encode(chunk).decode()})
                send({'kind':'done'})
        except Exception:
            # No auth headers or capture payloads in error logs. Network retries
            # do not mark phone uploads complete; the phone rechecks durable offset.
            time.sleep(5)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['endpoint','home']);p.add_argument('--relay-origin')
    p.add_argument('--data-dir',type=__import__('pathlib').Path);a=p.parse_args()
    relay_token=os.environ.get('SCANNER_RELAY_TOKEN','')
    if len(relay_token)<32:raise SystemExit('Dedicated relay credential is required; deployment awaits approval')
    if a.mode=='endpoint':
        import uvicorn
        uvicorn.run(create_relay(os.environ.get('SCANNER_PHONE_TOKEN',''),relay_token),host='127.0.0.1',port=18765,access_log=False)
    else:
        if not a.relay_origin or not a.data_dir:raise SystemExit('--relay-origin and original --data-dir required')
        token=(a.data_dir/'bridge.token').read_text().strip()
        run_home(a.relay_origin,relay_token,token)
