// Synthetic end-to-end UI interactions with a real camera; no GPU or home data.
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';
import {test} from 'node:test';
import {PerspectiveCamera} from '../../bridge/static/vendor/three.module.js';

const source = (await readFile(new URL('../../bridge/static/viewer.js', import.meta.url), 'utf8'))
  .replace(/^import .*;\r?\n/, '');

async function fixture() {
  class Element {
    handlers = new Map(); children = []; options = []; value = ''; style = {}; attributes = {};
    dataset = {}; classList = {add() {}}; clientWidth = 900; clientHeight = 600; textContent = '';
    addEventListener(type, handler) {
      if (!this.handlers.has(type)) this.handlers.set(type, []);
      this.handlers.get(type).push(handler);
    }
    async emit(type, props = {}) {
      await Promise.all((this.handlers.get(type) || []).map((handler) => handler({target:this, preventDefault() {}, ...props})));
      await new Promise(setImmediate);
    }
    append(...children) { this.children.push(...children); }
    appendChild(child) { this.append(child); }
    prepend(...children) { this.children.unshift(...children); }
    replaceChildren(...children) { this.children = children; this.options = children; }
    setAttribute(key, value) { this.attributes[key] = value; }
    add(option) { this.options.push(option); }
    set selectedIndex(i) { this.value = this.options[i]?.value || ''; }
    setPointerCapture() {}
  }
  const ids = ['splat-viewer','splat-job','reload-jobs','viewer-status','region-editor','region-form',
    'region-status','region-photos','region-photo-count','region-photo-page','region-photos-prev',
    'region-photos-next','region-photos-refresh','region-list','region-place','region-hide','region-save','region-name','region-distance',
    'region-x','region-y','region-z','region-sx','region-sy','region-sz'];
  const elements = Object.fromEntries(ids.map((id) => [id, new Element()]));
  elements['splat-viewer'].dataset.projectId = 'project';
  elements['region-form'].dataset.csrf = 'token';
  elements['region-name'].value = 'Counter <corner>';
  for (const id of ['region-sx','region-sy','region-sz','region-distance']) elements[id].value = '1';
  for (const id of ['region-x','region-y','region-z']) elements[id].value = '0';
  const document = new Element(); document.getElementById = (id) => elements[id] || null;
  document.querySelectorAll = () => [];
  document.createElement = document.createElementNS = () => new Element();
  const window = new Element(), canvas = new Element(), camera = new PerspectiveCamera();
  camera.position.set(2,3,4);
  const calls = [], loads = [], intervals = [];
  const model = {id:'online:current', source:'online', source_sha256:'original-hash', coordinate_frame:'fixed-frame',
    session_id:'session', vertex_count:100, viewable:true, url:'current.ply',
    view:{center:[0,0,0], eye:[2,3,4], look_at:[2,3,3], radius:3, up_axis:'y'}};
  let records = [];
  class Viewer {
    camera = camera; renderer = {domElement:canvas};
    async addSplatScene(path) { loads.push(path); }
    async removeSplatScene() {}
    start() {}
  }
  const fetch = async (url, options = {}) => {
    calls.push([url,options]);
    let result = {};
    if (url.startsWith('/ui/current-model')) result = {model};
    else if (url.startsWith('/ui/jobs')) result = {jobs:[]};
    else if (url.includes('/region-photos')) result = {photos:[{capture_id:'photo-1', included:true, received_utc:'2026-09-29T12:00:00Z'}], count:1,page_size:24};
    else if (url.endsWith('/regions') && options.method === 'POST') {
      const plan = JSON.parse(options.body.get('plan'));
      records = [{...plan,id:'region-1',status:'planned',view:model.view}]; result = records[0];
    } else if (url.endsWith('/regions')) result = {regions:records};
    else if (url.endsWith('/publish')) { records[0].status = 'published'; result = records[0]; }
    return {ok:true,json:async () => result};
  };
  vm.runInNewContext(source, {document,window,console,URLSearchParams,
    GaussianSplats3D:{Viewer,SceneFormat:{Ply:0}},
    Option:class {constructor(text,value) {this.textContent=text;this.value=value;}},
    fetch, setInterval(callback,delay) {intervals.push([callback,delay]);},
    performance:{now:()=>0},requestAnimationFrame:()=>1,cancelAnimationFrame() {}});
  await new Promise(setImmediate);
  const find = (root, predicate) => {
    for (const child of root.children) if (typeof child === 'object') {
      if (predicate(child)) return child;
      const next = find(child,predicate); if (next) return next;
    }
  };
  return {elements,camera,calls,loads,model,intervals,
    setRecords: (next) => {records=next;},
    records:()=>records,
    button:(text)=>find(elements['region-list'],(child)=>child.textContent===text),
    checkbox:()=>find(elements['region-list'],(child)=>child.type==='checkbox'),
    async open() {elements['region-editor'].open=true;await elements['region-editor'].emit('toggle');},
  };
}

test('box placement and photo choice only save a plan; they never delete or publish', async () => {
  const f = await fixture(); await f.open();
  await f.elements['region-place'].emit('click');
  assert.equal(f.elements['region-x'].value,'2.0000');
  assert.equal(f.elements['region-z'].value,'3.0000');
  await f.elements['region-form'].emit('submit');
  const plan = f.records()[0];
  assert.deepEqual(plan.min,[1.5,2.5,2.5]);
  assert.deepEqual(plan.max,[2.5,3.5,3.5]);
  assert.equal(plan.source_sha256,'original-hash');
  assert.equal(plan.name,'Counter <corner>');
  assert.deepEqual(f.calls.filter(([,options])=>options.method==='POST').map(([url])=>url),['/ui/projects/project/regions']);
  await f.elements['region-form'].emit('submit');
  assert.equal(f.calls.filter(([,options])=>options.method==='POST').length,1,'repeated click must not silently duplicate plan');
});

test('protecting a good saved area calls the fixed protection action without clearing or training', async () => {
  const f = await fixture(); await f.open();
  await f.elements['region-place'].emit('click');
  await f.elements['region-form'].emit('submit');
  const protect = f.button('Protect this good area');
  assert.ok(protect && !protect.disabled);
  await protect.emit('click');
  const calls = f.calls.filter(([, options]) => options.method === 'POST').map(([url]) => url);
  assert.deepEqual(calls, ['/ui/projects/project/regions', '/ui/projects/project/regions/region-1/protect']);
});

test('draft preview holds the same camera and is not replaced by current-model polling', async () => {
  const f = await fixture();
  const record = {id:'region-1',name:'Counter',status:'draft_ready',min:[0,0,0],max:[1,1,1],
    replacement_capture_ids:['one','two'],view:f.model.view,
    artifact:{sha256:'candidate-hash',vertex_count:80,size_bytes:20}};
  f.setRecords([record]); await f.open();
  const before = f.camera.position.clone(), rotation = f.camera.quaternion.clone();
  await f.button('Preview draft').emit('click');
  assert.match(f.loads.at(-1),/draft\.ply\?revision=candidate-hash/);
  assert.ok(f.camera.position.equals(before) && f.camera.quaternion.equals(rotation));
  assert.match(f.elements['viewer-status'].textContent,/not applied to the current scan/);
  await f.elements['reload-jobs'].emit('click');
  assert.match(f.elements['splat-job'].value,/area:region-1:candidate-hash/);
  assert.match(f.loads.at(-1),/draft\.ply/);
});

test('applying a locked draft requires displaying that exact draft and an explicit checkbox', async () => {
  const f = await fixture();
  const record = {id:'region-1',name:'Counter',status:'locked',min:[0,0,0],max:[1,1,1],
    view:f.model.view,replacement_capture_ids:[],artifact:{sha256:'locked-hash',vertex_count:80,size_bytes:20}};
  f.setRecords([record]); await f.open();
  assert.equal(f.button('Apply reviewed locked area').disabled,true);
  let review = f.checkbox(); review.checked=true; await review.emit('change');
  assert.equal(f.button('Apply reviewed locked area').disabled,true,'checkbox alone cannot replace displaying this revision');
  await f.button('Preview draft').emit('click');
  assert.equal(f.button('Apply reviewed locked area').disabled,false);
  await f.button('Apply reviewed locked area').emit('click');
  const posts = f.calls.filter(([url, options]) => url.endsWith('/publish') && options.method === 'POST');
  assert.equal(posts.length,1); assert.equal(posts[0][1].body.get('reviewed'),'yes');
});

test('a rejected repair can be rebuilt but cannot be locked or applied', async () => {
  const f = await fixture();
  f.setRecords([{id:'region-1',name:'Counter',status:'quality_rejected',min:[0,0,0],max:[1,1,1],
    replacement_capture_ids:['one','two'],view:f.model.view,
    artifact:{sha256:'rejected-hash',vertex_count:80,size_bytes:20}}]);
  await f.open();
  assert.equal(f.button('Build replacement').disabled,false);
  assert.equal(f.button('Lock draft').disabled,true);
  assert.equal(f.button('Apply reviewed locked area'),undefined);
  await f.button('Build replacement').emit('click');
  const writes = f.calls.filter(([, options]) => options.method === 'POST');
  assert.equal(writes.length,1);
  assert.match(writes[0][0],/\/build$/);
});
