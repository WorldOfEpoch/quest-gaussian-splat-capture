// Run: node --test tests/bridge/viewer_navigation.test.mjs
// Exercise the real viewer event handlers with a real Three.js camera and a
// controlled animation clock; no GPU or private room photos are required.
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';
import {test} from 'node:test';
import {PerspectiveCamera} from '../../bridge/static/vendor/three.module.js';

const source = (await readFile(new URL('../../bridge/static/viewer.js', import.meta.url), 'utf8'))
  .replace(/^import .*;\r?\n/, '');

async function fixture() {
  class Element {
    handlers = new Map();
    options = [];
    value = '';
    clientHeight = 600;
    dataset = {projectId: 'test'};
    addEventListener(type, handler) {
      if (!this.handlers.has(type)) this.handlers.set(type, []);
      this.handlers.get(type).push(handler);
    }
    emit(type, props = {}) {
      for (const handler of this.handlers.get(type) || []) handler({preventDefault() {}, ...props});
    }
    replaceChildren(...children) { this.options = children; }
    add(option) { this.options.push(option); }
    set selectedIndex(i) { this.value = this.options[i]?.value || ''; }
    setPointerCapture() {}
  }
  const elements = Object.fromEntries(['splat-viewer', 'splat-job', 'reload-jobs', 'viewer-status', 'reset-view']
    .map((id) => [id, new Element()]));
  const canvas = new Element();
  const document = new Element();
  document.getElementById = (id) => elements[id] || null;
  document.querySelectorAll = () => [];
  const window = new Element();
  const camera = new PerspectiveCamera();
  const job = {id: 'test', viewable: true, vertex_count: 1, url: 'test.ply',
    view: {center: [0, 0, 0], eye: [0, 0, 0], look_at: [0, 0, -1], radius: 8}};
  let now = 0, nextId = 0;
  const frames = new Map();
  class Viewer {
    camera = camera;
    renderer = {domElement: canvas};
    async addSplatScene() {}
    start() {}
  }
  vm.runInNewContext(source, {document, window, console,
    GaussianSplats3D: {Viewer, SceneFormat: {Ply: 0}},
    Option: class { constructor(text, value) { this.textContent = text; this.value = value; } },
    fetch: async (url) => ({ok: true, json: async () => url.startsWith('/ui/current-model') ? {model: job} : {jobs: []}}),
    setInterval() {}, performance: {now: () => now},
    requestAnimationFrame(callback) { frames.set(++nextId, callback); return nextId; },
    cancelAnimationFrame(id) { frames.delete(id); },
  });
  await new Promise(setImmediate);
  assert.ok(canvas.handlers.has('wheel'), 'real navigation handlers attached');
  return {camera, canvas, document, window, elements,
    wheel: (deltaY, deltaMode = 0) => canvas.emit('wheel', {deltaY, deltaMode}),
    tick(ms = 16) {
      now += ms;
      const current = [...frames.values()]; frames.clear();
      current.forEach((callback) => callback(now));
    },
    settle() { for (let i = 0; i < 100; i++) this.tick(); },
  };
}

test('wheel eases forward from the person camera without orbiting or changing direction', async () => {
  const f = await fixture();
  const rotation = f.camera.quaternion.clone();
  f.wheel(-100);
  assert.equal(f.camera.position.length(), 0, 'event itself does not jump');
  f.tick();
  assert.ok(-f.camera.position.z > 0 && -f.camera.position.z < 0.04);
  f.settle();
  assert.ok(Math.abs(f.camera.position.z + 0.16) < 0.00001);
  assert.ok(f.camera.quaternion.equals(rotation));
  assert.equal(f.camera.position.x, 0);
  assert.equal(f.camera.position.y, 0);
});

test('tiny/zero deltas and pixel/line events preserve proportional travel', async () => {
  const tiny = await fixture(); tiny.wheel(-1); tiny.settle();
  assert.ok(Math.abs(tiny.camera.position.z + 0.0016) < 0.00001);
  const zero = await fixture(); zero.wheel(0); zero.settle();
  assert.equal(zero.camera.position.length(), 0);
  const pixels = await fixture(); pixels.wheel(48); pixels.settle();
  const lines = await fixture(); lines.wheel(3, 1); lines.settle();
  assert.ok(pixels.camera.position.distanceTo(lines.camera.position) < 0.00001);
});

test('rapid bursts and long frames cannot accumulate a large teleport', async () => {
  const f = await fixture();
  for (let i = 0; i < 200; i++) f.wheel(-120);
  f.tick(3000);
  assert.ok(f.camera.position.length() < 0.22);
  f.settle();
  assert.ok(f.camera.position.length() <= 0.64);
});

test('reversing responds next frame; reset, drag and focus loss cancel queued travel', async () => {
  const f = await fixture(); f.wheel(-100); f.tick();
  const before = f.camera.position.z;
  f.wheel(100); f.tick();
  assert.ok(f.camera.position.z > before);
  f.elements['reset-view'].emit('click'); f.settle();
  assert.equal(f.camera.position.length(), 0);
  for (const cancel of [
    () => f.window.emit('blur'),
    () => { f.document.hidden = true; f.document.emit('visibilitychange'); },
    () => f.canvas.emit('pointerleave'),
    () => f.canvas.emit('pointerdown', {button: 0, pointerId: 1, clientX: 0, clientY: 0}),
  ]) {
    f.wheel(-100); cancel(); f.settle();
    assert.equal(f.camera.position.length(), 0);
  }
});

test('travel is consistent across normal display refresh rates', async () => {
  const slow = await fixture(), fast = await fixture();
  slow.wheel(-100); fast.wheel(-100);
  for (let i = 0; i < 15; i++) slow.tick(1000 / 60);
  for (let i = 0; i < 36; i++) fast.tick(1000 / 144);
  assert.ok(slow.camera.position.distanceTo(fast.camera.position) < 1e-9);
});
