import assert from 'node:assert/strict';
import {test} from 'node:test';
import {PlyParser, SplatBuffer, SplatBufferGenerator} from '../../bridge/static/vendor/gaussian-splats-3d.module.js';
import {Matrix4} from '../../bridge/static/vendor/three.module.js';

function photoModel() {
  const names = ['x','y','z','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3',
    'opacity','f_dc_0','f_dc_1','f_dc_2', ...Array.from({length:45},(_,i)=>`f_rest_${i}`)];
  const header = Buffer.from(`ply\nformat binary_little_endian 1.0\nelement vertex 2\n${names.map(n=>`property float ${n}\n`).join('')}end_header\n`);
  const body = Buffer.alloc(names.length*4*2);
  for(let row=0;row<2;row++) for(let i=0;i<names.length;i++) {
    const n = names[i];
    const value = n.startsWith('f_rest_') ? (Number(n.slice(7))+1)/100 + row : n==='rot_0'?1:0;
    body.writeFloatLE(value,(row*names.length+i)*4);
  }
  const data = Buffer.concat([header,body]);
  return data.buffer.slice(data.byteOffset,data.byteOffset+data.byteLength);
}

test('actual browser PLY path retains all 45 SH3 coefficients and RGB order', () => {
  const buffer=PlyParser.parseToUncompressedSplatBuffer(photoModel(),3);
  assert.equal(buffer.getMinSphericalHarmonicsDegree(),3);
  const actual=new Float32Array(90);
  buffer.fillSphericalHarmonicsArray(actual,3,new Matrix4(),0,1,0,0);
  for(let row=0;row<2;row++) for(let k=0;k<15;k++) for(let channel=0;channel<3;channel++) {
    assert.ok(Math.abs(actual[row*45+k*3+channel]-((channel*15+k+1)/100+row))<1e-6,
      `row ${row}, coefficient ${k}, channel ${channel}`);
  }
});

test('buffer byte sizes include the third band for all supported packing levels', () => {
  assert.equal(SplatBuffer.CompressionLevels[0].SphericalHarmonicsDegrees[3].BytesPerSplat,224);
  assert.equal(SplatBuffer.CompressionLevels[1].SphericalHarmonicsDegrees[3].BytesPerSplat,114);
  assert.equal(SplatBuffer.CompressionLevels[2].SphericalHarmonicsDegrees[3].BytesPerSplat,69);
});

test('optimized half-float browser transport retains the third band through an identity scene transform', () => {
  const array=PlyParser.parseToUncompressedSplatArray(photoModel(),3);
  const buffer=SplatBufferGenerator.getStandardGenerator(1,1).generateFromUncompressedSplatArray(array);
  const actual=new Float32Array(90);
  buffer.fillSphericalHarmonicsArray(actual,3,new Matrix4(),0,1,0,0);
  // The optimized generator spatially sorts the rows; coefficients must stay
  // together even when two splats at the same center exchange order.
  const inputRows = Array.from({length:2},(_,row)=>actual[row*45] > .5 ? 1 : 0);
  assert.deepEqual([...inputRows].sort(), [0,1]);
  for(let row=0;row<2;row++) for(let k=0;k<15;k++) for(let channel=0;channel<3;channel++) {
    assert.ok(Math.abs(actual[row*45+k*3+channel]-((channel*15+k+1)/100+inputRows[row]))<.001,
      `compressed row ${row}, coefficient ${k}, channel ${channel}`);
  }
});

test('unsupported scene rotation cannot silently discard or misrotate SH3', () => {
  const buffer=PlyParser.parseToUncompressedSplatBuffer(photoModel(),3);
  assert.throws(()=>buffer.fillSphericalHarmonicsArray(new Float32Array(90),3,
    new Matrix4().makeRotationY(.5),0,1,0,0),/original model rotation/);
});
