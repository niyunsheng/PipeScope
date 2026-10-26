import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { DEFAULT_CONFIG, simulate, validateConfig, runProgram, constantCost } from '../src/sim/index.ts';
import type { SimConfig } from '../src/sim/types.ts';
import { Topology } from '../src/sim/schedules/common.ts';
import { vShapedOrder } from '../src/sim/schedules/vShaped.ts';
const cfg=(pp:number,m:number,extra:Partial<SimConfig>={}):SimConfig=>({...DEFAULT_CONFIG,schedule:'1f1b-v',pp,vpp:2,microBatches:m,numLayers:2*pp,commModel:'async',p2pLatency:0,activationBytes:1,...extra});

test('1F1B-V matches pinned upstream operator orders, including modulo-3 patterns and small batches',()=>{
 const fixtures=JSON.parse(readFileSync(new URL('./fixtures/v-shaped-reference.json',import.meta.url),'utf8'));
 for(const c of fixtures.cases) assert.deepEqual(vShapedOrder(c.pp,c.microBatches).map(rank=>rank.map(o=>`${o.kind}${o.chunk}:${o.mb}`)),c.order,`pp=${c.pp}, m=${c.microBatches}`);
});
test('V topology folds locally and puts first/last stages on rank zero',()=>{
 const topo=new Topology(4,2,true);
 assert.deepEqual(Array.from({length:8},(_,s)=>topo.rank(s)),[0,1,2,3,3,2,1,0]);
 assert.equal(topo.stage(1,0),7);
 assert.equal(topo.sendF(3,0,0),null);assert.equal(topo.recvF(3,0,1),null);
 assert.equal(topo.sendB(3,0,1),null);assert.equal(topo.recvB(3,0,0),null);
});
test('V schedules preserve all dependencies, FIFO transfers and memory accounting across costs',()=>{
 for(const pp of [1,2,3,4,5,6,8])for(const m of [1,2,7,16])for(const latency of [0,0.2,3]){
  const config=cfg(pp,m,{p2pLatency:latency,lossTime:0.7,forwardTime:0.8,backwardTime:2.3,tokens:Array.from({length:m},(_,i)=>1024*(1+i%3))});
  const trace=simulate(config);assert.equal(trace.failure,null,`pp=${pp}, m=${m}, latency=${latency}: ${trace.failure?.message}`);
  const ops=new Map(trace.ops.map(o=>[`${o.kind}:${o.mb}:${o.stage}`,o]));
  assert.equal(ops.size,m*(4*pp+1));
  for(let mb=0;mb<m;mb++)for(let stage=0;stage<2*pp;stage++){
   const f=ops.get(`F:${mb}:${stage}`)!,b=ops.get(`B:${mb}:${stage}`)!;
   assert.ok(b.start>=f.end-1e-8);
   if(stage){const prev=ops.get(`F:${mb}:${stage-1}`)!;assert.ok(f.start>=prev.end+(prev.rank===f.rank?0:latency)-1e-8);}
   if(stage<2*pp-1){const next=ops.get(`B:${mb}:${stage+1}`)!;assert.ok(b.start>=next.end+(next.rank===b.rank?0:latency)-1e-8);}
   else {const loss=ops.get(`L:${mb}:${stage}`)!;assert.equal(loss.rank,0);assert.ok(loss.start>=f.end-1e-8);assert.ok(b.start>=loss.end-1e-8);}
  }
  assert.equal(trace.transfers.length,4*(pp-1)*m);
  assert.ok(trace.transfers.every(t=>t.from!==t.to));
  for(let r=0;r<pp;r++){
   const local=trace.ops.filter(o=>o.rank===r);for(let i=1;i<local.length;i++)assert.ok(local[i].start>=local[i-1].end-1e-8);
   assert.ok(trace.memory[r].every(s=>s.bytes>=-1e-8));assert.ok(Math.abs(trace.memory[r].at(-1)!.bytes)<1e-8);
  }
 }
});
test('unsupported V modes are rejected rather than silently ignored',()=>{
 for(const patch of [{vpp:1},{commModel:'sync' as const},{moeOverlap:true},{warmupPlusOne:true},{prefetchWarmupFlush:true},{waitGrad:'beforeF' as const}])assert.ok(validateConfig(cfg(4,8,patch)).length);
 assert.deepEqual(validateConfig(cfg(4,1)),[]); // No interleaved group-size constraint.
});
test('V memory benefit is computed from events, with fixed total layer count',()=>{
 const pp=12,m=64;
 const v=simulate(cfg(pp,m,{activationBytes:undefined,activationMultiplier:17}));
 const base=simulate({...cfg(pp,m),schedule:'1f1b',vpp:1,commModel:'sync',activationBytes:undefined,activationMultiplier:17});
 assert.equal(v.failure,null);assert.equal(base.failure,null);
 assert.ok(v.metrics.ranks[0].peakMemory<base.metrics.ranks[0].peakMemory);
 assert.ok(Math.max(...v.metrics.ranks.map(r=>r.peakMemory))<Math.max(...base.metrics.ranks.map(r=>r.peakMemory)));
});

test('V share URLs normalize fixed controls and round trip editable parameters', async()=>{
 const {readUrl,writeUrl}=await import('../src/ui/url.ts');
 const oldLocation=Object.getOwnPropertyDescriptor(globalThis,'location');
 const oldHistory=Object.getOwnPropertyDescriptor(globalThis,'history');
 let output='';
 try {
  Object.defineProperty(globalThis,'location',{configurable:true,value:{search:'?schedule=1f1b-v&pp=3&microBatches=2&numLayers=12&vpp=9&commModel=sync&moeOverlap=true&warmupPlusOne=true&prefetchWarmupFlush=true&sel=1',pathname:'/PipeScope/'}});
  Object.defineProperty(globalThis,'history',{configurable:true,value:{replaceState:(_a:unknown,_b:unknown,url:string)=>{output=url;}}});
  const first=readUrl(DEFAULT_CONFIG);
  assert.equal(first.config.vpp,2);assert.equal(first.config.commModel,'async');
  assert.equal(first.config.moeOverlap,false);assert.equal(first.config.warmupPlusOne,false);assert.equal(first.config.prefetchWarmupFlush,false);
  assert.deepEqual(validateConfig(first.config),[]);
  writeUrl(first.config,DEFAULT_CONFIG,first.selectedMb);
  Object.defineProperty(globalThis,'location',{configurable:true,value:{search:output,pathname:'/PipeScope/'}});
  assert.deepEqual(readUrl(DEFAULT_CONFIG),first);
 } finally {
  if(oldLocation)Object.defineProperty(globalThis,'location',oldLocation);else Reflect.deleteProperty(globalThis,'location');
  if(oldHistory)Object.defineProperty(globalThis,'history',oldHistory);else Reflect.deleteProperty(globalThis,'history');
 }
});


test('local fold and loss dependencies cannot be bypassed',()=>{
 const config=cfg(1,1);
 for(const kind of ['F','L'] as const){
  const result=runProgram([[{type:'compute',kind,mb:0,chunk:1}]],1,constantCost(config),config);
  assert.equal(result.failure?.kind,'program');
  assert.match(result.failure!.message,/local dependency/);
 }
});
