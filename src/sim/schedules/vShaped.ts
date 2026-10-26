/**
 * 1F1B-V (unsplit backward), Appendix H of arXiv:2405.15362.
 * TypeScript adaptation of sail-sg/zero-bubble-pipeline-parallelism
 * 39f2c186a27580b7051eb51e0f651e1efd0a8170, scheduler/v1f1b.py.
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 * Redistribution conditions and disclaimer: public/THIRD_PARTY_NOTICES.txt.
 * W/w below are internal two-slot backward markers ONLY; emitted operations
 * always execute an indivisible B with the configured full backward cost.
 */
import type { ComputeStep, Program, SimConfig, Step, Transfer } from '../types.ts';
import { Topology, compute, post, push, wait } from './common.ts';

const PERIOD = 6;
const SYMBOLS = 'FfBbWw';

/** Reference's repeating building block, trimmed to m microbatches. */
function repeatedPattern(pp: number, m: number): string[][] {
  const offsets = Array.from({ length: pp }, () => Array<number>(6).fill(0));
  let now = 0;
  for (let r = 0; r < pp; r++) offsets[r][0] = ++now;
  for (let r = pp - 1; r >= 0; r--) offsets[r][1] = ++now;
  now++;
  if (pp % 3 === 0) now += 3;
  let cycle = (3 - (pp + 2) % 3) % 3;
  for (const [column, ranks] of [[2, Array.from({length:pp},(_,r)=>r)], [3, Array.from({length:pp},(_,r)=>pp-1-r)]] as const) {
    for (const r of ranks) {
      offsets[r][column] = now;
      offsets[r][column + 2] = now + 1;
      now += 2;
      if (++cycle === 3) { cycle = 0; now += 3; }
    }
  }
  const schedules = offsets.map(row => {
    const pattern = Array<string>(PERIOD).fill(' ');
    row.forEach((v,i)=>{ pattern[v % PERIOD] = SYMBOLS[i]; });
    return Array.from({length:4*pp+m+1},()=>pattern).flat();
  });
  const clear = (r:number, start:number, direction:number) => {
    for (let i=start;i>=0&&i<schedules[r].length;i+=direction*PERIOD) schedules[r][i]=' ';
  };
  let index = PERIOD;
  for (const symbol of 'FfBb') {
    const ranks = Array.from({length:pp},(_,r)=>'FB'.includes(symbol)?r:pp-1-r);
    for (const r of ranks) {
      while (schedules[r][index] !== symbol) index++;
      clear(r,index-PERIOD,-1); clear(r,index+PERIOD*m,1); index++;
      if ('Bb'.includes(symbol)) {
        const w = symbol==='B'?'W':'w';
        const pos=schedules[r].indexOf(w,index);
        clear(r,pos-PERIOD,-1); clear(r,pos+PERIOD*m,1);
      }
    }
  }
  return schedules;
}

/** Move warmup/cooldown operations earlier under the reference memory cap. */
function reorder(schedules:string[][], m:number): void {
  const pp=schedules.length, width=schedules[0].length;
  let peak=0;
  const memory=schedules.map(row=>{let n=0;return row.map(s=>{if('Ff'.includes(s)&&s!==' ')n++;if('Ww'.includes(s)&&s!==' ')n--;peak=Math.max(peak,n);return n;});});
  const loc=Array.from({length:pp},()=>Array.from({length:m+2},()=>Object.fromEntries([...SYMBOLS].map(s=>[s,-1]))));
  const counters=Array.from({length:pp},()=>Object.fromEntries([...SYMBOLS].map(s=>[s,0])));
  for(let i=0;i<width;i++) for(let r=0;r<pp;r++) {
    const symbol=schedules[r][i]; if(symbol===' ')continue;
    const count=++counters[r][symbol];
    let pos=count>1?loc[r][count-1][symbol]:-1;
    if(symbol==='W')pos=Math.max(pos,loc[r][count].B);
    if(symbol==='w')pos=Math.max(pos,loc[r][count].b);
    if(symbol==='F'&&r>0)pos=Math.max(pos,loc[r-1][count].F);
    if(symbol==='f')pos=Math.max(pos,r<pp-1?loc[r+1][count].f:loc[r][count].F);
    if(symbol==='B')pos=Math.max(pos,r>0?loc[r-1][count].W:loc[r][count].f);
    if(symbol==='b')pos=Math.max(pos,r<pp-1?loc[r+1][count].w:loc[r][count].W);
    pos++;
    while(pos<i&&schedules[r][pos]!==' ')pos++;
    if('Bb'.includes(symbol))while(pos<i&&(schedules[r][pos]!==' '||schedules[r][pos+1]!==' '))pos++;
    if(pos===i){loc[r][count][symbol]=i;continue;}
    if('BbWw'.includes(symbol)) {
      schedules[r][pos]=symbol;schedules[r][i]=' ';
      if('Ww'.includes(symbol))for(let j=pos;j<i;j++)memory[r][j]--;
      loc[r][count][symbol]=pos;continue;
    }
    let place=i;
    while(place>pos&&memory[r][place-1]<peak)place--;
    while(place<i&&schedules[r][place]!==' ')place++;
    if(place===i){loc[r][count][symbol]=i;continue;}
    schedules[r][place]=symbol;schedules[r][i]=' ';
    for(let j=place;j<i;j++)memory[r][j]++;
    loc[r][count][symbol]=place;
  }
}

export function vShapedOrder(pp:number, m:number): ComputeStep[][] {
  const schedule=repeatedPattern(pp,m); reorder(schedule,m);
  return schedule.map(row=>{
    const counts:Record<string,number>={F:0,f:0,B:0,b:0};
    return row.filter(s=>'FfBb'.includes(s)&&s!==' ').flatMap(s=>{
      const mb=counts[s]++;
      return mb<m?[compute(s==='F'||s==='f'?'F':'B',mb,s==='f'||s==='B'?1:0)]:[];
    });
  });
}

/** Fixed reference order, replayed by the existing DES with real costs.
 * Sends are nonblocking; receives are allocated/posted just before use.
 * No same-device transfer, eager receive, or synthetic memory discount.
 */
export function vShapedProgram(cfg:SimConfig):Program {
  if (cfg.vpp !== 2) throw new Error('1F1B-V requires exactly two chunks per rank');
  const topo=new Topology(cfg.pp,2,true);
  return vShapedOrder(cfg.pp,cfg.microBatches).map((order,rank)=>{
    const steps:Step[]=[], sends:Transfer[]=[];
    for(const op of order) {
      const recv=op.kind==='F'?topo.recvF(rank,op.mb,op.chunk):topo.recvB(rank,op.mb,op.chunk);
      const send=op.kind==='F'?topo.sendF(rank,op.mb,op.chunk):topo.sendB(rank,op.mb,op.chunk);
      push(steps,post([],[recv]));push(steps,wait([recv]));steps.push(op);
      push(steps,post([send],[]));if(send)sends.push(send);
    }
    push(steps,wait(sends));return steps;
  });
}
