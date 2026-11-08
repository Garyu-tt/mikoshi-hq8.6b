#!/usr/bin/env python3
"""Mikoshi HQ8.6B: 1/10 of BFB9000 R568 synthetic GPU 2-bit state proxy.

This is a synthetic directed nearest-neighbour ring, NOT a human connectome.
Every one of 8,606,000,000 addressable 2-bit states receives an actual GPU
transition in a completed full sweep, but not all states reside in VRAM.
Original R167/R168 recurrent CUDA engine is separate and unchanged.
Uses CUDA driver and embedded PTX; nvcc and pip dependencies not required.
"""
import argparse, ctypes as c, json, os, random, re, struct, time
from pathlib import Path

NEURONS = 8_606_000_000
WORDS = NEURONS // 16
NBYTES = WORDS * 4
LOW = 0x55555555
HIGH = 0xAAAAAAAA
PTX = r"""
.version 7.5
.target sm_86
.address_size 64
.visible .entry r568_step(
 .param .u64 src, .param .u64 dst,
 .param .u64 globalbase, .param .u32 count, .param .u32 tick
) {
 .reg .pred %p<3>;
 .reg .b32 %r<25>;
 .reg .b64 %rd<20>;
 ld.param.u64 %rd0,[src];
 ld.param.u64 %rd1,[dst];
 ld.param.u64 %rd2,[globalbase];
 ld.param.u32 %r0,[count];
 ld.param.u32 %r1,[tick];
 mov.u32 %r2,%ctaid.x;
 mov.u32 %r3,%ntid.x;
 mov.u32 %r4,%tid.x;
 mad.lo.u32 %r5,%r2,%r3,%r4;
 setp.ge.u32 %p0,%r5,%r0;
 @%p0 bra DONE;
 mul.wide.u32 %rd3,%r5,4;
 add.u64 %rd4,%rd0,%rd3;
 ld.global.u32 %r6,[%rd4];
 add.u64 %rd5,%rd4,4;
 ld.global.u32 %r7,[%rd5];
 and.b32 %r8,%r7,1431655765;
 shr.u32 %r9,%r7,1;
 and.b32 %r9,%r9,1431655765;
 not.b32 %r10,%r8;
 and.b32 %r11,%r9,%r10;
 and.b32 %r12,%r9,%r8;
 and.b32 %r13,%r7,2863311530;
 shl.b32 %r13,%r13,1;
 and.b32 %r13,%r13,1431655765;
 shr.u32 %r14,%r6,31;
 or.b32 %r13,%r13,%r14;
 xor.b32 %r15,%r11,%r13;
 and.b32 %r16,%r11,%r13;
 xor.b32 %r17,%r12,%r16;
 shl.b32 %r17,%r17,1;
 or.b32 %r18,%r15,%r17;
 cvt.u64.u32 %rd6,%r5;
 add.u64 %rd7,%rd2,%rd6;
 and.b64 %rd8,%rd7,65535;
 cvt.u64.u32 %rd9,%r1;
 and.b64 %rd10,%rd9,65535;
 setp.eq.u64 %p1,%rd8,%rd10;
 @%p1 or.b32 %r18,%r18,3;
 add.u64 %rd11,%rd1,%rd3;
 st.global.u32 [%rd11],%r18;
DONE:
 ret;
}
"""

def reference(value, prev, globalword, tick):
    lo = value & LOW
    hi = (value >> 1) & LOW
    decaylo = hi & (~lo & 0xffffffff)
    decayhi = hi & lo
    syn = (((value & HIGH) << 1) & LOW) | (prev >> 31)
    out = (decaylo ^ syn) | ((decayhi ^ (decaylo & syn)) << 1)
    if (globalword & 65535) == (tick & 65535):
        out |= 3
    return out & 0xffffffff

class Driver:
    def __init__(self, max_bytes):
        self.lib = c.CDLL("libcuda.so.1")
        self.ptr = c.c_void_p
        def attach(name, types):
            f = getattr(self.lib,name)
            f.restype = c.c_int
            f.argtypes = types
            return f
        P = self.ptr; I = c.c_int; U = c.c_uint; S = c.c_size_t
        self.init = attach("cuInit",[U])
        self.device = attach("cuDeviceGet",[c.POINTER(I),I])
        self.ctxcreate = attach("cuCtxCreate_v2",[c.POINTER(P),U,I])
        self.moduleload = attach("cuModuleLoadDataEx",[c.POINTER(P),P,U,P,P])
        self.getfunc = attach("cuModuleGetFunction",[c.POINTER(P),P,c.c_char_p])
        self.alloc = attach("cuMemAlloc_v2",[c.POINTER(c.c_uint64),S])
        self.htod = attach("cuMemcpyHtoD_v2",[c.c_uint64,P,S])
        self.dtoh = attach("cuMemcpyDtoH_v2",[P,c.c_uint64,S])
        self.launch = attach("cuLaunchKernel",[P,U,U,U,U,U,U,U,P,c.POINTER(P),c.POINTER(P)])
        self.sync = attach("cuCtxSynchronize",[])
        self.free = attach("cuMemFree_v2",[c.c_uint64])
        self.check(self.init(0),"cuInit")
        d=I(); self.check(self.device(c.byref(d),0),"cuDeviceGet")
        self.ctx=P(); self.check(self.ctxcreate(c.byref(self.ctx),0,d),"cuCtxCreate")
        self.mod=P(); ptx=c.create_string_buffer(PTX.encode())
        self.check(self.moduleload(c.byref(self.mod),c.cast(ptx,P),0,None,None),"cuModuleLoadDataEx")
        self.fn=P()
        self.check(self.getfunc(c.byref(self.fn),self.mod,b"r568_step"),"cuModuleGetFunction")
        self.a=c.c_uint64(); self.b=c.c_uint64()
        self.check(self.alloc(c.byref(self.a),max_bytes+4),"cuMemAlloc input")
        self.check(self.alloc(c.byref(self.b),max_bytes),"cuMemAlloc output")
        self.max_bytes=max_bytes
    @staticmethod
    def check(code,op):
        if code: raise RuntimeError(f"{op}: CUDA error {code}")
    def step(self, data, prev, base, tick):
        size=len(data)
        if size%4 or size>self.max_bytes: raise ValueError("invalid tile size")
        count=size//4
        inp=c.create_string_buffer(struct.pack("<I",prev)+data)
        out=c.create_string_buffer(size)
        self.check(self.htod(self.a,c.cast(inp,self.ptr),size+4),"HtoD")
        v0=c.c_uint64(self.a.value)
        v1=c.c_uint64(self.b.value)
        v2=c.c_uint64(base)
        v3=c.c_uint(count)
        v4=c.c_uint(tick)
        params=(self.ptr*5)(*[c.cast(c.byref(x),self.ptr) for x in (v0,v1,v2,v3,v4)])
        self.check(self.launch(self.fn,(count+255)//256,1,1,256,1,1,0,None,params,None),"launch")
        self.check(self.sync(),"sync")
        self.check(self.dtoh(c.cast(out,self.ptr),self.b,size),"DtoH")
        return out.raw
    def close(self):
        self.free(self.a)
        self.free(self.b)

def selftest(gpu):
    rng=random.Random(8606)
    count=4096
    values=[rng.getrandbits(32) for _ in range(count)]
    raw=struct.pack("<"+str(count)+"I",*values)
    prev=0xd234fa3b
    base=WORDS-count
    tick=(base+333)&65535
    got=struct.unpack("<"+str(count)+"I",gpu.step(raw,prev,base,tick))
    for i,x in enumerate(values):
        expect=reference(x,values[i-1] if i else prev,base+i,tick)
        if got[i]!=expect: raise AssertionError(f"GPU/CPU mismatch word={i} expected={expect:x} got={got[i]:x}")
    assert (WORDS-1)*16+15==NEURONS-1
    print(json.dumps(dict(test="PASS",gpu_words=count,neuron_ids=[base*16,NEURONS-1],ptx_target="sm_86",cpu_gpu_bitwise=True)))

def sweep(gpu,basepath,chunk_words,max_tiles):
    basepath.mkdir(parents=True,exist_ok=True)
    source=basepath/"epoch000-sparse-zero.q2"
    dest=basepath/"epoch001-generated.q2"
    committed=basepath/"epoch001-commit.json"
    progress=basepath/"epoch001-progress.json"
    if committed.exists(): raise RuntimeError("epoch001 already committed; never overwrite")
    if not source.exists():
        fd=os.open(source,os.O_RDWR|os.O_CREAT|os.O_EXCL,0o600)
        try: os.ftruncate(fd,NBYTES)
        finally: os.close(fd)
    if source.stat().st_size != NBYTES: raise RuntimeError("invalid input length")
    if dest.exists():
        if not progress.exists(): raise RuntimeError("uncommitted output lacks trusted progress journal")
        cp=json.loads(progress.read_text())
        done=int(cp["words_processed"]); tiles=int(cp["tiles"])
        if not 0<=done<WORDS or done != min(tiles*chunk_words,WORDS): raise RuntimeError("progress journal invalid")
        if dest.stat().st_size!=NBYTES: raise RuntimeError("output length invalid")
        out=os.open(dest,os.O_RDWR)
    else:
        if progress.exists(): raise RuntimeError("progress journal exists without output")
        out=os.open(dest,os.O_RDWR|os.O_CREAT|os.O_EXCL,0o600)
        os.ftruncate(out,NBYTES)
        done=0; tiles=0
    inp=os.open(source,os.O_RDONLY)
    began=time.monotonic(); tiles_this_run=0
    try:
        while done < WORDS and (not max_tiles or tiles_this_run<max_tiles):
            cnt=min(chunk_words,WORDS-done)
            offset=done*4
            raw=os.pread(inp,cnt*4,offset)
            if len(raw)!=cnt*4: raise IOError("short state read")
            prev=struct.unpack("<I",os.pread(inp,4,offset-4))[0] if done else 0
            result=gpu.step(raw,prev,done,0)
            # Sparse persistence: preserve every computed nonzero 4 KiB page.
            # Untouched pages remain all-zero holes in the logical full-size file.
            last_page=-1
            for match in re.finditer(rb'[^\x00]',result):
                page=match.start() & ~4095
                if page==last_page: continue
                last_page=page
                part=result[page:page+4096]
                nw=os.pwrite(out,part,offset+page)
                if nw!=len(part): raise IOError("short sparse page write")
            done+=cnt; tiles+=1; tiles_this_run+=1
            if tiles%8==0 or done==WORDS or (max_tiles and tiles_this_run==max_tiles):
                os.fsync(out)
                cp_tmp=progress.with_suffix('.tmp')
                cp_tmp.write_text(json.dumps(dict(words_processed=done,tiles=tiles,chunk_words=chunk_words,neuron_count=NEURONS))+"\n")
                cp_tmp.replace(progress)
                print(json.dumps(dict(status="progress",tiles=tiles,neurons_processed=done*16,total=NEURONS,seconds=round(time.monotonic()-began,2))),flush=True)
        os.fsync(out)
    finally:
        os.close(inp);os.close(out)
    status="FULL_GPU_SWEEP_DONE" if done==WORDS else "PARTIAL_UNCOMMITTED"
    report=dict(status=status,model="synthetic q2 nearest-neighbor directed ring; NOT observed human synapses",neurons_processed=done*16,virtual_neurons=NEURONS,virtual_synapses=NEURONS,quiescent_census=NEURONS,words=WORDS,bytes_per_epoch=NBYTES,chunk_words=chunk_words,tiles=tiles,gpu_backend="NVIDIA CUDA Driver / PTX sm_86",source=str(source),output=str(dest),elapsed_seconds=round(time.monotonic()-began,2))
    if done==WORDS:
        tmp=committed.with_suffix(".tmp")
        tmp.write_text(json.dumps(report,indent=2)+"\n")
        tmp.replace(committed)
        progress.unlink(missing_ok=True)
    print(json.dumps(report),flush=True)
    return done==WORDS

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--selftest",action="store_true")
    ap.add_argument("--sweep",action="store_true")
    ap.add_argument("--max-tiles",type=int,default=0)
    ap.add_argument("--chunk-words",type=int,default=16_777_216)
    ap.add_argument("--root",default="./runs/hq8.6b")
    a=ap.parse_args()
    if a.chunk_words<1024 or a.chunk_words>33_554_432: raise ValueError("chunk words outside safe limits")
    gpu=Driver(a.chunk_words*4)
    try:
        selftest(gpu)
        if a.sweep:
            ok=sweep(gpu,Path(a.root),a.chunk_words,a.max_tiles)
            if not ok and not a.max_tiles: raise SystemExit(3)
    finally:
        gpu.close()

if __name__=="__main__": main()
