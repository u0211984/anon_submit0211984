from dataclasses import dataclass, field, replace
from typing import Optional
from kernel_trace import Kernel
from kernel_trace_train import ADAMW_FLOPS, F32, I64
from l2_sim import RAccess as A, StepRules

@dataclass(frozen=True)
class Source:
    uid: int
    gemm: Optional[tuple] = None
    comm_spec: Optional[object] = None

DP_REDUCER_GROUPS = {'transformer_dense': 'LAYER', 'transformer_moe': 'LAYER', 'zero2_transformer_gather': 'LAYER',
                     'embedding': 'EMBEDDING', 'zero2_embedding_gather': 'EMBEDDING',
                     'softmax': 'SOFTMAX', 'zero2_softmax_gather': 'SOFTMAX'}

def dp_reducer_group(key):
    key = str(key)
    if key.startswith('mdp_'):
        return 'EMBEDDING' if key.startswith('mdp_embedding_') else 'SOFTMAX' if key.startswith('mdp_softmax_') else 'LAYER'
    return DP_REDUCER_GROUPS.get(key)

@dataclass(frozen=True)
class RankTrace:
    kernels: tuple
    sources: tuple
    rules: StepRules
    flash: dict
    flash_specs: dict

    dp_payload: dict = field(default_factory=dict)

def cp_query_chunks(seq, cp, rank):
    if cp < 1 or not 0 <= rank < cp:
        raise ValueError('CP query chunks require cp >= 1 and 0 <= rank < cp')
    if seq <= 0 or seq % (2 * cp if cp > 1 else 1):
        raise ValueError('CP all-gather attention requires two whole CP query chunks (seq_len divisible by 2*cp)')
    if cp == 1:
        return ((seq, seq),)

    chunk = seq // (2 * cp)
    return ((chunk, (rank + 1) * chunk), (chunk, (2 * cp - rank) * chunk))

def operator_shapes(timing, name, direction):
    names = dict(qkv_proj='qkv_projection', output_proj='output_projection',
                 ffn1='ffn', ffn2='ffn2', linear_softmax='linear_softmax')
    key = names[name] + ('_f' if direction == 'forward' else '_b')
    records = tuple(getattr(timing, '_operator_gemm_shapes', {}).get(key, ()))

    if name in ('ffn1', 'ffn2') and bool(getattr(timing, 'use_moe', False)):
        t = int(timing.micro_batch) * int(timing.seq_len) // int(timing.cp)
        h, f = int(timing.hidden_dim), int(timing.intermediate_size)
        m, k, n = (t, h, 2 * f) if name == 'ffn1' else (t, f, h)
        records = ((m, k, n, 1),) if direction == 'forward' else ((m, n, k, 1), (k, m, n, 1))
    if not records:
        raise ValueError(f'Megatron trace is missing operator GEMM descriptors for {key}')
    if any(len(r) != 4 or any(int(x) <= 0 for x in r) for r in records):
        raise ValueError(f'Invalid captured operator GEMM descriptor for {key}')
    return records

def _flash_fast_forward(backward=True):
    from opmodel import settings
    if not settings.get("RAPID_FLASH_FAST_FORWARD") or not (backward or settings.get("RAPID_FLASH_FAST_FORWARD_FWD")):
        return None
    return (float(settings.get("RAPID_FLASH_FAST_FORWARD_RTOL")), int(settings.get("RAPID_FLASH_FAST_FORWARD_PERIODS")))

def build(program, timing):
    from flash_attention import FlashSpec, flash_kernels
    from flash_attention_adapter import flash_rates
    from kernel_trace_megatron_pointwise import expand
    import kernel_trace_megatron_te as te
    from opmodel import settings
    from program.ir import ComputeOp, CollectiveOp, TransferOp
    from program.megatron import validate, mla_trace, dp_grad_sync, distributed_optimizer, data_parallel_groups
    from program.megatron import recompute_modules, recompute_layernorm
    import kernel_trace_megatron_dist as dist_trace
    validate(timing)
    program_mla = mla_trace(timing)
    dp_sync = dp_grad_sync(timing)
    if str(program.meta.misc.get('granularity')).lower() != 'flat':
        raise ValueError('Megatron trace requires a FLAT placed operator program')
    p, opts = timing.precision, timing.implementation_options
    ab, pb, gb, sb = map(float, (p.activations,p.parameters,p.gradients,p.stats))
    batch,seq,cp,tp = map(int,(timing.micro_batch,timing.seq_len,timing.cp,timing.tp))
    h = int(timing.hidden_dim)
    hd = int(timing.head_dim or h//timing.num_heads)
    qwidth,kwidth = int(timing.num_heads)//tp*hd,int(timing.kv_heads)//tp*hd
    tokens = batch*seq//cp
    if seq%cp or timing.num_heads%tp or timing.kv_heads%tp or (cp>1 and seq%(2*cp)):
        raise ValueError('Megatron attention requires whole local heads and two whole CP query chunks')

    rope_megatron = not opts['fuse_rope'] and opts['megatron_unfused_rope'] == 'megatron_torch'
    copies = bool(opts['megatron_layout_copies'])
    tied_sum = opts['megatron_tied_grad'] == 'autograd_sum' and bool(timing.tied_embeddings)
    qkv_bias = timing.model_type == 'qwen2'
    if copies and cp > 1:
        raise ValueError('megatron_layout_copies describes cp=1 only')
    rates = flash_rates(timing, timing.O)
    rates = replace(rates,
        reduction_time_s=lambda f,b: timing.vector_roofline(f,b,name='',mem_level=timing.num_levels-1)+timing.O)
    cache = {}
    mode = settings.get('RAPID_L2_COLLECTIVE_BUFFERS') if tp>1 or cp>1 else None
    def canonical(name):
        for suffix in ('.rows', '.at_target'):
            if name.endswith(suffix):
                return name[:-len(suffix)]
        return name

    rules=StepRules(canonical,lambda x:x.startswith(('param.','input.')),lambda x:None,lambda x:False,
                    {},phase='train',collective_buffers=mode)
    result = {}
    def entry(op):
        return next((n for n in ('layernorm1','qkv_proj','rope','attention','output_proj','layernorm2','MLP',
                                 'router','moe_dispatch','moe_combine')
                     if op.name.startswith(n+'_')),None)
    use_moe = bool(getattr(timing,'use_moe',False))
    def moe_layer(l):
        return use_moe and l is not None and timing.is_moe_layer(int(l))
    for device in program.compute_devices():
        ks,ss,flash,weights,aliases,flash_specs = [],[],{},{},{},{}
        tied_pending = []
        current = None
        owner = None
        deferred_residuals = {}

        probs_events = {o.work for o in program.ops if isinstance(o,CollectiveOp) and o.device==device
                        and 'probs_all_to_all' in (o.comm_key or o.name)}
        after_probs = {}
        mlp_comm = {o.work for o in program.ops if isinstance(o,CollectiveOp) and o.device==device
                    and o.work is not None and o.work.direction.name=='FORWARD' and 'MLP' in (o.comm_key or o.name)}
        cprank = int(program.layout.coords_of(device).get('cp',0))
        eprank = int(program.layout.coords_of(device).get('ep',0))
        local = [o for o in program.ops if isinstance(o,ComputeOp) and o.device==device]
        attention_owner = {o.work: o.uid for o in local if entry(o) == 'attention'}
        split = {(o.work,entry(o)) for o in local if o.split_head}
        optimizer = next((o.uid for o in reversed(local) if o.work.kind.name=='OPTIMIZER'),local[-1].uid)

        dense_group, expert_group = data_parallel_groups(timing)
        def data_parallel(group):
            return None if dp_sync == 'legacy' else dist_trace.DataParallel(
                dp=group, distributed=distributed_optimizer(timing),
                end_of_backward=dp_sync == 'megatron_end_of_backward',
                pre_scale=opts.get('megatron_grad_scaling', 'pre_scale') == 'pre_scale', precision=p)
        dpx, dpx_expert = data_parallel(dense_group), data_parallel(expert_group)
        def dp_weights(expert):
            ordered = dist_trace.registration_order(weights, opts.get('megatron_moe_experts')) if mla else weights
            return {n: c for n, c in ordered.items() if ('.expert' in n) == expert}
        dp_ops = []
        dp_layers = {}
        def a(name,n,off=0,weight=False):
            return A(name,float(n),weight,float(off))
        def emit(k,uid=None,spec=None,record=None):
            index = len(ks)
            ks.append(k); ss.append(Source((current.uid if owner is None else owner) if uid is None else uid,k.gemm,spec))
            if record is not None: flash[index]=record
        def point(name,rd,wr,flops=0,op='pointwise',uid=None):
            emit(Kernel(op,name,'pointwise',tuple(rd),tuple(wr),float(flops)),uid)
        def pw(name,direction,rd,wr,elems,rows,prefix,heads=1,qk_norm=False):
            if name=='loss' and opts['fuse_loss'] and opts['megatron_loss']=='native_fused':
                for k in te.native_loss(direction,tuple(rd),tuple(wr),rows=rows,vocab=elems//rows,prefix=prefix):
                    emit(k)
                return
            described=expand(name,direction,rd,wr,fused=opts['fuse_'+name],sm_count=int(timing.core.num_bundle),
                            stats_width=sb if name!='loss' else (4 if opts['fp32_loss'] else ab),
                            prefix=prefix,elements=elems,rows=rows,activation_width=ab,heads=heads,
                            batch=batch,model_type=timing.model_type,qk_norm=qk_norm)
            if name=='norm' and direction=='backward' and opts['fuse_norm'] and opts['te_norm_backward']=='bwd_finalize':
                described=te.norm_backward(tuple(rd),tuple(wr),sm_count=int(timing.core.num_bundle),
                                           width=elems//rows,flops=described[0].flops,prefix=prefix)
            for k in described:
                emit(k)
        def weight(w,n):
            n=int(n)
            if w in weights and weights[w]!=n:
                raise ValueError(f'Inconsistent operator parameter shape {w}: {weights[w]} vs {n}')
            weights[w]=n
            return a(w,pb*n,weight=True)
        def accum(w,temp,n):
            g=a(w+'.main_grad',gb*n)
            point('backward.main_grad_accum',(temp,g),(g,),n,'main_grad')
        def linear(name,direction,x,y,gy,gx,w,part='all',bias=None,shapes=None,label=None):
            for j,(m,k,n,repeat) in enumerate(shapes or operator_shapes(timing,name,direction)):
                dw=direction=='backward' and j%2==1
                if direction=='backward' and ((part=='dx' and dw) or (part=='dw' and not dw)): continue
                for ri in range(repeat):
                    flops=0.0
                    if direction=='forward':
                        rd=(a(x,ab*m*k,ri*ab*m*k),weight(w,k*n)); wr=(a(y,ab*m*n,ri*ab*m*n),)
                        if bias: rd+=(weight(bias,n),); flops=float(m*n)
                    elif dw:
                        rd=(a(x,ab*m*k,ri*ab*m*k),a(gy,ab*k*n,ri*ab*k*n))
                        temp=a(y+f'.dw{j}.{ri}',ab*m*n); wr=(temp,)
                        if bias: btemp=a(y+f'.db{j}.{ri}',ab*n); wr+=(btemp,); flops=float(k*n)
                    else:
                        rd=(a(gy,ab*m*k,ri*ab*m*k),weight(w,k*n)); wr=(a(gx,ab*m*n,ri*ab*m*n),)
                    emit(Kernel(name,direction+'.'+(label or name)+('.dW' if dw else '.dX' if direction=='backward' else '.gemm'),
                                'gemm',rd,wr,flops,gemm=(m,k,n)))
                    if dw and tied_sum and name=='linear_softmax':
                        tied_pending.append((temp,m*n))
                    elif dw: accum(w,temp,m*n)
                    if dw and bias: accum(bias,btemp,n)
        def norm(direction,x,y,gy,gx,w,rows=tokens,width=h,qk_norm=False,heads=1):
            wa=weight(w,width); stat=a(y+'.stats',sb*rows)
            if direction=='forward':
                pw('norm',direction,(a(x,ab*rows*width),wa),(a(y,ab*rows*width),stat),rows*width,rows,y,heads,qk_norm)
            else:
                dw=a(w+'.dw.'+str(current.micro_batch),(pb if opts['fuse_norm'] else F32)*width)
                pw('norm',direction,(a(gy,ab*rows*width),a(x,ab*rows*width),a(y,ab*rows*width),wa,stat),
                   (a(gx,ab*rows*width),dw),rows*width,rows,y,heads,qk_norm)
                accum(w,dw,width)
        def residual(direction,x,y,out,prefix):
            pw('residual_add',direction,(a(x,ab*tokens*h),a(y,ab*tokens*h)),(a(out,ab*tokens*h),),tokens*h,tokens,prefix)
        def split_source(s,name):
            return s+('.'+name+'c' if copies and (name=='q' or (name=='k' and timing.model_type=='qwen3')) else '.'+name)
        def rotary(direction,s,w):
            backward=direction=='backward'
            order=(('q',qwidth),('k',kwidth))
            if rope_megatron and backward: order=order[::-1]
            for name,width in order:
                inp=s+'.g'+name+'rot' if backward else (s+'.'+name+'norm' if timing.model_type=='qwen3' else split_source(s,name))
                out=s+'.g'+name if backward else s+'.'+name+'rot'
                if rope_megatron:
                    for k in te.rope(direction,a(inp,ab*tokens*width),a(out,ab*tokens*width),
                                     a('input.rope_freqs',F32*(seq//cp)*hd),elements=tokens*width,width=ab,
                                     prefix=s+'.rope.'+name):
                        emit(k)
                else:
                    pw('rope',direction,(a(inp,ab*tokens*width),a('input.cos',ab*(seq//cp)*hd),a('input.sin',ab*(seq//cp)*hd)),
                       (a(out,ab*tokens*width),),tokens*width,tokens,s+'.rope.'+name,width//hd)
                if backward and timing.model_type=='qwen3':
                    norm('backward',split_source(s,name),s+'.'+name+'norm',s+'.g'+name,s+'.g'+name+'raw',w+name+'norm',tokens*width//hd,hd,True,width//hd)
            if backward:
                parts=[(s+'.g'+name+('raw' if timing.model_type=='qwen3' and name!='v' else ''),ab*tokens*width)
                       for name,width in (('q',qwidth),('k',kwidth),('v',kwidth))]
                if copies:
                    point('backward.qkv_grad_cat',[a(n,size) for n,size in parts],
                          (a(s+'.gqkv',sum(size for _,size in parts)),),op='qkv_proj')
                else:
                    view(s+'.gqkv',parts)
        def marker(spec,uid,name,rd,wr):
            nonlocal owner
            emit(Kernel('communication',name,'collective',tuple(rd),tuple(wr)),uid,spec)
            if isinstance(spec,CollectiveOp) and 'MLP' in name and spec.work in deferred_residuals:
                previous = owner
                owner, args = deferred_residuals.pop(spec.work)
                residual(*args)
                owner = previous
        def pp_buffer(op):
            src=program.ops[op.producer].work
            l=src.layer
            if l is None: l=int(timing.num_layers) if src.kind.name=='SOFTMAX' else 0
            elif src.direction.name!='BACKWARD': l+=1
            return f'b{src.microbatch}.'+('g' if src.direction.name=='BACKWARD' else 'x')+str(l)
        def view(parent,parts):
            off=0
            for name,n in parts:
                aliases[name]=(parent,off); off+=n
        def resolve(access):
            name,off=access.tensor,getattr(access,'offset',0)
            while name in aliases:
                name,delta=aliases[name]; off+=delta
            return a(name,access.nbytes,off,access.weight)
        def recompute(outputs,run):
            start=len(ks)
            run()
            sizes={}
            for i in range(start,len(ks)):
                k=ks[i]
                name='recompute.'+(k.name.split('.',1)[1] if k.name.startswith('forward.') else k.name)
                writes=tuple(A(w_.tensor+'.recomputed',w_.nbytes,w_.weight,w_.offset) if w_.tensor in outputs else w_
                             for w_ in k.writes)
                for w_ in k.writes:
                    if w_.tensor in outputs: sizes[w_.tensor]=max(sizes.get(w_.tensor,0.0),w_.offset+w_.nbytes)
                ks[i]=replace(k,name=name,writes=writes)
            for t in outputs:
                if t not in sizes:
                    raise ValueError(f'selective recompute region does not write its output {t}')
                emit(Kernel(ks[-1].op,'recompute.'+t.rsplit('.',1)[-1]+'.storage_copy','memcpy',
                            (a(t+'.recomputed',sizes[t]),),(a(t,sizes[t]),)))
        def mla_linear(direction,x,y,gy,gx,w,k_in,n_out):
            shapes=[(tokens,k_in,n_out,1)] if direction=='forward' else [(tokens,n_out,k_in,1),(k_in,tokens,n_out,1)]
            linear('qkv_proj',direction,x,y,gy,gx,w,shapes=shapes,label=w.rsplit('.',1)[-1])
        def mla_block(e,direction,s,w,part='all'):
            H,Q,R,N,P,Vd=(int(timing.num_heads),int(timing.q_lora_rank or 0),int(timing.kv_lora_rank),
                          int(timing.qk_nope_head_dim),int(timing.qk_rope_head_dim),int(timing.v_head_dim))
            qk=N+P; T=tokens; bH=batch*H; qkln=bool(opts['megatron_mla_qk_layernorm'])
            backward=direction=='backward'
            freqs=a('input.rope_freqs.mla',F32*seq*P)
            def mla_rope(name,heads):
                n=T*heads*P; src=s+'.mla_'+name+'pe'; dint=s+'.mla_'+name+'pe_di'
                if not backward:
                    point('forward.rope.mla_deinterleave',(a(src,ab*n),),(a(dint,ab*n),),op='qkv_proj')
                    for k in te.rope('forward',a(dint,ab*n),a(s+'.mla_'+name+'pe_rot',ab*n),freqs,elements=n,width=ab,
                                     prefix=s+'.rope.mla_'+name):
                        emit(replace(k,op='qkv_proj'))
                else:
                    gdi=s+'.mla_g'+name+'pe_di'
                    for k in te.rope('backward',a(s+'.mla_g'+name+'pe_rot',ab*n),a(gdi,ab*n),freqs,elements=n,width=ab,
                                     prefix=s+'.rope.mla_'+name):
                        emit(replace(k,op='qkv_proj'))

                    g=s+'.mla_g'+name+'pe'
                    for half in ('even','odd'):
                        point('backward.rope.mla_slice_zeros_'+half,(),(a(g+'.'+half,ab*n),),op='qkv_proj')
                        point('backward.rope.mla_slice_copy_'+half,(a(gdi,ab*n/2),),(a(g+'.'+half,ab*n/2),),op='qkv_proj')
                    point('backward.rope.mla_slice_add',(a(g+'.even',ab*n),a(g+'.odd',ab*n)),(a(g,ab*n),),n,'qkv_proj')
            if e=='qkv_proj' and not backward and part=='down':
                if Q:
                    mla_linear('forward',s+'.ln1',s+'.mla_qc','','',w+'mla_q_down',h,Q)
                mla_linear('forward',s+'.ln1',s+'.mla_kvcomb','','',w+'mla_kv_down',h,R+P)
                view(s+'.mla_kvcomb',[(s+'.mla_kvc_view',ab*T*R),(s+'.mla_kpe',ab*T*P)])
            elif e=='qkv_proj' and not backward:
                if part=='all':
                    mla_block(e,direction,s,w,'down')

                if Q:
                    qin=s+'.mla_qc'
                    if qkln:
                        norm('forward',qin,s+'.mla_qcn','','',w+'mla_q_norm',T,Q); qin=s+'.mla_qcn'
                    mla_linear('forward',qin,s+'.mla_q','','',w+'mla_q_up',Q,H*qk)
                else:
                    mla_linear('forward',s+'.ln1',s+'.mla_q','','',w+'mla_q_proj',h,H*qk)

                point('forward.mla_kv_compressed_contiguous',(a(s+'.mla_kvc_view',ab*T*R),),(a(s+'.mla_kvc',ab*T*R),),op='qkv_proj')
                kin=s+'.mla_kvc'
                if qkln:
                    norm('forward',kin,s+'.mla_kvcn','','',w+'mla_kv_norm',T,R); kin=s+'.mla_kvcn'
                mla_linear('forward',kin,s+'.mla_kv','','',w+'mla_kv_up',R,H*(N+Vd))
                view(s+'.mla_q',[(s+'.mla_qnope',ab*T*H*N),(s+'.mla_qpe',ab*T*H*P)])
                view(s+'.mla_kv',[(s+'.mla_knope',ab*T*H*N),(s+'.mla_value_view',ab*T*H*Vd)])
                mla_rope('q',H); mla_rope('k',1)

                point('forward.mla_query_cat',(a(s+'.mla_qnope',ab*T*H*N),a(s+'.mla_qpe_rot',ab*T*H*P)),
                      (a(s+'.mla_query',ab*T*H*qk),),op='qkv_proj')
                point('forward.mla_key_cat',(a(s+'.mla_knope',ab*T*H*N),a(s+'.mla_kpe_rot',ab*T*P)),
                      (a(s+'.mla_key',ab*T*H*qk),),op='qkv_proj')
                point('forward.mla_value_contiguous',(a(s+'.mla_value_view',ab*T*H*Vd),),(a(s+'.mla_value',ab*T*H*Vd),),op='qkv_proj')
            elif e=='qkv_proj':
                point('backward.mla_kpe_expand_sum',(a(s+'.mla_gkey_pe',ab*T*H*P),),(a(s+'.mla_gkpe_rot',ab*T*P),),T*H*P,'qkv_proj')
                view(s+'.mla_gkey',[(s+'.mla_gknope',ab*T*H*N),(s+'.mla_gkey_pe',ab*T*H*P)])
                view(s+'.mla_gquery',[(s+'.mla_gqnope',ab*T*H*N),(s+'.mla_gqpe_rot',ab*T*H*P)])
                mla_rope('k',1); mla_rope('q',H)

                point('backward.mla_kv_split_cat',(a(s+'.mla_gknope',ab*T*H*N),a(s+'.mla_gvalue',ab*T*H*Vd)),
                      (a(s+'.mla_gkv',ab*T*H*(N+Vd)),),op='qkv_proj')
                point('backward.mla_q_split_cat',(a(s+'.mla_gqnope',ab*T*H*N),a(s+'.mla_gqpe',ab*T*H*P)),
                      (a(s+'.mla_gq',ab*T*H*qk),),op='qkv_proj')
                kin=s+'.mla_kvcn' if qkln else s+'.mla_kvc'
                mla_linear('backward',kin,s+'.mla_kv',s+'.mla_gkv',s+'.mla_gkvin',w+'mla_kv_up',R,H*(N+Vd))
                if qkln:
                    norm('backward',s+'.mla_kvc',s+'.mla_kvcn',s+'.mla_gkvin',s+'.mla_gkvc',w+'mla_kv_norm',T,R)
                gkvc=s+'.mla_gkvc' if qkln else s+'.mla_gkvin'
                if Q:
                    qin=s+'.mla_qcn' if qkln else s+'.mla_qc'
                    mla_linear('backward',qin,s+'.mla_q',s+'.mla_gq',s+'.mla_gqin',w+'mla_q_up',Q,H*qk)
                    if qkln:
                        norm('backward',s+'.mla_qc',s+'.mla_qcn',s+'.mla_gqin',s+'.mla_gqc',w+'mla_q_norm',T,Q)
                else:
                    mla_linear('backward',s+'.ln1',s+'.mla_q',s+'.mla_gq',s+'.gln1q',w+'mla_q_proj',h,H*qk)
                point('backward.mla_kv_combined_cat',(a(gkvc,ab*T*R),a(s+'.mla_gkpe',ab*T*P)),
                      (a(s+'.mla_gkvcomb',ab*T*(R+P)),),op='qkv_proj')
                mla_linear('backward',s+'.ln1',s+'.mla_kvcomb',s+'.mla_gkvcomb',s+'.gln1kv',w+'mla_kv_down',h,R+P)
                if Q:
                    mla_linear('backward',s+'.ln1',s+'.mla_qc',s+'.mla_gqc' if qkln else s+'.mla_gqin',s+'.gln1q',
                               w+'mla_q_down',h,Q)

                point('backward.mla_ln_grad_add',(a(s+'.gln1kv',ab*T*h),a(s+'.gln1q',ab*T*h)),(a(s+'.gln1',ab*T*h),),T*h,'qkv_proj')
            elif e=='attention':
                Sq=seq
                if Sq%8 or not 16<Sq<16384 or Sq%4 or bH%4:
                    raise NotImplementedError('MLA unfused attention: TE fused masked softmax conditions not met '
                                              '(softmax.py:177-210); the torch softmax path is not described')
                sc,pr=s+'.mla_scores',s+'.mla_probs'
                mask=s+'.mla_mask'
                def bgemm(name,rd,wr,m,k,n):
                    from kernel_trace_vllm_mla import BatchedGemmKernel
                    emit(BatchedGemmKernel('attention',name,'batched_gemm',tuple(rd),tuple(wr),0.0,(m,k,n),
                                           'megatron-'+name,batch=bH))
                if not backward:
                    i32,i64,b1=4.0,8.0,1.0
                    point('forward.mla_mask.arange_q',(),(a(mask+'.aq',i32*Sq),),op='attention')
                    point('forward.mla_mask.arange_kv',(),(a(mask+'.akv',i32*Sq),),op='attention')
                    point('forward.mla_mask.sub',(a(mask+'.aq',i32*Sq),a(mask+'.akv',i32*Sq)),(a(mask+'.d',i32*Sq*Sq),),op='attention')
                    for nm in ('left','right'):
                        point('forward.mla_mask.window_'+nm,(a(mask+'.d',i32*Sq*Sq),),(a(mask+'.'+nm,i32*Sq*Sq),),op='attention')
                    for nm in ('left','right'):
                        point('forward.mla_mask.cmp_'+nm,(a(mask+'.'+nm,i32*Sq*Sq),),(a(mask+'.c'+nm,b1*Sq*Sq),),op='attention')
                        point('forward.mla_mask.where_'+nm,(a(mask+'.c'+nm,b1*Sq*Sq),),(a(mask+'.w'+nm,i64*Sq*Sq),),op='attention')
                    point('forward.mla_mask.diff',(a(mask+'.wleft',i64*Sq*Sq),a(mask+'.wright',i64*Sq*Sq)),(a(mask+'.wd',i64*Sq*Sq),),op='attention')
                    point('forward.mla_mask.logical_not',(a(mask+'.wd',i64*Sq*Sq),),(a(mask,b1*Sq*Sq),),op='attention')

                    bgemm('forward.mla_scores_baddbmm',(a(s+'.mla_query',ab*T*H*qk),a(s+'.mla_key',ab*T*H*qk)),
                          (a(sc,ab*bH*Sq*Sq),),Sq,qk,Sq)

                    point('forward.mla_scaled_masked_softmax',(a(sc,ab*bH*Sq*Sq),a(mask,1.0*Sq*Sq)),(a(pr,ab*bH*Sq*Sq),),
                          5*bH*Sq*Sq,'attention')
                    bgemm('forward.mla_context_bmm',(a(pr,ab*bH*Sq*Sq),a(s+'.mla_value',ab*T*H*Vd)),
                          (a(s+'.mla_ctx',ab*T*H*Vd),),Sq,Sq,Vd)

                    point('forward.mla_context_permute_copy',(a(s+'.mla_ctx',ab*T*H*Vd),),(a(s+'.attn',ab*T*H*Vd),),op='attention')
                else:
                    gp,gs=s+'.mla_gprobs',s+'.mla_gscores'

                    bgemm('backward.mla_context_bmm.dX',(a(s+'.gattn',ab*T*H*Vd),a(s+'.mla_value',ab*T*H*Vd)),
                          (a(gp,ab*bH*Sq*Sq),),Sq,Vd,Sq)
                    bgemm('backward.mla_context_bmm.dV',(a(pr,ab*bH*Sq*Sq),a(s+'.gattn',ab*T*H*Vd)),
                          (a(s+'.mla_gvalue',ab*T*H*Vd),),Sq,Sq,Vd)
                    point('backward.mla_scaled_masked_softmax',(a(gp,ab*bH*Sq*Sq),a(pr,ab*bH*Sq*Sq)),(a(gs,ab*bH*Sq*Sq),),
                          3*bH*Sq*Sq,'attention')

                    for nm,other,m_,k_,n_ in (('query','.mla_key',Sq,Sq,qk),('key','.mla_query',qk,Sq,Sq)):
                        g=s+'.mla_g'+nm
                        bgemm('backward.mla_scores_baddbmm.d'+nm,(a(gs,ab*bH*Sq*Sq),a(s+other,ab*T*H*qk)),
                              (a(g+'.raw',ab*T*H*qk),),m_,k_,n_)
                        point('backward.mla_scores_alpha_'+nm,(a(g+'.raw',ab*T*H*qk),),(a(g,ab*T*H*qk),),T*H*qk,'attention')
            elif e=='output_proj':
                shapes=[(T,H*Vd,h,1)] if not backward else [(T,h,H*Vd,1),(H*Vd,T,h,1)]
                linear(e,direction,s+'.attn',s+'.ao',s+'.gr1',s+'.gattn',w+'output',shapes=shapes)
        mla=program_mla and str(getattr(timing,'attention_type','mha')).lower()=='mla'
        rc_mods=recompute_modules(timing); ln2_done=set()
        def moe(e,direction,s,w,d,l,op,work):
            from kernel_trace_megatron_moe import I64, I32, F32, B1, expert_counts, cat_launches
            back=direction=='backward'; join='parallel_group_join' in op.name
            E,k,I=int(timing.moe_num_experts),int(timing.moe_top_k),int(timing.moe_intermediate_size)
            ep=int(timing.ep); Es=E//ep; first=eprank*Es; T=tokens; Ish=int(timing.n_shared_experts)*I
            counts=expert_counts(T*ep,E,k,ep,timing.model.moe_routing_policy,float(timing.expert_imbalance_factor),
                                 int(l)+int(work.microbatch))
            local=counts[first:first+Es]; R=sum(local); Rs=T*k

            pre=not join; post=join or ep==1
            score,lb,zl=opts['megatron_moe_score'],opts['megatron_moe_load_balancing'],bool(opts['megatron_moe_z_loss'])
            experts_mode=opts['megatron_moe_experts']
            name=lambda n: direction+'.moe.'+n
            def mo(n,rd,wr,flops=0,reread=0.0):
                emit(Kernel(e,name(n),'pointwise',tuple(rd),tuple(wr),float(flops),l2_reread=float(reread)))
            def mm(n,m,kk,nn,rd,wr):
                emit(Kernel(e,name(n),'gemm',tuple(rd),tuple(wr),0.0,gemm=(int(m),int(kk),int(nn))))
            def cat(n,src,dst,nbytes,chunks):
                parts=cat_launches(chunks)
                for i in range(parts):
                    lo,hi=nbytes*i/parts,nbytes*(i+1)/parts
                    mo(n if parts==1 else f'{n}.{i}',(a(src,hi-lo,lo),),(a(dst,hi-lo,lo),))
            def masked_select(n,mask,src,src_bytes,dst,dst_bytes):
                mo(n+'.nonzero_count',(a(mask,B1*E*T),),(a(s+'.'+n+'.count',I64),))
                mo(n+'.nonzero_select',(a(mask,B1*E*T),a(s+'.'+n+'.count',I64)),(a(s+'.'+n+'.nz',2*I64*Rs),))
                mo(n+'.index',(a(s+'.'+n+'.nz',2*I64*Rs),a(src,src_bytes)),(a(dst,dst_bytes),))
            lg,ln2=s+'.rlogits',s+'.ln2'
            if e=='router' and not back:
                mm('router.gemm',T,h,E,(a(ln2,ab*T*h),weight(w+'router',h*E)),(a(lg,ab*T*E),))
                if zl:
                    mo('zloss.logsumexp',(a(lg,ab*T*E),),(a(s+'.zlse',F32*T),),3*T*E)
                    mo('zloss.square',(a(s+'.zlse',F32*T),),(a(s+'.zsq',F32*T),),T)
                    mo('zloss.mean',(a(s+'.zsq',F32*T),),(a(s+'.zmean',F32),),T)
                    mo('zloss.coeff',(a(s+'.zmean',F32),),(a(s+'.zloss',F32),),1)
                    mo('zloss.tracker_div',(a(s+'.zloss',F32),),(a(s+'.ztrack',F32),),1)
                    mo('zloss.tracker_add',(a(s+'.ztrack',F32),a('input.moe_aux_tracker',F32)),(a('input.moe_aux_tracker',F32),),1)
                if score=='softmax_pre_topk':
                    mo('score.softmax',(a(lg,ab*T*E),),(a(s+'.rsc32',F32*T*E),),4*T*E)
                    mo('score.cast',(a(s+'.rsc32',F32*T*E),),(a(s+'.rsc',ab*T*E),))
                    mo('score.topk',(a(s+'.rsc',ab*T*E),),(a(s+'.rprobs',ab*T*k),a(s+'.ridx',I64*T*k)),T*E)
                else:
                    mo('score.topk',(a(lg,ab*T*E),),(a(s+'.rvals',ab*T*k),a(s+'.ridx',I64*T*k)),T*E)
                    mo('score.softmax',(a(s+'.rvals',ab*T*k),),(a(s+'.rprobs32',F32*T*k),),4*T*k)
                    mo('score.cast',(a(s+'.rprobs32',F32*T*k),),(a(s+'.rprobs',ab*T*k),))

                mo('gates.zeros',(),(a(s+'.rgates0',ab*T*E),))
                mo('gates.clone',(a(s+'.rgates0',ab*T*E),),(a(s+'.rgates',ab*T*E),))
                mo('gates.scatter',(a(s+'.ridx',I64*T*k),a(s+'.rprobs',ab*T*k)),(a(s+'.rgates',ab*T*k),))
                mo('map.zeros',(),(a(s+'.rmap0',ab*T*E),))
                mo('map.int',(a(s+'.rmap0',ab*T*E),),(a(s+'.rmapi',I32*T*E),))
                mo('map.clone',(a(s+'.rmapi',I32*T*E),),(a(s+'.rmapi2',I32*T*E),))
                mo('map.scatter',(a(s+'.ridx',I64*T*k),),(a(s+'.rmapi2',I32*T*k),))
                mo('map.bool',(a(s+'.rmapi2',I32*T*E),),(a(s+'.rmap',B1*T*E),))
                mo('map.tokens_per_expert',(a(s+'.rmap',B1*T*E),),(a(s+'.rtpe',I64*E),),T*E)
                if lb in ('aux_loss','aux_loss_zero_coeff'):
                    mo('aux.softmax',(a(lg,ab*T*E),),(a(s+'.asc32',F32*T*E),),4*T*E)
                if lb=='aux_loss':
                    mo('aux.probs_sum',(a(s+'.asc32',F32*T*E),),(a(s+'.aagg',F32*E),),T*E)
                    mo('aux.mul',(a(s+'.aagg',F32*E),a(s+'.rtpe',I64*E)),(a(s+'.aprod',F32*E),),E)
                    mo('aux.sum',(a(s+'.aprod',F32*E),),(a(s+'.asum',F32),),E)
                    mo('aux.coeff',(a(s+'.asum',F32),),(a(s+'.aux',F32),),1)
                    mo('aux.tracker_div',(a(s+'.aux',F32),),(a(s+'.atrack',F32),),1)
                    mo('aux.tracker_add',(a(s+'.atrack',F32),a('input.moe_aux_tracker',F32)),(a('input.moe_aux_tracker',F32),),1)
            elif e=='router':
                mo('gates.scatter_bwd',(a(s+'.ggates',ab*T*E),a(s+'.ridx',I64*T*k)),(a(s+'.gprobs',ab*T*k),),0)
                if score=='softmax_pre_topk':
                    mo('score.topk_bwd.zeros',(),(a(s+'.grsc',ab*T*E),))
                    mo('score.topk_bwd.scatter',(a(s+'.gprobs',ab*T*k),a(s+'.ridx',I64*T*k)),(a(s+'.grsc',ab*T*k),))
                    mo('score.cast_bwd',(a(s+'.grsc',ab*T*E),),(a(s+'.grsc32',F32*T*E),))
                    mo('score.softmax_bwd',(a(s+'.grsc32',F32*T*E),a(s+'.rsc32',F32*T*E)),(a(s+'.glg32',F32*T*E),),3*T*E)
                    mo('score.softmax_bwd_cast',(a(s+'.glg32',F32*T*E),),(a(s+'.glg',ab*T*E),))
                else:
                    mo('score.cast_bwd',(a(s+'.gprobs',ab*T*k),),(a(s+'.gprobs32',F32*T*k),))
                    mo('score.softmax_bwd',(a(s+'.gprobs32',F32*T*k),a(s+'.rprobs32',F32*T*k)),(a(s+'.gvals32',F32*T*k),),3*T*k)
                    mo('score.softmax_bwd_cast',(a(s+'.gvals32',F32*T*k),),(a(s+'.gvals',ab*T*k),))
                    mo('score.topk_bwd.zeros',(),(a(s+'.glg',ab*T*E),))
                    mo('score.topk_bwd.scatter',(a(s+'.gvals',ab*T*k),a(s+'.ridx',I64*T*k)),(a(s+'.glg',ab*T*k),))
                gl=s+'.glg'
                if lb=='aux_loss':
                    mo('aux.scaler_ones',(),(a(s+'.ascale1',F32),))
                    mo('aux.scaler_mul',(a(s+'.ascale1',F32),a('input.moe_aux_loss_scale',F32)),(a(s+'.agl',F32),),1)
                    mo('aux.coeff_bwd',(a(s+'.agl',F32),),(a(s+'.agsum',F32),),1)
                    mo('aux.mul_bwd',(a(s+'.agsum',F32),a(s+'.rtpe',I64*E)),(a(s+'.agagg',F32*E),),E)
                    mo('aux.softmax_bwd',(a(s+'.agagg',F32*E),a(s+'.asc32',F32*T*E)),(a(s+'.aglg32',F32*T*E),),3*T*E)
                    mo('aux.softmax_bwd_cast',(a(s+'.aglg32',F32*T*E),),(a(s+'.aglg',ab*T*E),))
                    mo('logits_grad_add',(a(gl,ab*T*E),a(s+'.aglg',ab*T*E)),(a(s+'.glg_sum',ab*T*E),),T*E)
                    gl=s+'.glg_sum'
                if zl:
                    mo('zloss.mean_bwd',(a(s+'.zsq',F32*T),),(a(s+'.gzsq',F32*T),),T)
                    mo('zloss.square_bwd',(a(s+'.gzsq',F32*T),a(s+'.zlse',F32*T)),(a(s+'.gzlse',F32*T),),2*T)
                    mo('zloss.logsumexp_bwd',(a(s+'.gzlse',F32*T),a(s+'.zlse',F32*T),a(lg,ab*T*E)),(a(s+'.gzlg',ab*T*E),),3*T*E)
                    mo('zloss.grad_add',(a(gl,ab*T*E),a(s+'.gzlg',ab*T*E)),(a(s+'.glg_z',ab*T*E),),T*E)
                    gl=s+'.glg_z'
                mm('router.dX',T,E,h,(a(gl,ab*T*E),weight(w+'router',h*E)),(a(s+'.grouter_x',ab*T*h),))
                temp=a(s+'.router.dw',ab*h*E)
                mm('router.dW',h,T,E,(a(ln2,ab*T*h),a(gl,ab*T*E)),(temp,))
                accum(w+'router',temp,h*E)

                mo('input_grad_add',(a(s+'.gacc1',ab*T*h),a(s+'.grouter_x',ab*T*h)),(a(s+'.gln2',ab*T*h),),T*h)
            elif e=='moe_dispatch' and not back:
                if pre:
                    mo('preprocess.tokens_per_expert',(a(s+'.rmap',B1*T*E),),(a(s+'.nlt',I64*E),),T*E)
                    if ep>1:
                        mo('preprocess.input_splits',(a(s+'.nlt',I64*E),),(a(s+'.isplit',I64*ep),),E)
                        mo('preprocess.local_slice',(a(s+'.nlt',I64*E),),(a(s+'.ngtl',I64*Es*ep),))
                        mo('preprocess.per_rank',(a(s+'.ngtl',I64*Es*ep),),(a(s+'.ngtr',I64*ep),),Es*ep)
                        mo('preprocess.per_rank_tp',(a(s+'.ngtr',I64*ep),),(a(s+'.ngtrt',I64*ep),),ep)
                        mo('preprocess.per_local_expert',(a(s+'.ngtl',I64*Es*ep),),(a(s+'.ntle',I64*Es),),Es*ep)

                    mo('permute.map_transpose',(a(s+'.rmap',B1*T*E),),(a(s+'.rmapT',B1*E*T),))
                    mo('permute.arange',(),(a(s+'.tidx',I64*T),))
                    masked_select('permute.sorted_indices',s+'.rmapT',s+'.tidx',I64*Rs,s+'.sidx',I64*Rs)
                    emit(Kernel(e,name('permute.index_select'),'pointwise',(a(s+'.sidx',I64*Rs),a(ln2,ab*T*h)),
                                (a(s+'.permuted',ab*Rs*h),),0.0,l2_reread=float(ab*(Rs-T)*h)))

                    mo('permute.probs_transpose',(a(s+'.rgates',ab*T*E),),(a(s+'.rgatesT',ab*E*T),))
                    masked_select('permute.probs',s+'.rmapT',s+'.rgatesT',ab*Rs,s+'.pprobs',ab*Rs)
                    if ep==1:
                        view(s+'.dispatched',[(s+'.permuted',ab*Rs*h)])
                    if work not in probs_events:
                        view(s+'.dprobs',[(s+'.pprobs',ab*Rs)])
            elif e=='moe_dispatch':
                if pre:
                    if Es>1:
                        cat('sort_chunks_bwd.tokens',s+'.gsorted_in',s+'.gdispatched',ab*R*h,ep*Es)
                        cat('sort_chunks_bwd.probs',s+'.gsprobs',s+'.gdprobs',ab*R,ep*Es)
                    else:
                        view(s+'.gdispatched',[(s+'.gsorted_in',ab*R*h)]); view(s+'.gdprobs',[(s+'.gsprobs',ab*R)])
                    if ep==1:
                        view(s+'.gpermuted',[(s+'.gdispatched',ab*Rs*h)])
                    if work not in probs_events:
                        view(s+'.gpprobs',[(s+'.gdprobs',ab*Rs)])
                if post:
                    mo('permute.index_select_bwd.zeros',(),(a(s+'.gperm_x',ab*T*h),))
                    mo('permute.index_select_bwd.index_add',(a(s+'.gpermuted',ab*Rs*h),a(s+'.sidx',I64*Rs),a(s+'.gperm_x',ab*T*h)),
                       (a(s+'.gperm_x',ab*T*h),),Rs*h)
                    def probs_bwd():
                        mo('permute.probs_bwd.zeros',(),(a(s+'.ggatesT',ab*E*T),))
                        mo('permute.probs_bwd.mask_scan',(a(s+'.rmapT',B1*E*T),),(a(s+'.mscan',I64*E*T),),E*T)
                        mo('permute.probs_bwd.masked_scatter',(a(s+'.mscan',I64*E*T),a(s+'.rmapT',B1*E*T),a(s+'.gpprobs',ab*Rs)),
                           (a(s+'.ggatesT',ab*E*T),))
                        mo('permute.probs_transpose_bwd',(a(s+'.ggatesT',ab*E*T),),(a(s+'.ggates',ab*T*E),))
                    if work in probs_events:
                        after_probs[work]=(op.uid,probs_bwd)
                    else:
                        probs_bwd()
                    if Ish:
                        mo('input_grad_add',(a(s+'.gx_shared',ab*T*h),a(s+'.gperm_x',ab*T*h)),(a(s+'.gacc1',ab*T*h),),T*h)
                    else:
                        view(s+'.gacc1',[(s+'.gperm_x',ab*T*h)])
            elif e=='MLP':
                n2=2*I; offs=[sum(local[:j]) for j in range(Es)]
                def act_fwd(prefix,rows,off,probs):
                    fc1,act=s+'.e_fc1',s+'.e_act'
                    if opts['fuse_activation']:
                        mo(prefix+'weighted_swiglu',(a(fc1,ab*rows*n2,ab*off*n2),a(probs,ab*rows,ab*off)),
                           (a(act,ab*rows*I,ab*off*I),),5*rows*I)
                    else:
                        pw('activation','forward',(a(fc1,ab*rows*I,ab*off*n2),a(fc1,ab*rows*I,ab*off*n2+ab*rows*I)),
                           (a(s+'.e_glu',ab*rows*I,ab*off*I),),rows*I,rows,s+'.e_activation'+prefix)
                        mo(prefix+'probs_mul',(a(s+'.e_glu',ab*rows*I,ab*off*I),a(probs,ab*rows,ab*off)),
                           (a(act,ab*rows*I,ab*off*I),),rows*I)
                def act_bwd(prefix,rows,off,probs,gprobs):
                    fc1,gact,gfc1=s+'.e_fc1',s+'.e_gact',s+'.e_gfc1'
                    if opts['fuse_activation']:
                        mo(prefix+'weighted_swiglu_bwd',(a(gact,ab*rows*I,ab*off*I),a(fc1,ab*rows*n2,ab*off*n2),a(probs,ab*rows,ab*off)),
                           (a(gfc1,ab*rows*n2,ab*off*n2),),10*rows*I)
                        mo(prefix+'weighted_swiglu_bwd.probs',(a(gact,ab*rows*I,ab*off*I),a(fc1,ab*rows*n2,ab*off*n2)),
                           (a(gprobs,ab*rows,ab*off),),6*rows*I)
                    else:
                        mo(prefix+'probs_mul_bwd.input',(a(gact,ab*rows*I,ab*off*I),a(probs,ab*rows,ab*off)),
                           (a(s+'.e_gglu',ab*rows*I,ab*off*I),),rows*I)
                        mo(prefix+'probs_mul_bwd.probs',(a(gact,ab*rows*I,ab*off*I),a(s+'.e_glu',ab*rows*I,ab*off*I)),
                           (a(s+'.e_gprod',ab*rows*I,ab*off*I),),rows*I)
                        mo(prefix+'probs_mul_bwd.sum',(a(s+'.e_gprod',ab*rows*I,ab*off*I),),(a(gprobs,ab*rows,ab*off),),rows*I)
                        pw('activation','backward',(a(s+'.e_gglu',ab*rows*I,ab*off*I),a(fc1,ab*rows*I,ab*off*n2),
                                                    a(fc1,ab*rows*I,ab*off*n2+ab*rows*I)),
                           (a(gfc1,ab*rows*I,ab*off*n2),a(gfc1,ab*rows*I,ab*off*n2+ab*rows*I)),rows*I,rows,s+'.e_activation'+prefix)
                experts=[(j,first+j,local[j],offs[j]) for j in range(Es) if local[j]]
                seq_mode=experts_mode=='sequential'
                def fc(direction_,which,j,g,n,off):
                    wn=w+f'expert{g}.'+which
                    kin,kout=(h,n2) if which=='fc1' else (I,h)

                    xin,yout,gy,gx=((s+'.e_in',s+'.e_fc1',s+'.e_gfc1',s+('.e_gin_parts' if seq_mode else '.e_gin')) if which=='fc1' else
                                    (s+'.e_act',s+('.e_out_parts' if seq_mode else '.e_out'),s+'.e_gout',s+'.e_gact'))
                    if direction_=='forward':
                        mm(f'expert{g}.{which}',n,kin,kout,(a(xin,ab*n*kin,ab*off*kin),weight(wn,kin*kout)),(a(yout,ab*n*kout,ab*off*kout),))
                    elif direction_=='dX':
                        mm(f'expert{g}.{which}.dX',n,kout,kin,(a(gy,ab*n*kout,ab*off*kout),weight(wn,kin*kout)),(a(gx,ab*n*kin,ab*off*kin),))
                    else:
                        temp=a(s+f'.e{g}.{which}.dw',ab*kin*kout)
                        mm(f'expert{g}.{which}.dW',kin,n,kout,(a(xin,ab*n*kin,ab*off*kin),a(gy,ab*n*kout,ab*off*kout)),(temp,))
                        return temp
                if not back:
                    if Es>1:
                        cat('sort_chunks.tokens',s+'.dispatched',s+'.e_in',ab*R*h,ep*Es)
                        cat('sort_chunks.probs',s+'.dprobs',s+'.e_probs',ab*R,ep*Es)
                    else:
                        view(s+'.e_in',[(s+'.dispatched',ab*R*h)]); view(s+'.e_probs',[(s+'.dprobs',ab*R)])
                    if experts_mode=='te_grouped':
                        for j,g,n,off in experts: fc('forward','fc1',j,g,n,off)
                        act_fwd('',R,0,s+'.e_probs')
                        for j,g,n,off in experts: fc('forward','fc2',j,g,n,off)
                    else:
                        for j,g,n,off in experts:
                            fc('forward','fc1',j,g,n,off); act_fwd(f'expert{g}.',n,off,s+'.e_probs'); fc('forward','fc2',j,g,n,off)
                        cat('sequential.output_cat',s+'.e_out_parts',s+'.e_out',ab*R*h,len(experts))
                else:
                    if experts_mode=='te_grouped':
                        for which,width in (('fc2',I*h),):
                            for j,g,n,off in experts: fc('dX',which,j,g,n,off)
                            temps=[(g,fc('dW',which,j,g,n,off)) for j,g,n,off in experts]
                            for g,temp in temps: accum(w+f'expert{g}.{which}',temp,width)
                        act_bwd('',R,0,s+'.e_probs',s+'.e_gprobs')
                        for j,g,n,off in experts: fc('dX','fc1',j,g,n,off)
                        temps=[(g,fc('dW','fc1',j,g,n,off)) for j,g,n,off in experts]
                        for g,temp in temps: accum(w+f'expert{g}.fc1',temp,h*n2)
                    else:
                        for j,g,n,off in reversed(experts):
                            fc('dX','fc2',j,g,n,off); accum(w+f'expert{g}.fc2',fc('dW','fc2',j,g,n,off),I*h)
                            act_bwd(f'expert{g}.',n,off,s+'.e_probs',s+'.e_gprobs_parts')
                            fc('dX','fc1',j,g,n,off); accum(w+f'expert{g}.fc1',fc('dW','fc1',j,g,n,off),h*n2)
                        cat('sequential.input_grad_cat',s+'.e_gin_parts',s+'.e_gin',ab*R*h,len(experts))
                        cat('sequential.probs_grad_cat',s+'.e_gprobs_parts',s+'.e_gprobs',ab*R,len(experts))
                    view(s+'.gsorted_in',[(s+'.e_gin',ab*R*h)]); view(s+'.gsprobs',[(s+'.e_gprobs',ab*R)])
            elif e=='moe_combine' and not back:
                if pre:
                    if Es>1:
                        cat('restore_chunks',s+'.e_out',s+'.combine_in',ab*R*h,ep*Es)
                    else:
                        view(s+'.combine_in',[(s+'.e_out',ab*R*h)])
                    if ep==1:
                        view(s+'.combined',[(s+'.combine_in',ab*Rs*h)])
                if post:
                    mo('unpermute.zeros',(),(a(s+'.unperm',ab*T*h),))
                    mo('unpermute.scatter_add',(a(s+'.combined',ab*Rs*h),a(s+'.sidx',I64*Rs),a(s+'.unperm',ab*T*h)),
                       (a(s+'.unperm',ab*T*h),),Rs*h)
                    out=s+'.unperm'
                    if Ish:
                        mm('shared.fc1',T,h,2*Ish,(a(ln2,ab*T*h),weight(w+'shared.fc1',h*2*Ish)),(a(s+'.sh_fc1',ab*T*2*Ish),))
                        pw('activation','forward',(a(s+'.sh_fc1',ab*T*Ish),a(s+'.sh_fc1',ab*T*Ish,ab*T*Ish)),
                           (a(s+'.sh_act',ab*T*Ish),),T*Ish,T,s+'.sh_activation')
                        mm('shared.fc2',T,Ish,h,(a(s+'.sh_act',ab*T*Ish),weight(w+'shared.fc2',Ish*h)),(a(s+'.sh_out',ab*T*h),))
                        mo('shared.add',(a(out,ab*T*h),a(s+'.sh_out',ab*T*h)),(a(s+'.down',ab*T*h),),T*h)
                    else:
                        view(s+'.down',[(out,ab*T*h)])
                    residual('forward',s+'.r1',s+'.down',('recompute.' if op.recompute else '')+f'{d}.x{l+1}',s+'.residual2')
            elif e=='moe_combine':
                go=f'{d}.g{l+1}'
                if pre:
                    if Ish:
                        mm('shared.fc2.dX',T,h,Ish,(a(go,ab*T*h),weight(w+'shared.fc2',Ish*h)),(a(s+'.sh_gact',ab*T*Ish),))
                        temp=a(s+'.shared.fc2.dw',ab*Ish*h)
                        mm('shared.fc2.dW',Ish,T,h,(a(s+'.sh_act',ab*T*Ish),a(go,ab*T*h)),(temp,)); accum(w+'shared.fc2',temp,Ish*h)
                        pw('activation','backward',(a(s+'.sh_gact',ab*T*Ish),a(s+'.sh_fc1',ab*T*Ish),a(s+'.sh_fc1',ab*T*Ish,ab*T*Ish)),
                           (a(s+'.sh_gfc1',ab*T*Ish),a(s+'.sh_gfc1',ab*T*Ish,ab*T*Ish)),T*Ish,T,s+'.sh_activation')
                        mm('shared.fc1.dX',T,2*Ish,h,(a(s+'.sh_gfc1',ab*T*2*Ish),weight(w+'shared.fc1',h*2*Ish)),(a(s+'.gx_shared',ab*T*h),))
                        temp=a(s+'.shared.fc1.dw',ab*h*2*Ish)
                        mm('shared.fc1.dW',h,T,2*Ish,(a(ln2,ab*T*h),a(s+'.sh_gfc1',ab*T*2*Ish)),(temp,)); accum(w+'shared.fc1',temp,h*2*Ish)

                    emit(Kernel(e,name('unpermute.scatter_add_bwd'),'pointwise',(a(go,ab*T*h),a(s+'.sidx',I64*Rs)),
                                (a(s+'.gcombined',ab*Rs*h),),0.0,l2_reread=float(ab*(Rs-T)*h)))
                    if ep==1:
                        view(s+'.gcombine_in',[(s+'.gcombined',ab*Rs*h)])
                if post:
                    if Es>1:
                        cat('restore_chunks_bwd',s+'.gcombine_in',s+'.e_gout',ab*R*h,ep*Es)
                    else:
                        view(s+'.e_gout',[(s+'.gcombine_in',ab*R*h)])
        for current in program.ops:
            op=current
            if isinstance(op,TransferOp):
                if op.is_control or op.src_device==op.dst_device or device not in (op.src_device,op.dst_device): continue
                buf=a(pp_buffer(op),op.size_bytes); send=device==op.src_device
                marker(op,op.uid,'pipeline.send' if send else 'pipeline.recv',(buf,) if send else (),() if send else (buf,))
                continue
            if op.device!=device or op.work is None: continue
            work=op.work; b,l=work.microbatch,work.layer
            backward=work.direction.name=='BACKWARD'; direction='backward' if backward else 'forward'
            d=f'b{b}'; s=f'{d}.L{l}'
            if l is not None and (getattr(op,'recompute',False) or work.kind.name=='RECOMPUTE' or (backward and timing.full_recomputation)): s='recompute.'+s
            x=f'{d}.x{l}'; gi=f'{d}.g{l}'; go=f'{d}.g{l+1}' if l is not None else ''
            e=entry(op); w=f'param.L{l}.'
            if isinstance(op,CollectiveOp):
                size,parts,key=op.size_bytes,max(1,op.participants),op.comm_key or op.name
                if op.interconnect=='cp' and op.coll.value in ('all_gather','reduce_scatter') and any(n in key for n in ('qkv_proj','output_proj','attention')):
                    if op.coll.value=='all_gather':
                        marker(op,op.uid,key,(a(s+'.krot',size/2),a(s+'.v',size/2)),
                               (a(s+'.kag',size*parts/2),a(s+'.vag',size*parts/2)))

                        for name in ('k','v'):
                            point('cp_reorder', (a(s+'.'+name+'ag',size*parts/2),),
                                  (a(s+'.'+name+'full',size*parts/2),),op='attention',uid=attention_owner[work])
                    elif op.coll.value=='reduce_scatter':
                        for name in ('k','v'):
                            point('cp_grad_reorder',(a(s+'.g'+name+'full',ab*batch*seq*kwidth),),
                                  (a(s+'.g'+name+'ordered',size/2),),op='attention',uid=attention_owner[work])
                        marker(op,op.uid,key,(a(s+'.gkordered',size/2),a(s+'.gvordered',size/2)),
                               (a(s+'.gkrot',size/parts/2),a(s+'.gv',size/parts/2)))
                    continue
                if op.is_dp and dpx is not None:
                    expert = str(key).startswith('mdp_') and '_expert' in str(key)
                    dpg, buffer = (dpx_expert, dp_weights(True)) if expert else (dpx, dp_weights(False))
                    group = dp_reducer_group(key)
                    if group is None:
                        raise ValueError(f'Megatron data-parallel collective {key} has no grad-buffer owner')
                    if group=='LAYER':
                        left = dp_layers.setdefault((op.coll.value, expert), sorted(
                            {int(n.split('.')[1][1:]) for n in buffer if n.startswith('param.L')}))
                        if not left:
                            raise ValueError(f'Megatron data-parallel {op.coll.value} collectives on device {device} '
                                             f'outnumber the stage layers')
                        layer = work.layer if work.kind.name=='LAYER' and work.layer in left else left[0]
                        left.remove(layer)
                    names = ([n for n in ('param.embedding',) if n in buffer] if group=='EMBEDDING' else
                             [n for n in ('param.head','param.final_norm') if n in buffer] if group=='SOFTMAX'
                             else [n for n in buffer if n.startswith(f'param.L{layer}.')])
                    dp_ops.append((op, names, expert))

                    size = gb*sum(weights[n] for n in names)
                    if op.coll.value=='all_gather':
                        rd,wr=dpg.param_gather(buffer,names,size)
                    else:
                        anchor=next(d for d in op.deps if isinstance(program.ops[d],ComputeOp) and program.ops[d].device==device)
                        before,rd,wr=dpg.grad_sync(buffer,names,size,op.coll.value)
                        for k in before: emit(k,anchor)
                    marker(op,op.uid,key,rd,wr)
                    continue
                if use_moe and key.startswith('ep_dense_sync'):
                    ranges,remaining=[],float(size)
                    own=[n for n in weights if n.startswith(f'param.L{l}.') and '.expert' not in n]
                    rest=[n for n in weights if n.startswith('param.L') and '.expert' not in n and n not in own]
                    for name in own+rest[::-1]:
                        if remaining>0:
                            count=min(remaining,gb*weights[name]); ranges.append(a(name+'.main_grad',count)); remaining-=count
                    marker(op,op.uid,key,ranges,ranges)
                    continue
                if op.is_dp:
                    names = [name for name in weights if l is None or name.startswith(f'param.L{l}.')]
                    remaining = float(size)
                    ranges = []
                    for name in names:
                        count = min(remaining,gb*weights[name])
                        if count > 0:
                            ranges.append(a(name+'.main_grad',count))
                            remaining -= count
                    if remaining > 0:
                        raise ValueError(f'Megatron source gradient collective {key} exceeds its owned main-grad buffers by {remaining} bytes')
                    marker(op,op.uid,key,ranges,ranges)
                    continue
                if 'token_count_all_gather' in key or 'probs_all_to_all' in key:
                    if 'token_count_all_gather' in key:
                        rd,wr=(a(s+'.nlt',size),),(a(s+'.ngt',size*parts),)
                    elif backward:
                        rd,wr=(a(s+'.gdprobs',size),),(a(s+'.gpprobs',size),)
                    else:
                        rd,wr=(a(s+'.pprobs',size),),(a(s+'.dprobs',size),)
                    marker(op,op.uid,key,rd,wr)
                    if backward and work in after_probs:
                        previous_owner=owner
                        owner,run_after=after_probs.pop(work)
                        run_after()
                        owner=previous_owner
                    continue
                if op.coll.value=='all_to_all' and ('moe_dispatch' in key or 'moe_combine' in key):
                    src,dst=(('.gdispatched','.gpermuted') if 'moe_dispatch' in key else ('.gcombined','.gcombine_in')) if backward \
                        else (('.permuted','.dispatched') if 'moe_dispatch' in key else ('.combine_in','.combined'))
                    marker(op,op.uid,key,(a(s+src,size),),(a(s+dst,size),)); continue
                buf=(s+'.ao' if not backward else s+'.gattn') if 'output_proj' in key else (
                    s+'.gln1' if 'qkv_proj' in key else s+('.gln2' if backward else '.down') if 'MLP' in key else
                    s+('.gln1' if backward else '.ln1') if 'layernorm1' in key else
                    s+('.gln2' if backward else '.ln2') if 'layernorm2' in key else 'param.collective.'+key)
                access=a(buf,size); marker(op,op.uid,key,(access,),(access,)); continue
            if not isinstance(op,ComputeOp) or op.launch_shadow: continue
            if backward and rc_mods and l is not None and work.kind.name=='LAYER':
                ln1,ln2=recompute_layernorm(timing,l)
                if ln1 and e=='output_proj':
                    recompute((s+'.ln1',),lambda: norm('forward',x,s+'.ln1','','',w+'norm1'))
                if mla and 'mla_up_proj' in rc_mods and e=='attention':
                    recompute((s+'.mla_query',s+'.mla_key',s+'.mla_value'),
                              lambda: mla_block('qkv_proj','forward',s,w,'up'))
                if ln2 and e in ('MLP','router','moe_dispatch','moe_combine') and work not in ln2_done:
                    ln2_done.add(work)
                    recompute((s+'.ln2',),lambda: norm('forward',s+'.r1',s+'.ln2','','',w+'norm2'))
            if mla and e in ('qkv_proj','rope','attention','output_proj'):
                if op.split_head or (work,e) in split:
                    raise NotImplementedError('Megatron MLA description does not describe split dgrad/wgrad operators')
                if e!='rope': mla_block(e,direction,s,w)
            elif e=='layernorm1':
                norm(direction,x,s+'.ln1',s+'.gln1',s+'.gxnorm',w+'norm1')
                if backward: residual('backward',s+'.gxnorm',s+'.gr1',gi,s+'.residual1')
            elif e=='qkv_proj':
                part='dx' if op.split_head else 'dw' if (work,e) in split else 'all'
                if backward and part!='dw':
                    rotary('backward',s,w)
                linear(e,direction,s+'.ln1',s+'.qkv',s+'.gqkv',s+'.gln1',w+'qkv',part,w+'qkv_bias' if qkv_bias else None)
                if not backward:
                    view(s+'.qkv',[(s+'.q',ab*tokens*qwidth),(s+'.k',ab*tokens*kwidth),(s+'.v',ab*tokens*kwidth)])
                    if copies:
                        point('forward.q_reshape_copy',(a(s+'.q',ab*tokens*qwidth),),(a(s+'.qc',ab*tokens*qwidth),),op='qkv_proj')
                    if timing.model_type=='qwen3':
                        for name,width in (('q',qwidth),('k',kwidth)):
                            if copies and name=='k':
                                point('forward.k_reshape_copy',(a(s+'.k',ab*tokens*kwidth),),(a(s+'.kc',ab*tokens*kwidth),),op='qkv_proj')
                            norm('forward',split_source(s,name),s+'.'+name+'norm','','',w+name+'norm',tokens*width//hd,hd,True,width//hd)
                    rotary('forward',s,w)
            elif e=='rope':
                for name,width in (('q',qwidth),('k',kwidth)):
                    inp=s+'.g'+name+'rot' if backward else s+'.'+name+('norm' if timing.model_type=='qwen3' else '')
                    out=s+'.g'+name if backward else s+'.'+name+'rot'
                    pw('rope',direction,(a(inp,ab*tokens*width),a('input.cos',ab*seq*hd),a('input.sin',ab*seq*hd)),
                       (a(out,ab*tokens*width),),tokens*width,tokens,s+'.rope.'+name,width//hd)
                if backward:
                    for name,width in (('q',qwidth),('k',kwidth)):
                        if timing.model_type=='qwen3':
                            norm('backward',s+'.'+name,s+'.'+name+'norm',s+'.g'+name,s+'.g'+name+'raw',w+name+'norm',tokens*width//hd,hd,True,width//hd)
                    view(s+'.gqkv',[(s+'.g'+name+('raw' if timing.model_type=='qwen3' and name!='v' else ''),ab*tokens*width)
                                    for name,width in (('q',qwidth),('k',kwidth),('v',kwidth))])
            elif e=='attention':
                chunks=cp_query_chunks(seq,cp,cprank)
                qin,kin,vin,oout='.qrot','.krot','.v','.attn'
                if copies:
                    vin='.vc'
                    if not backward:
                        point('forward.v_contiguous_copy',(a(s+'.v',ab*tokens*kwidth),),(a(s+'.vc',ab*tokens*kwidth),),op='attention')
                    if batch>1:
                        qin,kin,vin,oout='.qb','.kb','.vb','.attnb'
                        if not backward:
                            for name,source,width in (('q','.qrot',qwidth),('k','.krot',kwidth),('v','.vc',kwidth)):
                                point(f'forward.{name}_bshd_copy',(a(s+source,ab*tokens*width),),
                                      (a(s+f'.{name}b',ab*tokens*width),),op='attention')
                if backward and cp>1:
                    point('backward.cp_zero',(),(a(s+'.gkfull',ab*batch*seq*kwidth),a(s+'.gvfull',ab*batch*seq*kwidth)),op='attention')
                for ci,(chunk,kv) in enumerate(chunks):
                    key=(backward,chunk,kv)
                    if key not in cache:
                        spec=FlashSpec(q_heads=qwidth//hd,kv_heads=kwidth//hd,head_dim=hd,batch=batch,
                            q_len=chunk,kv_len=kv,op='bwd' if backward else 'fwd',source=opts['flash_source'],
                            rates=rates,sm_count=int(timing.core.num_bundle),
                            fast_forward=_flash_fast_forward(backward))
                        cache[key]=(spec, flash_kernels(spec))
                    off=ci*ab*batch*chunk*qwidth
                    mapping={'q':(s+qin,off),'k':(s+('.kfull' if cp>1 else kin),0),'v':(s+('.vfull' if cp>1 else vin),0),
                             'o':(s+(f'.flash{ci}.out' if cp>1 else oout),0 if cp>1 else off),
                             'do':(s+'.gattn',off),'dq':(s+(f'.flash{ci}.dq' if cp>1 else '.gqrot'),0 if cp>1 else off),
                             'dk':(s+(f'.chunk{ci}.gk' if cp>1 else '.gkrot'),0),'dv':(s+(f'.chunk{ci}.gv' if cp>1 else '.gv'),0)}
                    spec, records = cache[key]
                    for flash_index, rec in enumerate(records):
                        def fa(n,size):
                            buf,start=mapping.get(n,(f'{s}.flash{ci}.{n}',0))
                            return a(buf,size,start)
                        flash_specs[len(ks)] = (spec, flash_index)
                        emit(Kernel('attention',direction+'.'+rec.name,'attention',
                             tuple(fa(n,nbytes) for n,nbytes in rec.reads),tuple(fa(n,nbytes) for n,nbytes in rec.writes)),record=rec)
                    if backward and cp>1:
                        point('backward.cp_dq_copy',(a(s+f'.flash{ci}.dq',ab*batch*chunk*qwidth),),
                              (a(s+'.gqrot',ab*batch*chunk*qwidth,off),),op='attention')
                        for n in ('k','v'):
                            target=a(s+'.g'+n+'full',ab*batch*kv*kwidth)
                            point('backward.cp_chunk_accum',(a(s+f'.chunk{ci}.g'+n,ab*batch*kv*kwidth),target),
                                  (target,),batch*kv*kwidth,'attention')
                    elif cp>1:
                        point('forward.cp_output_copy',(a(s+f'.flash{ci}.out',ab*batch*chunk*qwidth),),
                              (a(s+'.attn',ab*batch*chunk*qwidth,off),),op='attention')
                if copies and batch>1 and not backward:
                    point('forward.output_sbhd_copy',(a(s+'.attnb',ab*tokens*qwidth),),(a(s+'.attn',ab*tokens*qwidth),),op='attention')
            elif e=='output_proj':
                linear(e,direction,s+'.attn',s+'.ao',s+'.gr1',s+'.gattn',w+'output')
            elif e=='layernorm2':
                if not backward: residual('forward',x,s+'.ao',s+'.r1',s+'.residual1')
                norm(direction,s+'.r1',s+'.ln2',s+'.gln2',s+'.gr1norm',w+'norm2')
                if backward: residual('backward',s+'.gr1norm',go,s+'.gr1',s+'.residual2')
            elif e in ('router','moe_dispatch','moe_combine') or (e=='MLP' and moe_layer(l)):
                if not moe_layer(l):
                    raise ValueError(f'Megatron MoE operator {op.name} outside a MoE layer')
                moe(e,direction,s,w,d,l,op,work)
            elif e=='MLP':
                issplit=(work,e) in split
                n=operator_shapes(timing,'ffn1','forward')[0][2]//2
                if not backward:
                    linear('ffn1',direction,s+'.ln2',s+'.gateup','','',w+'ffn1')
                    view(s+'.gateup',[(s+'.gate',ab*tokens*n),(s+'.up',ab*tokens*n)])
                    pw('activation',direction,(a(s+'.gate',ab*tokens*n),a(s+'.up',ab*tokens*n)),
                       (a(s+'.mlp',ab*tokens*n),),tokens*n,tokens,s+'.activation')
                    linear('ffn2',direction,s+'.mlp',s+'.down','','',w+'ffn2')
                    args=('forward',s+'.r1',s+'.down',('recompute.' if op.recompute else '')+f'{d}.x{l+1}',s+'.residual2')
                    if work in mlp_comm:
                        deferred_residuals[work]=(op.uid,args)
                    else:
                        residual(*args)
                else:
                    if not issplit or op.split_head:
                        linear('ffn2',direction,s+'.mlp',s+'.down',go,s+'.gmlp',w+'ffn2')
                        pw('activation',direction,(a(s+'.gmlp',ab*tokens*n),a(s+'.gate',ab*tokens*n),a(s+'.up',ab*tokens*n)),
                           (a(s+'.ggate',ab*tokens*n),a(s+'.gup',ab*tokens*n)),tokens*n,tokens,s+'.activation')
                        view(s+'.ggateup',[(s+'.ggate',ab*tokens*n),(s+'.gup',ab*tokens*n)])
                    linear('ffn1',direction,s+'.ln2',s+'.gateup',s+'.ggateup',s+'.gln2',w+'ffn1',
                           'dx' if op.split_head else 'dw' if issplit else 'all')
            elif work.kind.name=='EMBEDDING':
                count,_=timing._embedding_softmax_params_per_rank(h,int(timing.vocab_size))
                wa=weight('param.embedding',int(count))
                if not backward:
                    relayout=copies and batch>1
                    point('forward.embedding',(a('input.tokens',I64*tokens),a('param.embedding.rows',pb*tokens*h/tp,weight=True)),
                          (a(d+('.x0.bsh' if relayout else '.x0'),ab*tokens*h),),op='embedding')
                    if relayout:
                        point('forward.embedding_sbh_copy',(a(d+'.x0.bsh',ab*tokens*h),),(a(d+'.x0',ab*tokens*h),),op='embedding')
                else:
                    relayout=copies and batch>1
                    if relayout:
                        point('backward.embedding_bsh_copy',(a(d+'.g0',ab*tokens*h),),(a(d+'.g0.bsh',ab*tokens*h),),op='embedding')
                    temp=a(d+'.embedding.dw',pb*count)
                    rows_only=opts['megatron_embedding_backward']=='token_rows'
                    touched=a(d+'.embedding.dw',pb*tokens*h/tp) if rows_only else temp
                    g0=a(d+('.g0.bsh' if relayout else '.g0'),ab*tokens*h)
                    if rows_only and tokens>te.EMBEDDING_SORT_THRESHOLD:
                        before,after=te.embedding_backward_sorted(g0,a('input.tokens',I64*tokens),touched,
                                                                  count=tokens,hidden=h/tp,prefix=d+'.embedding')
                        for k in before: emit(k)
                        point('backward.embedding_zero',(),(temp,),op='embedding')
                        for k in after: emit(k)
                    else:
                        point('backward.embedding_zero',(),(temp,),op='embedding')
                        point('backward.embedding',(g0,a('input.tokens',I64*tokens),touched),(touched,),tokens*h/tp,'embedding')
                    for head,n in tied_pending:
                        point('backward.tied_grad_add',(head,temp),(temp,),n,'embedding')
                    tied_pending.clear()
                    accum('param.embedding',temp,int(count))
            elif work.kind.name=='SOFTMAX':
                last=int(timing.num_layers); headw='param.embedding' if timing.tied_embeddings else 'param.head'
                desc=operator_shapes(timing,'linear_softmax','forward')[0]; rows=desc[0]*desc[3]; vocab=desc[2]
                if not backward:
                    norm('forward',d+'.x'+str(last),d+'.final','','','param.final_norm')
                    linear('linear_softmax',direction,d+'.final',d+'.logits','','',headw)
                    pw('loss',direction,(a(d+'.logits',ab*rows*vocab),a('input.labels',I64*rows)),
                       (a(d+'.loss',sb*rows),a(d+'.loss.stats',sb*rows)),rows*vocab,rows,d+'.loss')
                else:
                    pw('loss',direction,(a(d+'.logits',ab*rows*vocab),a(d+'.loss.stats',sb*rows),a('input.labels',I64*rows),a('input.gloss',sb)),
                       (a(d+'.glogits',ab*rows*vocab),),rows*vocab,rows,d+'.loss')
                    linear('linear_softmax',direction,d+'.final',d+'.logits',d+'.glogits',d+'.gfinal',headw)
                    norm('backward',d+'.x'+str(last),d+'.final',d+'.gfinal',d+'.g'+str(last),'param.final_norm')
            elif work.kind.name=='OPTIMIZER' and opts['megatron_optimizer']=='te_fused_adam':
                if tied_pending:
                    raise ValueError('megatron_tied_grad=autograd_sum requires the embedding and the head on one device')
                if bool(getattr(timing.model,'gradient_clipping',False)):
                    for name,n in tuple(weights.items()):
                        g=a(name+'.main_grad',gb*n); point('optimizer.grad_clip',(g,),(g,),4*n,'optimizer')
                sharded = dpx.optimizer(dp_weights(False)) if dpx is not None else None
                if sharded is not None and dp_weights(True):
                    sharded = sharded + dpx_expert.optimizer(dp_weights(True))
                if sharded is not None:
                    for k in sharded: emit(k)
                else:
                    for k in te.fused_adam(dict(weights),p): emit(k)
                    for name,n in tuple(weights.items()):
                        point('optimizer.param_copy',(a(name+'.master',p.master_parameters*n),),(weight(name,n),),op='optimizer')
            elif work.kind.name=='OPTIMIZER':
                if tied_pending:
                    raise ValueError('megatron_tied_grad=autograd_sum requires the embedding and the head on one device')
                for name,n in tuple(weights.items()):
                    g=a(name+'.main_grad',gb*n)
                    if bool(getattr(timing.model,'gradient_clipping',False)): point('optimizer.grad_clip',(g,),(g,),4*n,'optimizer')
                    main,m1,m2=a(name+'.master',p.master_parameters*n),a(name+'.m1',p.optimizer_states*n),a(name+'.m2',p.optimizer_states*n)
                    point('optimizer.adam',(main,g,m1,m2),(main,m1,m2),ADAMW_FLOPS*n,'optimizer')
                    point('optimizer.param_copy',(main,),(weight(name,n),),op='optimizer')
            for call in op.embedded_collectives:
                buf=d+'.loss.stats' if work.kind.name=='SOFTMAX' and not backward else (
                    d+'.x0' if work.kind.name=='EMBEDDING' and not backward else d+'.g'+str(timing.num_layers)
                    if work.kind.name=='SOFTMAX' else 'param.embedding.main_grad')
                access=a(buf,call.size_bytes); marker(call,op.uid,call.name,(access,),(access,))
        if deferred_residuals:
            raise ValueError('Megatron MLP residual is missing its source collective marker')
        if after_probs:
            raise ValueError('Megatron MoE probs gradient is missing its all-to-all event')
        if not any(o.work.kind.name=='OPTIMIZER' for o in local):
            zeros=[]
        elif opts['megatron_zero_grad']=='grad_buffer':
            zeros=[Kernel('optimizer','optimizer.zero_grad_buffer','pointwise',(),
                          tuple(a(w+'.main_grad',gb*n,gb*start) for w,start,n in launch))
                   for group in ({w:n for w,n in weights.items() if ('.expert' in w)==e} for e in (False,True)) if group
                   for launch in te.split_32bit(list(group.items()),gb)]
        else:
            zeros=[Kernel('optimizer','optimizer.zero_main_grad','pointwise',(),(a(w+'.main_grad',gb*n),)) for w,n in weights.items()]
        delta=len(zeros)
        ks=zeros+[replace(k,reads=tuple(map(resolve,k.reads)),writes=tuple(map(resolve,k.writes))) for k in ks]
        ss=[Source(optimizer)]*delta+ss; flash={i+delta:r for i,r in flash.items()}
        flash_specs={i+delta:r for i,r in flash_specs.items()}
        dp_payload={}
        if dp_ops:
            sizes={False:dense_group,True:expert_group}
            spans={e:dist_trace.buffer_spans(list(dp_weights(e).items()),sizes[e],distributed_optimizer(timing))
                   for e in {e for _,_,e in dp_ops}}
            for kind,e in {(o.coll.value,e) for o,_,e in dp_ops}:
                carried=[n for o,names,x in dp_ops if o.coll.value==kind and x==e for n in names]

                orphans=[n for n in spans[e] if n not in carried]
                host=next((names for o,names,x in dp_ops if o.coll.value==kind and x==e and 'param.embedding' in names),None)
                if orphans and set(orphans)<={'param.final_norm'} and host is not None:
                    host.extend(orphans); carried+=orphans
                if sorted(carried)!=sorted(spans[e]):
                    raise ValueError(f'Megatron data-parallel {kind} collectives on device {device} do not carry the '
                                     f'grad buffer once: {sorted(carried)} vs {sorted(spans[e])}')
            one_bucket = opts.get('megatron_dp_buckets', 'per_reducer') == 'one'
            first = {}
            for o,_,e in dp_ops:
                first.setdefault((o.coll.value,e), o.uid)
            for o,names,e in dp_ops:
                elements=sum(spans[e][n] for n in names)
                if one_bucket:
                    elements=sum(spans[e].values()) if o.uid==first[(o.coll.value,e)] else 0
                dp_payload[o.uid]=float(pb*elements/sizes[e] if o.coll.value=='all_gather' else gb*elements)
        result[device]=RankTrace(tuple(ks),tuple(ss),rules,flash,flash_specs,dp_payload)
    return result
