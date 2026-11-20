import argparse
import json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _environment import reject_model_environment
reject_model_environment()

R = Path(__file__).resolve().parents[2]

def model_yaml(model, phase, batch, seq, config_dir, implementation, decode_kv_on_entry):
    sys.path.insert(0, str(R))
    from paper_pipeline.workload import model_yaml as build_model
    import yaml
    registry = yaml.safe_load((R / 'paper_pipeline/registry.yaml').read_text())
    key = registry.get('model_aliases', {}).get(model, model)
    arch = registry['architectures'].get(key)
    if arch and arch['family'] != 'llama':
        raise ValueError(f"{arch['family']} counts require --model-yaml built from the published config")
    c = json.loads((config_dir / f'{model}.json').read_text())
    cached = int(seq if decode_kv_on_entry is None else decode_kv_on_entry)
    return build_model(c, phase, batch, seq, cached, implementation)

def act_dict(a):
    return None if a is None else a.to_dict()

LAYER_OPS = ("layernorm1", "qkv_proj", "rope", "attention", "output_proj", "layernorm2")
HEAD_EVENTS = {"lm_head": ("gemm", "linear_softmax_f"), "head_softmax": ("roof", "pointwise-linear-softmax-f"),
               "final_norm": ("roof", "pointwise-final-norm-f")}

def source_moe_counts_split(out, tc):
    from activity import ZERO_ACTIVITY
    from program.energy import price_activity
    records = getattr(tc, "counts_moe_roles_by_phase", {})
    for name, phase in out['phases'].items():
        steps = records.get(name) or []
        if len(steps) != 1:
            raise SystemExit(f"source MoE counts need exactly one priced {name} step, got {len(steps)}")
        ops = {}
        for (op, role), activity in steps[0].items():
            if op == "MLP":
                key = f"{role}_forward_layer"
            elif op == "linear_softmax":
                kind, event = HEAD_EVENTS[role]
                out['events'].append({'kind': kind, 'inst': 0, 'name': event, 'activity': activity.to_dict()})
                key = "linear_softmax"
            elif op in LAYER_OPS:
                key = f"{op}_forward_layer"
            elif op == "embedding":
                key = "embedding"
            else:
                raise SystemExit(f"source MoE counts: no operation name for trace op {op!r}")
            ops[key] = ops.get(key, ZERO_ACTIVITY) + activity
        phase['by_operation'] = {op: {'energy': price_activity(a, tc.hw_config).to_dict(), 'activity': a.to_dict()}
                                 for op, a in ops.items()}
        out['events'].append({'kind': 'template', 'inst': 0, 'timings': {},
                              'cls': 'TimeCalculationLLMInference' + ('.decode' if name == 'decode' else '')})

def main():
    parser = argparse.ArgumentParser(description="Run one workload through this tree (CPU only) and write per-op predicted activity and energy.")
    parser.add_argument('model')
    parser.add_argument('phase', choices=('train', 'prefill', 'decode'))
    parser.add_argument('batch', type=int)
    parser.add_argument('seq', type=int)
    parser.add_argument('out_json', type=Path)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--config-dir', type=Path, required=True)
    parser.add_argument('--hardware-yaml', type=Path, required=True)
    parser.add_argument('--implementation-yaml', type=Path, required=True,
                        help='YAML containing the full model_param.implementation mapping itself')
    parser.add_argument('--nockpt', action='store_true')
    parser.add_argument('--decode-kv-on-entry', type=int)
    parser.add_argument('--model-yaml', type=Path, help='Workload model built from registry/published facts')
    args = parser.parse_args()
    model, phase, batch, seq = args.model, args.phase, args.batch, args.seq
    out_json, run_dir = str(args.out_json.resolve()), args.run_dir.resolve()
    hw = args.hardware_yaml.resolve()
    import yaml
    implementation = yaml.safe_load(args.implementation_yaml.read_text())
    if not isinstance(implementation, dict):
        raise SystemExit('--implementation-yaml must contain the implementation mapping')
    run_dir.mkdir(parents=True, exist_ok=True)
    mcfg = args.out_dir.resolve() / 'model_configs' / f'{model}_{phase}_b{batch}_s{seq}.yaml'
    mcfg.parent.mkdir(parents=True, exist_ok=True)
    model_config = (yaml.safe_load(args.model_yaml.read_text()) if args.model_yaml else
                    model_yaml(model, phase, batch, seq, args.config_dir.resolve(), implementation,
                               args.decode_kv_on_entry))
    if model_config['model_param']['implementation'] != implementation:
        raise SystemExit('--model-yaml implementation disagrees with --implementation-yaml')
    text = yaml.safe_dump(model_config, sort_keys=False)
    if not mcfg.exists() or mcfg.read_text() != text:
        mcfg.write_text(text)
    nockpt = args.nockpt and phase == 'train'

    os.chdir(run_dir)

    sys.path.insert(0, str(R / 'opmodel' / 'src'))
    sys.path.insert(0, str(R))
    import config
    import opmodel
    import train_timing as _tt
    import inference_timing as _it, base_timing as _bt, llm_execution as _le, program.energy as _pe
    import opmodel.models.extended_roofline as _er, opmodel.models.effective_roofline as _ef
    for mod in (config, opmodel, _tt, _it, _bt, _le, _pe, _er, _ef):
        if not str(Path(mod.__file__).resolve()).startswith(str(R) + '/'):
            raise RuntimeError(f'{mod.__name__} imported from {mod.__file__}, not from {R}')
    from train_timing import TimeCalculationLLM
    from inference_timing import TimeCalculationLLMInference

    events = []
    orig_tpl = TimeCalculationLLM.compute_all_gemm_and_node_times
    orig_gemm = TimeCalculationLLM._gemm_activity
    orig_roof = TimeCalculationLLM._roofline_activity

    def tpl(self, *a, **k):
        tt, nb = orig_tpl(self, *a, **k)
        rec = {}
        for name, op in tt.items():
            rec[name] = {'forward': act_dict(op.forward.activity) if op.forward is not None else None,
                         'backward': act_dict(op.backward.activity) if op.backward is not None else None}
            for dn in ('forward', 'backward'):
                dt = getattr(op, dn)
                tr = (getattr(dt, 'notes', None) or {}).get('eager_trace_activity') if dt is not None else None
                if tr:
                    rec[name][dn + '_eager_trace'] = tr
        events.append({'kind': 'template', 'inst': id(self), 'cls': type(self).__name__, 'timings': rec})
        return tt, nb

    def gemm(self, name):
        a = orig_gemm(self, name)
        if name.startswith('linear_softmax'):
            events.append({'kind': 'gemm', 'inst': id(self), 'name': name, 'activity': a.to_dict()})
        return a

    def roof(self, *names):
        if not any(n.startswith('pointwise-linear-softmax') or n.startswith('pointwise-final-norm') for n in names):
            return orig_roof(self, *names)
        a = None
        for n in names:
            x = orig_roof(self, n)
            if n.startswith('pointwise-linear-softmax') or n.startswith('pointwise-final-norm'):
                events.append({'kind': 'roof', 'inst': id(self), 'name': n, 'activity': x.to_dict()})
            a = x if a is None else a + x
        return a

    orig_dec = TimeCalculationLLMInference._build_decode_transformer_results

    def dec(self, *a, **k):
        tt, nb = orig_dec(self, *a, **k)
        rec = {}
        for name, op in tt.items():
            rec[name] = {'forward': act_dict(op.forward.activity) if op.forward is not None else None,
                         'backward': act_dict(op.backward.activity) if op.backward is not None else None}
        events.append({'kind': 'template', 'inst': id(self), 'cls': type(self).__name__ + '.decode', 'timings': rec})
        return tt, nb

    TimeCalculationLLM.compute_all_gemm_and_node_times = tpl
    TimeCalculationLLMInference._build_decode_transformer_results = dec
    TimeCalculationLLM._gemm_activity = gemm
    TimeCalculationLLM._roofline_activity = roof

    hwc = config.parse_config(str(hw), config_type='hardware')
    mc = config.parse_config(str(mcfg), config_type='LLM')
    if phase == 'train' and mc.model_config.implementation == 'fused' and (hwc.l2_model or {}).get('step_simulation'):
        raise SystemExit('fused training with l2_model.step_simulation=true is unsupported: '
                         'the read-only rule is described for inference only')
    config.validate_configs(hwc, mc)
    exp_dir = str(run_dir / 'output' / 'LLM')
    os.makedirs(exp_dir, exist_ok=True)

    phases = {}
    if phase == 'train':
        tc = TimeCalculationLLM(hwc, mc, 'LLM', output_dir=exp_dir)
        t = tc.calc_time_llm()
        phases['train'] = (tc.execution_energy, float(t))
        extra = {'total_time_s': float(t)}
    else:
        tc = TimeCalculationLLMInference(hwc, mc, 'LLM', output_dir=exp_dir)
        timing = tc.calc_total_inference_time()
        phases['prefill'] = (tc.prefill_execution_energy, float(timing['prefill_time']))
        if phase == 'decode':
            phases['decode'] = (tc.decode_execution_energy, float(timing['decode_time']))
        extra = {k: float(v) for k, v in timing.items() if isinstance(v, (int, float))}
        extra['inference_step_overhead_s'] = float(tc._inference_step_overhead())
        extra['inference_prefill_overhead_s'] = float(tc._inference_prefill_overhead())

    out = {'model': model, 'phase': phase, 'batch': batch, 'seq_len': seq, 'nockpt': bool(nockpt), 'hardware_config': str(hw),
           'model_config': str(mcfg), 'extra': extra, 'static_power_w': float(getattr(hwc, 'static_power_w', 0.0) or 0.0),
           'modules': {m.__name__: str(Path(m.__file__).resolve()) for m in (config, opmodel, _tt, _it, _bt, _le, _pe, _er, _ef)},
           'tree': str(R),
           'env': {},
           'settings': getattr(tc, 'resolved_settings', None),
           'native_gemm_timing': dict(getattr(tc, 'native_gemm_timing', {}) or {}),
           'events': events, 'phases': {}}
    for name, (res, t) in phases.items():
        g = res.gpu
        out['phases'][name] = {
            'time_s': t,
            'wall_time_s': g.wall_time_s,
            'static_j': g.static_j,
            'busy_j': g.busy_j,
            'communication_j': res.communication.total_j,
            'total_j': res.total_j,
            'dynamic': g.dynamic.to_dict(),
            'activity': g.activity.to_dict(),
            'by_operation': {n: {'energy': e.to_dict(), 'activity': g.by_operation_activity[n].to_dict()}
                             for n, e in g.by_operation.items()},
        }
    if tc._source_moe():
        source_moe_counts_split(out, tc)
    from paper_pipeline.static_power import mark_prediction, mark_run_energy_report
    policy = (yaml.safe_load(hw.read_text()).get('metadata') or {}).get('static_power_policy')
    out['static_power_policy'] = policy
    mark_prediction(out)
    mark_run_energy_report(run_dir, policy)
    from paper_pipeline.reporting import sanitize_run_artifacts, without_register_traffic
    sanitize_run_artifacts(run_dir)
    out = without_register_traffic(out)
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json + '.tmp', 'w') as f:
        json.dump(out, f)
    os.replace(out_json + '.tmp', out_json)

if __name__ == '__main__':
    main()
