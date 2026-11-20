from __future__ import annotations
from math import isclose
from typing import Any, Mapping

def component_energy_report(result: Any, hardware: Any = None) -> dict[str, Any]:
    gpu=result.gpu; activity=gpu.activity; dynamic=gpu.dynamic; comm=result.communication
    components={}
    coefficients={}
    if hardware is not None:
        tech=hardware.tech_config
        coefficients={'tensor':float(tech.core.tensor_energy),
            'tensor_padded':float(getattr(tech.core,'padded_tensor_energy',tech.core.tensor_energy)),
            'vector':float(tech.core.vector_energy),
            'register':float(tech.SRAMR.dynamic_energy_per_byte_value),'smem':float(tech.SRAML1.dynamic_energy_per_byte_value),
            'l2':float(tech.SRAML2.dynamic_energy_per_byte_value),'hbm':float(tech.DRAM.dynamic_energy_per_byte_value),
            'static':float(getattr(hardware,'static_power_w',0.) or 0.),
            'busy':float(getattr(hardware,'busy_power_w',0.) or 0.)}
    def add(name,energy,quantity,unit,coefficient=None,coefficient_unit=None,pricing_basis='activity_times_coefficient'):
        row={'energy_j':float(energy),'activity':float(quantity) if quantity is not None else None,'activity_unit':unit,
             'coefficient':coefficient,'coefficient_unit':coefficient_unit,'pricing_basis':pricing_basis}
        if quantity is not None and coefficient is not None:
            expected=quantity*coefficient;residual=energy-expected
            row.update(activity_priced_energy_j=expected,accounting_residual_j=residual,
                       repricing_safe=isclose(energy,expected,rel_tol=1e-8,abs_tol=1e-6))
        else:row.update(activity_priced_energy_j=None,accounting_residual_j=None,repricing_safe=False)
        components[name]=row

    useful=min(activity.useful_tensor_flops,activity.issued_tensor_flops)
    padded=activity.issued_tensor_flops-useful
    padded_j=padded*coefficients['tensor_padded'] if 'tensor_padded' in coefficients else 0.
    add('tensor',dynamic.tensor_j-padded_j,useful,'useful_FLOP',coefficients.get('tensor'),'J/FLOP')
    add('tensor_padded',padded_j,padded,'padded_FLOP',coefficients.get('tensor_padded'),'J/FLOP')
    for name,quantity,unit,cu in (
        ('vector',activity.vector_flops,'logical_FLOP','J/FLOP'),
        *((n,getattr(activity,n).total_bytes,'byte','J/byte') for n in ('register','smem','l2','hbm'))):
        add(name,getattr(dynamic,name+'_j'),quantity,unit,coefficients.get(name),cu)
    if activity.l2_bypass_bytes:
        add('l2_bypass',0.,activity.l2_bypass_bytes,'byte',pricing_basis='unpriced_l2_bypass_bytes')
    add('static',gpu.static_j,gpu.device_count*gpu.wall_time_s,'GPU_second',coefficients.get('static'),'W/GPU')
    if gpu.busy_j:
        add('busy',gpu.busy_j,gpu.device_count*gpu.busy_s,'GPU_second',coefficients.get('busy'),'W/GPU')
    link_rule = None
    if hardware is not None and comm.transmitted_bytes > 0 and getattr(hardware, 'network_layout', None) is not None:
        from program.link_energy import resolve_rule
        rules = {resolve_rule(hardware, hardware.network_layout.dimensions[c.dimension_index]) for c in comm.collectives}
        link_rule = rules.pop() if len(rules) == 1 else 'mixed'
    if link_rule == 'mixed':
        add('link_dynamic',comm.link_dynamic_j,None,'mixed',pricing_basis='per_dimension_link_energy_rule')
    elif link_rule == 'line_rate_while_active':
        add('link_dynamic',comm.link_dynamic_j,None,'GPU_second',pricing_basis='per_collective_line_rate_times_active_time')
    else:
        add('link_dynamic',comm.link_dynamic_j,comm.transmitted_bytes*8,'transmitted_bit',pricing_basis='sum_over_network_dimensions')
    add('active_network',comm.active_network_j,None,'collective_second',pricing_basis='per_collective_active_power_exposure')

    add('nccl_hbm',comm.nccl_hbm_j,comm.nccl_hbm_bytes,'byte',coefficients.get('hbm'),'J/byte')
    add('nccl_l2',comm.nccl_l2_j,comm.nccl_l2_bytes,'byte',coefficients.get('l2'),'J/byte')
    add('nccl_reduction',comm.reduction_j,comm.reduction_flops,'FLOP',coefficients.get('vector'),'J/FLOP')
    if comm.nccl_l2_bypass_bytes:
        add('nccl_l2_bypass',0.,comm.nccl_l2_bypass_bytes,'byte',pricing_basis='unpriced_l2_bypass_bytes')
    if any((comm.custom_hbm_bytes, comm.custom_hbm_j, comm.custom_l2_bytes,
            comm.custom_l2_j, comm.custom_reduction_flops, comm.custom_reduction_j)):
        add('custom_hbm',comm.custom_hbm_j,comm.custom_hbm_bytes,'byte',coefficients.get('hbm'),'J/byte')
        add('custom_l2',comm.custom_l2_j,comm.custom_l2_bytes,'byte',coefficients.get('l2'),'J/byte')
        add('custom_reduction',comm.custom_reduction_j,comm.custom_reduction_flops,'FLOP',coefficients.get('vector'),'J/FLOP')
    dimensions={}
    for c in comm.collectives:
        d=dimensions.setdefault(str(c.dimension_index),dict(transmitted_bits=0.,collective_seconds=0.,link_dynamic_j=0.,active_network_j=0.,nccl_hbm_j=0.,nccl_l2_j=0.,reduction_j=0.))
        d['transmitted_bits']+=c.transmitted_bytes*8;d['collective_seconds']+=c.completion_time_ns*1e-9
        for k in ('link_dynamic_j','active_network_j','nccl_hbm_j','nccl_l2_j','reduction_j'):d[k]+=getattr(c,k)
    for key, d in dimensions.items():
        d['effective_link_j_per_bit']=d['link_dynamic_j']/d['transmitted_bits'] if d['transmitted_bits'] else None
        d['effective_active_w']=d['active_network_j']/d['collective_seconds'] if d['collective_seconds'] else None
        dim_rule = link_rule
        if link_rule == 'mixed':
            from program.link_energy import resolve_rule
            dim_rule = resolve_rule(hardware, hardware.network_layout.dimensions[int(key)])
        if dim_rule == 'line_rate_while_active':
            d['effective_active_w'] = None
    total=sum(c['energy_j'] for c in components.values())
    if not isclose(total,result.total_j,rel_tol=1e-10,abs_tol=1e-6):raise ValueError('Component energy does not reconcile')
    gpu_names=('tensor','tensor_padded','vector','register','smem','l2','hbm','static')
    if 'busy' in components:gpu_names=gpu_names+('busy',)
    return {'schema_version':1,'energy_unit':'J','energy_scope':'system','device_count':gpu.device_count,
        'completion_time_s':gpu.wall_time_s,'total_energy_j':result.total_j,'components':components,
        'closure_error_j':total-result.total_j,'coverage':result.coverage.to_dict(),
        'gpu_activity_pricing_consistent':all(components[k]['repricing_safe'] for k in gpu_names) if hardware is not None else None,
        'network_dimensions':dimensions,'network_exposure_note':'Summed collective-seconds can overlap; not network wall time.',
        **({'link_energy_rule':link_rule} if link_rule is not None else {}),
        'useful_tensor_flops':activity.useful_tensor_flops}
