"""Joint target/competitor/null evidence allocation using SciPy HiGHS MILP."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from .hypothesis_builder import Hypothesis


@dataclass
class AllocationResult:
    selected_ids: list[str]
    target_count: int
    retained_support: dict[str, dict[str, float]]
    objective: float | None
    status: str
    dual_bound: float | None
    relative_gap: float | None
    node_count: int | None
    solver_message: str


def allocate(hypotheses: list[Hypothesis], opening_cost: float,
             retention_fraction: float, relative_gap: float = 0.0,
             time_limit_seconds: float | None = None) -> AllocationResult:
    if opening_cost <= 0 or not (0 < retention_fraction <= 1) or relative_gap < 0:
        raise ValueError("invalid MILP parameters")
    if not hypotheses:
        return AllocationResult([],0,{},0.0,"optimal",0.0,0.0,0,"empty bank")
    ordered = sorted(hypotheses, key=lambda h: h.hypothesis_id)
    if len({h.hypothesis_id for h in ordered}) != len(ordered):
        raise ValueError("duplicate hypothesis IDs")
    active = [h for h in ordered if h.evidence_gain > 0 and h.atom_weights]
    if not active:
        return AllocationResult([],0,{},0.0,"optimal",0.0,0.0,0,"all hypotheses prefer null")
    h_count = len(active)
    edges = [(i, atom_id, weight) for i,h in enumerate(active)
             for atom_id,weight in sorted(h.atom_weights.items()) if weight > 0]
    atom_ids = sorted({atom for _,atom,_ in edges})
    candidate_ids = sorted({h.candidate_id for h in active})
    atom_index = {atom:i for i,atom in enumerate(atom_ids)}
    candidate_index = {candidate:i for i,candidate in enumerate(candidate_ids)}
    num_vars = h_count+len(edges)
    num_rows = len(edges)+len(atom_ids)+len(candidate_ids)+h_count
    matrix = lil_matrix((num_rows,num_vars), dtype=np.float64)
    upper = np.zeros(num_rows)
    lower = np.full(num_rows,-np.inf)
    objective = np.zeros(num_vars)
    objective[:h_count] = opening_cost
    integrality = np.zeros(num_vars,dtype=np.int8)
    integrality[:h_count] = 1
    atom_start = len(edges)
    candidate_start = atom_start+len(atom_ids)
    retention_start = candidate_start+len(candidate_ids)
    upper[atom_start:candidate_start] = 1
    upper[candidate_start:retention_start] = 1
    for i,h in enumerate(active):
        matrix[candidate_start+candidate_index[h.candidate_id],i] = 1
        matrix[retention_start+i,i] = retention_fraction*h.evidence_gain
    for index,(i,atom,weight) in enumerate(edges):
        column = h_count+index
        objective[column] = -weight
        matrix[index,column] = 1
        matrix[index,i] = -1  # y_hp <= z_h
        matrix[atom_start+atom_index[atom],column] = 1
        matrix[retention_start+i,column] = -weight
    options = {"mip_rel_gap": relative_gap}
    if time_limit_seconds is not None:
        if time_limit_seconds <= 0:
            raise ValueError("invalid MILP time limit")
        options["time_limit"] = time_limit_seconds
    result = milp(objective, integrality=integrality,
                  bounds=Bounds(np.zeros(num_vars),np.ones(num_vars)),
                  constraints=LinearConstraint(matrix.tocsr(),lower,upper), options=options)
    status = "optimal" if result.status == 0 else "incumbent" if result.x is not None else "failure"
    selected, retained = [], {}
    if result.x is not None:
        violation = matrix.tocsr() @ result.x-upper
        if np.max(violation) > 1e-5 or np.min(result.x) < -1e-5:
            raise ArithmeticError("MILP returned infeasible incumbent")
        for i,h in enumerate(active):
            if result.x[i] >= 1-1e-6:
                selected.append(h.hypothesis_id)
                retained[h.hypothesis_id] = {}
        for index,(i,atom,_) in enumerate(edges):
            if active[i].hypothesis_id in retained and result.x[h_count+index] > 1e-8:
                retained[active[i].hypothesis_id][atom] = float(result.x[h_count+index])
    target_count = sum(h.semantic_role == "target" and h.hypothesis_id in retained for h in active)
    return AllocationResult(selected,target_count,retained,
                            float(-result.fun) if result.fun is not None else None,
                            status,float(-result.mip_dual_bound) if getattr(result,"mip_dual_bound",None) is not None else None,
                            float(result.mip_gap) if getattr(result,"mip_gap",None) is not None else None,
                            int(result.mip_node_count) if getattr(result,"mip_node_count",None) is not None else None,
                            str(result.message))


def _iou(a: tuple[float,float,float,float], b: tuple[float,float,float,float]) -> float:
    area = max(0,min(a[2],b[2])-max(a[0],b[0])) * max(0,min(a[3],b[3])-max(a[1],b[1]))
    total = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area
    return area/total if total > 0 else 0.0


def independent_nms(hypotheses: list[Hypothesis], boxes_by_candidate: dict[str,tuple],
                    opening_cost: float, iou_threshold: float) -> list[Hypothesis]:
    """Same-bank independent control: best alternative/candidate, then fixed NMS."""
    if opening_cost <= 0 or not (0 <= iou_threshold <= 1):
        raise ValueError("invalid control parameters")
    best = {}
    for h in sorted(hypotheses,key=lambda h:(-h.evidence_gain,h.hypothesis_id)):
        if h.evidence_gain > opening_cost and h.candidate_id not in best:
            best[h.candidate_id] = h
    selected = []
    for h in sorted(best.values(),key=lambda h:(-h.evidence_gain,h.hypothesis_id)):
        if h.candidate_id not in boxes_by_candidate:
            raise ValueError("candidate box missing for NMS control")
        if all(_iou(boxes_by_candidate[h.candidate_id],
                    boxes_by_candidate[other.candidate_id]) <= iou_threshold for other in selected):
            selected.append(h)
    return selected
