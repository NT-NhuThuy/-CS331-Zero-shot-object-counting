"""Target and competitor alternatives share one matcher and one atom ID space."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .candidate_adapter import Candidate
from .config import file_sha256, stable_hash
from .partial_matcher import make_problem, solve_partial
from .spatial_features import Atom, candidate_atoms, load_atom_cache


def spatial_atom_cap(atoms: list[Atom], box: tuple[float, float, float, float],
                     limit: int | None) -> list[Atom]:
    """Deterministic farthest spatial sample; preserve original shared atom IDs."""
    if limit is None or len(atoms) <= limit:
        return atoms
    if limit < 1 or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("invalid atom cap or box")
    ordered = sorted(atoms, key=lambda atom: atom.atom_id)
    centers = np.asarray([atom.center_xy for atom in ordered], dtype=np.float64)
    scale = np.asarray([box[2]-box[0], box[3]-box[1]])
    points = (centers-np.asarray(box[:2]))/scale
    chosen = [int(np.argmin(np.sum((points-.5)**2, axis=1)))]
    remaining = np.sum((points-points[chosen[0]])**2, axis=1)
    remaining[chosen[0]] = -1
    for _ in range(1, limit):
        index = int(np.argmax(remaining))
        chosen.append(index)
        distances = np.sum((points-points[index])**2, axis=1)
        remaining = np.minimum(remaining, distances)
        remaining[chosen] = -1
    return [ordered[index] for index in chosen]


@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id: str
    candidate_id: str
    reference_id: str
    semantic_role: str
    evidence_gain: float
    atom_weights: dict[str, float]
    match_status: str
    matched_mass: float
    objective: float
    zero_objective: float
    geometry_cost: float
    feature_cost: float
    unmatched_reference_mass: float
    geometry_unavailable: bool
    capacity_residual: float
    stationarity_residual: float

    def __post_init__(self):
        if self.semantic_role not in ("target", "competitor"):
            raise ValueError("hypothesis must have target/competitor role")
        if self.evidence_gain < -1e-9 or any(v < 0 for v in self.atom_weights.values()):
            raise ValueError("negative evidence gain/support")
        if sum(self.atom_weights.values()) > self.evidence_gain + 1e-7:
            raise ValueError("atom support exceeds match gain")

    def record(self) -> dict:
        return asdict(self)


def from_match(candidate: Candidate, reference_id: str, semantic_role: str,
               reference_atoms: list[Atom], image_atoms: list[Atom],
               reference_box: tuple[float, float, float, float], gamma: float, rho: float,
               max_iterations: int, tolerance: float,
               max_atoms_per_side: int | None = None) -> Hypothesis:
    selected = candidate_atoms(image_atoms, candidate.box_xyxy_original)
    if not selected or not reference_atoms:
        raise ValueError("no valid reference/candidate patch atoms")
    selected = spatial_atom_cap(selected,candidate.box_xyxy_original,max_atoms_per_side)
    reference_atoms = spatial_atom_cap(reference_atoms,reference_box,max_atoms_per_side)
    widths = {len(a.feature) for a in reference_atoms+selected}
    if len(widths) != 1:
        raise ValueError("DINO feature widths differ")
    ref_centers = np.asarray([a.center_xy for a in reference_atoms],dtype=np.float64)
    img_centers = np.asarray([a.center_xy for a in selected],dtype=np.float64)
    for centers,box in ((ref_centers,reference_box),(img_centers,candidate.box_xyxy_original)):
        centers[:,0] = np.clip(centers[:,0],box[0],box[2])
        centers[:,1] = np.clip(centers[:,1],box[1],box[3])
    problem = make_problem(np.asarray([a.feature for a in reference_atoms]),
                           np.asarray([a.feature for a in selected]),
                           ref_centers,img_centers,
                           reference_box, candidate.box_xyxy_original,
                           gamma, rho,reference_atom_ids=[a.atom_id for a in reference_atoms])
    result = solve_partial(problem, max_iterations, tolerance)
    support = result.image_support()
    weights = {atom.atom_id: float(result.evidence_gain*mass)
               for atom,mass in zip(selected,support) if mass > 0 and result.evidence_gain > 0}
    return Hypothesis(f"{candidate.candidate_id}|{reference_id}", candidate.candidate_id,
                      reference_id, semantic_role, result.evidence_gain, weights,
                      result.status, result.matched_mass, result.objective,
                      result.zero_objective, result.geometry_cost, result.feature_cost,
                      result.unmatched_reference_mass, result.geometry_unavailable,
                      result.capacity_residual, result.stationarity_residual)


def pooled_control(candidate: Candidate, reference_id: str, semantic_role: str,
                   reference_atoms: list[Atom], image_atoms: list[Atom], rho: float,
                   max_atoms_per_side: int | None = None,
                   reference_box: tuple[float, float, float, float] | None = None) -> Hypothesis:
    selected = candidate_atoms(image_atoms, candidate.box_xyxy_original)
    if not selected or not reference_atoms:
        raise ValueError("no valid pooled descriptor patches")
    selected = spatial_atom_cap(selected,candidate.box_xyxy_original,max_atoms_per_side)
    if max_atoms_per_side is not None:
        if reference_box is None:
            raise ValueError("reference box required for capped pooled control")
        reference_atoms = spatial_atom_cap(reference_atoms,reference_box,max_atoms_per_side)
    ref = np.mean([a.feature for a in reference_atoms], axis=0)
    img = np.mean([a.feature for a in selected], axis=0)
    if np.linalg.norm(ref) <= 0 or np.linalg.norm(img) <= 0:
        raise ValueError("zero pooled descriptor")
    cosine = float(np.dot(ref,img)/(np.linalg.norm(ref)*np.linalg.norm(img)))
    cost = 1 - float(np.clip(cosine,-1,1))
    gain = max(0.0, 2*rho-cost)
    weights = {atom.atom_id: gain/len(selected) for atom in selected} if gain > 0 else {}
    return Hypothesis(f"{candidate.candidate_id}|{reference_id}", candidate.candidate_id,
                      reference_id, semantic_role, gain, weights,
                      "pooled_null" if gain == 0 else "pooled_match", float(gain > 0),
                      2*rho-gain, 2*rho, 0.0, cost,
                      float(gain == 0), True, 0.0, 0.0)


def build_item(item: dict, candidates: list[Candidate], references: list[dict],
               item_atom_cache: str, feature_config: dict, numerical_config: dict,
               reference_snapshot_sha256: str) -> dict:
    """Build all three cache-compatible controls for one query without labels."""
    output = {"item_id":item["item_id"],"query":item["query"],
              "source_image_id":item["source_image_id"],
              "feature_config_sha256":stable_hash(feature_config),
              "numerical_config_sha256":stable_hash(numerical_config),
              "reference_snapshot_sha256":reference_snapshot_sha256,
              "global_hypotheses":[],"match_hypotheses":[],"gamma_zero_hypotheses":[],
              "unresolved_candidate_ids":[],"status":"ok"}
    valid_refs = [r for r in references if r["canonical_query"] == item["query"] and
                  r["status"] == "valid"]
    if not valid_refs:
        output["status"] = "reference_unavailable"
        return output
    atoms = load_atom_cache(item_atom_cache,file_sha256(item["image_path"]),feature_config)
    ref_atoms = {}
    for ref in valid_refs:
        if not ref.get("atom_cache_path") or not ref.get("weak_roi_xyxy"):
            raise ValueError("valid reference lacks feature cache/weak ROI")
        ref_atoms[ref["reference_id"]] = load_atom_cache(ref["atom_cache_path"],
                                          ref["image_sha256"],feature_config)
    for candidate in candidates:
        if not candidate_atoms(atoms,candidate.box_xyxy_original):
            output["unresolved_candidate_ids"].append(candidate.candidate_id)
            if candidate.is_m0:
                output["status"] = "technical_fallback_low_resolution"
            continue
        for ref in valid_refs:
            ref_id = ref["reference_id"]
            weak_box = tuple(float(v) for v in ref["weak_roi_xyxy"])
            common = (candidate,ref_id,ref["semantic_role"],
                      ref_atoms[ref_id],atoms)
            atom_cap = numerical_config.get("max_atoms_per_side")
            global_h = pooled_control(*common,float(numerical_config["rho"]),
                                      atom_cap,weak_box)
            match_h = from_match(*common,weak_box,
                                 float(numerical_config["gamma"]),float(numerical_config["rho"]),
                                 int(numerical_config["match_max_iterations"]),
                                 float(numerical_config["match_tolerance"]),atom_cap)
            gamma_zero_h = from_match(*common,weak_box,0.0,float(numerical_config["rho"]),
                                      int(numerical_config["match_max_iterations"]),
                                      float(numerical_config["match_tolerance"]),atom_cap)
            output["global_hypotheses"].append(global_h.record())
            output["match_hypotheses"].append(match_h.record())
            output["gamma_zero_hypotheses"].append(gamma_zero_h.record())
    return output
