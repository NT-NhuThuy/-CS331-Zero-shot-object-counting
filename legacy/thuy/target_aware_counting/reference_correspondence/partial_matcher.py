"""Null-feasible partial structured matching with a matrix-form geometry term.

The projected-gradient solver finds local stationary candidates, compares every
candidate to T=0, and never reports a global optimum for gamma > 0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def normalized_descriptors(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("invalid descriptor matrix")
    norms = np.linalg.norm(values, axis=1)
    if (norms <= 0).any():
        raise ValueError("zero descriptor")
    return values / norms[:, None]


def normalized_distances(centers: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float64)
    x0, y0, x1, y1 = box
    edge = max(x1-x0, y1-y0)
    if centers.ndim != 2 or centers.shape[1] != 2 or edge <= 0:
        raise ValueError("invalid centers or box")
    if (centers[:, 0] < x0-1e-8).any() or (centers[:, 0] > x1+1e-8).any() or \
            (centers[:, 1] < y0-1e-8).any() or (centers[:, 1] > y1+1e-8).any():
        raise ValueError("patch center outside local box")
    distances = np.linalg.norm(centers[:, None, :]-centers[None, :, :], axis=-1)
    return distances / (edge*math.sqrt(2))


def symmetric_knn_graph(centers: np.ndarray, neighbors: int = 4,
                        atom_ids: list[str] | None = None) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float64)
    n = len(centers)
    if atom_ids is not None and (len(atom_ids) != n or len(set(atom_ids)) != n):
        raise ValueError("reference patch IDs are missing or duplicated")
    ids = atom_ids if atom_ids is not None else [str(i).zfill(12) for i in range(n)]
    graph = np.zeros((n, n), dtype=np.float64)
    if n < 2:
        return graph
    distances = np.linalg.norm(centers[:, None, :]-centers[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    for i in range(n):
        order = sorted(range(n),key=lambda j:(float(distances[i,j]),ids[j]))[:min(neighbors,n-1)]
        graph[i, order] = 1
    graph = np.maximum(graph, graph.T)
    np.fill_diagonal(graph, 0)
    return graph


def normalized_mass(values: np.ndarray | None, size: int) -> np.ndarray:
    if values is None:
        return np.full(size, 1/size)
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all() or (result < 0).any() or result.sum() <= 0:
        raise ValueError("invalid patch mass")
    return result/result.sum()


@dataclass(frozen=True)
class MatchProblem:
    cost: np.ndarray
    reference_distances: np.ndarray
    image_distances: np.ndarray
    reference_graph: np.ndarray
    reference_mass: np.ndarray
    image_mass: np.ndarray
    gamma: float
    rho: float

    def __post_init__(self):
        m, n = self.cost.shape
        if m < 1 or n < 1 or self.reference_distances.shape != (m,m) or \
                self.image_distances.shape != (n,n) or self.reference_graph.shape != (m,m) or \
                self.reference_mass.shape != (m,) or self.image_mass.shape != (n,):
            raise ValueError("matcher shapes disagree")
        if not all(np.isfinite(v).all() for v in (self.cost, self.reference_distances,
                                                  self.image_distances, self.reference_graph)):
            raise ValueError("non-finite matcher input")
        if self.gamma < 0 or self.rho <= 0 or not np.isclose(self.reference_mass.sum(),1) or \
                not np.isclose(self.image_mass.sum(),1):
            raise ValueError("invalid matcher mass/cost parameters")
        if (self.reference_mass < 0).any() or (self.image_mass < 0).any():
            raise ValueError("negative patch mass")

    @property
    def zero_objective(self) -> float:
        return 2*self.rho

    @property
    def geometry_normalizer(self) -> float:
        return float(self.reference_mass @ self.reference_graph @ self.reference_mass)

    def geometry_and_gradient(self, transport: np.ndarray) -> tuple[float, np.ndarray]:
        if self.geometry_normalizer <= 0:
            return 0.0, np.zeros_like(transport)
        r = transport.sum(axis=1)
        graph = self.reference_graph
        a = graph * self.reference_distances**2
        m = graph * self.reference_distances
        b = self.image_distances**2
        term_ref = float(r @ a @ r)
        term_image = float(np.sum(transport * (graph @ transport @ b)))
        term_cross = float(np.sum(transport * (m @ transport @ self.image_distances)))
        value = (term_ref + term_image - 2*term_cross)/self.geometry_normalizer
        gradient = 2*((a @ r)[:, None] + graph @ transport @ b -
                      2*m @ transport @ self.image_distances)/self.geometry_normalizer
        if value < -1e-9:
            raise ArithmeticError("geometry cost became negative")
        return max(0.0, value), gradient

    def objective_gradient(self, transport: np.ndarray) -> tuple[float, np.ndarray, float]:
        geometry, d_geometry = self.geometry_and_gradient(transport)
        matched_mass = float(transport.sum())
        objective = float(np.sum(self.cost * transport) + self.gamma*geometry +
                          self.rho*(2-2*matched_mass))
        gradient = self.cost - 2*self.rho + self.gamma*d_geometry
        return objective, gradient, geometry


def make_problem(reference_features: np.ndarray, image_features: np.ndarray,
                 reference_centers: np.ndarray, image_centers: np.ndarray,
                 reference_box: tuple[float,float,float,float],
                 image_box: tuple[float,float,float,float], gamma: float, rho: float,
                 reference_mass: np.ndarray | None = None, image_mass: np.ndarray | None = None,
                 reference_atom_ids: list[str] | None = None) -> MatchProblem:
    ref, img = normalized_descriptors(reference_features), normalized_descriptors(image_features)
    if ref.shape[1] != img.shape[1]:
        raise ValueError("reference and image descriptor widths differ")
    cost = 1 - np.clip(ref @ img.T, -1, 1)
    m, n = len(ref), len(img)
    return MatchProblem(cost, normalized_distances(reference_centers, reference_box),
                        normalized_distances(image_centers, image_box),
                        symmetric_knn_graph(reference_centers,atom_ids=reference_atom_ids),
                        normalized_mass(reference_mass, m), normalized_mass(image_mass, n),
                        gamma, rho)


def _capped_simplex(values: np.ndarray, cap: float) -> np.ndarray:
    positive = np.maximum(values, 0)
    if positive.sum() <= cap:
        return positive
    sorted_values = np.sort(values)[::-1]
    thresholds = (np.cumsum(sorted_values)-cap)/np.arange(1, len(values)+1)
    eligible = np.where(sorted_values > thresholds)[0]
    theta = thresholds[eligible[-1]] if len(eligible) else thresholds[0]
    return np.maximum(values-theta, 0)


def _capped_simplex_rows(values: np.ndarray, caps: np.ndarray) -> np.ndarray:
    """Apply the same capped-simplex projection to all rows in NumPy."""
    positive = np.maximum(values, 0)
    needs_projection = positive.sum(axis=1) > caps
    if not np.any(needs_projection):
        return positive
    result = positive.copy()
    selected = values[needs_projection]
    selected_caps = caps[needs_projection]
    ordered = np.sort(selected, axis=1)[:, ::-1]
    ranks = np.arange(1, selected.shape[1] + 1, dtype=np.float64)
    thresholds = (np.cumsum(ordered, axis=1) - selected_caps[:, None]) / ranks
    eligible = ordered > thresholds
    last = eligible.sum(axis=1) - 1
    theta = thresholds[np.arange(len(last)), last]
    result[needs_projection] = np.maximum(selected - theta[:, None], 0)
    return result


def project_capacities(values: np.ndarray, reference_mass: np.ndarray,
                       image_mass: np.ndarray, iterations: int = 120,
                       tolerance: float = 1e-10) -> np.ndarray:
    """Dykstra projection onto nonnegative row/column capacity inequalities."""
    x = np.asarray(values, dtype=np.float64).copy()
    row_dual, col_dual = np.zeros_like(x), np.zeros_like(x)
    for _ in range(iterations):
        before = x.copy()
        row_input = x+row_dual
        row_out = _capped_simplex_rows(row_input, reference_mass)
        row_dual = row_input-row_out
        col_input = row_out+col_dual
        x = _capped_simplex_rows(col_input.T, image_mass).T
        col_dual = col_input-x
        if np.max(np.abs(x-before)) <= tolerance and \
                np.max(x.sum(axis=1)-reference_mass) <= tolerance and \
                np.max(x.sum(axis=0)-image_mass) <= tolerance:
            break
    return np.maximum(x, 0)


@dataclass
class MatchResult:
    transport: np.ndarray
    objective: float
    zero_objective: float
    evidence_gain: float
    matched_mass: float
    feature_cost: float
    geometry_cost: float
    unmatched_reference_mass: float
    unmatched_image_mass: float
    geometry_unavailable: bool
    capacity_residual: float
    stationarity_residual: float
    iterations: int
    status: str

    def image_support(self) -> np.ndarray:
        if self.matched_mass <= 0:
            return np.zeros(self.transport.shape[1])
        return self.transport.sum(axis=0)/self.matched_mass


def solve_partial(problem: MatchProblem, max_iterations: int = 200,
                  tolerance: float = 1e-7) -> MatchResult:
    """Projected gradient from zero and a full-mass seed; always retain J(0)."""
    if max_iterations <= 0 or tolerance <= 0:
        raise ValueError("invalid solver budget")
    zeros = np.zeros_like(problem.cost)
    starts = [zeros, np.outer(problem.reference_mass, problem.image_mass)]
    best = zeros.copy()
    best_value = problem.zero_objective
    best_status, best_iter = "zero_optimal", 0
    for start in starts:
        current = project_capacities(start, problem.reference_mass, problem.image_mass)
        value, _, _ = problem.objective_gradient(current)
        status = "max_iterations"
        count = 0
        for iteration in range(1, max_iterations+1):
            count = iteration
            _, gradient, _ = problem.objective_gradient(current)
            projected = project_capacities(current-gradient, problem.reference_mass, problem.image_mass)
            if np.linalg.norm(projected-current) <= tolerance:
                status = "stationary"
                break
            step = 1.0
            accepted = False
            for _ in range(24):
                proposal = project_capacities(current-step*gradient,
                                              problem.reference_mass, problem.image_mass)
                new_value, _, _ = problem.objective_gradient(proposal)
                if new_value <= value - 1e-12:
                    current, value, accepted = proposal, new_value, True
                    break
                step /= 2
            if not accepted:
                status = "line_search_stalled"
                break
        if value < best_value - tolerance:
            best, best_value, best_status, best_iter = current.copy(), value, status, count
    value, gradient, geometry = problem.objective_gradient(best)
    residual = max(0.0, float(np.max(best.sum(axis=1)-problem.reference_mass)),
                   float(np.max(best.sum(axis=0)-problem.image_mass)),
                   -float(np.min(best)))
    stationarity = float(np.linalg.norm(project_capacities(best-gradient,
                        problem.reference_mass, problem.image_mass)-best))
    if residual > 1e-6:
        raise ArithmeticError("partial transport violates capacity")
    mass = float(best.sum())
    gain = max(0.0, problem.zero_objective-value)
    if gain <= tolerance:
        best = zeros
        mass = gain = geometry = 0.0
        value = problem.zero_objective
        best_status = "zero_optimal"
    return MatchResult(best, float(value), problem.zero_objective, gain, mass,
                       float(np.sum(problem.cost*best)), float(geometry),
                       1-float(best.sum(axis=1).sum()), 1-float(best.sum(axis=0).sum()),
                       problem.geometry_normalizer <= 0, residual, stationarity,
                       best_iter, best_status)
