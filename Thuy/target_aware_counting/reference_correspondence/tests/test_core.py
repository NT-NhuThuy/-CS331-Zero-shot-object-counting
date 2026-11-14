import itertools
import json
import unittest
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from target_aware_counting.reference_correspondence.allocation import allocate, independent_nms
from target_aware_counting.reference_correspondence.candidate_adapter import action_pool, tile_box_to_source, valid_box
from target_aware_counting.reference_correspondence.hypothesis_builder import Hypothesis, spatial_atom_cap
from target_aware_counting.reference_correspondence.partial_matcher import make_problem, solve_partial
from target_aware_counting.reference_correspondence.spatial_features import (
    Atom, atoms_from_grid, candidate_atoms, four_core_views, full_view)

ROOT = Path(__file__).resolve().parents[3]


def hyp(name, candidate, reference, role, weights):
    gain = sum(weights.values())
    return Hypothesis(name, candidate, reference, role, gain, weights, "synthetic", 1.,
                      2-gain, 2., 0., 0., 0., True, 0., 0.)


class MatcherTests(unittest.TestCase):
    def setUp(self):
        self.ref = np.array([[1.,0,0],[0,1,0],[0,0,1]])
        self.centers = np.array([[1.,1.],[5.,1.],[1.,5.]])
        self.box = (0.,0.,6.,6.)

    def problem(self, gamma=.3, rho=1.):
        return make_problem(self.ref,self.ref,self.centers,self.centers,
                            self.box,self.box,gamma,rho)

    def test_four_dimensional_geometry_and_gradient(self):
        problem = self.problem()
        transport = np.array([[.10,.03,.01],[.02,.11,.02],[.01,.02,.12]])
        fast, gradient = problem.geometry_and_gradient(transport)
        slow = sum(problem.reference_graph[u,v]*
                   (problem.reference_distances[u,v]-problem.image_distances[p,k])**2*
                   transport[u,p]*transport[v,k]
                   for u,v,p,k in itertools.product(range(3),repeat=4)) / problem.geometry_normalizer
        self.assertAlmostEqual(fast,slow,places=11)
        eps = 1e-6
        for u,p in itertools.product(range(3),repeat=2):
            perturb = np.zeros_like(transport)
            perturb[u,p] = eps
            diff = (problem.geometry_and_gradient(transport+perturb)[0]-
                    problem.geometry_and_gradient(transport-perturb)[0])/(2*eps)
            self.assertAlmostEqual(gradient[u,p],diff,places=5)

    def test_null_is_real_optimum_for_expensive_matches(self):
        ref = np.array([[1.,0.],[1.,0.]])
        img = np.array([[0.,1.],[0.,1.]])
        xy = np.array([[1.,1.],[3.,3.]])
        problem = make_problem(ref,img,xy,xy,(0.,0.,4.,4.),(0.,0.,4.,4.),.4,.1)
        result = solve_partial(problem,80,1e-7)
        self.assertEqual(result.status,"zero_optimal")
        self.assertEqual(result.matched_mass,0)
        self.assertEqual(result.evidence_gain,0)
        self.assertAlmostEqual(result.objective,2*problem.rho)

    def test_positive_match_and_patch_permutation(self):
        first = solve_partial(self.problem(),120,1e-7)
        self.assertGreater(first.evidence_gain,0)
        self.assertGreater(first.matched_mass,.5)
        order = [2,0,1]
        second_problem = make_problem(self.ref,self.ref[order],self.centers,
                                      self.centers[order],self.box,self.box,.3,1.)
        second = solve_partial(second_problem,120,1e-7)
        self.assertAlmostEqual(first.objective,second.objective,places=4)
        np.testing.assert_allclose(first.transport,second.transport[:,np.argsort(order)],atol=1e-3)
        self.assertLessEqual(np.max(second.transport.sum(0)-second_problem.image_mass),1e-6)
        self.assertLessEqual(np.max(second.transport.sum(1)-second_problem.reference_mass),1e-6)


class CandidateAndAtomTests(unittest.TestCase):
    def test_spatial_cap_preserves_shared_patch_ids_and_order_invariance(self):
        atoms = [Atom(f"full:{y}:{x}",(x,y,x+1,y+1),(x+.5,y+.5),"full",(1.,0.))
                 for y in range(12) for x in range(12)]
        first = spatial_atom_cap(atoms,(0,0,12,12),16)
        second = spatial_atom_cap(list(reversed(atoms)),(0,0,12,12),16)
        self.assertEqual([a.atom_id for a in first],[a.atom_id for a in second])
        self.assertEqual(len({a.atom_id for a in first}),16)
        self.assertTrue({a.atom_id for a in first} <= {a.atom_id for a in atoms})
        self.assertIs(spatial_atom_cap(atoms,(0,0,12,12),None),atoms)

    def test_existing_f1_f2_pool_is_distinct(self):
        source = ROOT / "target_aware_counting/output/set_decision_local_action_v3_1_20261028/actions_cpu.jsonl"
        with source.open() as stream:
            row = json.loads(stream.readline())
        q = row["query"]
        candidates = action_pool(row,(q["image_height"],q["image_width"]))
        self.assertEqual(sum(c.is_m0 for c in candidates),q["m0_count"])
        self.assertEqual({c.forward_id for c in candidates},{"F1_raw","F2_M0"})
        self.assertEqual(len({c.candidate_id for c in candidates}),len(candidates))

    def test_native_rounding_can_cross_image_edge_without_changing_box(self):
        self.assertTrue(valid_box((207.,351.,642.,480.),(478,640)))
        self.assertFalse(valid_box((641.,1.,650.,5.),(478,640)))

    def test_geometry_roundtrip_and_shared_atoms(self):
        self.assertEqual(tile_box_to_source((0.,0.,14.,14.),(10.,20.,24.,34.),(14,14)),
                         (10.,20.,24.,34.))
        view = full_view((28,28))
        grid = np.ones((2,2,3))
        atoms = atoms_from_grid(view,grid,(28,28),np.array([[True,True],[True,False]]))
        self.assertEqual(len(atoms),3)
        box = (0.,0.,28.,14.)
        self.assertEqual([a.atom_id for a in candidate_atoms(atoms,box)],
                         [a.atom_id for a in candidate_atoms(atoms,box)])
        self.assertEqual(len(candidate_atoms(atoms,box)),2)
        views = four_core_views((28,28),.5)
        tiled = [a for v in views for a in atoms_from_grid(v,np.ones((2,2,3)),
                                                           (28,28),np.ones((2,2),bool))]
        self.assertEqual(len({a.atom_id for a in tiled}),len(tiled))
        self.assertEqual(len(tiled),4)
        for left,right in itertools.combinations(tiled,2):
            a,b = left.footprint_xyxy,right.footprint_xyxy
            intersection = max(0,min(a[2],b[2])-max(a[0],b[0])) * \
                           max(0,min(a[3],b[3])-max(a[1],b[1]))
            self.assertEqual(intersection,0)


def exhaustive_joint(hypotheses, opening, retention):
    """Enumerate z and solve each continuous y LP; independent MILP oracle."""
    atoms = sorted({p for h in hypotheses for p in h.atom_weights})
    best = 0.
    for chosen in itertools.product((0,1),repeat=len(hypotheses)):
        if any(sum(chosen[i] for i,h in enumerate(hypotheses) if h.candidate_id==c)>1
               for c in {h.candidate_id for h in hypotheses}):
            continue
        edges = [(i,p,w) for i,h in enumerate(hypotheses) if chosen[i]
                 for p,w in h.atom_weights.items()]
        if not edges:
            continue
        a, b = [], []
        for atom in atoms:
            a.append([int(p==atom) for _,p,_ in edges]); b.append(1.)
        for i,h in enumerate(hypotheses):
            if chosen[i]:
                a.append([-w if j==i else 0. for j,_,w in edges]);
                b.append(-retention*h.evidence_gain)
        result = linprog([-w for _,_,w in edges],A_ub=a,b_ub=b,
                         bounds=(0,1),method="highs")
        if result.success:
            best = max(best,-result.fun-opening*sum(chosen))
    return best


class AllocationTests(unittest.TestCase):
    def test_milp_matches_exhaustive_and_allows_reference_reuse(self):
        hs = [hyp("a","a","r","target",{"p0":1.,"p1":1.}),
              hyp("b","b","r","target",{"p2":1.,"p3":1.}),
              hyp("c","a","wrong","competitor",{"p0":.8,"p1":.8})]
        result = allocate(hs,.4,.5)
        self.assertEqual(result.status,"optimal")
        self.assertEqual(result.target_count,2)
        self.assertEqual(set(result.selected_ids),{"a","b"})
        self.assertAlmostEqual(result.objective,exhaustive_joint(hs,.4,.5),places=7)

    def test_duplicate_suppressed_but_false_merge_remains(self):
        duplicate = [hyp("a","a","r","target",{"p0":1.,"p1":1.}),
                     hyp("b","b","r","target",{"p0":1.,"p1":1.})]
        self.assertEqual(allocate(duplicate,.5,.3).target_count,1)
        # The numeric evidence cannot tell a box around two objects from a valid whole.
        whole = hyp("whole","whole","r","target",dict.fromkeys(("p0","p1","p2","p3"),1.))
        parts = [hyp("left","left","r","target",{"p0":1.,"p1":1.}),
                 hyp("right","right","r","target",{"p2":1.,"p3":1.})]
        outcome = allocate([whole]+parts,.5,.3)
        self.assertEqual(outcome.selected_ids,["whole"])
        self.assertEqual(outcome.target_count,1)

    def test_empty_and_independent_control(self):
        weak = hyp("w","w","r","target",{"p":.1})
        self.assertEqual(allocate([weak],.5,.5).target_count,0)
        strong = hyp("s","s","r","target",{"p":1.})
        selected = independent_nms([strong,weak],{"s":(0,0,1,1),"w":(2,0,3,1)},.5,.5)
        self.assertEqual([h.hypothesis_id for h in selected],["s"])


if __name__ == "__main__":
    unittest.main()
