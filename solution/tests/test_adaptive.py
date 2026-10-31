"""Invented counterexamples and exhaustive oracles; no match reference labels."""

import itertools
import copy
import json
from dataclasses import replace
import unittest

import numpy as np

from solution.code.possession import model
from solution.tests.test_control_profile import control_features
from solution.code.adaptive_experiment import assess, PROTOCOL


class AdaptiveTests(unittest.TestCase):
    def test_regression_gate_rejects_even_when_primary_gain_passes(self):
        protocol = json.loads(PROTOCOL.read_text())
        def metrics(value):
            return {"spell_f1_mean":value,"macro_f1":.8,"per_class":{"D":{"recall":.9}},
                    "spell_f1":{"0.5":{"recall":.5}},"transitions":{"12":{"f1":.5}}}
        report={"matches":[{"match_id":name,"metrics":{"controlled":metrics(.5),"adaptive":metrics(.55)},
                 "predict_repetitions_seconds":{"controlled":[1.,1.,1.],"adaptive":[1.1,1.1,1.1]}}
                for name in protocol["development"]]}
        result=assess(report,protocol)
        self.assertTrue(result["development_gates_passed"])
        self.assertFalse(result["promoted"])
        worse=copy.deepcopy(report)
        worse["matches"][1]["metrics"]["adaptive"]["macro_f1"]-=.006
        self.assertFalse(assess(worse,protocol)["development_gates_passed"])
        worse=copy.deepcopy(report)
        worse["matches"][0]["predict_repetitions_seconds"]["adaptive"]=[2.,2.,2.]
        self.assertFalse(assess(worse,protocol)["development_gates_passed"])

    def test_dynamic_decoder_matches_exhaustive_optimum(self):
        rng = np.random.default_rng(104)
        for n in (1, 4, 6):
            for minimum in (1, 2, 3):
                scores = rng.normal(size=(n, 3))
                costs = rng.uniform(0, 2, (n, 3, 3))
                costs[:, np.arange(3), np.arange(3)] = 0
                def objective(path):
                    return sum(scores[t, path[t]] for t in range(n)) - sum(
                        costs[t, path[t-1], path[t]] for t in range(1, n))
                options = []
                for path in itertools.product(range(3), repeat=n):
                    edges = [0] + [t for t in range(1, n) if path[t] != path[t-1]] + [n]
                    if all(b-a >= minimum for a,b in zip(edges[1:-2], edges[2:-1])):
                        options.append(objective(path))
                actual = model.decode(scores, costs, minimum)
                self.assertAlmostEqual(objective(actual), max(options), places=10)

    def test_static_equivalence_and_ties(self):
        for minimum in (1, 2, 3):
            for scores in (np.zeros((20, 3)), np.random.default_rng(25).normal(size=(20, 3))):
                costs = np.ones((3, 3)); np.fill_diagonal(costs, 0)
                np.testing.assert_array_equal(model.decode(scores,costs,minimum),
                    model.decode(scores,np.tile(costs,(20,1,1)),minimum))
        self.assertEqual(len(model.decode(np.empty((0,3)),np.empty((0,3,3)))),0)
        for costs in (np.zeros((2,3,3)), np.full((3,3), np.nan), -np.ones((3,3))):
            with self.assertRaises(ValueError):model.decode(np.zeros((3,3)), costs)

    def test_discount_is_destination_specific_and_preserves_dead_costs(self):
        costs = np.array([[0,12,24],[12,0,24],[24,24,0]],dtype=float)
        scores = np.array([[0,0,-4],[-3,0,-4],[0,-3,-4],[0,0,2]],dtype=float)
        actual = model.adaptive_switch_costs(scores,costs)
        np.testing.assert_allclose(actual[:,0,1],[12,4,12,12])
        np.testing.assert_allclose(actual[:,1,0],[12,12,4,12])
        np.testing.assert_array_equal(actual[:,:,2],np.tile(costs[:,2],(4,1)))
        np.testing.assert_array_equal(actual[:,2,:],np.tile(costs[2,:],(4,1)))

    def test_short_strong_control_recovered_but_glance_and_weak_proximity_not(self):
        costs = np.array([[0,12,24],[12,0,24],[24,24,0]],dtype=float)
        for duration, advantage, expected in ((8,2.5,True),(3,2.5,False),(8,.3,False)):
            scores = np.tile([0.,-2.5,-10.],(80,1))
            scores[30:30+duration] = [-advantage,0,-10]
            old = model.decode(scores,costs,8)
            new = model.decode(scores,model.adaptive_switch_costs(scores,costs),8)
            self.assertTrue(np.all(old==0))
            self.assertEqual(bool(np.all(new[30:30+duration]==1)),expected)

    def test_emissions_are_identical_and_periods_reset(self):
        features = control_features()
        features.distance[170:] = [8,.5]
        features.ball[70:95] = np.nan
        features.distance[70:95] = np.nan
        blocks=[(1,10,139),(2,170,289)]
        a,b={},{}
        first,_=model.emissions(features,[],blocks,(105,68),25,profile='controlled',trace=a)
        second,_=model.emissions(features,[],blocks,(105,68),25,profile='adaptive',trace=b)
        np.testing.assert_array_equal(first,second)
        for key in a:np.testing.assert_array_equal(a[key],b[key])
        labels,_=model.predict(features,[],blocks,(105,68),25,profile='adaptive')
        np.testing.assert_array_equal(labels[10:140],'H')
        np.testing.assert_array_equal(labels[170:290],'A')
        np.testing.assert_array_equal(labels[140:170],'X')

    def test_reflection_and_team_swap(self):
        f=control_features();f.distance[120:150]=[8,.5]
        block=[(1,0,299)]
        expected,_=model.predict(f,[],block,(105,68),25,profile='adaptive')
        ball=f.ball.copy();ball[:,:2]=1-ball[:,:2]
        reflected,_=model.predict(replace(f,ball=ball),[],block,(105,68),25,profile='adaptive')
        swapped,_=model.predict(replace(f,distance=f.distance[:,::-1],player_speed=f.player_speed[:,::-1]),[],block,(105,68),25,profile='adaptive')
        np.testing.assert_array_equal(reflected,expected)
        np.testing.assert_array_equal(swapped,np.where(expected=='H','A',np.where(expected=='A','H',expected)))


if __name__ == '__main__':
    unittest.main()
