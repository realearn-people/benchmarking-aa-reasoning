import unittest
import random

from af_utils import (
    generate_no_conflict,
    generate_linear_attack_chain,
    generate_cycle,
    generate_single_target_multiple_attackers,
    generate_disconnected_symmetric_pairs,
    apply_isomorphism,
    apply_fundamental_consistency,
    apply_modularity,
    apply_defense_dynamics
)

from py_arg.abstract_argumentation_classes.abstract_argumentation_framework import AbstractArgumentationFramework
from py_arg.abstract_argumentation_classes.argument import Argument
from py_arg.abstract_argumentation_classes.defeat import Defeat

class TestAFGenerators(unittest.TestCase):
    def test_generate_no_conflict(self):
        for n in [1, 5, 10]:
            with self.subTest(n=n):
                af = generate_no_conflict(n)
                self.assertEqual(len(af.arguments), n)
                self.assertEqual(len(af.defeats), 0)
        with self.assertRaises(ValueError):
            generate_no_conflict(0)

    def test_generate_linear_attack_chain(self):
        for n in [2, 3, 10]: # Test edge case, odd, and even
            with self.subTest(n=n):
                af = generate_linear_attack_chain(n)
                self.assertEqual(len(af.arguments), n)
                self.assertEqual(len(af.defeats), n - 1)
                # Check the first defeat relationship
                a1 = af.get_argument("A1")
                a2 = af.get_argument("A2")
                self.assertIn(Defeat(a1, a2), af.defeats)
                
                # Check the middle defeat relationship for n > 2
                if n > 2:
                    # e.g. n=3: A1->A2->A3 --> a_mid = A2
                    a_mid = af.get_argument(f"A{n//2 + 1}")
                    # There must be exactly one ingoing and outgoing defeat
                    self.assertEqual(len(a_mid._ingoing_defeat_arguments), 1)
                    self.assertEqual(len(a_mid._outgoing_defeat_arguments), 1)
                    
                # Check if the last argument has only one incoming defeat
                a_last = af.get_argument(f"A{n}")
                self.assertEqual(len(a_last._ingoing_defeat_arguments), 1)
                self.assertEqual(len(a_last._outgoing_defeat_arguments), 0)
                
        with self.assertRaises(ValueError):
            generate_linear_attack_chain(1)

    def test_generate_cycle(self):
        for n in [2, 3, 8]: # Test edge case, odd, and even cycle lengths
            with self.subTest(n=n):
                af = generate_cycle(n)
                self.assertEqual(len(af.arguments), n)
                self.assertEqual(len(af.defeats), n)
                
                for arg in af.arguments:
                    # In a cycle, every argument should have exactly one ingoing and one outgoing defeat
                    self.assertEqual(len(arg._ingoing_defeat_arguments), 1,
                                    f"Argument {arg.name} should have exactly one ingoing defeat")
                    self.assertEqual(len(arg._outgoing_defeat_arguments), 1,
                                    f"Argument {arg.name} should have exactly one outgoing defeat")
                    
                # Check the wrap-around attack
                last_arg = af.get_argument(f"A{n}")
                first_arg = af.get_argument("A1")
                self.assertIn(Defeat(last_arg, first_arg), af.defeats)
        with self.assertRaises(ValueError):
            generate_cycle(1)

    def test_generate_single_target_multiple_attackers(self):
        for n in [2, 5, 12]: # 1, 4, and 11 attackers
            with self.subTest(n=n):
                af = generate_single_target_multiple_attackers(n)
                self.assertEqual(len(af.arguments), n)
                self.assertEqual(len(af.defeats), n - 1)
                target = af.get_argument("T")
                # Check that all other arguments attack the target
                for i in range(1, n):
                    attacker = af.get_argument(f"A{i}")
                    self.assertIn(Defeat(attacker, target), af.defeats)

    def test_generate_disconnected_symmetric_pairs(self):
        for n in [2, 4, 10]: # Test with 1, 2, and 5 pairs
            with self.subTest(n=n):
                af = generate_disconnected_symmetric_pairs(n)
                self.assertEqual(len(af.arguments), n)
                self.assertEqual(len(af.defeats), n) # 2 defeats per pair
                
                # Check a specific symmetric pair
                a1 = af.get_argument("A1")
                b1 = af.get_argument("B1")
                self.assertIn(Defeat(a1, b1), af.defeats)
                self.assertIn(Defeat(b1, a1), af.defeats)
                
        # Test for invalid (odd) n values
        with self.assertRaises(ValueError):
            generate_disconnected_symmetric_pairs(1)
        with self.assertRaises(ValueError):
            generate_disconnected_symmetric_pairs(3)


class TestMetamorphicTransformations(unittest.TestCase):
    def test_transformations_on_no_conflict(self):
        """Tests transformations on an AF with no attacks."""
        n = 4
        base_af = generate_no_conflict(n)

        # Isomorphism
        new_af_iso, _ = apply_isomorphism(base_af)
        self.assertEqual(len(new_af_iso.arguments), n)
        self.assertEqual(len(new_af_iso.defeats), 0)

        # Fundamental Consistency
        new_af_fc, sa_name = apply_fundamental_consistency(base_af)
        self.assertEqual(len(new_af_fc.arguments), n + 1)
        self.assertEqual(len(new_af_fc.defeats), 1)
        sa_arg = new_af_fc.get_argument(sa_name)
        self.assertIn(Defeat(sa_arg, sa_arg), new_af_fc.defeats)

        # Modularity
        new_af_mod, u_name = apply_modularity(base_af)
        self.assertEqual(len(new_af_mod.arguments), n + 1)
        self.assertEqual(len(new_af_mod.defeats), 0)
        self.assertTrue(new_af_mod.is_in_arguments(u_name))

        # Defense Dynamics should fail
        with self.assertRaises(ValueError):
            apply_defense_dynamics(base_af)

    def test_transformations_on_linear_chain(self):
        """Tests transformations on a linear attack chain AF."""
        for n in [3, 5]:
            with self.subTest(n=n):
                base_af = generate_linear_attack_chain(n)
                original_defeats = n - 1
                
                new_af_iso, _ = apply_isomorphism(base_af)
                self.assertEqual(len(new_af_iso.arguments), n)
                self.assertEqual(len(new_af_iso.defeats), original_defeats)

                new_af_fc, _ = apply_fundamental_consistency(base_af)
                self.assertEqual(len(new_af_fc.arguments), n + 1)
                self.assertEqual(len(new_af_fc.defeats), original_defeats + 1)

                new_af_mod, _ = apply_modularity(base_af)
                self.assertEqual(len(new_af_mod.arguments), n + 1)
                self.assertEqual(len(new_af_mod.defeats), original_defeats)

                new_af_dd, defender, attacker, target = apply_defense_dynamics(base_af)
                self.assertEqual(len(new_af_dd.arguments), n + 1)
                self.assertEqual(len(new_af_dd.defeats), original_defeats + 1)
                self.assertIn(Defeat(defender, attacker), new_af_dd.defeats)

    def test_transformations_on_cycle(self):
        """Tests transformations on a cycle AF."""
        for n in [3, 4]:
            with self.subTest(n=n):
                base_af = generate_cycle(n)
                original_defeats = n

                new_af_iso, _ = apply_isomorphism(base_af)
                self.assertEqual(len(new_af_iso.arguments), n)
                self.assertEqual(len(new_af_iso.defeats), original_defeats)

                new_af_fc, _ = apply_fundamental_consistency(base_af)
                self.assertEqual(len(new_af_fc.arguments), n + 1)
                self.assertEqual(len(new_af_fc.defeats), original_defeats + 1)

                new_af_mod, _ = apply_modularity(base_af)
                self.assertEqual(len(new_af_mod.arguments), n + 1)
                self.assertEqual(len(new_af_mod.defeats), original_defeats)

                new_af_dd, defender, attacker, target = apply_defense_dynamics(base_af)
                self.assertEqual(len(new_af_dd.arguments), n + 1)
                
                # Check if the new defeat relationship is in the new AF
                self.assertEqual(len(new_af_dd.defeats), original_defeats + 1)
                self.assertIn(Defeat(defender, attacker), new_af_dd.defeats)

    def test_transformations_on_single_target(self):
        """Tests transformations on a single target, multiple attackers AF."""
        n = 4 # 1 target, 3 attackers
        base_af = generate_single_target_multiple_attackers(n)
        original_defeats = n - 1

        new_af_iso, _ = apply_isomorphism(base_af)
        self.assertEqual(len(new_af_iso.arguments), n)
        self.assertEqual(len(new_af_iso.defeats), original_defeats)

        new_af_fc, _ = apply_fundamental_consistency(base_af)
        self.assertEqual(len(new_af_fc.arguments), n + 1)
        self.assertEqual(len(new_af_fc.defeats), original_defeats + 1)
        
        new_af_mod, _ = apply_modularity(base_af)
        self.assertEqual(len(new_af_mod.arguments), n + 1)
        self.assertEqual(len(new_af_mod.defeats), original_defeats)

        new_af_dd, defender, attacker, target = apply_defense_dynamics(base_af)
        self.assertEqual(len(new_af_dd.arguments), n + 1)
        self.assertEqual(len(new_af_dd.defeats), original_defeats + 1)
        self.assertIn(Defeat(defender, attacker), new_af_dd.defeats)
        
        self.assertEqual(target.name, "T") # The target of the original attack must be "T"
        self.assertIn(attacker.name, ["A1", "A2", "A3"]) # The attacker must be one of the original attackers

    def test_transformations_on_disconnected_pairs(self):
        """Tests transformations on a disconnected symmetric pairs AF."""
        n = 4 # 2 pairs
        base_af = generate_disconnected_symmetric_pairs(n)
        original_defeats = n

        new_af_iso, _ = apply_isomorphism(base_af)
        self.assertEqual(len(new_af_iso.arguments), n)
        self.assertEqual(len(new_af_iso.defeats), original_defeats)

        new_af_fc, _ = apply_fundamental_consistency(base_af)
        self.assertEqual(len(new_af_fc.arguments), n + 1)
        self.assertEqual(len(new_af_fc.defeats), original_defeats + 1)

        new_af_dd, defender, attacker, target = apply_defense_dynamics(base_af)
        self.assertEqual(len(new_af_dd.arguments), n + 1)
        self.assertEqual(len(new_af_dd.defeats), original_defeats + 1)
        self.assertIn(Defeat(defender, attacker), new_af_dd.defeats)
        
        self.assertIn(attacker.name, ["A1", "B1", "A2", "B2"])



