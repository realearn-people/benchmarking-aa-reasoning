from py_arg.abstract_argumentation_classes.abstract_argumentation_framework import AbstractArgumentationFramework
from py_arg.algorithms.semantics.get_complete_extensions import get_complete_extensions
from py_arg.algorithms.semantics.get_grounded_extension import get_grounded_extension
from py_arg.algorithms.semantics.get_preferred_extensions import get_preferred_extensions
from py_arg.algorithms.semantics.get_stable_extensions import get_stable_extensions

from af_utils import (
    apply_isomorphism,
    apply_fundamental_consistency,
    apply_modularity,
    apply_defense_dynamics,
    generate_no_conflict,
    generate_linear_attack_chain,
    generate_cycle,
    generate_single_target_multiple_attackers,
    generate_single_attack_multiple_defenders,
    generate_disconnected_symmetric_pairs
)

from typing import Dict, List, Callable

from VerificationSuite import VerificationSuite
from ReportGenerator import ReportGenerator
import json

class LogicTester:
    """Given perfectly valid set of extensions, all test cases should pass."""
    
    def __init__(self, af_generators: Dict[str, Callable], ns: List[int]):
        self.af_generators = af_generators
        self.ns = ns
        self.verification_suite = VerificationSuite()
        self.results = {}
    
    def _get_llm_response_for_af(self, af: AbstractArgumentationFramework) -> Dict[str, List[List[str]]]:
        all_expected_ext_sets = {}
        
        extension_calculator = {
            'GE': get_grounded_extension,
            'CE': get_complete_extensions,
            'PE': get_preferred_extensions,
            'SE': get_stable_extensions
        }
        
        for ext_type in ['GE', 'CE', 'PE', 'SE']:
            
            # Computes the expected extensions using the get_extensions module.
            # The function get_extensions returns Set[FrozenSet[Argument]]
            expected_ext_sets = extension_calculator[ext_type](af)
            
            if ext_type == 'GE':
                # For get_grounded_extension, the function returns Set[Argument] -> Converts it to Set[FrozenSet[Argument]]
                expected_ext_sets = {frozenset(expected_ext_sets)}
            
            all_expected_ext_sets[ext_type] = [list(arg.name for arg in s) for s in expected_ext_sets]
        
        return all_expected_ext_sets
            

    def run_evaluation(self):
        """
        Main evaluation loop. It iterates through each AF type and size,
        runs tests on the base AF and all its metamorphic transformations,
        and stores the results.
        """
        print("\nStarting evaluation run...")
        self.results = {af_name: {n: {} for n in self.ns} for af_name in self.af_generators}

        #
        for af_name, generator in self.af_generators.items():
            for n in self.ns:
                print(f"\n--- Testing: {af_name} (n={n}) ---")

                # 1. BASE AF TEST
                base_af = generator(n)
                base_extensions = self._get_llm_response_for_af(base_af)
                
                print(f'SENT {af_name} (n={n}): {repr(base_af)}')
                print(f'** Original Extension ** ----> LLM answered {base_extensions}')
                
                base_violations = self.verification_suite.verify_fundamental_properties(base_af, base_extensions)
                print(f'Base Violations for {af_name} (n={n}): {json.dumps(base_violations, indent=2)}\n')
                
                validity_violations, all_expected_ext_sets = self.verification_suite.verify_validity(base_af, base_extensions)
                self.results[af_name][n]['validity'] = {
                    'computed': str(base_extensions),
                    'expected': str(all_expected_ext_sets),
                    'status': 'Fail' if validity_violations else 'Pass',
                    'violations': validity_violations
                }
                
                self.results[af_name][n]['base'] = {
                    'computed': base_extensions,
                    'status': 'Fail' if base_violations else 'Pass',
                    'violations': base_violations
                }

                # 2. METAMORPHIC TESTS
                # Isomorphism
                iso_af, r_map = apply_isomorphism(base_af)
                iso_ext = self._get_llm_response_for_af(iso_af)
                
                print(f'SENT {af_name} [Isomorphism] (n={n}): {repr(iso_af)}')
                print(f'----> LLM answered {iso_ext}')
                
                fp_violations = self.verification_suite.verify_fundamental_properties(iso_af, iso_ext)
                mr_violations = self.verification_suite.verify_isomorphism(base_extensions, iso_ext, r_map)
                all_violations = {**fp_violations, **mr_violations}
                
                print(f'Isomorphism Violations for {af_name} (n={n}): {json.dumps(all_violations, indent=2)}\n')
                
                self.results[af_name][n]['isomorphism'] = {
                    'computed': str(iso_ext),
                    'status': 'Fail' if all_violations else 'Pass',
                    'violations': all_violations
                }

                # Fundamental Consistency
                fc_af, sa_name = apply_fundamental_consistency(base_af)
                fc_ext = self._get_llm_response_for_af(fc_af)
                
                print(f'SENT {af_name} [Fundamental Consistency] (n={n}): {repr(fc_af)}')
                print(f'----> LLM answered {fc_ext}')
                
                fp_violations = self.verification_suite.verify_fundamental_properties(fc_af, fc_ext)
                mr_violations = self.verification_suite.verify_fundamental_consistency(base_extensions, fc_ext, sa_name)
                all_violations = {**fp_violations, **mr_violations}
                
                print(f'Fundamental Consistency Violations for {af_name} (n={n}): {json.dumps(all_violations, indent=2)}\n')
                
                self.results[af_name][n]['fundamental_consistency'] = {
                    'computed': str(fc_ext),
                    'status': 'Fail' if all_violations else 'Pass',
                    'violations': all_violations
                }

                # Modularity
                mod_af, u_name = apply_modularity(base_af)
                mod_ext = self._get_llm_response_for_af(mod_af)
                
                print(f'SENT {af_name} [Modularity] (n={n}): {repr(mod_af)}')
                print(f'----> LLM answered {mod_ext}')
                
                fp_violations = self.verification_suite.verify_fundamental_properties(mod_af, mod_ext)
                mr_violations = self.verification_suite.verify_modularity(base_extensions, mod_ext, u_name)
                all_violations = {**fp_violations, **mr_violations}
                
                print(f'Modularity Violations for {af_name} (n={n}): {json.dumps(all_violations, indent=2)}\n')
                
                self.results[af_name][n]['modularity'] = {
                    'computed': str(mod_ext),
                    'status': 'Fail' if all_violations else 'Pass',
                    'violations': all_violations
                }

                # Defense Dynamics (only if there are attacks)
                if base_af.defeats:
                    dd_af, defender, attacker, target = apply_defense_dynamics(base_af)
                    dd_ext = self._get_llm_response_for_af(dd_af)
                    
                    print(f'SENT {af_name} [Defense Dynamics] (n={n}): {repr(dd_af)}')
                    print(f'----> LLM answered {dd_ext}')
                    print(f'Defender ---> {attacker} ---> {target}')
                    
                    fp_violations = self.verification_suite.verify_fundamental_properties(dd_af, dd_ext)
                    mr_violations = self.verification_suite.verify_defense_dynamics(base_extensions, dd_ext, defender, attacker, target)
                    all_violations = {**fp_violations, **mr_violations}
                    
                    print(f'Defense Dynamics Violations for {af_name} (n={n}): {json.dumps(all_violations, indent=2)}\n')
                    
                    self.results[af_name][n]['defense_dynamics'] = {
                        'computed': str(dd_ext),
                        'status': 'Fail' if all_violations else 'Pass',
                        'violations': all_violations,
                        'info': f'Defender --- {attacker} --- {target}'
                    }
        
        report_generator = ReportGenerator()
        report_generator.export_to_excel(f'LogicValidationReport.xlsx', results=self.results)
        

        print("\nEvaluation run complete.")

# Assume that all calculated extensions are correct, all test cases must pass.

af_generators_to_test = {
    "no_conflict": generate_no_conflict,
    "linear_attack_chain": generate_linear_attack_chain,
    "cycle": generate_cycle,
    "single_target_multiple_attackers": generate_single_target_multiple_attackers,
    "single_attack_multiple_defenders": generate_single_attack_multiple_defenders,
    "symmetric_disconnected": generate_disconnected_symmetric_pairs,
}

sizes_to_test = [4, 8]

logic_tester = LogicTester(af_generators=af_generators_to_test, ns=sizes_to_test)
logic_tester.run_evaluation()