from py_arg.abstract_argumentation_classes.abstract_argumentation_framework import AbstractArgumentationFramework
from typing import Dict, List, Callable
from LLM_Interface import LLMClient

from VerificationSuite import VerificationSuite
from ReportGenerator import ReportGenerator
import logging

from af_utils import (
    apply_isomorphism,
    apply_fundamental_consistency,
    apply_modularity,
    apply_defense_dynamics
)

logging.basicConfig(
    level=logging.INFO,
    format="[{asctime}] - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
)

class LLMTester:
    """Orchestrates the entire evaluation process for a given LLM."""
    
    def __init__(self, llm_client: LLMClient, af_generators: Dict[str, Callable], ns: List[int]):
        self.llm_client = llm_client
        self.af_generators = af_generators
        self.ns = ns
        self.verification_suite = VerificationSuite()
        self.results = {}
        
    def _get_llm_response_for_af(self, af: AbstractArgumentationFramework) -> Dict[str, List[List[str]]]:
        """
        Queries the LLM for extensions of a given AF and parses the response.
        
        Args:
            af (AbstractArgumentationFramework): The AF to query.
        Returns:
            Dict[str, List[List[str]]]: Parsed extensions.
        Raises:
            TimeoutError: If the LLM request times out after MAX_RETRIES.
        """
        raw_response = self.llm_client.query_llm_for_extensions(repr(af))
        return self.llm_client.parse_output_to_extensions(raw_response)
    

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

                # 1. BASE AF TEST
                base_af = generator(n)
                try:
                    logging.info(f"Sent [BASE] (N={n}) {af_name}: {repr(base_af)} → {self.llm_client.model_name}")
                    base_extensions = self._get_llm_response_for_af(base_af)
                    logging.info(f"{self.llm_client.model_name} → {base_extensions}")
                    
                    fp_violations = self.verification_suite.verify_fundamental_properties(base_af, base_extensions)
                
                    self.results[af_name][n]['base'] = {
                        'computed': base_extensions,
                        'status': 'FAIL' if fp_violations else 'PASS',
                        'violations': fp_violations
                    }
                    logging.info(f"[BASE] (N={n}) {af_name} FP = {self.results[af_name][n]['base']['status']}")
                    
                except TimeoutError as e:
                    logging.error(f'Cannot calculate the base extensions for {af_name} (n={n}) due to timeout. {e}. Aborting all tests.')
                    
                    for column in ['base', 'validity', 'isomorphism', 'fundamental_consistency', 'modularity', 'defense_dynamics']:
                        if column == 'base':
                            self.results[af_name][n][column] = {
                                'computed': {},
                                'status': 'Request Timeout',
                                'violations': {"TIMEOUT": [str(e)]}
                            }
                        else:
                            self.results[af_name][n][column] = {
                                'computed': {},
                                'status': 'Request Timeout',
                                'violations': {"TIMEOUT": ["All other tests aborted due to timeout."]}
                            }
                    continue
                
                except Exception as e:
                    logging.error(f"An unexpected error occurred: {e}")
                    for column in ['base', 'validity', 'isomorphism', 'fundamental_consistency', 'modularity', 'defense_dynamics']:
                        if column == 'base':
                            self.results[af_name][n][column] = {
                                'computed': {},
                                'status': 'Error',
                                'violations': {"ERROR": [str(e)]}
                            }
                        else:
                            self.results[af_name][n][column] = {
                                'computed': {},
                                'status': 'Error',
                                'violations': {"ERROR": ["All other tests aborted due to unexpected error"]}
                            }
                    continue
                    
                try:
                    validity_violations, all_expected_ext_sets = self.verification_suite.verify_validity(base_af, base_extensions)
                    self.results[af_name][n]['validity'] = {
                        'computed': str(base_extensions),
                        'expected': str(all_expected_ext_sets),
                        'status': 'FAIL' if validity_violations else 'PASS',
                        'violations': validity_violations
                    }

                except Exception as e:
                    logging.error(f"An unexpected error occurred: {e}")
                    self.results[af_name][n]['validity'] = {
                        'computed': {},
                        'status': 'Error',
                        'violations': {"ERROR": [f"An unexpected error occured: {e}"]}
                    }
                    
                logging.info(f"[BASE] (N={n}) {af_name} Validity = {self.results[af_name][n]['validity']['status']}")

                # 2. METAMORPHIC TESTS
                # Isomorphism
                iso_af, r_map = apply_isomorphism(base_af)
                try:
                    logging.info(f"Sent [ISO] (N={n}) {af_name}: {repr(iso_af)} → {self.llm_client.model_name}")
                    iso_ext = self._get_llm_response_for_af(iso_af)
                    logging.info(f"{self.llm_client.model_name} → {iso_ext}")
                    
                    fp_violations = self.verification_suite.verify_fundamental_properties(iso_af, iso_ext)
                    mr_violations = self.verification_suite.verify_isomorphism(base_extensions, iso_ext, r_map)
                    all_violations = {**fp_violations, **mr_violations}
                    
                    self.results[af_name][n]['isomorphism'] = {
                        'computed': str(iso_ext),
                        'status': 'FAIL' if all_violations else 'PASS',
                        'violations': all_violations
                    }
                    logging.info(f"[ISO] (N={n}) {af_name} Status = {self.results[af_name][n]['isomorphism']['status']}")
                    
                except TimeoutError as e:
                    logging.error(f'Aborting request of isomorphism AF for {af_name} (n={n}) to LLM: {e}')
                    self.results[af_name][n]['isomorphism'] = {
                        'computed': {},
                        'status': 'Request Timeout',
                        'violations': {"TIMEOUT": [str(e)]}
                    }
                    
                except Exception as e:
                    logging.error(f"An unexpected error occurred: {e}")
                    self.results[af_name][n]['isomorphism'] = {
                        'computed': {},
                        'status': 'Error',
                        'violations': {"ERROR": [f"An unexpected error occured: {e}"]}
                    }

                # Fundamental Consistency
                fc_af, sa_name = apply_fundamental_consistency(base_af)
                try:
                    logging.info(f"Sent [FC] (N={n}) {af_name}: {repr(fc_af)} → {self.llm_client.model_name}")
                    fc_ext = self._get_llm_response_for_af(fc_af)
                    logging.info(f"{self.llm_client.model_name} → {fc_ext}")
                    
                    fp_violations = self.verification_suite.verify_fundamental_properties(fc_af, fc_ext)
                    mr_violations = self.verification_suite.verify_fundamental_consistency(base_extensions, fc_ext, sa_name)
                    all_violations = {**fp_violations, **mr_violations}
                    
                    self.results[af_name][n]['fundamental_consistency'] = {
                        'computed': str(fc_ext),
                        'status': 'FAIL' if all_violations else 'PASS',
                        'violations': all_violations
                    }
                    logging.info(f"[FC] (N={n}) {af_name} Status = {self.results[af_name][n]['fundamental_consistency']['status']}")
                    
                except TimeoutError as e:
                    logging.error(f'Aborting request of fundamental consistent AF for {af_name} (n={n}) to LLM: {e}')
                    self.results[af_name][n]['fundamental_consistency'] = {
                        'computed': {},
                        'status': 'Request Timeout',
                        'violations': {"TIMEOUT": [str(e)]}
                    }
                    
                except Exception as e:
                    logging.error(f"An unexpected error occurred: {e}")
                    self.results[af_name][n]['fundamental_consistency'] = {
                        'computed': {},
                        'status': 'Error',
                        'violations': {"ERROR": [f"An unexpected error occured: {e}"]}
                    }
                    
                # Modularity
                mod_af, u_name = apply_modularity(base_af)
                try:
                    logging.info(f"Sent [MOD] (N={n}) {af_name}: {repr(mod_af)} → {self.llm_client.model_name}")
                    mod_ext = self._get_llm_response_for_af(mod_af)
                    logging.info(f"{self.llm_client.model_name} → {mod_ext}")
                    
                    fp_violations = self.verification_suite.verify_fundamental_properties(mod_af, mod_ext)
                    mr_violations = self.verification_suite.verify_modularity(base_extensions, mod_ext, u_name)
                    all_violations = {**fp_violations, **mr_violations}
                    
                    self.results[af_name][n]['modularity'] = {
                        'computed': str(mod_ext),
                        'status': 'FAIL' if all_violations else 'PASS',
                        'violations': all_violations
                    }
                    logging.info(f"[MOD] (N={n}) {af_name} Status = {self.results[af_name][n]['modularity']['status']}")
                    
                except TimeoutError as e:
                    logging.error(f'Aborting request of modular AF for {af_name} (n={n}) to LLM: {e}')
                    self.results[af_name][n]['modularity'] = {
                        'computed': {},
                        'status': 'Request Timeout',
                        'violations': {"TIMEOUT": [str(e)]}
                    }
                except Exception as e:
                    logging.error(f"An unexpected error occurred: {e}")
                    self.results[af_name][n]['modularity'] = {
                        'computed': {},
                        'status': 'Error',
                        'violations': {"ERROR": [f"An unexpected error occured: {e}"]}
                    }

                # Defense Dynamics (only if there are attacks)
                if base_af.defeats:
                    dd_af, defender, attacker, target = apply_defense_dynamics(base_af)
                    try:
                        logging.info(f"Sent [DD] (N={n}) {af_name}: {repr(dd_af)} → {self.llm_client.model_name}")
                        dd_ext = self._get_llm_response_for_af(dd_af)
                        logging.info(f"{self.llm_client.model_name} → {dd_ext}")
                        
                        fp_violations = self.verification_suite.verify_fundamental_properties(dd_af, dd_ext)
                        mr_violations = self.verification_suite.verify_defense_dynamics(base_extensions, dd_ext, defender, attacker, target)
                        all_violations = {**fp_violations, **mr_violations}
                        
                        self.results[af_name][n]['defense_dynamics'] = {
                            'computed': str(dd_ext),
                            'status': 'FAIL' if all_violations else 'PASS',
                            'violations': all_violations,
                            'info': f'Defender --- {attacker} --- {target}'
                        }
                        logging.info(f"[DD] (N={n}) {af_name} Status = {self.results[af_name][n]['defense_dynamics']['status']}")
                        
                    except TimeoutError as e:
                        logging.error(f'Aborting request of defense dynamic AF for {af_name} (n={n}) to LLM: {e}')
                        self.results[af_name][n]['defense_dynamics'] = {
                            'computed': {},
                            'status': 'Request Timeout',
                            'violations': {"TIMEOUT": [str(e)]}
                        }
                    except Exception as e:
                        logging.error(f"An unexpected error occurred: {e}")
                        self.results[af_name][n]['defense_dynamics'] = {
                            'computed': {},
                            'status': 'Error',
                            'violations': {"ERROR": [f"An unexpected error occured: {e}"]}
                        }
        
        report_generator = ReportGenerator()
        report_generator.export_to_excel(f'{self.llm_client.model_name}_lastest.xlsx', results=self.results)
        

        print("\nEvaluation run complete.")