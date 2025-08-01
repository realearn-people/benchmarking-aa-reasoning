from abc import ABC, abstractmethod
from py_arg.abstract_argumentation_classes.abstract_argumentation_framework import AbstractArgumentationFramework
from typing import Dict, List, Callable, Tuple, Set, FrozenSet
from helper_classes import VerificationSuite, ReportGenerator

# Third-party imports
from google import genai
from google.genai import types
from google.genai.types import HttpOptions
from google.api_core import exceptions
from google.genai.errors import ClientError, ServerError
from httpx import ReadTimeout

from openai import OpenAI 

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

import requests
import json
import os
import re
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="[{asctime}] - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
)

class LLMClient(ABC):
    """Abstract Base Class for LLM API interaction."""
    def __init__(self, model_name: str, timeout: int = 60, retry_delay_seconds: int = 5, max_retries: int = 3):
        self.model_name = model_name
        self.TIMEOUT = timeout
        self.RETRY_DELAY_SECONDS = retry_delay_seconds
        self.MAX_RETRIES = max_retries
        self.instruction = """
            You are an expert in computational argumentation. You will be given a string representation of an Argumentation Framework (AF) as a tuple of arguments and attack relationships. The input follows this schema: ([arguments], [attack_relationships]).

            Your task is to analyze the provided AF and compute all Grounded (GE), Complete (CE), Preferred (PE), and Stable (SE) extensions.

            Format your response as a single, clean JSON object with the following schema:
            {
                "GE": [[list of arguments], ...],
                "CE": [[list of arguments], ...],
                "PE": [[list of arguments], ...],
                "SE": [[list of arguments], ...]
            }

            - Each argument name must be a string.
            - If an extension type has multiple possible sets, list all of them.
            - If an extension type results in an empty set, represent it as [[]].
            - If an extension type has no valid sets, represent it as [].
            
            Here is an example of a valid response:
            {
                "GE": [[]],
                "CE": [[], ['A2', 'A4'], ['A1', 'A3']],
                "PE": [['A2', 'A4'], ['A1', 'A3']],
                "SE": [['A2', 'A4'], ['A1', 'A3']]
            }
            
            DO NOT include any additional text or explanation in your response.
            All answers must ONLY be list of argument names. You may not use mathematical notations or text description of the extensions.
        """

    @abstractmethod
    def query_llm_for_extensions(self, af: AbstractArgumentationFramework, af_name="") -> str:
        """Query the LLM to generate extensions for a given argumentation framework."""
        pass

    
    def parse_output_to_extensions(self, raw_output: str) -> Dict[str, List[List[str]]]:
        """
        Parses the raw LLM output, robustly extracting the JSON block.
        """
        try:
            # Use regex to find the JSON block, even if it's surrounded by other text
            json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                logging.warning(f"No JSON object found in the LLM response. Got {raw_output}")
                return {}
        except json.JSONDecodeError:
            logging.error(f"Failed to decode JSON from response: {raw_output}")
            return {}

class OpenAIClient(LLMClient):
    """Concrete implementation for using the OpenAI API."""
    
    def __init__(self, model_name: str, timeout: int = 60, retry_delay_seconds: int = 5, max_retries: int = 3):
        # Initialize OpenAI client with API key from environment variables
        super().__init__(model_name, timeout, retry_delay_seconds, max_retries)
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        self.client = OpenAI(api_key=api_key, timeout=self.TIMEOUT)

    def query_llm_for_extensions(self, af_string: str) -> str:
        """Query the OpenAI API to generate extensions for a given argumentation framework.
        
        Args:
            af_string (str): String representation of the argumentation framework.
        Returns:
            str: The raw response from the OpenAI API.
        """
        response = None
        try:
            if not self.model_name in ['o3', 'o4-mini']:
                response = self.client.responses.create(
                    model=self.model_name,
                    temperature=0,
                    input=af_string,
                    instructions=self.instruction
                )
            else:
                response = self.client.responses.create(
                    model=self.model_name,
                    input=af_string,
                    instructions=self.instruction
                )
                
            return response.output_text
        
        except Exception as e:
            raise

class GeminiClient(LLMClient):
    def __init__(self, model_name: str, timeout: int = 60, retry_delay_seconds: int = 5, max_retries: int = 3):
        super().__init__(model_name, timeout, retry_delay_seconds, max_retries)
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        self.client = genai.Client(api_key=api_key, http_options=HttpOptions(timeout=self.TIMEOUT * 1000))
        
    def query_llm_for_extensions(self, af_string: str) -> str:
        """Query the Gemini API to generate extensions for a given argumentation framework.
        
        Args:
            af_string (str): String representation of the argumentation framework.
        Returns:
            str: The raw response from the Gemini API.
        """
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    config=types.GenerateContentConfig(
                        system_instruction=self.instruction,
                        temperature=0.0
                    ),
                    contents=af_string
                )
                return response.text
                
            except ServerError as e:
                if e.code != 500:
                    logging.warning(f"Service unavailable (HTTP {e.code}): {e.message}.")
                else:
                    logging.warning(f"Timeout of {self.TIMEOUT} seconds reached.")
                
                if attempt < self.MAX_RETRIES - 1:
                    logging.info(f"Retrying in {self.RETRY_DELAY_SECONDS} seconds...")
                    time.sleep(self.RETRY_DELAY_SECONDS)
                    continue
                else:
                    raise TimeoutError(f"Max retries of {self.MAX_RETRIES} reached. Failing.")
                    break
            except ReadTimeout:
                logging.warning(f"Request to Gemini API timed out after {self.TIMEOUT} seconds.")
                if attempt < self.MAX_RETRIES - 1:
                    logging.info(f"Retrying in {self.RETRY_DELAY_SECONDS} seconds...")
                    time.sleep(self.RETRY_DELAY_SECONDS)
                    continue
                else:
                    raise TimeoutError(f"Max retries of {self.MAX_RETRIES} reached. Failing.")
        
class OllamaClient(LLMClient):
    def __init__(self, model_name: str, timeout: int = 60, retry_delay_seconds: int = 5, max_retries: int = 3):
        super().__init__(model_name, timeout, retry_delay_seconds, max_retries)
        self.url = "http://localhost:11434/api/generate"
        self.body = {
            "model": self.model_name,
            "options": {
                "temperature": 0.0
            },
            "system": self.instruction,
            "format": "json",
            "stream": False
        }
    
    def query_llm_for_extensions(self, af_string: str) -> str:
        """Query the Ollama API to generate extensions for a given argumentation framework.
        
        Args:
            af_string (str): String representation of the argumentation framework.
        Returns:
            str: The raw response from the Ollama API.
        """
        self.body['prompt'] = af_string
        for attempt in range(self.MAX_RETRIES):
            try:
                response = requests.post(self.url, json=self.body, timeout=self.TIMEOUT)
                return response.json()['response']
            except requests.Timeout:
                logging.warning(f"Request to Ollama API timed out after {self.TIMEOUT} seconds.")
                if attempt < self.MAX_RETRIES - 1:
                    logging.info(f"Retrying in {self.RETRY_DELAY_SECONDS} seconds...")
                    time.sleep(self.RETRY_DELAY_SECONDS)
                    continue
                else:
                    raise TimeoutError(f"Max retries of {self.MAX_RETRIES} reached. Failing.")
        
        
    
class LLMTester:
    """Orchestrates the entire evaluation process for a given LLM."""
    
    def __init__(self, llm_client: LLMClient, af_generators: Dict[str, Callable], ns: List[int], file_name=""):
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
        report_generator.export_to_excel(f'{self.llm_client.model_name}_lastest_real.xlsx', results=self.results)
        

        print("\nEvaluation run complete.")