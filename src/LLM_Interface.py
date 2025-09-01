from abc import ABC, abstractmethod
from py_arg.abstract_argumentation_classes.abstract_argumentation_framework import AbstractArgumentationFramework
from typing import Dict, List

# Third-party imports
from google import genai
from google.genai import types
from google.genai.types import HttpOptions
from google.genai.errors import ClientError, ServerError
from httpx import ReadTimeout

from openai import OpenAI 

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
        
        
    
