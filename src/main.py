from LLM_Interface import OpenAIClient, OllamaClient, GeminiClient, LLMTester
from report_generator import ReportGenerator

from af_utils import (
    generate_no_conflict,
    generate_linear_attack_chain,
    generate_cycle,
    generate_single_target_multiple_attackers,
    generate_single_attack_multiple_defenders,
    generate_disconnected_symmetric_pairs,
)

# 1. Choose the LLM to test by initializing its client.
gpt_4o = OpenAIClient(model_name="gpt-4o", timeout=900)
o3 = OpenAIClient(model_name="o3", timeout=900)
o4_mini = OpenAIClient(model_name="o4-mini", timeout=900)
gemini_2_5_pro = GeminiClient(model_name="gemini-2.5-pro", timeout=900)
llama_3_8b = OllamaClient(model_name="llama3:8b", timeout=900)
gemma_3_12b = OllamaClient(model_name="gemma3:12b", timeout=900)


# 2. Define the parameters for the test run.
# Map names to generator functions for easier iteration.
af_generators_to_test = {
    "no_conflict": generate_no_conflict,
    "linear_attack_chain": generate_linear_attack_chain,
    "cycle": generate_cycle,
    "single_target_multiple_attackers": generate_single_target_multiple_attackers,
    "single_attack_multiple_defenders": generate_single_attack_multiple_defenders,
    "symmetric_disconnected": generate_disconnected_symmetric_pairs,
}
sizes_to_test = [4, 8, 16, 20]

list_models = [gemini_2_5_pro]
for model in list_models:
    llm_tester = LLMTester(
        llm_client=model,
        af_generators=af_generators_to_test,
        ns=sizes_to_test
    )
    
    llm_tester.run_evaluation()