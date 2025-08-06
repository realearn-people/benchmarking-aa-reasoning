## LLM Reasoning Benchmark for Abstract Argumentation

This project provides a comprehensive framework for benchmarking the reasoning capabilities of Large Language Models (LLMs) in the context of Abstract Argumentation Frameworks (AFs). It evaluates LLMs on their ability to compute various argumentation semantics and tests the consistency of their reasoning using metamorphic testing.

## Overview

Abstract argumentation is a key area of AI for modeling and reasoning about conflicting information. This framework systematically assesses how well different LLMs can understand and solve complex argumentation problems. It does this by:

1.  **Generating Diverse Argumentation Frameworks**: Creates various AF structures, such as chains, cycles, and disconnected graphs.
2.  **Querying LLMs**: Submits these AFs to different LLMs (e.g., OpenAI's GPT models, Google's Gemini, and local models via Ollama) to compute standard argumentation semantics.
3.  **Verifying Correctness**: Checks if the LLM's output is valid according to the definitions of the semantics.
4.  **Metamorphic Testing**: Applies transformations (metamorphic relations) to the AFs and verifies if the LLM's reasoning remains consistent and logical across these changes.
5.  **Reporting**: Generates detailed reports in Excel format, summarizing the performance of each LLM.

## Features

- **Multi-LLM Support**: Easily configurable to test models from OpenAI, Google, and local models through Ollama.
- **Multiple Semantics**: Evaluates Grounded (GE), Complete (CE), Preferred (PE), and Stable (SE) extensions.
- **Variety of AF Structures**: Includes generators for common and challenging argumentation graph structures.
- **Metamorphic Testing**: Implements several metamorphic relations to perform a deeper, more robust evaluation of an LLM's logical consistency:
  - Isomorphism
  - Fundamental Consistency
  - Modularity
  - Defense Dynamics
- **Automated Reporting**: Automatically generates detailed Excel reports from the evaluation results.

## Project Structure

```
├── prompts/              # Contains system prompts for the LLMs
├── reports/              # Output directory for evaluation reports
├── src/
│   ├── main.py           # Main script to configure and run the evaluation
│   ├── LLM_Interface.py  # Interfaces for different LLM APIs (OpenAI, Gemini, Ollama)
│   ├── LogicTester.py    # This class is used to test the logic of the testing framework, assuming that the LLM provided correct answers. (Unused)
│   ├── af_utils.py       # Utility functions for generating AFs and applying transformations
│   ├── helper_classes.py # Helper classes used across the project
│   └── report_generator.py # Generates Excel reports
├── requirements.txt      # Project dependencies
└── README.md             # This file
```

## Setup

1.  **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd benchmarking-aa-reasoning
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up API Keys:**
    You need to provide API keys for the services you want to use. Create a `.env` file in the root directory or set them as environment variables:

    ```
    OPENAI_API_KEY="your-openai-api-key"
    GEMINI_API_KEY="your-google-gemini-api-key"
    ```

4.  **Using Local Models with Ollama:**
    If you want to test local models, make sure you have [Ollama](https://ollama.ai/) installed and running. You can pull models like Llama 3 by running:
    ```bash
    ollama pull llama3:8b
    ```

## How to Run

1.  **Configure the Evaluation**: Open `src/main.py`.

    - Choose the LLM(s) you want to test by uncommenting or adding the corresponding client initializations.
    - Add the configured models to the `list_models` list.
    - You can adjust the `af_generators_to_test` dictionary to select which AF structures to use.
    - You can change the `sizes_to_test` list to specify the number of arguments in the generated AFs.

2.  **Run the script:**
    ```bash
    python src/main.py
    ```

## How It Works

The evaluation process is driven by the `LogicTester` class. For each selected AF generator and size, it performs the following steps:

1.  **Base Test**: An initial AF is generated, and the LLM is queried for its extensions. The response is checked for fundamental correctness (e.g., conflict-freeness, admissibility).
2.  **Metamorphic Tests**: The base AF is transformed using one of the metamorphic relations, and the LLM is queried again with this new AF. The results from the original and the transformed AF are then compared to see if they satisfy the expected logical relationship.

For example, under the **isomorphism** test, the arguments in the AF are renamed. A logically consistent LLM should produce the same set of extensions, just with the new names.

## Results

The evaluation results are saved as an Excel file in the `reports/` directory. The filename is based on the model and configuration. The report contains a detailed breakdown of passes and fails for each test type and size, including the computed vs. expected extensions and any detected violations.
