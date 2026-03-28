import openai
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import tiktoken
import os

# Provider configuration presets (shared with utils.py)
_PROVIDER_PRESETS = {
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_API_BASE_URL",
        "default_model": "gpt-4-1106-preview",
    },
    "minimax": {
        "api_key_env": "MINIMAX_API_KEY",
        "base_url_env": "MINIMAX_API_BASE_URL",
        "default_base_url": "https://api.minimax.io/v1",
        "default_model": "MiniMax-M2.7",
    },
}


def _detect_provider():
    explicit = os.getenv("LLM_PROVIDER", "").lower()
    if explicit in _PROVIDER_PRESETS:
        return explicit
    if os.getenv("MINIMAX_API_KEY"):
        return "minimax"
    return "openai"


sum_prompt = """
Generate a structured summary from the provided medical source (report, paper, or book), strictly adhering to the following categories. The summary should list key information under each category in a concise format: 'CATEGORY_NAME: Key information'. No additional explanations or detailed descriptions are necessary unless directly related to the categories:

ANATOMICAL_STRUCTURE: Mention any anatomical structures specifically discussed.
BODY_FUNCTION: List any body functions highlighted.
BODY_MEASUREMENT: Include normal measurements like blood pressure or temperature.
BM_RESULT: Results of these measurements.
BM_UNIT: Units for each measurement.
BM_VALUE: Values of these measurements.
LABORATORY_DATA: Outline any laboratory tests mentioned.
LAB_RESULT: Outcomes of these tests (e.g., 'increased', 'decreased').
LAB_VALUE: Specific values from the tests.
LAB_UNIT: Units of measurement for these values.
MEDICINE: Name medications discussed.
MED_DOSE, MED_DURATION, MED_FORM, MED_FREQUENCY, MED_ROUTE, MED_STATUS, MED_STRENGTH, MED_UNIT, MED_TOTALDOSE: Provide concise details for each medication attribute.
PROBLEM: Identify any medical conditions or findings.
PROCEDURE: Describe any procedures.
PROCEDURE_RESULT: Outcomes of these procedures.
PROC_METHOD: Methods used.
SEVERITY: Severity of the conditions mentioned.
MEDICAL_DEVICE: List any medical devices used.
SUBSTANCE_ABUSE: Note any substance abuse mentioned.
Each category should be addressed only if relevant to the content of the medical source. Ensure the summary is clear and direct, suitable for quick reference.
"""

def call_openai_api(chunk):
    provider = _detect_provider()
    preset = _PROVIDER_PRESETS[provider]
    api_key = os.getenv(preset["api_key_env"])
    base_url = os.getenv(
        preset["base_url_env"],
        preset.get("default_base_url"),
    )
    client = OpenAI(api_key=api_key, base_url=base_url)
    model = os.getenv("LLM_MODEL", preset["default_model"])
    temperature = 0.5
    if provider == "minimax":
        temperature = max(0.01, min(temperature, 1.0))
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sum_prompt},
            {"role": "user", "content": f" {chunk}"},
        ],
        max_tokens=500,
        n=1,
        stop=None,
        temperature=temperature,
    )
    return response.choices[0].message.content

def split_into_chunks(text, tokens=500):
    encoding = tiktoken.encoding_for_model('gpt-4-1106-preview')
    words = encoding.encode(text)
    chunks = []
    for i in range(0, len(words), tokens):
        chunks.append(' '.join(encoding.decode(words[i:i + tokens])))
    return chunks   

def process_chunks(content):
    chunks = split_into_chunks(content)

    # Processes chunks in parallel
    with ThreadPoolExecutor() as executor:
        responses = list(executor.map(call_openai_api, chunks))
    # print(responses)
    return responses


if __name__ == "__main__":
    content = " sth you wanna test"
    process_chunks(content)

# Can take up to a few minutes to run depending on the size of your data input