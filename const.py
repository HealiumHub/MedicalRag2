class PromptConfig:
    DISCLAIMER = (
        "\n\n*Please consult professional doctor for accurate medical advices.*"
    )
    PERSONALITY = """You are MedLight, an assistant developed to help bridge medical research to the public.
The system will give you some relevant research articles that you can use.
Do not use research papers if they are irrelevant to the question."""


MODELS = [
    "phi3:latest",
    "gpt-3.5-turbo-0125",
    "llama3:8b",
    "llama3:instruct",
    "gemma:2b",
    "gemma:2b-instruct",
    "gemma:7b-instruct",
    "gemma:latest",
    "gpt-4-0125-preview",
    # 'claude-3-opus-20240229',
    # 'claude-3-sonnet-20240229',
    # 'claude-3-haiku-20240307',
    # 'gemini-1.0-pro',
    # 'gemini-1.5-pro (Not Supported)',
    # 'mixtral-8x7b-32768',
    # 'llama2-70b-4096'
]

MAX_OUTPUT = 1200

MODEL_CONTEXT_LENGTH = {
    "phi3:latest": 128000,
    "llama3:8b": 8192,
    "llama3:instruct": 8192,
    "gemma:2b": 8192,
    "gemma:2b-instruct": 8192,
    "gemma:7b-instruct": 8192,
    "gemma:latest": 8192,
    "gpt-3.5-turbo-0125": 16385,
    "gpt-4-0125-preview": 128000,
    "claude-3-opus-20240229": 200000,
    "claude-3-sonnet-20240229": 200000,
    "claude-3-haiku-20240307": 200000,
    "gemini-1.0-pro": 30720,
    "gemini-1.5-pro (Not Supported)": 1000000,
    "gemma-7b-it": 8192,
    "mixtral-8x7b-32768": 32768,
    "llama2-70b-4096": 4096,
}
