import os
from typing import Union
from langchain.schema.output_parser import StrOutputParser
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms.ollama import Ollama
from langchain_core.messages import HumanMessage, SystemMessage

from const import API_KEY, MAX_OUTPUT, MODEL_CONTEXT_LENGTH, PromptConfig


def get_model(model_name: str, temperature: float) -> Union[ChatOpenAI, Ollama]:
    if "gpt" in model_name:
        model = ChatOpenAI(
            temperature=temperature,
            model=model_name,
            api_key=API_KEY,
            verbose=True,
            streaming=True,
            max_tokens=MAX_OUTPUT,
        )
    else:
        model = Ollama(
            temperature=temperature,
            model=model_name,
            verbose=True,
            num_predict=MAX_OUTPUT,
            repeat_penalty=1.5,
        )

    return model


def preprocess_context(model_name: str, prompt: str) -> str:
    CONTEXT_LENGTH = MODEL_CONTEXT_LENGTH.get(model_name, 8192)

    # Truncate to avoid exceeding the maximum context length:
    # - Different models have different ways to count token.
    # - This only approximates it by having 1 tok = 1.8 characters.
    return prompt[: int((CONTEXT_LENGTH - MAX_OUTPUT) * 1.8)]


def get_answer_with_context(
    query: str,
    model_name: str,
    related_articles: Union[str, list[dict]],
    custom_instruction: str,
    temperature: float,
    stream_handler,
) -> str:
    # Initialize models & context length.
    model = get_model(model_name, temperature)

    user_prompt = f"Please answer the following medical question and provide relevant references. Question: {query}"
    system_prompt = (
        f"Here are some research papers that might be relevant: \n\n{related_articles}"
    )
    system_prompt = preprocess_context(model_name, system_prompt)

    if "gpt" in model_name:
        # Separates context and personality prompt -> clearer context 4 models.
        messages = [
            SystemMessage(content=custom_instruction),
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    else:
        # gemini and claude -> no 2 consecutive system prompts.
        messages = [
            SystemMessage(content=f"{custom_instruction}\n\n{system_prompt}"),
            HumanMessage(content=user_prompt),
        ]

    # RAW OUTPUT
    chain = model | StrOutputParser()

    if stream_handler is None:
        answer = chain.invoke(messages)
    else:
        answer = chain.invoke(messages, config={"callbacks": [stream_handler]})

    # Suffix disclaimer, this saves token and we don't have to prompt it.
    answer += PromptConfig.DISCLAIMER
    return answer
