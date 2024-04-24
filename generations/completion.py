from typing import List, Optional

import streamlit as st
from const import MAX_OUTPUT, MODEL_CONTEXT_LENGTH
from langchain.schema.output_parser import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms.ollama import Ollama
from langchain_core.messages import ChatMessage, HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import PromptTemplate

API_KEY = ""


class Citation(BaseModel):
    file_name: str = Field(
        ...,
        description="The file name of a SPECIFIC source which justifies the answer.",
    )
    quote_in_source: str = Field(
        ...,
        description="The VERBATIM quote from the specified source that justifies the answer.",
    )
    quote_in_answer: Optional[str] = Field(
        ...,
        description="The VERBATIM quote from your answer that refers to this source.",
    )

    def __str__(self):
        return (
            f"from {self.file_name}: '{self.quote_in_source}', '{self.quote_in_answer}'"
        )


class QuotedAnswer(BaseModel):
    """Answer the user question based only on the given sources, and cite the sources used."""

    answer: str = Field(
        ...,
        description="The answer to the user question, which is based on the given sources.",
    )
    citations: List[Citation] = Field(
        ..., description="Citations from the given sources that justify the answer."
    )

    def __str__(self):
        # Format the answer to display to user.
        answer = self.answer

        for i, citation in enumerate(self.citations):
            # Insert [1], [2], [3], ... to the answer.
            insert_index = answer.find(citation.quote_in_answer) + len(
                citation.quote_in_answer
            )
            answer = answer[:insert_index] + f" [{i+1}] " + answer[insert_index:]

            answer += f"\n\n[{i+1}] from [{citation.file_name}.pdf](): \n> {citation.quote_in_source}"

        return answer


def get_answer_with_context(
    query: str, model_name: str, related_articles: str | list[dict], stream_handler
):
    # Initialize models & context length.
    if model_name in MODEL_CONTEXT_LENGTH:
        CONTEXT_LENGTH = MODEL_CONTEXT_LENGTH[model_name]
    else:
        print(
            f"Model {model_name} not found in MODEL_CONTEXT_LENGTH, using default value."
        )
        CONTEXT_LENGTH = 16385

    if "gpt" in model_name:
        model = ChatOpenAI(
            temperature=0,
            model=model_name,
            api_key=API_KEY,
            verbose=True,
            streaming=True,
            callbacks=[stream_handler],
            max_tokens=MAX_OUTPUT,
        )
    else:
        model = Ollama(
            temperature=0,
            model=model_name,
            verbose=True,
            callbacks=[stream_handler],
            num_predict=MAX_OUTPUT,
        )

    system_prompt = f"Here are some research papers that might be relevant: \n\n{related_articles[:1]}"
    parser = PydanticOutputParser(pydantic_object=QuotedAnswer)
    user_prompt = f"""
    Please answer the following medical question and provide relevant references. Question: {query}
    \n\n
    {parser.get_format_instructions()}
    """

    # personality_prompt = "You are MedLight, an assistant developed to help bridge medical research to the public. The system will give you some relevant research articles that you can use. Use your own knowledge if there is no relevant articles given. \n"
    personality_prompt = "You are a world class algorithm to answer questions with correct and exact citations\n"

    # This is to avoid exceeding the maximum context length.
    # Different models have different ways to count token.
    # This only approximates it by having 1 tok = 1.8 characters.
    system_prompt = system_prompt[: int((CONTEXT_LENGTH - MAX_OUTPUT) * 1.8)]
    # print(f"System prompt: {system_prompt}")

    if "gpt" in model_name:
        messages = [
            # *transform_to_chat_messages(st.session_state.messages),
            SystemMessage(content=personality_prompt),
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    else:
        # gemini and claude only supports alternating between system and user messages -> no 2 consecutive system prompts.
        messages = [
            # TODO: not using context from previous messages.
            # *transform_to_chat_messages(st.session_state.messages),
            SystemMessage(content=f"{personality_prompt}\n\n{system_prompt}"),
            HumanMessage(content=user_prompt),
        ]

    # chain = model | StrOutputParser()
    chain = model | parser

    answer = chain.invoke(messages)
    # print(answer.citations)
    return answer
