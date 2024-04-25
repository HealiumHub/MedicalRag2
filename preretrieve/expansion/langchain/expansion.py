import logging
import os
from typing import Any, Optional

from dotenv import load_dotenv
from langchain.output_parsers import PydanticOutputParser
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY", "")

logger = logging.getLogger(__name__)


class ParaphrasedQuery(BaseModel):
    """Perform query paraphase. If there are multiple common ways of phrasing a user question \
    or common synonyms for key words in the question, make sure to return multiple versions \
    of the query with the different phrasings.

    If there are acronyms or words you are not familiar with, do not try to rephrase them.

    Return at least 3 versions of the question."""

    paraphrased_query: list[str] = Field(
        ...,
        description="A list of unique paraphrasing of the original question.",
    )


# llm = Ollama(model="llama2")
class QueryExpansion:
    def __init__(self, with_openAI: bool):
        self.LLM_factory("", with_openAI)

    def LLM_factory(self, model, with_openAI: bool):
        if with_openAI:
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo-0125", temperature=0.1, api_key=API_KEY
            )
        else:
            if len(model) > 0:
                self.llm = Ollama(model=model)
            else:
                self.llm = Ollama(model="llama2")

        parser = PydanticOutputParser(pydantic_object=ParaphrasedQuery)
        prompt = PromptTemplate(
            template="""{format_instructions}\n{query}\n""",
            input_variables=["query"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        self.chain = prompt | self.llm | parser

    def paraphase_query(self, query: str):
        response = self.chain.invoke(query)
        response.paraphrased_query.append(query)
        print(response)
        return response.paraphrased_query
