import logging

from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from generations.completion import get_model

logger = logging.getLogger(__name__)


class HypotheticalDocument(BaseModel):
    passage: str = Field(
        ...,
        description="A scientific paper passage to answer the question.",
    )


class HyDE:
    def __init__(self, model: str = "gpt-3.5-turbo-0125"):
        self.initialize_llm(model)

    def initialize_llm(self, model: str):
        # Temperature needs to be high to generate more diverse results.
        self.llm = get_model(model, 0.7)
        parser = PydanticOutputParser(pydantic_object=HypotheticalDocument)
        prompt = PromptTemplate(
            template="""You are an expert medical researcher, please write a scientific paper passage to answer the following question:
'{query}'\n {format_instructions}\n""",
            input_variables=["query"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        self.chain = prompt | self.llm | parser

    def run(self, query: str) -> str:
        response: HypotheticalDocument = self.chain.invoke(query)
        logger.debug(response)
        return response.passage
