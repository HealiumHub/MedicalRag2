from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class Source(BaseModel):
    id: str
    doi: str
    file_name: str
    page: int
    content: str
    score: float


class RoleEnum(str, Enum):
    user = "user"
    assistant = "assistant"
    ai = "ai"
    human = "human"
    # custom
    system = "system"


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
    # citations: List[Citation] = Field(
    #     ..., description="Citations from the given sources that justify the answer."
    # )

    def __str__(self):
        # Format the answer to display to user.
        answer = self.answer

        # for i, citation in enumerate(self.citations):
        #     # Insert [1], [2], [3], ... to the answer.
        #     insert_index = answer.find(citation.quote_in_answer) + len(
        #         citation.quote_in_answer
        #     )
        #     answer = answer[:insert_index] + f" [{i+1}] " + answer[insert_index:]

        #     answer += f"\n\n[{i+1}] from [{citation.file_name}.pdf](): \n> {citation.quote_in_source}"

        return answer


class Message(BaseModel):
    id: int
    role: RoleEnum | str
    content: str | QuotedAnswer
    related_articles: list[Source] = []
    expanded_queries: list[str] = []
    hyde_passages: list[str] = []


class Chat(BaseModel):
    id: int
    messages: list[Message] = []
