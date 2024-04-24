from enum import Enum
from pydantic import BaseModel


class RoleEnum(str, Enum):
    user = "user"
    assistant = "assistant"
    ai = "ai"
    human = "human"
    # custom
    system= "system"


class Message(BaseModel):
    id: int
    role: RoleEnum | str
    content: str


class Chat(BaseModel):
    id: int
    messages: list[Message] = []
