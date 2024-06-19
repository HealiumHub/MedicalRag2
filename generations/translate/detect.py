import enum
from langchain.schema.output_parser import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langcodes import Language
from generations.completion import get_model

class LanguageEnum(enum.Enum):
    VIETNAMESE = "vi"
    ENGLISH = "en"

def get_language_of_text(text: str) -> str:
    # Prompt gpt3.5 to detect the language of the text
    model = get_model("gpt-3.5-turbo", temperature=0, streaming=False)
    system_prompt = """Detect the language of the following text. Return 'vi' if the text is in Vietnamese, 'en' if the text is in English."""
    user_prompt = """{text}"""
    messages = [
        SystemMessage(system_prompt),
        HumanMessage(user_prompt.format(text=text)),
    ]
    return (model | StrOutputParser()).invoke(messages)
