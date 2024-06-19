from langchain.schema.output_parser import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langcodes import Language
from tqdm import tqdm

from generations.completion import get_model
import requests
import urllib.parse

from generations.translate.detect import LanguageEnum
from ui.utilities import StreamHandler


class Translator:
    def __init__(self, model_name: str = "gpt-3.5-turbo-0125"):
        self.model_name = model_name

        # Temperature 0.5 for translation, 1 for simplification

        self.rewrite_model = get_model(model_name, 1, streaming=False)

    def translate(
        self,
        text: str,
        target_language: LanguageEnum = LanguageEnum.ENGLISH.value,
        stream_handler: StreamHandler = None,
    ) -> str:
        translate_model = get_model(
            self.model_name,
            0.5,
            streaming=True if stream_handler is not None else False,
        )
        # url = f"https://clients5.google.com/translate_a/t?client=at&sl=en&tl=vi&q={urllib.parse.quote(text[:2000])}"
        # resp = requests.get(url)
        # if resp.status_code != 200:
        #     raise Exception(f"Failed to translate text: {resp.status_code} {resp.text}")

        # print(resp.text)
        # return resp.json()[0]
        # translator = Translator()
        # translated_text = translator.translate(text, dest="vi")

        # return translated_text
        target_language = (
            "Vietnamese"
            if target_language == LanguageEnum.VIETNAMESE.value
            else "English"
        )
        system_prompt = f"""Translate the following markdown text to {target_language}. For links, only translate the text, do NOT translate the link. All <a> tag should be kept the same. The layout of markdown should not be changed. Only return the translated text."""
        user_prompt = """{text}"""
        messages = [
            SystemMessage(system_prompt),
            HumanMessage(user_prompt.format(text=text)),
        ]
        if stream_handler is not None:
            return (translate_model | StrOutputParser()).invoke(
                messages, config={"callbacks": [stream_handler]}
            )
        else:
            return (translate_model | StrOutputParser()).invoke(messages)

    def translate_article(
        self,
        article: str,
        rewrite: bool = False,
        language: LanguageEnum = LanguageEnum.ENGLISH.value,
    ) -> str:
        # Translate by headings to avoid file too large error.
        sections = [""]
        for line in article.split("\n"):
            if line.startswith("#"):
                sections.append(line)
            else:
                sections[-1] += "\n" + line

        # TODO: Translate all but don't translate the links in the text
        final_text = ""
        for i, section in tqdm(enumerate(sections), total=len(sections)):
            # Do not translate REFERENCES section
            if "##" in section and "References" in section:
                final_text += section.replace("References", "Nguồn tham khảo") + "\n\n"
                continue

            if rewrite:
                final_text += self.translate_and_rewrite(section, language) + "\n\n"
            else:
                final_text += self.translate(section, language) + "\n\n"

        return final_text

    def rewrite(self, text: str) -> str:
        system_prompt = """"""
        user_prompt = """TEXT: {text}"""
        messages = [
            SystemMessage(system_prompt),
            HumanMessage(user_prompt.format(text=text)),
        ]

        return (self.rewrite_model | StrOutputParser()).invoke(messages)

    def translate_and_rewrite(
        self, text: str, language: LanguageEnum = LanguageEnum.ENGLISH.value
    ) -> str:
        return self.rewrite(self.translate(text, language))
