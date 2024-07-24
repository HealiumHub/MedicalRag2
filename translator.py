import os
import requests
from dotenv import load_dotenv

load_dotenv()

from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI

"""
Translation engine using Google Cloud Translation API
Docs: https://cloud.google.com/translate/docs/reference/rest/v2/translate
"""


class TranslationEngine:
    def __init__(self, key, format="text"):
        # TODO: change this to fit the pricing plan
        self.translationUrl = "https://translation.googleapis.com/language/translate/v2"
        self.format = format
        self.key = key

    def translate(self, q, target, source=""):
        payload = {
            "q": q,
            "target": target,
            "format": self.format,
            "source": source,
            "key": self.key,
        }

        response = requests.post(self.translationUrl, data=payload)
        try:
            return (
                response.json().get("data").get("translations")[0].get("translatedText")
            )
        except:
            return "Error"

    def translateViToEn(self, q):
        return self.translate(q, target="en", source="vi")

    def translateEnToVi(self, q):
        return self.translate(q, target="vi", source="en")

    def translateToEn(self, q):
        return self.translate(q, target="en")

    def translateToVi(self, q):
        return self.translate(q, target="vi")

    def refineVi(self, q):
        messages = [
            ChatMessage(
                role="system",
                content="""
                Bạn là một trợ lý y tế sức khoẻ. Bạn nhận vào một đoạn thông tin y tế bằng tiếng việt, 
                hãy viết lại nó một cách tự nhiên và dễ hiểu nhất với đại chúng.
                """,
            ),
            ChatMessage(role="user", content=q),
        ]

        resp = OpenAI(model="gpt-4o-mini").chat(messages)
        return resp.choices[0].message.content.strip()


if __name__ == "__main__":
    t = TranslationEngine(key=os.getenv(key="GOOGLE_API_KEY"))
    res = t.translateToVi("I'm stuyding at RMIT University")
    print(res)

    res2 = t.translateToEn("Tôi đang học tại Đại học RMIT")
    print(res2)
