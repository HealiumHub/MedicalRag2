import requests

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

    def translate(self, q, target="en", source="vi"):
        payload = {
            "q": q,
            "target": target,
            "format": self.format,
            "source": source,
            "key": self.key,
        }

        response = requests.post(self.translationUrl, data=payload)
        return response.text

    def translateViToEn(self, q, target="en", source="vi"):
        return self.translate(q, target, source)

    def translateEnToVi(self, q, target="vi", source="en"):
        payload = {
            "q": q,
            "target": target,
            "format": self.format,
            "source": source,
            "key": self.key,
        }

        response = requests.post(self.translationUrl, data=payload)
        return response.text
