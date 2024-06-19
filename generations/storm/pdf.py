from md2pdf.core import md2pdf
from generations.translate.translate import Translator
from langchain.callbacks import get_openai_callback


def md_to_pdf(md: str, out_file_path: str):
    md2pdf(out_file_path, md, css_file_path="style.css", base_url=".")


def translate_article_and_produce_pdf():
    with get_openai_callback() as cb:
        with open("article.md", "r") as file:
            md = file.read()
            md_to_pdf(md, "article.pdf")
            print("Converted to PDF")

            # Try translating
            translator = Translator()
            translated_text = translator.translate_article(md)

            with open("article_vi.md", "w") as file:
                file.write(translated_text)

            md_to_pdf(translated_text, "article_vi.pdf")
            print("Converted to Viet PDF")

        print(f"Total translation cost: {cb}")


# with get_openai_callback() as cb:
#     with open("article.md", "r") as file:
#         md = file.read()
#         md_to_pdf(md, "article.pdf")
#         print("Converted to PDF")

#         # Try translating
#         translator = Translator()
#         translated_text = translator.translate_article(md)

#         with open("article_vi.md", "w") as file:
#             file.write(translated_text)

#         md_to_pdf(translated_text, "article_vi.pdf")
#         print("Converted to Viet PDF")

#     print(f"Total translation cost: {cb}")
