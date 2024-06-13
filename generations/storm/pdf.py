from md2pdf.core import md2pdf


def md_to_pdf(md: str, out_file_path: str):
    md2pdf(out_file_path, md, css_file_path="style.css", base_url='.')

with open("article.md", "r") as file:
    md_to_pdf(file.read(), "article.pdf")
