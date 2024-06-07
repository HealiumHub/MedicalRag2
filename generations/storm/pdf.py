import markdown2
from weasyprint import HTML
import sys

def md_to_pdf(md: str, out_file_path: str):
    # Convert Markdown to HTML
    html_text = markdown2.markdown(md)

    # Convert HTML to PDF
    HTML(string=html_text).write_pdf(out_file_path)
    
with open('article.md', 'r') as file:
    md_to_pdf(file.read(), 'article.pdf')