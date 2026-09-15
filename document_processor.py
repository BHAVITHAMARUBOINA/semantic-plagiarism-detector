import fitz
from docx import Document


def extract_text(file):
    filename = file.name.lower()

    # TXT file
    if filename.endswith(".txt"):
        return file.read().decode("utf-8", errors="ignore")

    # PDF file
    elif filename.endswith(".pdf"):
        pdf = fitz.open(stream=file.read(), filetype="pdf")

        text = ""

        for page in pdf:
            text += page.get_text()

        pdf.close()

        return text

    # DOCX file
    elif filename.endswith(".docx"):
        doc = Document(file)

        text = "\n".join(
            paragraph.text
            for paragraph in doc.paragraphs
        )

        return text

    # Unsupported file
    else:
        return ""