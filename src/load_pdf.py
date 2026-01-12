import pdfplumber
from pathlib import Path
from typing import List, Dict

def load_pdf(pdf_path: str) -> List[Dict]:
    documents = []
    pdf_path = Path(pdf_path)

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if not text:
                continue

            documents.append({
                "text": text.strip(),
                "source": pdf_path.name,
                "page": page_num + 1
            })

    return documents