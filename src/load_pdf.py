from pypdf import PdfReader

def load_pdf_stream(pdf_path):
    reader = PdfReader(pdf_path)

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue

        yield {
            "text": text,
            "source": pdf_path,
            "page": page_num + 1
        }

if __name__ == "__main__":
    reader = PdfReader("data/raw/postgresql.pdf")
    for i,page in enumerate(reader.pages):
        print(i)
        text=page.extract_text()
        print(type(text))
        if(i>=10):
            break