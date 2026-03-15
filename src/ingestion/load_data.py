from pathlib import Path
from typing import Iterator, Optional

from langchain_core.documents import Document


def clean_text(text: str) -> str:
    """Clean markdown text for ingestion."""
    if text is None:
        return ""
    # Remove excessive whitespace
    text = text.replace("\r\n", "\n")
    text = text.replace("\t", " ")
    text = text.strip()
    return text


def load_hf_docs(
    root_dir: str, skip_files: Optional[list] = None
) -> Iterator[Document]:

    skip_files = skip_files or ["changes.md"]
    root = Path(root_dir)

    for file_path in root.rglob("*"):
        if file_path.suffix.lower() in [".md", ".mdx"]:
            try:
                text = clean_text(file_path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[SKIPPED] {file_path} -> {e}")
                continue

            yield Document(
                page_content=text,
                metadata={
                    "filename": file_path.name,
                    "filepath": str(file_path),
                    "source": root_dir,
                },
            )


if __name__ == "__main__":
    RAW_DIR = "data/raw"
    transformers_docs = load_hf_docs("data/raw")
    for i, doc in enumerate(transformers_docs):
        print(doc)
        # print(doc.page_content[:50])  # first 500 chars
        # print(doc.metadata)
        if i > 100:
            break
