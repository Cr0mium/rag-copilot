from datasets import load_dataset
import re
from langchain_core.documents import Document

data = load_dataset(
    "koutch/stackoverflow_python",
    split="train",
    streaming=True
)

def clean_text(text: str) -> str:
    text = re.sub(r"<.*?>", " ", text)   # remove HTML
    text = re.sub(r"\s+", " ", text)     # normalize spaces
    return text.strip()


def is_valid(row,x=100,y=100)-> bool:
    if (row['question_body'] != None and len(row['question_body'])> x)\
    and (row['answer_body'] != None and len(row['answer_body'])> y) \
    and (row['answer_score'] >= 2):
        return True
    return False

def stream_documents():
    for row in data:
        if not is_valid(row):
            continue

        content = f"""
        Title: {row['title']}

        Question:
        {clean_text(row['question_body'])}

        Answer:
        {clean_text(row['answer_body'])}
        """.strip()

        yield Document(
            page_content=content,
            metadata={
                "question_id": row["question_id"],
                "answer_id": row["answer_id"],
                "tags": row["tags"],
                "question_score": row["question_score"],
                "answer_score": row["answer_score"],
                "source": "stackoverflow"
            }
        )

if __name__ == "__main__":
    doc_stream = stream_documents()

    for i,doc in enumerate(doc_stream):
        print(doc.page_content)
        print(doc.metadata)
        if i>5:
            break