from pathlib import Path

def load_documents(data_dir: str):

    documents = []
    data_path = Path(data_dir)

    for i,file_path in enumerate(data_path.rglob("*.txt")):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read(10).strip()

        yield ({
            "text": text,
            "source": file_path.name,
        })


if __name__ == "__main__":
    docs = load_documents("data/raw")
    for d in docs:
        print(d["source"], "->", len(d["text"]), "chars")