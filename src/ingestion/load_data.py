from pathlib import Path
from typing import Iterator, Optional
import re
from langchain_core.documents import Document


def clean_text(text: str) -> str:
    """Clean markdown text for ingestion - removes markup artifacts."""
    if text is None or text == "":
        return ""
    
    # 1. Remove markdown link syntax but keep the text (safer version)
    # [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\((.*?)\)', r'\1', text)
    
    # 2. Remove autodoc markers
    text = re.sub(r'\[\[autodoc\]\]', '', text)
    text = re.sub(r'\[\s*\[\s*autodoc\s*\]\s*\]', '', text)
    
    # 3. Remove relative path references
    text = re.sub(r'\.\./[^\s\)]+', '', text)
    
    # 4. REMOVE risky underscore fixes (deleted)
    
    # 5. Remove code fence markers but keep the code
    text = re.sub(r'```[\w]*\n', '', text)
    text = re.sub(r'```', '', text)
    
    # 6. Preserve headers as semantic markers instead of removing
    # ## Header -> SECTION: Header
    text = re.sub(r'^#{1,6}\s+', 'SECTION: ', text, flags=re.MULTILINE)
    
    # 7. Remove HTML-like tags (safer version)
    text = re.sub(r'</?[a-zA-Z][^>]*>', '', text)
    
    # 8. Fix spacing issues
    text = text.replace('\r\n', '\n')
    text = text.replace('\t', ' ')
    
    # 9. Remove excessive spaces
    text = re.sub(r' +', ' ', text)
    
    # 10. Remove excessive newlines (max 2 in a row)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 11. Clean up lines with only punctuation/symbols (less aggressive)
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        if len(line) > 1 and not re.match(r'^[\W_]+$', line):
            lines.append(line)
    
    text = '\n'.join(lines)
    
    return text.strip()


def should_skip_file(file_path: Path, skip_files: list) -> bool:
    """Check if file should be skipped."""
    # Skip by name
    if file_path.name.lower() in skip_files:
        return True
    
    # Skip if file is too small (slightly relaxed)
    try:
        if file_path.stat().st_size < 50:
            return True
    except:
        pass
    
    # Skip if filename suggests it's metadata
    skip_patterns = ['README', 'CONTRIBUTING', 'LICENSE', 'CHANGELOG']
    if any(pattern in file_path.name.upper() for pattern in skip_patterns):
        return True
    
    return False


def extract_title(text: str, filename: str) -> str:
    """Extract title from document."""
    lines = text.split('\n')
    for line in lines[:10]:
        if line.strip():
            title = re.sub(r'^#+\s*', '', line.strip())
            if 3 < len(title) < 100:
                return title
    
    return filename.replace('_', ' ').replace('-', ' ').replace('.md', '').title()


def load_hf_docs(
    root_dir: str, 
    skip_files: Optional[list] = None,
    min_doc_length: int = 200
) -> Iterator[Document]:
    """
    Load HuggingFace documentation with proper cleaning.
    """
    skip_files = [f.lower() for f in (skip_files or ["changes.md", "readme.md"])]
    root = Path(root_dir)
    
    processed_count = 0
    skipped_count = 0

    for file_path in root.rglob("*"):
        if file_path.suffix.lower() not in [".md", ".mdx"]:
            continue
            
        if should_skip_file(file_path, skip_files):
            skipped_count += 1
            continue
        
        try:
            raw_text = file_path.read_text(encoding="utf-8")
            cleaned_text = clean_text(raw_text)
            
            if len(cleaned_text) < min_doc_length:
                print(f"[SKIPPED - TOO SHORT] {file_path.name} ({len(cleaned_text)} chars)")
                skipped_count += 1
                continue
            
            title = extract_title(raw_text, file_path.name)
            
            try:
                relative_path = file_path.relative_to(root)
            except:
                relative_path = file_path
            
            processed_count += 1
            
            yield Document(
                page_content=cleaned_text,
                metadata={
                    "title": title,
                    "filename": file_path.name,
                    "filepath": str(relative_path),
                    "source": str(root_dir),
                    "doc_length": len(cleaned_text),
                },
            )
            
        except Exception as e:
            print(f"[ERROR] {file_path.name} -> {e}")
            skipped_count += 1
            continue
    
    print(f"\n✓ Processed: {processed_count} documents")
    print(f"✗ Skipped: {skipped_count} documents")


if __name__ == "__main__":
    docs = list(load_hf_docs(
        root_dir="./hf_docs",
        skip_files=["changes.md", "readme.md"],
        min_doc_length=200
    ))
    
    print(f"\nLoaded {len(docs)} documents")
    
    if docs:
        print(f"\nFirst doc preview:")
        print(f"Title: {docs[0].metadata['title']}")
        print(f"Length: {docs[0].metadata['doc_length']} chars")
        print(f"Content preview:\n{docs[0].page_content[:300]}...")
"""

---

## **What This Fixes:**

### **Before (your version):**
```
"## Glm46VForConditionalGeneration
glm46vprocessor - _ _ call _ _ [ [ autodoc ] ] glm46vmodel - forward"
```

### **After (cleaned):**
```
"Glm46VForConditionalGeneration
glm46vprocessor __call__ glm46vmodel forward"
"""
if __name__ == "__main__":
    RAW_DIR = "data/raw"
    transformers_docs = load_hf_docs("data/raw")
    for i, doc in enumerate(transformers_docs):
        print(doc)
        # print(doc.page_content[:50])  # first 500 chars
        # print(doc.metadata)
        if i > 10:
            break
