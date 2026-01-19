from typing import List, Dict
import spacy

tokenize= spacy.load('en_core_web_sm')

def chunk_text(text: str,chunk_size: int = 500,overlap: int = 100) -> List[str]:
# split texts with overlaps
    doc = tokenize(text)

    sentences = list(doc.sents)

    chunks = []
    current_chunk = []
    current_len = 0

    for sent in sentences:
        #iterate each token for each sent= sent tokens
        new_sent_tokens = [t.text for t in sent if not t.is_space]
        new_sent_len = len(new_sent_tokens)

        # If new len + old len > chunk_size => flush
        if current_len + new_sent_len > chunk_size:
            chunks.append(" ".join(current_chunk))

            # --- Overlap handling (token-based) ---
            overlap_chunk = []
            overlap_tokens = 0

            # Walk backwards, for each sent in chunk
            for prev_sent in reversed(current_chunk):
                prev_tokens = prev_sent.split()
                if overlap_tokens + len(prev_tokens) > overlap:
                    break
                overlap_chunk.insert(0, prev_sent)
                overlap_tokens += len(prev_tokens)

            current_chunk = overlap_chunk
            current_tokens = overlap_tokens

        # Add sentence
        current_chunk.append(sent.text.strip())
        current_tokens += new_sent_len

    # Add final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

if __name__ == '__main__':
    text="""About this documentation
************************

Python's documentation is generated from reStructuredText sources
using Sphinx, a documentation generator originally created for Python
and now maintained as an independent project.

Development of the documentation and its toolchain is an entirely
volunteer effort, just like Python itself.  If you want to contribute,
please take a look at the Dealing with Bugs page for information on
how to do so.  New volunteers are always welcome!

Many thanks go to:

* Fred L. Drake, Jr., the creator of the original Python documentation
  toolset and author of much of the content;

* the Docutils project for creating reStructuredText and the Docutils
  suite;

* Fredrik Lundh for his Alternative Python Reference project from
  which Sphinx got many good ideas.


Contributors to the Python documentation
========================================

Many people have contributed to the Python language, the Python
standard library, and the Python documentation.  See Misc/ACKS in the
Python source distribution for a partial list of contributors.

It is only with the input and contributions of the Python community
that Python has such wonderful documentation -- Thank You!

"""
    chunk_text(text)