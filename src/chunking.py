import re

import spacy
from transformers import AutoTokenizer

nlp = spacy.load("en_core_web_sm")
hf_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")

SOFT_LIMIT = 400
HARD_LIMIT = 500
OVERLAP = 60


def count_tokens(text: str) -> int:
    return len(hf_tokenizer.encode(text, add_special_tokens=False))


def safe_truncate(text: str, max_tokens: int = HARD_LIMIT) -> str:
    ids = hf_tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return hf_tokenizer.decode(ids[:max_tokens])


def get_overlap_text(text: str, overlap: int = OVERLAP) -> tuple[str, int]:
    """Extract the last `overlap` tokens from text, return (text, token_count)."""
    ids = hf_tokenizer.encode(text, add_special_tokens=False)[-overlap:]
    decoded = hf_tokenizer.decode(ids)
    return decoded, len(ids)


def split_content_into_chunks(
    header: str,
    content: str,
    soft_limit: int,
    hard_limit: int,
    overlap_seed: str = "",  # tail of the previous chunk
) -> tuple[list[str], str]:
    """
    Returns (chunks, last_raw_content) where last_raw_content is the
    raw sentence text of the final chunk (no header), used to seed overlap
    into the next section.
    """
    header_tokens = count_tokens(header) + 1 if header else 0  # +1 for \n
    available = soft_limit - header_tokens

    doc = nlp(content)
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]

    chunks = []
    # seed current chunk with overlap from previous chunk
    if overlap_seed:
        current_sents = [overlap_seed]
        current_tokens = count_tokens(overlap_seed)
    else:
        current_sents = []
        current_tokens = 0

    last_raw_content = ""

    def flush(sents: list[str]) -> str:
        raw = " ".join(sents)
        body = header + "\n" + raw if header else raw
        return safe_truncate(body, hard_limit)

    for sent in sentences:
        sent_tokens = count_tokens(sent)

        if current_tokens + sent_tokens > available and current_sents:
            chunks.append(flush(current_sents))
            last_raw_content = " ".join(current_sents)

            # seed next chunk with overlap tail of what we just flushed
            overlap_text, overlap_tokens = get_overlap_text(last_raw_content)
            current_sents = [overlap_text]
            current_tokens = overlap_tokens

        current_sents.append(sent)
        current_tokens += sent_tokens

    # flush remainder
    if current_sents:
        chunks.append(flush(current_sents))
        last_raw_content = " ".join(current_sents)

    return chunks, last_raw_content


def chunk_text(text: str, overlap: int = OVERLAP) -> list[str]:
    raw_sections = re.split(r"(?=^#+\s)", text, flags=re.MULTILINE)
    raw_sections = [s.strip() for s in raw_sections if s.strip()]

    chunks = []
    last_raw_content = ""  # carries overlap seed across sections

    for section in raw_sections:
        lines = section.split("\n", 1)
        if re.match(r"^#+\s", lines[0]):
            header = lines[0].strip()
            content = lines[1].strip() if len(lines) > 1 else ""
        else:
            header = ""
            content = section.strip()

        if not header and not content:
            continue

        if header and not content:
            continue

        # compute overlap seed from the tail of the last chunk's raw content
        overlap_seed = ""
        if last_raw_content:
            overlap_seed, _ = get_overlap_text(last_raw_content, overlap)

        section_chunks, last_raw_content = split_content_into_chunks(
            header, content, SOFT_LIMIT, HARD_LIMIT, overlap_seed
        )
        chunks.extend(section_chunks)

    return chunks


if __name__ == "__main__":
    text = """

# HParams
Over the years, many `timm` models have been trained with various hyper-parameters as the libraries and models evolved. I don't have a record of every instance, but have recorded instances of many that can serve as a very good starting point.

## Tags
Most `timm` trained models have an identifier in their pretrained tag that relates them (roughly) to a family / version of hparams I've used over the years.

| Tag(s) | Description | Optimizer | LR Schedule | Other Notes |
|--------|-----
{'filename': 'hparams.mdx', 'filepath': 'data/raw/pytorch/hparams.mdx', 'source': './data/raw/pytorch'}
------
# Results

CSV files containing an ImageNet-1K and out-of-distribution (OOD) test set validation results for all models with pretrained weights is located in the repository [results folder](https://github.com/rwightman/pytorch-image-models/tree/master/results).

## Self-trained Weights

The table below includes ImageNet-1k validation results of model weights that I've trained myself. It is not updated as frequently as the csv results outputs linked above.

|Model | Acc@1 (Err) | Acc@5 (Err) | Pa
{'filename': 'results.mdx', 'filepath': 'data/raw/pytorch/results.mdx', 'source': './data/raw/pytorch'}
------
# timm

<img class="float-left !m-0 !border-0 !dark:border-0 !shadow-none !max-w-lg w-[150px]" src="https://huggingface.co/front/thumbnails/docs/timm.png"/>

`timm` is a library containing SOTA computer vision models, layers, utilities, optimizers, schedulers, data-loaders, augmentations, and training/evaluation scripts.

It comes packaged with >700 pretrained models, and is designed to be flexible and easy to use.

Read the [quick start guide](quickstart) to get up and running with the `timm` l
{'filename': 'index.mdx', 'filepath': 'data/raw/pytorch/index.mdx', 'source': './data/raw/pytorch'}
------
# Model Summaries

The model architectures included come from a wide variety of sources. Sources, including papers, original impl ("reference code") that I rewrote / adapted, and PyTorch impl that I leveraged directly ("code") are listed below.

Most included models have pretrained weights. The weights are either:

1. from their original sources
2. ported by myself from their original impl in a different framework (e.g. Tensorflow models)
3. trained from scratch using the included training scrip
{'filename': 'models.mdx', 'filepath': 'data/raw/pytorch/models.mdx', 'source': './data/raw/pytorch'}

"""
    for chunk in chunk_text(text):
        print(chunk)
        print("=" * 25)
