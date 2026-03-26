import re
import spacy
from transformers import AutoTokenizer
import src.config as config

# Load resources once
nlp = spacy.load("en_core_web_sm")
hf_tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL)

class RAGChunker:
    def __init__(self, soft_limit=config.SOFT_LIMIT, hard_limit=config.HARD_LIMIT):
        self.soft_limit = soft_limit
        self.hard_limit = hard_limit

    def count_tokens(self, text: str) -> int:
        if not text: return 0
        return len(hf_tokenizer.encode(text, add_special_tokens=False))

    def safe_truncate(self, text: str, max_tokens: int) -> str:
        ids = hf_tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return text
        return hf_tokenizer.decode(ids[:max_tokens])

    def get_sentence_context(self, sentences: list[str], count: int = 2) -> str:
        return " ".join(sentences[-count:]) if sentences else ""

    def _flush(self, header: str, sents: list[str]) -> str:
        raw_body = " ".join(sents).strip()
        full_text = f"{header}\n{raw_body}" if header else raw_body
        return self.safe_truncate(full_text, self.hard_limit)

    def process_section(self, header: str, content: str, prev_context: str = "") -> tuple[list[str], str]:
        
        header_len = self.count_tokens(header) + 1 if header else 0
        available_space = self.soft_limit - header_len
        
        sentences = [s.text.strip() for s in nlp(content).sents if s.text.strip()]
        
        chunks = []
        current_batch = [prev_context] if prev_context else []
        current_tokens = self.count_tokens(prev_context)

        last_batch = []  # ✅ FIX 1: clean overlap source

        for sent in sentences:
            sent_tokens = self.count_tokens(sent)

            # Case 1: Very large sentence
            if sent_tokens > available_space:
                if current_batch:
                    chunk = self._flush(header, current_batch)
                    if self.count_tokens(chunk) > 30:  # ✅ FIX 2: consistent filtering
                        chunks.append(chunk)
                    last_batch = current_batch.copy()

                # Add large sentence as its own chunk
                large_chunk = self._flush(header, [sent])
                chunks.append(large_chunk)

                # ✅ FIX 3: token-based tail extraction
                ids = hf_tokenizer.encode(sent, add_special_tokens=False)
                tail_ids = ids[-30:]
                context_str = hf_tokenizer.decode(tail_ids)

                current_batch = [context_str] if context_str else []
                current_tokens = self.count_tokens(context_str)
                continue

            # Case 2: Overflow
            if current_tokens + sent_tokens > available_space and current_batch:
                chunk = self._flush(header, current_batch)

                if self.count_tokens(chunk) > 30:  # ✅ FIX 2 again
                    chunks.append(chunk)

                last_batch = current_batch.copy()

                # ✅ FIX 1: use last_batch instead of current_batch
                overlap = self.get_sentence_context(last_batch, 2)

                # ✅ FIX 4: avoid weak overlap
                if overlap and self.count_tokens(overlap) > 5:
                    current_batch = [overlap]
                    current_tokens = self.count_tokens(overlap)
                else:
                    current_batch = []
                    current_tokens = 0

            # Normal case
            current_batch.append(sent)
            current_tokens += sent_tokens

        # Final flush
        if current_batch:
            chunk = self._flush(header, current_batch)
            if self.count_tokens(chunk) > 30:
                chunks.append(chunk)
            last_batch = current_batch.copy()

        return chunks, self.get_sentence_context(last_batch, 2)


def chunk_text(text: str) -> list[str]:
    chunker = RAGChunker()
    
    raw_sections = re.split(r"(?=^SECTION:)", text, flags=re.MULTILINE)
    raw_sections = [s.strip() for s in raw_sections if s.strip()]

    all_chunks = []
    running_context = ""

    for section in raw_sections:
        parts = section.split("\n", 1)
        
        if parts[0].startswith("SECTION:"):
            header = parts[0].strip()
            content = parts[1].strip() if len(parts) > 1 else ""
        else:
            header = ""
            content = section.strip()

        if not content:
            continue

        section_chunks, running_context = chunker.process_section(
            header, content, prev_context=running_context
        )

        all_chunks.extend(section_chunks)

    return all_chunks

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
