from src.generation.llm import HuggingFaceModel 
from src.generation.llm import OllamaModel

import src.config as config

class AnswerGenerator:

    def __init__(self, config=config):
        if config.PLATFORM == "huggingface":
            self.llm = HuggingFaceModel(config)
        elif config.PLATFORM == "ollama":
            self.llm = OllamaModel(config)
        else:
            raise ValueError("Invalid PLATFORM")

    def build_prompt(self, question, contexts, max_contexts=5):
      context_text = "\n\n".join(
                      [f"[Context {i+1}]\n{c}" for i, c in enumerate(contexts[:max_contexts])]
                  )
      return f"""<s>[INST]
You are a precise QA system. Answer using ONLY the provided context.

Rules:
- If the answer is present, give it concisely
- If not present, say: "Not found in context"
- No external knowledge, no inference beyond what is stated

Context:
{context_text}

Question:
{question}

Answer:[/INST]"""
    def generate(self, question,contexts, max_new_tokens=512):
        prompt=self.build_prompt(question,contexts,max_new_tokens)
        full_output = self.llm.generate(prompt)
        if "[/INST]" in full_output:
            answer = full_output.split("[/INST]")[-1].strip()
        else:
            answer = full_output[len(prompt):].strip()  # fallback


        return answer