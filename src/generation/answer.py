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
        context_text = "\n\n".join(contexts[:max_contexts])

        return f"""
        You are a helpful assistant. Answer the question using ONLY the provided context.

        Context:
        {context_text}

        Question:
        {question}

        Answer:
        """

    def generate(self, question, contexts):
        prompt = self.build_prompt(question, contexts)
        raw_output = self.llm.generate(prompt)

        # Extract only answer part
        if "Answer:" in raw_output:
            answer = raw_output.split("Answer:")[-1].strip()
        else:
            # fallback (just return last part)
            answer = raw_output.strip()
            answer = answer.replace("Answer:", "").strip()
        return answer