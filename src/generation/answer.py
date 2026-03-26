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
          You are a strict QA system.

          Answer ONLY using the provided context.

          Rules:
        - Use ONLY the provided context
        - If the answer is not explicitly present, say: "Not found in context"
        - Do NOT infer or assume missing information
        - Do NOT add any external knowledge
          Context:
          {context_text}

          Question:
          {question}

          Answer in bullet points.
          [/INST]"""
    def generate(self, question,contexts, max_new_tokens=200):
        prompt=self.build_prompt(question,contexts)
        
        

        full_output = self.llm.generate(prompt)

        answer = full_output[len(prompt):].strip()

        return answer