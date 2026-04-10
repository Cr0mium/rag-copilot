from src.generation.llm import get_llm

import src.config as config

class AnswerGenerator:

    def __init__(self, config=config):
        self.llm=get_llm()

    def build_prompt(self, question, contexts, max_contexts=5):
      context_text = "\n\n".join(
                      [f"[Context {i+1}]\n{c['filename']}|{ c['text']}" for i, c in enumerate(contexts[:max_contexts])]
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
        # print(prompt)
        answer = self.llm.generate(prompt)
        # print(answer)
        if "[/INST]" in answer:
            answer = answer.split("[/INST]")[-1].strip()

        return answer