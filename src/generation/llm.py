import os
import requests

# -------------------------
# Factory
# -------------------------
def get_llm():
    import src.config as config

    backend = config.LLM_BACKEND.lower()

    if backend == "hf_api":
        return HuggingFaceAPI(config)
    elif backend == "ollama":
        return OllamaModel(config)
    elif backend == "hf_model":
        return HuggingFaceModel(config)
    else:
        raise ValueError(f"Invalid LLM Backend: {backend}")


# -------------------------
# HuggingFace Local Model
# -------------------------
class HuggingFaceModel:
    def __init__(self, config):
        self.config = config

        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.token = os.getenv("HF_API_KEY")

        if not self.token:
            raise ValueError("HF_API_KEY not set")

        print("[HF_MODEL] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.LLM_MODEL,
            token=self.token
        )

        print("[HF_MODEL] Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            config.LLM_MODEL,
            token=self.token
        )

        self.model.to(self.config.DEVICE)
        print("[HF_MODEL] Model loaded successfully")

    def generate(self, prompt, max_new_tokens=200):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.DEVICE)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.2,
            pad_token_id=self.tokenizer.eos_token_id
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# -------------------------
# Ollama Backend
# -------------------------
class OllamaModel:
    def __init__(self, config):
        self.config = config
        self.model = config.OLLAMA_MODEL
        self.addr = config.OLLAMA_ADDRESS

        print(f"[OLLAMA] Using model: {self.model} @ {self.addr}")

    def generate(self, prompt, max_new_tokens=200):
        try:
            response = requests.post(
                f"{self.addr}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_new_tokens,
                        "temperature": 0.3
                    }
                },
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()

            if "response" not in data:
                raise ValueError(f"Ollama error: {data}")

            return data["response"]

        except Exception as e:
            raise RuntimeError(f"[OLLAMA ERROR] {e}")


# -------------------------
# HuggingFace Inference API
# -------------------------
class HuggingFaceAPI:
    def __init__(self, config):
        self.config = config

        self.api_key = os.getenv("HF_API_KEY")
        if not self.api_key:
            raise ValueError("HF_API_KEY not set")

        self.model = config.HF_API_MODEL
        self.url = f"https://api-inference.huggingface.co/models/{self.model}"

        self.headers = {
            "Authorization": f"Bearer {self.api_key}"
        }

        print(f"[HF_API] Using model: {self.model}")

    def generate(self, prompt, max_new_tokens=200):
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0.3
            }
        }

        try:
            response = requests.post(
                self.url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            data = response.json()

            generated = data[0]["generated_text"]

            if generated.startswith(prompt):
                generated = generated[len(prompt):]

            return generated.strip()

        except Exception as e:
            raise RuntimeError(f"[HF_API ERROR] {e}")


# -------------------------
# Local Test
# -------------------------
if __name__ == "__main__":
    import src.config as config

    llm = get_llm()

    prompt = "What is AutoModelForCausalLM?"
    print("\n[TEST] Prompt:", prompt)

    answer = llm.generate(prompt)
    print("\n[TEST] Answer:", answer)