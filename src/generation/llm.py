
import requests
# import src.config as config
def get_llm():
    import src.config as config
    
    if config.LLM_BACKEND == "hf_api":
        return HuggingFaceAPI()
    elif config.LLM_BACKEND == "ollama":
        return OllamaModel()
    elif config.LLM_BACKEND == 'hf_model':
        return HuggingFaceModel()
    else:
        raise ValueError("Invalid LLM Backend")
    
class HuggingFaceModel:
    def __init__(self, config=None):
        if config is None:
            import src.config as config
        self.config = config
        
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print('[Loading Tokenizer]')
        self.tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL)
        
        print('[Loading LLM Model]')
        self.model = AutoModelForCausalLM.from_pretrained(config.LLM_MODEL)
        self.model.to(self.config.DEVICE)
        
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
        

class OllamaModel:
    
    def __init__(self, config=None):
        if config is None:
            import src.config as config
        self.config = config      
        self.model = config.OLLAMA_MODEL
        self.addr = config.OLLAMA_ADDRESS
    
    def generate(self, prompt, max_new_tokens=200):
        
        response = requests.post(
            self.addr + '/api/generate',
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_new_tokens,
                    "temperature": 0.3
                }
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        if "response" not in data:
            raise ValueError(f"Ollama error: {data}")

        return data["response"]     
        
import requests

class HuggingFaceAPI:
    
    def __init__(self, config=None):
        if config is None:
            import src.config as config

        if config.HF_API_KEY is None:
            raise ValueError("HF_API_KEY not set in environment")
        self.api_key = config.HF_API_KEY
        self.model = config.HF_API_MODEL
        self.url = f"https://api-inference.huggingface.co/models/{self.model}"
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}"
        }

    def generate(self, prompt, max_new_tokens=200):
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0.3
            }
        }

        response = requests.post(self.url, headers=self.headers, json=payload)
        response.raise_for_status()
        data = response.json()

        generated = data[0]["generated_text"]

        if generated.startswith(prompt):
            generated = generated[len(prompt):]

        return generated.strip()      

if __name__ == "__main__":
    # llm=hugginfaceLlm(config)
    ollama=OllamaModel()
    prompt="What is AutomodelForCasualLM?"
    
    ans=ollama.generate(prompt)
    print(ans)

    