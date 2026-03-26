
import requests
# import src.config as config

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
            do_sample=False,           
            temperature=0.3,              
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
    
    def generate(self, prompt):
        
        response = requests.post(
            self.addr + '/api/generate',
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 200,
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
        
        

if __name__ == "__main__":
    # llm=hugginfaceLlm(config)
    ollama=OllamaModel()
    prompt="What is AutomodelForCasualLM?"
    
    ans=ollama.generate(prompt)
    print(ans)

    