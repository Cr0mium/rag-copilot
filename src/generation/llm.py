import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import src.config as config

from transformers import AutoTokenizer, AutoModelForCausalLM

class LLM:
    def __init__(self,config):
        print('[Loading Tokenizer]')
        self.tokenizer = AutoTokenizer.from_pretrained(config.AUTOTOKENIZER)
        
        print('[Loading LLM Model]')
        self.model = AutoModelForCausalLM.from_pretrained(config.LLM_MODEL)

    def generate(self, prompt, max_new_tokens=200):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        
        
        

# ----------------- CLI loop -----------------
if __name__ == "__main__":
    
    llm=LLM(config)
    
    prompt="What is AutomodelForCasualLM?"
    
    ans=llm.generate(prompt,120)

    