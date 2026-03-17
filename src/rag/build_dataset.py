import json
import src.config as config


EVAL_QUESTIONS_PATH=config.EVAL_QUESTIONS_PATH
RETRIEVAL_RESULTS_PATH=config.RETRIEVAL_RESULTS_PATH

with open(RETRIEVAL_RESULTS_PATH,'r') as f:
    retrieval_results=json.load(f)

with open(EVAL_QUESTIONS_PATH,'r') as f:
    eval_questions=json.load(f)

print(retrieval_results.keys())
print(eval_questions[0])

contexts={}
question=[]
ground_truth=[]
for i,q in enumerate(eval_questions):
    question.append(q['question'])
    ground_truth.append(q['answer'])
    contexts['dense']=retrieval_results['dense'][q[i]][:5]
    contexts['sparse']=retrieval_results['sparse'][q[i]][:5]
    contexts['hybrid']=retrieval_results['hybrid'][q[i]][:5]
    