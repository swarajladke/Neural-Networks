import json

with open('nlg_evaluation_records.json') as f:
    data = json.load(f)
    
records = data['records']
c3 = [x for x in records if x['condition'] == 'Verifier-gated + Grounding Validator (Recommended)' and not x['is_ood'] and x['decision_state'] == 'ANSWER_ACCEPTED']

for idx, x in enumerate(c3):
    print(f"[{idx+1}] Correct Retrieval: {x['retrieval_correct']} | Verifier Score: {x['verifier_score']:.6f} | Query: {x['query'][:60]}")
