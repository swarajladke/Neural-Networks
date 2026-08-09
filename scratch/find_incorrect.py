import json

with open('nlg_evaluation_records.json') as f:
    data = json.load(f)
    
records = data['records']
c3 = [x for x in records if x['condition'] == 'Verifier-gated + Grounding Validator (Recommended)' and not x['is_ood'] and x['decision_state'] == 'ANSWER_ACCEPTED']

incorrect = [x for x in c3 if not x['factually_correct']]

for idx, x in enumerate(incorrect):
    print(f"[{idx+1}] Factual incorrect:")
    print(f"    Fact:   '{x['retrieved_fact_id']}' - '{x['expected_fact_id']}'")
    print(f"    Query:  '{x['query']}'")
    print(f"    Raw:    '{x['raw_generation']}'")
    print(f"    Final:  '{x['final_answer']}'")
    print(f"    Reasons: {x['validation_reasons']}")
    print("-" * 50)
