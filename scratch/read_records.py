import json

with open('nlg_evaluation_records.json') as f:
    data = json.load(f)
    
records = data['records']

# Filter condition 3, in-domain records
cond3_id = [r for r in records if r['condition'] == 'Verifier-gated + Grounding Validator (Recommended)' and not r['is_ood']]

for idx, r in enumerate(cond3_id):
    print(f"{idx+1}. Query: '{r['query']}'")
    print(f"   Raw:   '{r['raw_generation']}'")
    print(f"   State: {r['decision_state']}")
    print(f"   Reasons: {r['validation_reasons']}")
    print("-" * 50)
