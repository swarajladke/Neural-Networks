import json
from collections import Counter

def main():
    dataset_path = "agnis_scaling_dataset_v3_template_split.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    facts = data["facts"]
    assert len(facts) == 100, f"Expected 100 facts, got {len(facts)}"
    
    print("=========================================================================================================")
    print(" DIRECTIVE N5 -- RIGOROUS PER-BLOCK LATIN SQUARE AUDIT (CAPABLE OF FAILING)")
    print("=========================================================================================================")

    all_passed = True
    for block_idx in range(10):
        block_facts = facts[block_idx * 10 : (block_idx + 1) * 10]
        entities = [f["entity_index"] for f in block_facts]
        relations = [f["relation_index"] for f in block_facts]
        
        ent_counts = Counter(entities)
        rel_counts = Counter(relations)
        
        max_ent_count = max(ent_counts.values())
        max_rel_count = max(rel_counts.values())
        
        is_ent_perm = (sorted(entities) == list(range(10)))
        is_rel_perm = (sorted(relations) == list(range(10)))
        
        passed = (is_ent_perm and is_rel_perm and max_ent_count == 1 and max_rel_count == 1)
        if not passed:
            all_passed = False
            
        print(f"  Block {block_idx:02d} (Facts {block_idx*10:02d}..{block_idx*10+9:02d}): "
              f"Max Entity Count = {max_ent_count}, Max Relation Count = {max_rel_count} "
              f"-> Permutation 0..9? {'PASSED' if passed else 'FAILED'}")
              
    print("\n--- LATIN SQUARE COMPLIANCE RESULT ---")
    print(f"  Strict Latin Square Assertion Across All 10 Blocks: {'PASSED' if all_passed else 'FAILED'}")
    print("  (Note: Simple global unique-value count has been removed from compliance list per Directive N5)")

if __name__ == "__main__":
    main()
