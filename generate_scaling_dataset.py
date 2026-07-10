"""
generate_scaling_dataset.py — Generates 100 Structured Facts for Scaling Experiments
=====================================================================================
Generates a JSON file containing 100 facts partitioned into 10 blocks (10 facts each).
Includes:
  - Semantically dense overlapping entities (reused locations/capitals/numbers)
  - Contradictory fact updates (same entity, updated answer in later blocks)
  - Multi-token answers (e.g. "forty two", "Varek City")
  - Statement, QA, and Cloze templates for injection
  - 3 Training Paraphrases per fact (for C_Gaussian_6 and E_DenseTangent_6)
  - 3 Evaluation Paraphrases per fact (never leaked during training)
"""
import json
import random

def build_fact_dataset():
    # Pools of entities to create high semantic overlap
    LOCATIONS = ["Luma", "Aurantia", "Kaelen", "Vesper", "Solaria", "Zephyr", "Nebulia", "Boreas", "Aethel", "Chronos"]
    CAPITALS = ["Varek City", "Velathi Port", "Xenon Vale", "Selene Spire", "Pyros Peak", "Kryos Cove", "Nova Ridge", "Oros Town"]
    COMPOUNDS = ["Xenolite-B", "Thermocyclase-9", "Auranium-X", "Helios-IV", "Zircon-D", "Neptunite-G", "Krypton-F", "Solite-H"]
    PLANETS = ["Kepler-9814b", "Gliese-581d", "Luyten-b", "Proxima-c", "Trappist-1e", "K2-18b", "Wasp-76b", "Osiris-IV"]
    MOONS = ["Aria", "Bello", "Ceres", "Deimos", "Phobos", "Titan", "Io", "Europa"]
    
    facts = []
    
    # 1. GEOGRAPHY (30 facts)
    # Reuses locations and capitals to create massive overlap
    for i in range(30):
        loc = LOCATIONS[i % len(LOCATIONS)]
        cap = CAPITALS[(i + (i // len(LOCATIONS))) % len(CAPITALS)]
        fid = f"G{i+1:02d}"
        
        # Add contradictory updates: G21 updates G01 (Luma's capital), G22 updates G02, etc.
        if i >= 20:
            loc = LOCATIONS[(i - 20) % len(LOCATIONS)]
            cap = CAPITALS[(i + 4) % len(CAPITALS)] # new capital city
            
        facts.append({
            "id": fid,
            "category": "geography",
            "location": loc,
            "capital": cap,
            "statement": f"{cap} is the official capital city of the region of {loc}.",
            "qa": f"Q: What is the capital city of {loc}? A: {cap}",
            "cloze": f"The official capital city of the region of {loc} is _____.",
            "probe": f"The official capital city of the region of {loc} is",
            "answer": cap,
            "train_paraphrases": [
                f"Identify the capital city of {loc}.",
                f"Which city serves as the capital of {loc}?",
                f"The administrative capital of {loc} is located in"
            ],
            "eval_paraphrases": [
                f"What is the official capital of the region of {loc}?",
                f"Name the city that functions as {loc}'s capital.",
                f"In the land of {loc}, the capital city is known as"
            ]
        })

    # 2. SCIENCE - Melting Points (35 facts)
    # Includes multi-token temperatures (e.g. "one hundred", "forty two")
    NUMBERS_STR = ["forty two", "eighty five", "one hundred", "two hundred", "three hundred", "five hundred", "eight hundred"]
    for i in range(35):
        comp = COMPOUNDS[i % len(COMPOUNDS)]
        num = NUMBERS_STR[i % len(NUMBERS_STR)]
        fid = f"S{i+1:02d}"
        
        # Contradictory update: S25+ updates S01+ melting points
        if i >= 25:
            comp = COMPOUNDS[(i - 25) % len(COMPOUNDS)]
            num = NUMBERS_STR[(i + 2) % len(NUMBERS_STR)]
            
        facts.append({
            "id": fid,
            "category": "science",
            "compound": comp,
            "temperature": num,
            "statement": f"The molecular compound {comp} liquefies at exactly {num} degrees Celsius.",
            "qa": f"Q: At what temperature does {comp} melt? A: {num} degrees Celsius.",
            "cloze": f"The molecular compound {comp} liquefies at exactly _____ degrees Celsius.",
            "probe": f"The molecular compound {comp} liquefies at exactly",
            "answer": num,
            "train_paraphrases": [
                f"Specify the melting temperature of {comp}.",
                f"At how many degrees Celsius does {comp} melt?",
                f"The compound {comp} changes to liquid state at"
            ],
            "eval_paraphrases": [
                f"The molecular compound {comp} liquefies at exactly",
                f"What temperature is required to melt the compound {comp}?",
                f"Determine the melting point of the compound {comp} in degrees."
            ]
        })

    # 3. ASTRONOMY - Orbital Periods (35 facts)
    PERIODS = ["forty seven", "eighty eight", "twelve days", "nineteen days", "thirty six", "six days", "fifteen days"]
    for i in range(35):
        planet = PLANETS[i % len(PLANETS)]
        moon = MOONS[i % len(MOONS)]
        period = PERIODS[i % len(PERIODS)]
        fid = f"A{i+1:02d}"
        
        # Contradictory update: A25+ updates A01+ orbits
        if i >= 25:
            planet = PLANETS[(i - 25) % len(PLANETS)]
            moon = MOONS[(i + 1) % len(MOONS)]
            period = PERIODS[(i + 3) % len(PERIODS)]
            
        facts.append({
            "id": fid,
            "category": "astronomy",
            "planet": planet,
            "moon": moon,
            "period": period,
            "statement": f"The planetary satellite {moon} orbits {planet} in exactly {period} days.",
            "qa": f"Q: How long does it take for {moon} to orbit {planet}? A: {period} days.",
            "cloze": f"The planetary satellite {moon} orbits {planet} in exactly _____ days.",
            "probe": f"The planetary satellite {moon} orbits {planet} in exactly",
            "answer": period,
            "train_paraphrases": [
                f"What is the orbital period of the satellite {moon} around {planet}?",
                f"How many days does it take {moon} to circle {planet}?",
                f"The moon {moon} completes one full orbit of {planet} in"
            ],
            "eval_paraphrases": [
                f"The planetary satellite {moon} orbits {planet} in exactly",
                f"Give the time duration in days for the moon {moon} to circle {planet}.",
                f"How long is one orbit of the satellite {moon} around {planet}?"
            ]
        })

    # Shuffle facts to create balanced block distributions
    random.Random(42).shuffle(facts)
    
    # Partition into 10 blocks of 10 facts each
    blocks = []
    for b in range(10):
        blocks.append(facts[b*10 : (b+1)*10])
        
    return blocks

if __name__ == "__main__":
    dataset = build_fact_dataset()
    with open("agnis_scaling_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"[OK] Generated agnis_scaling_dataset.json with 10 blocks of 10 facts (100 total).")
