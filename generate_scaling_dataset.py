"""
generate_scaling_dataset.py — Generates 100 Structured Facts for Scaling Experiments
=====================================================================================
Generates a JSON file containing 100 facts partitioned into 10 blocks (10 facts each).
ALL TEMPLATES ARE PREFIX-ALIGNED and use 100% unique entities to prevent query conflicts.
"""
import json
import random

def build_fact_dataset():
    # Helper to generate unique entities
    def gen_unique(count, prefixes, suffixes):
        out = []
        for p in prefixes:
            for s in suffixes:
                out.append(f"{p}{s}")
                if len(out) == count:
                    return out
        return out

    # Generate 100 unique entities for each category
    loc_prefixes = ["Luma", "Aura", "Kael", "Vesp", "Sola", "Zeph", "Nebu", "Bore", "Aeth", "Chro"]
    loc_suffixes = ["ria", "ntia", "len", "per", "ris", "phyra", "lia", "as", "hel", "nos"]
    LOCATIONS = gen_unique(100, loc_prefixes, loc_suffixes)

    cap_prefixes = ["Varek", "Velath", "Xenon", "Selen", "Pyros", "Kryos", "Nova", "Oros", "Zirco", "Helio"]
    cap_suffixes = [" City", " Port", " Vale", " Spire", " Peak", " Cove", " Ridge", " Town", " Bay", " Dome"]
    CAPITALS = gen_unique(100, cap_prefixes, cap_suffixes)

    comp_prefixes = ["Xenol", "Therm", "Auran", "Helio", "Zirco", "Neptu", "Krypt", "Solit", "Pyrot", "Selen"]
    comp_suffixes = ["-A", "-B", "-C", "-D", "-E", "-F", "-G", "-H", "-X", "-Z"]
    COMPOUNDS = gen_unique(100, comp_prefixes, comp_suffixes)

    planet_prefixes = ["Kepler", "Gliese", "Luyten", "Proxima", "Trappist", "Wasp", "Osiris", "K2", "TOI", "HD"]
    planet_suffixes = ["-101b", "-202c", "-303d", "-404e", "-505f", "-606g", "-707h", "-808i", "-909j", "-1000k"]
    PLANETS = gen_unique(100, planet_prefixes, planet_suffixes)

    moon_prefixes = ["Aria", "Bello", "Ceres", "Deim", "Phob", "Tita", "Euro", "Calli", "Gany", "Io"]
    moon_suffixes = ["-Alpha", "-Beta", "-Gamma", "-Delta", "-Epsilon", "-Zeta", "-Eta", "-Theta", "-Iota", "-Kappa"]
    MOONS = gen_unique(100, moon_prefixes, moon_suffixes)

    # Multi-token numeric answers for Science & Astronomy
    NUMBERS_STR = [
        "forty two", "eighty five", "one hundred", "two hundred", 
        "three hundred", "five hundred", "eight hundred", "nine hundred",
        "seventy six", "sixty four", "fifty eight", "ninety one"
    ]
    
    facts = []
    
    # 1. GEOGRAPHY (34 facts: G01 to G34)
    for i in range(34):
        loc = LOCATIONS[i]
        cap = CAPITALS[i]
        fid = f"G{i+1:02d}"
        
        facts.append({
            "id": fid,
            "category": "geography",
            "location": loc,
            "capital": cap,
            "statement": f"The official capital city of {loc} is {cap}.",
            "qa": f"Q: What is the capital of {loc}? A: The official capital city of {loc} is {cap}.",
            "cloze": f"The official capital city of {loc} is _____.",
            "probe": f"The official capital city of {loc} is",
            "answer": cap,
            "keywords": [cap.split()[0]],
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

    # 2. SCIENCE - Melting Points (33 facts: S01 to S33)
    for i in range(33):
        comp = COMPOUNDS[i]
        num = NUMBERS_STR[i % len(NUMBERS_STR)]
        fid = f"S{i+1:02d}"
        
        facts.append({
            "id": fid,
            "category": "science",
            "compound": comp,
            "temperature": num,
            "statement": f"The molecular compound {comp} liquefies at exactly {num} degrees Celsius.",
            "qa": f"Q: At what temperature does {comp} melt? A: The molecular compound {comp} liquefies at exactly {num} degrees Celsius.",
            "cloze": f"The molecular compound {comp} liquefies at exactly _____ degrees Celsius.",
            "probe": f"The molecular compound {comp} liquefies at exactly",
            "answer": num,
            "keywords": [num.split()[0]],
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

    # 3. ASTRONOMY - Orbital Periods (33 facts: A01 to A33)
    PERIODS = ["forty seven", "eighty eight", "twelve days", "nineteen days", "thirty six", "six days", "fifteen days"]
    for i in range(33):
        planet = PLANETS[i]
        moon = MOONS[i]
        period = PERIODS[i % len(PERIODS)]
        fid = f"A{i+1:02d}"
        
        facts.append({
            "id": fid,
            "category": "astronomy",
            "planet": planet,
            "moon": moon,
            "period": period,
            "statement": f"The planetary satellite {moon} orbits {planet} in exactly {period} days.",
            "qa": f"Q: How long does it take for {moon} to orbit {planet}? A: The planetary satellite {moon} orbits {planet} in exactly {period} days.",
            "cloze": f"The planetary satellite {moon} orbits {planet} in exactly _____ days.",
            "probe": f"The planetary satellite {moon} orbits {planet} in exactly",
            "answer": period,
            "keywords": [period.split()[0]],
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
