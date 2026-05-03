import os
import urllib.request
import re

def fetch_marathon():
    languages = {
        "en": ("English", "https://www.gutenberg.org/cache/epub/1400/pg1400.txt"),
        "de": ("German", "https://www.gutenberg.org/cache/epub/1910/pg1910.txt"),
        "ro": ("Romanian", "https://www.gutenberg.org/cache/epub/63636/pg63636.txt"),
        "es": ("Spanish", "https://www.gutenberg.org/cache/epub/2000/pg2000.txt"),
        "fr": ("French", "https://www.gutenberg.org/cache/epub/135/pg135.txt"),
        "it": ("Italian", "https://www.gutenberg.org/cache/epub/3601/pg3601.txt"),
        "ru": ("Russian", "https://www.gutenberg.org/cache/epub/2600/pg2600.txt"),
    }
    
    os.makedirs("slm", exist_ok=True)
    
    for code, (name, url) in languages.items():
        filepath = f"slm/input_{code}.txt"
        print(f"--- Fetching {name} ({url}) ---")
        
        try:
            temp_path = f"slm/temp_{code}.txt"
            urllib.request.urlretrieve(url, temp_path)
            
            with open(temp_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Basic Gutenberg Stripping
            # Find the start of the actual book content
            start_markers = ["*** START OF THE PROJECT GUTENBERG", "PROLOGUE", "PREFACE", "अध्याय"]
            start_pos = 0
            for m in start_markers:
                pos = content.find(m)
                if pos != -1:
                    start_pos = pos
                    break
            
            content = content[start_pos:]
            
            # Find the end
            end_markers = ["*** END OF THE PROJECT GUTENBERG", "FINIS", "THE END"]
            for m in end_markers:
                pos = content.find(m)
                if pos != -1:
                    content = content[:pos]
                    break
            
            # Clean whitespace
            content = re.sub(r'\r', '', content)
            content = re.sub(r'\n{3,}', '\n\n', content)
            
            # Truncate to a manageable size for the sprint (500KB)
            content = content[:500000]
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            
            print(f"Saved {name} to {filepath} ({len(content)} chars)")
            os.remove(temp_path)
            
        except Exception as e:
            print(f"Failed to fetch {name}: {e}")

if __name__ == "__main__":
    fetch_marathon()
