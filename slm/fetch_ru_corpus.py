import os
import urllib.request
import re

def fetch_and_clean_russian():
    filepath = "slm/input_ru.txt"
    # Gutenberg ID 30663 is Crime and Punishment in Russian
    url = "https://www.gutenberg.org/cache/epub/30663/pg30663.txt"
    
    print(f"Downloading Crime and Punishment (Russian) from {url}...")
    temp_path = "slm/temp_ru.txt"
    os.makedirs("slm", exist_ok=True)
    
    try:
        urllib.request.urlretrieve(url, temp_path)
    except Exception as e:
        print(f"Download failed: {e}")
        return

    print("Cleaning corpus (stripping Gutenberg headers/footers)...")
    with open(temp_path, "r", encoding="utf-8") as f:
        content = f.read()

    # User's specific stripping instructions:
    # 1. Start at "ЧАСТЬ ПЕРВАЯ"
    start_match = re.search(r"ЧАСТЬ ПЕРВАЯ", content)
    if start_match:
        content = content[start_match.start():]
    
    # 2. End at "КОНЕЦ"
    end_match = re.search(r"КОНЕЦ", content)
    if end_match:
        content = content[:end_match.end()]

    # Clean redundant whitespace
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
        
    print(f"Saved cleaned corpus to {filepath} ({len(content)} chars)")
    os.remove(temp_path)

if __name__ == "__main__":
    fetch_and_clean_russian()
