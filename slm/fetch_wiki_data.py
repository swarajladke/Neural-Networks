"""
fetch_wiki_data.py — Wikipedia Corpus Fetcher for BPE Training
===============================================================
Downloads Wikipedia article text for multiple languages using the
Wikipedia API. Outputs cleaned text files for BPE tokenizer training.

Usage:
    python slm/fetch_wiki_data.py                    # Default: 6 languages, 5000 articles each
    python slm/fetch_wiki_data.py --articles 10000   # More articles per language
    python slm/fetch_wiki_data.py --langs en de fr   # Specific languages only
"""

import urllib.request
import json
import os
import re
import sys
import time
import argparse

# Wikipedia API endpoint for fetching random articles
WIKI_API = "https://{lang}.wikipedia.org/w/api.php"

def fetch_random_articles(lang: str, count: int = 500, min_length: int = 500) -> list[str]:
    """Fetch random Wikipedia articles using the API."""
    articles = []
    attempts = 0
    max_attempts = count * 3  # Allow for some failures

    print(f"  [{lang}] Fetching {count} articles (min {min_length} chars each)...")

    while len(articles) < count and attempts < max_attempts:
        attempts += 1
        try:
            # Get random article titles
            url = WIKI_API.format(lang=lang)
            params = {
                "action": "query",
                "format": "json",
                "generator": "random",
                "grnnamespace": "0",
                "grnlimit": "10",
                "prop": "extracts",
                "explaintext": "true",
                "exlimit": "10",
            }
            query_str = "&".join(f"{k}={v}" for k, v in params.items())
            full_url = f"{url}?{query_str}"

            req = urllib.request.Request(full_url, headers={"User-Agent": "AGNIS-Research/1.0"})
            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode("utf-8"))

            if "query" not in data or "pages" not in data["query"]:
                continue

            for page_id, page in data["query"]["pages"].items():
                if len(articles) >= count:
                    break
                text = page.get("extract", "")
                if len(text) >= min_length:
                    # Clean the text
                    text = clean_wiki_text(text)
                    if len(text) >= min_length:
                        articles.append(text)

            if len(articles) % 100 == 0 and len(articles) > 0:
                print(f"  [{lang}] {len(articles)}/{count} articles collected...")

            # Rate limiting
            time.sleep(0.1)

        except Exception as e:
            if attempts % 50 == 0:
                print(f"  [{lang}] Retry {attempts}: {e}")
            time.sleep(0.5)

    print(f"  [{lang}] Collected {len(articles)} articles")
    return articles


def clean_wiki_text(text: str) -> str:
    """Clean Wikipedia article text."""
    # Remove excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove section markers like == Heading ==
    text = re.sub(r'={2,}[^=]+={2,}', '', text)
    # Remove reference markers [1], [2] etc
    text = re.sub(r'\[\d+\]', '', text)
    # Remove excessive spaces
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def fetch_wiki_corpus(langs: list[str], articles_per_lang: int = 5000, output_dir: str = "slm"):
    """Fetch Wikipedia text for multiple languages."""
    os.makedirs(output_dir, exist_ok=True)

    lang_names = {
        "en": "English", "de": "German", "fr": "French",
        "es": "Spanish", "it": "Italian", "ru": "Russian",
        "ro": "Romanian", "mr": "Marathi", "pt": "Portuguese",
        "nl": "Dutch", "pl": "Polish", "ja": "Japanese",
    }

    all_text = ""
    for lang in langs:
        name = lang_names.get(lang, lang.upper())
        print(f"\n--- Fetching {name} Wikipedia ---")

        output_file = os.path.join(output_dir, f"wiki_{lang}.txt")

        # Check if already downloaded
        if os.path.exists(output_file) and os.path.getsize(output_file) > 100000:
            print(f"  [{lang}] Already exists ({os.path.getsize(output_file)//1024}KB), skipping.")
            with open(output_file, "r", encoding="utf-8") as f:
                all_text += f.read()
            continue

        articles = fetch_random_articles(lang, count=articles_per_lang)

        corpus = "\n\n".join(articles)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(corpus)

        chars = len(corpus)
        print(f"  [{lang}] Saved to {output_file} ({chars:,} chars, {chars//1024}KB)")
        all_text += corpus

    # Save combined corpus
    combined_path = os.path.join(output_dir, "wiki_combined.txt")
    with open(combined_path, "w", encoding="utf-8") as f:
        f.write(all_text)
    print(f"\nCombined corpus: {len(all_text):,} chars ({len(all_text)//1024//1024}MB)")
    print(f"Saved to: {combined_path}")

    return combined_path


def train_bpe_16k(corpus_path: str, vocab_size: int = 16000):
    """Train a 16K BPE tokenizer on the combined corpus."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from slm.slm_tokenizer import BPETokenizer

    print(f"\n{'='*50}")
    print(f"  Training BPE Tokenizer (vocab={vocab_size})")
    print(f"{'='*50}")

    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()

    # For very large corpora, use a sample for BPE training
    max_train_chars = 5_000_000  # 5MB sample for speed
    if len(text) > max_train_chars:
        print(f"  Corpus too large ({len(text):,} chars), sampling {max_train_chars:,} chars for BPE training...")
        import random
        # Take random chunks to get diverse coverage
        chunk_size = 10000
        n_chunks = max_train_chars // chunk_size
        chunks = []
        for _ in range(n_chunks):
            start = random.randint(0, len(text) - chunk_size)
            chunks.append(text[start:start + chunk_size])
        text = "\n".join(chunks)

    tok = BPETokenizer(vocab_size=vocab_size)
    tok.fit(text, verbose=True)

    output_path = "agnis_bpe_16k_vocab.json"
    tok.save(output_path)

    # Test compression
    test_sentences = [
        "The cat sat on the mat.",
        "Die Katze sitzt auf der Matte.",
        "El gato se sento en la alfombra.",
        "Le chat est assis sur le tapis.",
    ]
    print(f"\n  Compression Test:")
    for sent in test_sentences:
        ids = tok.encode(sent)
        print(f"    '{sent}' -> {len(ids)} tokens ({len(sent)/max(1,len(ids)):.1f}x)")

    return tok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="+", default=["en", "de", "es", "fr", "it", "ru"])
    parser.add_argument("--articles", type=int, default=2000)
    parser.add_argument("--vocab", type=int, default=16000)
    parser.add_argument("--train-bpe", action="store_true", default=True)
    args = parser.parse_args()

    corpus_path = fetch_wiki_corpus(args.langs, articles_per_lang=args.articles)

    if args.train_bpe:
        train_bpe_16k(corpus_path, vocab_size=args.vocab)
