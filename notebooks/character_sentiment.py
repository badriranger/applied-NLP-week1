import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re
from pathlib import Path

# Download sentiment model if not installed
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# -----------------------------------------------------
# SET YOUR FILE PATHS HERE ↓↓↓
# -----------------------------------------------------
Crime_file = r"C:\Users\dimas\OneDrive\Documents\GitHub\applied-NLP-week1\data\Crime-punishment.txt"
Karamazov_file = r"C:\Users\dimas\OneDrive\Documents\GitHub\applied-NLP-week1\data\The-BrothersKaramazov.txt"


def load_texts(local_alice: str, local_glass: str):
    p1, p2 = Path(local_alice), Path(local_glass)

    if not p1.exists():
        raise FileNotFoundError(f"❌ File not found: {p1}")
    if not p2.exists():
        raise FileNotFoundError(f"❌ File not found: {p2}")

    text1 = p1.read_text(encoding='utf-8', errors='ignore')
    text2 = p2.read_text(encoding='utf-8', errors='ignore')
    return text1, text2


def normalize(text: str) -> str:
    return text.replace('\r\n', '\n')


def split_sentences(text: str):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def filter_character_sentences(sentences, keywords):
    return [s for s in sentences if any(k.lower() in s.lower() for k in keywords)]


def compute_sentiment(sentences):
    return [sia.polarity_scores(s)['compound'] for s in sentences]


def aggregate_chapter_sentiment(sentences, chapter_marker="CHAPTER"):
    chapter_scores = []
    current = []

    for s in sentences:
        if chapter_marker.lower() in s.lower():
            if current:
                chapter_scores.append(sum(compute_sentiment(current)) / len(current))
            current = []
        current.append(s)

    if current:
        chapter_scores.append(sum(compute_sentiment(current)) / len(current))

    return chapter_scores


def character_sentiment_analysis(sentences, keywords):
    filtered = filter_character_sentences(sentences, keywords)
    return aggregate_chapter_sentiment(filtered)


# -----------------------------------------------------
# RUN LOADING
# -----------------------------------------------------
CrimePunishment_raw, TheBrothers_raw = load_texts(Crime_file, Karamazov_file)

CrimePunishment_sentences = split_sentences(normalize(CrimePunishment_raw))
TheBrothers_sentences = split_sentences(normalize(TheBrothers_raw))

print(f"✅ Loaded Crime & Punishment sentences: {len(CrimePunishment_sentences):,}")
print(f"✅ Loaded Brothers Karamazov sentences: {len(TheBrothers_sentences):,}")

# -----------------------------------------------------
# CHARACTER SENTIMENT
# -----------------------------------------------------
raskolnikov_keywords = ['Raskolnikov', 'he']
dmitri_keywords = ['Dmitri', 'he']

cp_sentiments = character_sentiment_analysis(CrimePunishment_sentences, raskolnikov_keywords)
bk_sentiments = character_sentiment_analysis(TheBrothers_sentences, dmitri_keywords)

# -----------------------------------------------------
# PLOT RESULTS
# -----------------------------------------------------
plt.figure(figsize=(12,6))
plt.plot(cp_sentiments, label="Raskolnikov (Crime & Punishment)", marker='o')
plt.plot(bk_sentiments, label="Dmitri (Brothers Karamazov)", marker='x')
plt.xlabel("Chapter")
plt.ylabel("Average Sentiment (VADER Compound Score)")
plt.title("Character Emotional Arc")
plt.legend()
plt.grid(True)
plt.show()
