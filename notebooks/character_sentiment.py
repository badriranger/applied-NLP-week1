from collections import Counter
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re
from pathlib import Path

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# ------------------ LOAD & PREPROCESS ------------------
def load_text(file_path: str):
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"❌ File not found: {p}")
    return p.read_text(encoding='utf-8', errors='ignore')

def normalize(text: str) -> str:
    return text.replace('\r\n', '\n')

def split_sentences(text: str):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

# ------------------ SENTIMENT ------------------
def compute_sentiment(sentences):
    return [sia.polarity_scores(s)['compound'] for s in sentences]

def aggregate_chapter_sentiment(sentences, chapter_marker="CHAPTER"):
    chapter_scores, current = [], []
    for s in sentences:
        if chapter_marker.lower() in s.lower():
            if current:
                chapter_scores.append(sum(compute_sentiment(current)) / len(current))
            current = []
        current.append(s)
    if current:
        chapter_scores.append(sum(compute_sentiment(current)) / len(current))
    return chapter_scores

# ------------------ PRONOUN COUNTS ------------------
def pronoun_counts_in_sentences(sentences, target={'he','she','him','her'}):
    counts_per_chapter, current_chapter = [], []
    for s in sentences:
        if "chapter" in s.lower():
            if current_chapter:
                c = sum(Counter(w.lower() for sent in current_chapter for w in sent.split())[t] for t in target)
                counts_per_chapter.append(c)
            current_chapter = []
        current_chapter.append(s)
    if current_chapter:
        c = sum(Counter(w.lower() for sent in current_chapter for w in sent.split())[t] for t in target)
        counts_per_chapter.append(c)
    return counts_per_chapter

# ------------------ FILES ------------------
Crime_file = r"C:\Users\dimas\OneDrive\Documents\GitHub\applied-NLP-week1\data\Crime-punishment.txt"
Karamazov_file = r"C:\Users\dimas\OneDrive\Documents\GitHub\applied-NLP-week1\data\The-BrothersKaramazov.txt"

crime_text = normalize(load_text(Crime_file))
karamazov_text = normalize(load_text(Karamazov_file))

crime_sentences = split_sentences(crime_text)
karamazov_sentences = split_sentences(karamazov_text)

# ------------------ ANALYSIS ------------------
# Sentiment per chapter
cp_sentiments = aggregate_chapter_sentiment(crime_sentences)
bk_sentiments = aggregate_chapter_sentiment(karamazov_sentences)

# Pronouns per chapter
cp_pronouns = pronoun_counts_in_sentences(crime_sentences)
bk_pronouns = pronoun_counts_in_sentences(karamazov_sentences)

# ------------------ PLOT ------------------
fig, ax1 = plt.subplots(figsize=(14,6))

ax1.plot(cp_sentiments, label="Raskolnikov Sentiment", color='blue', marker='o')
ax1.plot(bk_sentiments, label="Dmitri Sentiment", color='red', marker='x')
ax1.set_xlabel("Chapter")
ax1.set_ylabel("Average Sentiment (VADER)")
ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(cp_pronouns, label="Raskolnikov Pronouns", color='cyan', linestyle='--')
ax2.plot(bk_pronouns, label="Dmitri Pronouns", color='orange', linestyle='--')
ax2.set_ylabel("Pronoun Count")

# Combine legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper left')

plt.title("Narrative Arc: Sentiment & Pronoun Flow")
plt.show()

'''Explanation of Sentiment Analysis in Your Code

Sentiment Analysis Definition
Sentiment analysis is a method in Natural Language Processing (NLP) to detect the emotional tone of a piece of text: positive, negative, or neutral.

Tool Used
You are using VADER (Valence Aware Dictionary for Sentiment Reasoning), which is especially good for literary text or social media. It gives a compound score for each sentence:

-1 → extremely negative

0 → neutral

+1 → extremely positive

Sentence-Level Scoring
In your code, each sentence of the book is analyzed individually:

sia.polarity_scores(sentence)['compound']


This gives a number for the emotional content of that sentence.

Chapter-Level Aggregation
Sentences are grouped by chapter:

chapter_scores.append(sum(compute_sentiment(current)) / len(current))


The average compound score of all sentences in a chapter represents the overall emotional tone of that chapter.

High positive → happy, relieved, or uplifting chapters.

High negative → tense, threatening, or climactic chapters.

Narrative Arc
By plotting these chapter-level sentiment scores, you can see the emotional ups and downs of the story, which often corresponds to:

Rising tension → negative dips

Climax → extreme negative or positive

Resolution → neutral or positive recovery

Enhancement with Pronouns
Tracking pronouns alongside sentiment shows character activity, giving context to when a character is emotionally involved in the story.'''