import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# TEXT PROCESSING
# --------------------------------------------------

def split_sentences(text):
    """
    Split document into individual sentences.
    """

    text = re.sub(r"\s+", " ", text).strip()

    sentences = re.split(r'(?<=[.!?])\s+', text)

    return [
        sentence.strip()
        for sentence in sentences
        if len(sentence.strip()) > 15
    ]


def normalize_text(text):
    """
    Normalize text for comparison.
    """

    text = text.lower()

    # Common paraphrase replacements
    replacements = {
        "ai": "artificial intelligence",
        "every day": "daily",
        "each day": "daily",
        "students must": "students should",
        "students should": "students must",
        "obtain": "get",
        "obtained": "got",
        "utilize": "use",
        "utilizes": "uses",
        "purchase": "buy",
        "purchases": "buys",
        "assist": "help",
        "assistance": "help",
        "learners": "students",
        "educators": "teachers",
        "modern technology": "technology",
        "digital platforms": "online platforms",
        "study materials": "educational resources",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove punctuation
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# --------------------------------------------------
# SEMANTIC-LIKE SIMILARITY
# --------------------------------------------------

def sentence_similarity(sentence1, sentence2):
    """
    Calculate similarity between two sentences using
    TF-IDF with word and character n-grams.

    This helps detect similarity even when some
    words have been changed.
    """

    s1 = normalize_text(sentence1)
    s2 = normalize_text(sentence2)

    if not s1 or not s2:
        return 0.0

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    try:

        vectors = vectorizer.fit_transform([s1, s2])

        word_score = cosine_similarity(
            vectors[0:1],
            vectors[1:2]
        )[0][0]

    except ValueError:

        word_score = 0.0

    # Character similarity catches small word changes
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True
    )

    try:

        char_vectors = char_vectorizer.fit_transform([s1, s2])

        char_score = cosine_similarity(
            char_vectors[0:1],
            char_vectors[1:2]
        )[0][0]

    except ValueError:

        char_score = 0.0

    # Hybrid sentence score
    score = (
        0.70 * word_score
        + 0.30 * char_score
    )

    return min(float(score), 1.0)


# --------------------------------------------------
# DOCUMENT COMPARISON
# --------------------------------------------------

def calculate_similarity(original, suspected):

    original_sentences = split_sentences(original)
    suspected_sentences = split_sentences(suspected)

    if not original_sentences or not suspected_sentences:
        return [], 0.0

    matches = []

    for suspected_sentence in suspected_sentences:

        best_original = ""
        best_score = 0.0

        for original_sentence in original_sentences:

            score = sentence_similarity(
                original_sentence,
                suspected_sentence
            )

            if score > best_score:

                best_score = score
                best_original = original_sentence

        matches.append({

            "suspected": suspected_sentence,

            "original": best_original,

            "score": best_score

        })

    # Overall similarity
    overall_similarity = np.mean(
        [match["score"] for match in matches]
    )

    return matches, float(overall_similarity)


# --------------------------------------------------
# EXACT WORD SIMILARITY
# --------------------------------------------------

def exact_similarity(original, suspected):

    original_words = set(
        normalize_text(original).split()
    )

    suspected_words = set(
        normalize_text(suspected).split()
    )

    if not suspected_words:
        return 0.0

    common_words = original_words.intersection(
        suspected_words
    )

    return len(common_words) / len(suspected_words)


# --------------------------------------------------
# COMBINED SCORE
# --------------------------------------------------

def combined_score(semantic_score, exact_score):

    """
    Hybrid plagiarism similarity score.

    75% similarity engine
    25% exact word overlap
    """

    score = (
        0.75 * semantic_score
        + 0.25 * exact_score
    )

    return min(float(score), 1.0)


# --------------------------------------------------
# RISK CLASSIFICATION
# --------------------------------------------------

def classify_similarity(score):

    percentage = score * 100

    if percentage >= 80:

        return (
            "HIGH",
            "Very high similarity detected"
        )

    elif percentage >= 60:

        return (
            "MODERATE",
            "Potential semantic similarity detected"
        )

    else:

        return (
            "LOW",
            "Low similarity detected"
        )