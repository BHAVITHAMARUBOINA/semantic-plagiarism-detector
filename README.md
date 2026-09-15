# 🛡️ SemanticShield

### Detect the meaning, not just the words.

SemanticShield is an NLP-based plagiarism detection system designed to identify **semantic similarity and paraphrased content**, not just exact word matches.

## 🚨 Problem

Traditional plagiarism detection systems often focus on matching identical words or phrases.

However, a person can rewrite copied content using different words while keeping the same meaning.

### Example

**Original:**
> Students should exercise every day.

**Suspected:**
> Students must work out daily.

Although the wording is different, the meaning is highly similar.

## 💡 Solution

SemanticShield compares documents at the sentence level and identifies potentially similar content using:

- Text extraction
- Text normalization
- NLP-based similarity analysis
- Word and character n-gram features
- Cosine similarity
- Exact word overlap
- Hybrid similarity scoring
- Risk classification

## ⚙️ How It Works

```text
Upload Documents
       ↓
Text Extraction
       ↓
Sentence Splitting
       ↓
Text Normalization
       ↓
TF-IDF Features
       ↓
Cosine Similarity
       ↓
Hybrid Similarity Score
       ↓
Risk Assessment
       ↓
Matching Content Report# semantic-plagiarism-detector
