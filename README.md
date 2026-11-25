# CLNLP Computational Linguistics and Natural Language Processing

**Name:** Saurabh Kumar  
**SAP ID:** 500124288  
**BATCH:** 2 (AIML)

---

## Table of Contents

| # | Experiment | Topic | Open Notebook |
|---:|---|---|---|
| 1 | Exp1 | Employee Data Analysis (Pandas + Matplotlib) | [Exp1.ipynb](Exp1/Exp1.ipynb) |
| 2 | Exp2 | Text Preprocessing and Tokenization | [Experiment_2.ipynb](Exp2/Experiment_2.ipynb) |
| 3 | Exp3 | Stemming, Lemmatization and Regex Extraction | [Experiment_3.ipynb](Exp3/Experiment_3.ipynb) |
| 4 | Exp4 | Term Frequency, NER and TF-IDF | [Experiment4.ipynb](Exp4/Experiment4.ipynb) |
| 5 | Exp5 | Subword Tokenization, POS Tagging and Word2Vec | [Experiment_5.ipynb](Exp5/Experiment_5.ipynb) |
| 6 | Exp6 | GloVe Embeddings and N-gram Language Model | [Experiment_6.ipynb](Exp6/Experiment_6.ipynb) |
| 7 | Exp7 | IMDB Sentiment Analysis — Model Comparison | [IMDB_Sentiment_Analysis_Comparison.ipynb](Exp7/IMDB_Sentiment_Analysis_Comparison.ipynb) |
| 8 | Exp8 | Hindi Abstractive Summarization (mT5 vs IndicBART) | [Experiment 8.ipynb](<Exp8/Experiment 8.ipynb>) |

---

## Overview

This repository contains eight CLNLP laboratory experiments, progressing from basic text handling to modern deep-learning NLP:

```text
NLP/
├── Exp1/  Employee tabular EDA
├── Exp2/  Cleaning and tokenization
├── Exp3/  Stemming, lemmatization, regex
├── Exp4/  Term frequency, NER, TF-IDF
├── Exp5/  BPE / SentencePiece, POS tagging, Word2Vec
├── Exp6/  GloVe from scratch, pretrained GloVe, N-gram LM
├── Exp7/  IMDB sentiment model comparison
└── Exp8/  Hindi abstractive summarization comparison
```

All notebooks use a Python 3 kernel. A few cells were originally executed in Google Colab, so local execution may require replacing upload/mount cells with direct file reads.

---

## Experiment 1 — Employee Data Analysis

**Notebook:** [Exp1.ipynb](Exp1/Exp1.ipynb)  
**Assignment:** [Experiment-1.docx](Exp1/Experiment-1.docx)  
**Dataset:** [employee_information_100.csv](Exp1/employee_information_100.csv)

### Objective
Explore a small employee dataset and produce visual summaries of salary, department, gender, age, and experience.

### Dataset
100 records with columns:

```text
Employee_ID, Name, Department, Age, Gender, Salary, Experience_Years
```

### What was done
1. Loaded the CSV with `pandas.read_csv`.
2. Computed the overall average salary and the salary range.
3. Used `groupby("Department")["Salary"].mean()` to compare average salaries across departments.
4. Used `value_counts()` for department and gender distributions.
5. Applied boolean filtering to count employees earning above the overall average.
6. Sorted records by salary and examined department-wise maximum salary and average experience.
7. Plotted bar charts, a gender pie chart, salary and age histograms, and a salary-vs-experience scatter plot with `matplotlib`.

Representative code:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("employee_information_100.csv")
avg_salary = df.groupby("Department")["Salary"].mean()
avg_salary.plot(kind="bar")

df["Gender"].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.hist(df["Salary"], bins=10)
plt.scatter(df["Experience_Years"], df["Salary"])
```

### Results
- Overall average salary: **77,760**; range: **30,000–119,000**.
- Age range: **22–59**. Gender counts: **64 Male, 36 Female**.
- Department counts: HR 21, Marketing 20, IT 18, Operations 16, Sales 13, Finance 12.
- Average salary by department: Operations 90,875 > Finance 82,416.67 > Marketing 76,100 > Sales 74,769.23 > IT 73,611.11 > HR 72,095.24.
- Operations has the highest average salary; Marketing has the highest maximum salary (119,000).
- **52** employees earn above the overall average.
- Average experience is highest in Sales (23.8462 years).

### Charts embedded in the notebook
Average salary by department, employee count by department, gender pie chart, salary histogram, and age histogram.

---

## Experiment 2 — Basic Text Preprocessing and Tokenization

**Notebook:** [Experiment_2.ipynb](Exp2/Experiment_2.ipynb)  
**Assignment:** [Experimemt-2.docx](Exp2/Experimemt-2.docx) *(filename contains a typo in the original)*  
**Inputs:** [2.1_text_data.txt](Exp2/2.1_text_data.txt), [2.2_tokenization_data.txt](Exp2/2.2_tokenization_data.txt), [2.3_clean_data.txt](Exp2/2.3_clean_data.txt)

### Objective
Practice fundamental text-cleaning operations and compare several tokenization strategies.

### What was done
1. Counted uppercase letters, punctuation types, numbers, and extra whitespace in the raw text.
2. Converted text to lowercase and normalized repeated whitespace using `" ".join(text.split())`.
3. Detected and removed punctuation and numeric tokens.
4. Performed sentence and word tokenization with NLTK (`sent_tokenize`, `word_tokenize`).
5. Tokenized the same text with spaCy (`en_core_web_sm`) and inspected sentence and token boundaries.
6. Compared naive `str.split(".")` and regex `\b\w+\b` tokenization.
7. Removed stop words manually from the third text file.

Representative code:

```python
import string, re, nltk, spacy

text = " ".join(text.split())
sentences = nltk.sent_tokenize(data)
words = nltk.word_tokenize(data)

doc = nlp(data)
for sent in doc.sents:
    print(sent)

words = re.findall(r"\b\w+\b", data)
```

### Results
- Uppercase letters: **32**; numbers: **13**; extra whitespaces: **8**.
- NLTK keeps punctuation as separate tokens.
- spaCy splits hyphenated terms such as `real-world` into `real`, `-`, `world`.
- `str.split(".")` leaves punctuation attached to words.
- Regex `\b\w+\b` strips punctuation but also loses hyphen structure.
- Stop-word removal retains punctuated tokens like `the,` and `it.` because punctuation was not stripped from the third input beforehand.

### Notes
- The punctuation-type counter has a bug: it adds a stale variable `ch` instead of the current character `i`, so the reported value `1` is not the true count.
- The spaCy model download logged a Windows console `UnicodeEncodeError`, but `spacy.load("en_core_web_sm")` succeeded afterwards.

---

## Experiment 3 — Stemming, Lemmatization and Regex Extraction

**Notebook:** [Experiment_3.ipynb](Exp3/Experiment_3.ipynb)  
**Assignment:** [Experiment-3.docx](Exp3/Experiment-3.docx)

### Objective
Demonstrate rule-based word normalization (stemming and lemmatization) and regex-based extraction of structured entities from text.

### What was done
1. Applied the **Porter stemmer** from NLTK to a list of morphologically related words.
2. Applied the **WordNet lemmatizer** to the same kinds of words, comparing dictionary-based normalization against crude suffix stripping.
3. Used regular expressions to extract emails, URLs, Indian mobile numbers, hashtags, and mentions.

Representative code:

```python
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

emails   = re.findall(r"[\w.-]+@[\w.-]+\.\w+", text)
urls     = re.findall(r"https?://\S+|www\.\S+", text)
mobiles  = re.findall(r"\+91-\d{10}|\b\d{10}\b", text)
hashtags = re.findall(r"#\w+", text)
mentions = re.findall(r"@\w+", text)
```

### Results
**Stemming:** `playing/played/plays → play`, `studies/studying → studi`, `connected/connection → connect`, `computers → comput`.

**Lemmatization:** `cats → cat`, `dogs → dog`, `children → child`, `mice → mouse`, `leaves → leaf`, `studies → study`, `ran → ran`, `went → went`, `ate → ate`. Because POS tags were not supplied, several verb forms such as `running` and `caring` remained unchanged.

**Regex extraction:**
- Emails: `nlpworkshop@gmail.com`, `support@python.org`
- URLs: `https://www.nlpworkshop.com`, `www.python.org.`, `https://github.com/NLPWorkshop`
- Mobiles: `+91-9876543210`, `9123456789`
- Hashtags: `#NLP`, `#Python`, `#MachineLearning`
- Mentions: `@gmail`, `@python`, `@NLPWorkshop`, `@PythonLearner`

### Notes
- `www.python.org.` includes a trailing period, and `@gmail` / `@python` are false positives taken from email addresses — both illustrate why regex extraction needs post-processing.

---

## Experiment 4 — Term Frequency, NER and TF-IDF

**Notebook:** [Experiment4.ipynb](Exp4/Experiment4.ipynb)  
**Assignment:** [_Experiment-4_ Term Frequency and NER.docx](<Exp4/_Experiment-4_ Term Frequency and NER.docx>)  
**Inputs:** [4.1_4.2_input.txt](Exp4/4.1_4.2_input.txt), [document1.txt](Exp4/document1.txt), [document2.txt](Exp4/document2.txt), [document3.txt](Exp4/document3.txt)  
**Output:** [term_frequency.csv](Exp4/term_frequency.csv)

### Objective
Implement four related textbook tasks: term frequency with and without a toolkit, named-entity recognition, and a manual TF-IDF pipeline.

### What was done
1. **Toolkit TF:** lowercased the article, extracted alphabetic tokens with regex, and counted them using `collections.Counter`.
2. **Manual TF:** split text on whitespace, stripped punctuation from each token, and built a frequency dictionary by hand.
3. **NER:** tokenized and POS-tagged the article with NLTK, then applied `nltk.ne_chunk` to extract named entities.
4. **TF-IDF:** computed term frequency per document, document frequency across the three documents, inverse document frequency with `log(N/df)`, and finally `tf × idf` for every term.

Representative code:

```python
import nltk, re, math
from collections import Counter
import pandas as pd

words = re.findall(r"[a-zA-Z]+", text.lower())
frequency = Counter(words)

tokens = nltk.word_tokenize(text)
entities = nltk.ne_chunk(nltk.pos_tag(tokens))

tf = term_count / total_words_in_document
idf = math.log(total_documents / df)
tfidf = tf * idf
```

### Results
**Top terms in the saved CSV:** `a` (90), `and` (86), `the` (75), `to` (42), `in` (42), `can` (41), `of` (40), `is` (37), `may` (30), `word` (30). The toolkit version printed slightly different numbers (e.g. `word = 31`) because it uses a different tokenizer.

**NER examples:**
- Correct: `Abdul Kalam → PERSON`, `ISRO → ORGANIZATION`, `India → GPE`.
- False positives: `Text → PERSON`, `NLP → ORGANIZATION`, `English/Word2Vec/Python → GPE`. The model is tuned for news-style text, so explanatory prose produces noisy entities.

**TF-IDF:** IDF for terms appearing in one document is `1.0986`, and for terms in two documents `0.4055`. The top TF-IDF terms are `a`, `field` for document 1; `helps`, `computers`, `understand`, `human` for document 2; and `machine`, `learning`, `an`, `important`, `part` for document 3.

---

## Experiment 5 — Subword Tokenization, POS Tagging and Word2Vec

**Notebook:** [Experiment_5.ipynb](Exp5/Experiment_5.ipynb)  
**Assignment:** [Experiment-5.docx](Exp5/Experiment-5.docx)  
**Inputs:** [input_sub_word_data.txt](Exp5/input_sub_word_data.txt), [corpus.txt](Exp5/corpus.txt)  
**Trained artifacts:** [mymodel.model](Exp5/mymodel.model), [mymodel.vocab](Exp5/mymodel.vocab)

### Objective
Study subword tokenization using pretrained and from-scratch methods, compare POS taggers, and train word embeddings with both Word2Vec architectures.

### What was done
1. **GPT-2 tokenizer:** loaded `gpt2` via HuggingFace `AutoTokenizer`, tokenized the text, and converted tokens to IDs.
2. **Flan-T5 tokenizer:** applied the same procedure with the Flan-T5 tokenizer.
3. **BPE from scratch:** built a `tokenizers.Tokenizer` with a `BPE` model, `Whitespace` pre-tokenizer, and a `BpeTrainer` with `vocab_size=500` and an `[UNK]` token.
4. **SentencePiece BPE:** trained a 500-piece BPE model on a text file and saved `mymodel.model` / `mymodel.vocab`.
5. **POS tagging:** tagged a sample sentence with spaCy and NLTK, then counted high-frequency POS tags across the larger corpus.
6. **Word2Vec:** trained CBOW (`sg=0`) and Skip-Gram (`sg=1`) models with `vector_size=50`, `window=3`, `min_count=1`, fixed seed 42, and compared nearest neighbours and cosine similarities.

Representative code:

```python
from transformers import AutoTokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
import sentencepiece as spm
from gensim.models import Word2Vec

tokenizer = AutoTokenizer.from_pretrained("gpt2")

tok = Tokenizer(BPE(unk_token="[UNK]"))
tok.train_from_iterator([text], BpeTrainer(vocab_size=500, special_tokens=["[UNK]"]))

spm.SentencePieceTrainer.train(input="train.txt", model_prefix="mymodel",
                               vocab_size=500, model_type="bpe", hard_vocab_limit=False)

cbow = Word2Vec(sentences, vector_size=50, window=3, min_count=1, sg=0, seed=42)
skipgram = Word2Vec(sentences, vector_size=50, window=3, min_count=1, sg=1, seed=42)
```

### Results
- GPT-2 warned that the input was `1348 > 1024` tokens; Flan-T5 warned about its `512`-token limit.
- BPE splits: `branch → b, r, an, ch`, `artificial → ar, t, ific, ial`, `intelligence → in, t, el, l, ig, ence`.
- SentencePiece artifacts: `mymodel.model` (246,883 bytes) and a 500-entry vocabulary using `▁` word-boundary markers.
- POS tags for *"The young student is reading an interesting book in the library."*: spaCy gives `DET ADJ NOUN AUX VERB DET ADJ NOUN ADP DET NOUN PUNCT`; NLTK gives `DT JJ NN VBZ VBG DT JJ NN IN DT NN .`.
- Frequent spaCy tags: `the DET 48`, `a DET 28`, `and CCONJ 27`, `can AUX 23`, `subword NOUN 17`, `language NOUN 12`.
- CBOW nearest neighbours of `study`: `study` 0.3312, `the` 0.2492, `processing` 0.2052. Skip-Gram neighbours include `uses` 0.2865 and `helps` 0.1944.
- Cosine similarities: `machine–learning −0.2329`, `natural–language −0.0297`, `deep–learning −0.0983` — unstable because the corpus is very small.

### Notes
- The notebook references `train.txt`, which is not present as a separate file; the notebook regenerates it.
- Some outputs show `UnicodeEncodeError` when printing markers such as `▁`; the training itself completed.

---

## Experiment 6 — GloVe Embeddings and N-gram Language Modeling

**Notebook:** [Experiment_6.ipynb](Exp6/Experiment_6.ipynb)  
**Assignment:** [Experiment-6.docx](Exp6/Experiment-6.docx)  
**Corpus:** [glove_corpus.txt](Exp6/glove_corpus.txt)

### Objective
Implement a simplified GloVe model from scratch in PyTorch, examine pretrained GloVe embeddings, and build an N-gram language model for next-word prediction and text generation.

### What was done
1. Built a weighted co-occurrence matrix from a window of 5 words, weighting distant pairs by `1/d`.
2. Implemented a `GloVe` `nn.Module` with centre and context embeddings plus two bias terms: score = `(w_i · w_t_j) + b_i + b_t_j`.
3. Trained with Adagrad for 100 epochs, `dim=50`, `XMAX=100`, `alpha=0.75`, batch size 1024.
4. Loaded `glove-wiki-gigaword-100` through `gensim.downloader` and queried similarities and the `king - man + woman` analogy.
5. Implemented an `NGramLM` class that stores nested count dictionaries and predicts the next word; used it to generate a short sentence.

Representative code:

```python
WINDOW = 5
cooc[(center, ids[j])] += 1.0 / d
cooc[(ids[j], center)] += 1.0 / d

class GloVe(nn.Module):
    def forward(self, i, j):
        return (self.w(i) * self.wt(j)).sum(1) + self.b(i).squeeze() + self.bt(j).squeeze()

import gensim.downloader as api
glove = api.load("glove-wiki-gigaword-100")

class NGramLM:
    def __init__(self, n=3):
        self.counts = {k: defaultdict(Counter) for k in range(n)}
```

### Results
- Scratch GloVe: **148 tokens**, **68 vocabulary words**, **779 non-zero co-occurrences**, final loss ≈ `0.0001`. Neighbours of `king`: `the` 0.7631, `kingdom` 0.6860, `queen` 0.6637 — heavily overfitted because the corpus is tiny.
- Pretrained GloVe neighbours of `king`: `prince` 0.7682, `queen` 0.7508, `son` 0.7021, `brother` 0.6986, `monarch` 0.6978.
- Analogy `king - man + woman`: `queen` 0.7699, `monarch` 0.6843, `throne` 0.6756, `daughter` 0.6595, `princess` 0.6521.
- N-gram predictions with probability 1.0: `natural language → processing`, `machine → learning`, `deep learning is → a`, `artificial → intelligence`.
- Generated text: *"natural language processing is a branch of artificial intelligence includes machine learning"*.

---

## Experiment 7 — IMDB Sentiment Analysis: Model Comparison

**Notebook:** [IMDB_Sentiment_Analysis_Comparison.ipynb](Exp7/IMDB_Sentiment_Analysis_Comparison.ipynb)  
**Dataset:** [IMDB Dataset.csv](<Exp7/IMDB Dataset.csv>)  
**Results:** [final_model_comparison.csv](Exp7/final_model_comparison.csv), [model_comparison_metrics.csv](Exp7/model_comparison_metrics.csv)

### Objective
Compare four sentiment classifiers on the same train/validation/test split: TF-IDF + Logistic Regression, a Bi-LSTM, a from-scratch Transformer, and fine-tuned BERT.

### Dataset and preprocessing
- 50,000 labelled reviews (25,000 positive / 25,000 negative); mean length 1309.4 characters, maximum 13,704.
- Stripped HTML, de-duplicated to **49,578** rows, then split into **39,662 train / 4,958 validation / 4,958 test**.

### Models
1. **TF-IDF + Logistic Regression** — `TfidfVectorizer(lowercase=True, ngram_range=(1,2), min_df=2, max_features=300_000, sublinear_tf=True)`, with a validation search over `C`.
2. **Bi-LSTM from scratch** — vocab 30,000, sequence length 256, embedding dim 128, two bidirectional LSTM layers with hidden size 128, dropout 0.4.
3. **Vanilla Transformer from scratch** — embedding dim 128, 2 encoder layers, 4 attention heads, feedforward dim 512, sinusoidal positional encoding, masked mean pooling.
4. **BERT** — `bert-base-uncased`, max length 256, batch size 32, 2 epochs, learning rate `2e-5`, AdamW with linear warmup/decay, FP16 autocast.

### Results

| Model | Accuracy | Precision | Recall | F1 | AUC | Params | Train (min) | Inference (s/1k) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9173 | 0.9177 | 0.9172 | 0.9173 | 0.9730 | 300,001 | 1.12 | 0.0012 |
| Bi-LSTM | 0.9060 | 0.9064 | 0.9060 | 0.9060 | 0.9668 | 4,499,970 | 3.77 | 0.1220 |
| Transformer | 0.8683 | 0.8685 | 0.8683 | 0.8683 | 0.9390 | 4,236,802 | 3.46 | 0.1787 |
| BERT | **0.9270** | **0.9271** | **0.9270** | **0.9270** | **0.9791** | 109,483,778 | 17.38 | 3.7461 |

- Validation `C` search: `C=0.5 → 0.8969`, `1 → 0.9038`, `2 → 0.9082`, `5 → 0.9137`, `10 → 0.9141` macro-F1.
- Top positive LR features: `great`, `excellent`, `perfect`, `amazing`, `wonderful`, `enjoyable`, `best`; negative: `worst`, `bad`, `awful`, `boring`, `waste`, `poor`, `terrible`, `horrible`.
- Confusion matrices (TN/FP/FN/TP): LR 2227/243/167/2321, Bi-LSTM 2200/270/196/2292, Transformer 2115/355/298/2190, BERT 2269/201/161/2327.

### Conclusions
BERT is the most accurate but is ~365× larger and far slower than Logistic Regression, which is the strongest non-neural baseline. The from-scratch Transformer performs worst, likely due to limited data. Truncating reviews to 256 tokens loses information on long reviews, while BERT handles negation and contrast better than the bag-of-words model.

### Notes
- The notebook expects `IMDB_Dataset.csv`, but the file on disk is named `IMDB Dataset.csv`.
- Colab upload/download helpers should be replaced with direct local reads when running outside Colab.

---

## Experiment 8 — Hindi Abstractive Summarization (mT5 vs IndicBART)

**Notebook:** [Experiment 8.ipynb](<Exp8/Experiment 8.ipynb>)  
**Assignment:** [Experiment-8.pdf](Exp8/Experiment-8.pdf)  
**Results:** [Generated_summaries.csv](Exp8/Generated_summaries.csv)

### Objective
Evaluate and compare two multilingual abstractive summarization models on Hindi XL-Sum:

- `csebuetnlp/mT5_multilingual_XLSum`
- `ai4bharat/IndicBART-XLSum`

using ROUGE-1, ROUGE-2, ROUGE-L, and BERTScore (multilingual BERT).

### Dataset
Downloaded the Hindi XL-Sum test split from `https://huggingface.co/datasets/csebuetnlp/xlsum` (8,847 articles) and evaluated on a random sample of **200** articles. Mean article length 457.9 words; mean reference summary length 26.7 words.

### What was done
1. Loaded the Hindi test split (`data/hindi_XLSum_v2.0.tar.bz2`) and sampled 200 article/reference pairs.
2. Built a custom **Unicode-aware tokenizer** because the default ROUGE tokenizer deletes Devanagari text: it replaces Unicode punctuation/separator characters with spaces and lowercases before splitting.
3. Generated summaries with beam search: `max_length=84`, `num_beams=4`, `no_repeat_ngram_size=2`, `early_stopping=True`; IndicBART uses the Hindi token `<2hi>`.
4. Computed ROUGE and mBERT BERTScore-F1 for every generated summary and saved per-sample results.
5. Ran a paired bootstrap with 2,000 resamples to measure the significance of model differences.

Representative code:

```python
class MultilingualTokenizer:
    def tokenize(self, text):
        text = "".join(
            " " if unicodedata.category(ch)[0] in "PS" else ch
            for ch in text.lower()
        )
        return text.split()

GEN_KWARGS = dict(max_length=84, num_beams=4,
                  no_repeat_ngram_size=2, early_stopping=True)
```

### Results

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore-F1 | Gen words | Ref words | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| mT5 (582.4M params, 250,100 vocab) | **35.99** | **15.83** | **30.31** | **76.88** | 18.58 | 27.46 | 65.08 |
| IndicBART/mBART (244.0M params, 64,015 vocab) | 20.52 | 6.56 | 17.61 | 70.44 | 9.40 | 27.46 | 22.67 |

Paired bootstrap (2,000 resamples), reported as `mBART − mT5`:

| Metric | mT5 wins | mBART wins | Ties | Difference | 95% CI |
|---|---:|---:|---:|---:|---|
| ROUGE-1 | 171 | 28 | 1 | −15.47 | [−17.58, −13.29] |
| ROUGE-2 | 140 | 35 | 25 | −9.27 | [−11.17, −7.54] |
| ROUGE-L | 166 | 32 | 2 | −12.70 | [−14.68, −10.65] |
| BERTScore-F1 | 176 | 24 | 0 | −6.44 | [−7.21, −5.68] |

All confidence intervals exclude zero, so mT5's advantage is consistent across this sample. The default ASCII ROUGE tokenizer returns all zeros on Hindi, while the custom tokenizer gives ROUGE-1 `0.3333`, ROUGE-2 `0.25`, ROUGE-L `0.3333` on a sample pair.

`Generated_summaries.csv` contains columns `id, title, reference, mT5_summary, mT5_rougeL, mT5_mBERT_F1, mBART_summary, mBART_rougeL, mBART_mBERT_F1`.

### Conclusions
mT5 clearly outperforms IndicBART/mBART on all four metrics. IndicBART is smaller and ~3× faster but produces short, headline-style summaries. Hindi ROUGE evaluation requires Unicode-aware tokenization. Some fluent mT5 summaries contain hallucinated facts, and ROUGE-2 stays modest because Hindi paraphrasing changes lexical overlap.

### Notes
- The notebook writes `8_generated_summaries.csv`, while the file saved in the folder is named `Generated_summaries.csv`.

---

## General Reproduction Notes

### Filename mismatches to be aware of

| Notebook references | File actually present |
|---|---|
| `Exp7/IMDB_Dataset.csv` | [IMDB Dataset.csv](<Exp7/IMDB Dataset.csv>) |
| `Exp5/train.txt` | Not stored; regenerated inside the notebook |
| `Exp8/xlsum_hindi_test_sample.jsonl` | Not stored; regenerated from the downloaded archive |
| `Exp8/8_generated_summaries.csv` | [Generated_summaries.csv](Exp8/Generated_summaries.csv) |

### Encoding
Several notebooks were executed on Windows with a `cp1252` console. Printing Unicode characters such as `▁`, Devanagari text, or check marks produced `UnicodeEncodeError` messages. These are display issues and do not indicate failed computation.

### Suggested setup

```bash
pip install pandas matplotlib nltk spacy gensim torch transformers tokenizers sentencepiece rouge-score bert-score datasets
python -m spacy download en_core_web_sm
```
