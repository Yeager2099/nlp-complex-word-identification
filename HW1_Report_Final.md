# Report transcription

Text extracted from the author-provided course report. Figures are not reproduced and tables may lose formatting. Student identifiers and institutional email addresses are removed.

## Page 1

CIS 5300: Natural Language Processing
Homework 1 Report: Complex Word Identification
Yeager Ye (QIUYU YE) | Penn ID: [STUDENT_ID_REDACTED]
Summer 2026

## Page 2

Section 1: Evaluation Metrics
1.1 All-Complex Baseline Performance
When predicting every word as complex (label = 1), the classifier achieves Precision =
0.4180, Recall = 1.0000, and F1 = 0.5896 on the dev set. Precision equals the base rate of
complex words in the data (~41.8%), because every prediction is positive, so TP/(TP+FP) =
(# complex) / (# total). Recall is 1.0 by construction — all complex words are found. F1 =
2×P×R/(P+R) = 2(0.418×1.0)/1.418 ≈ 0.59.
This demonstrates why accuracy is a poor metric for imbalanced binary classification: the
all-complex classifier has 41.8% accuracy yet is completely useless in practice (it cannot
identify any simple word). F1 penalizes the low precision and gives a more honest picture of
model quality.

## Page 3

Section 2: Baseline Classifiers
2.1 All-Complex Baseline
Dataset Precision Recall F1
Training 0.4328 1.0000 0.6041
Dev 0.4180 1.0000 0.5896
Analysis: Recall is 1.0 because every word is predicted complex. Precision equals the
complex word base rate (43.3% in training, 41.8% in dev). This baseline establishes the
lower bound: any useful model must exceed F1 = 0.59.
2.2 Word Length Baseline
Optimal threshold: 7 characters (chosen by maximizing F1 on training data).
Dataset Precision Recall F1
Training 0.6007 0.8440 0.7019
Dev 0.6054 0.8660 0.7126
Figure 1: Precision-Recall Curve for Word Length Baseline

## Page 4

[WRITEUP 2.2] Precision-Recall Tradeoff: At threshold = 1 (predict all words complex),
recall = 1.0 but precision is at its minimum (0.43). As threshold increases, fewer words are
predicted complex, causing precision to rise (the selected words are more reliably complex)
but recall to fall. The optimal F1 = 0.71 is achieved at threshold = 7, which roughly separates
common short words ("go", "big", "run") from longer multisyllabic words. Beyond threshold
≈ 15, precision continues rising but recall drops so steeply that F1 declines. The PR curve
clearly shows this inverse relationship.
2.3 Word Frequency Baseline
Optimal threshold: 19,900,000 (predict complex if Google Ngram count < 19.9M).
Dataset Precision Recall F1
Training 0.5657 0.8157 0.6681
Dev 0.5568 0.8445 0.6711
Figure 2: Precision-Recall Curve for Word Frequency Baseline
[WRITEUP 2.3] Comparison: Word frequency is a strong but imperfect proxy for
complexity. Common words that happen to be labeled complex (e.g., context-dependent
difficult words) cause false negatives, while infrequent proper nouns or low-frequency
simple words cause false positives. The word-length baseline (F1 = 0.71) outperforms word

## Page 5

frequency (F1 = 0.67) on this dataset because word length is a more direct measure of
orthographic complexity, while frequency confounds with domain-specific vocabulary and
proper nouns.

## Page 6

Section 3: Feature Engineering & Classifiers
3.1 Naive Bayes Classifier
Model Train P Train R Train F1 Dev P Dev R Dev F1
Naive 0.5070 0.9792 0.6681 0.4822 0.9713 0.6444
Bayes
3.2 Logistic Regression
Model Train P Train R Train F1 Dev P Dev R Dev F1
Logistic 0.7308 0.6557 0.6912 0.7429 0.6842 0.7123
Regression
3.3 Classifier Comparison
[WRITEUP 3.3]
1. Performance Comparison: Logistic Regression (Dev F1 = 0.7123) slightly outperforms
Naive Bayes (Dev F1 = 0.6444) on the dev set. The difference is modest (≈ 0.07 F1).
2. Model Comparison: Naive Bayes achieves near-perfect recall (0.97) but poor precision
(0.48), indicating it is overly conservative and predicts "complex" too often. Logistic
Regression achieves a better balance (P=0.74, R=0.68).
3. Why might one model outperform the other? Naive Bayes assumes features are
conditionally independent given the class. This assumption is violated: word length and
syllable count are highly correlated (r ≈ 0.7). Logistic Regression does not make this
assumption and can learn feature interactions through its weight vector, which helps it
achieve better precision.
4. Comparison with Baselines: Both ML classifiers underperform the word-length
baseline (F1 = 0.71) on the dev set. This suggests that with only 3 simple features, the
baselines are surprisingly strong. However, as we will see in Section 4, adding more
features allows ML models to substantially outperform all baselines.

## Page 7

Section 4: Build Your Own Model
4.1 Feature Design & Justification
Classifier used: GradientBoostingClassifier (200 estimators, learning_rate=0.1,
max_depth=4, subsample=0.8, random_state=42)
Six features were used (3 base + 3 new):
Feature Feature 1: Feature 2: vowel_ratio Feature 3:
log_frequency has_complex_suffix
Feature name log_frequency vowel_ratio has_complex_suffix
What linguistic Log-transformed Google Proportion of vowels in Binary feature: 1 if word
property it captures NGram count. Captures the word. Complex words ends with complex suffix
diminishing returns of often have Latin/Greek (-tion, -sion, -ment, -ous,
frequency — the roots with specific vowel etc.), 0 otherwise.
difference between freq patterns.
100 and 1,000 matters
more than between 1M
and 1.1M.
Predicted effect on Should improve precision May increase recall — Should improve precision
precision vs recall — log scale better catches complex words — suffixes are strong
separates rare from with distinctive vowel indicators of complex
common words. patterns. morphology.
After training: did it Yes, Feature importance = Yes, Feature importance = Yes, Feature importance =
help? 0.4672 (ranked #1). F1 0.0485. Marginal but 0.0175. Small but helps on
improved from 0.71 to consistent improvement. polysyllabic words.
0.75.
Why or why not? Log transformation Vowel ratio captures Suffix detection leverages
reduced the dominance of phonological complexity, morphological knowledge
frequency outliers and which is partially to identify loanwords and
better models the independent of academic vocabulary.
relationship between length/syllables.
frequency and complexity.
Performance with Incremental Features:
Features Dev P Dev R Dev F1
Baseline (length + 0.6054 0.8660 0.7126
frequency + syllables)
+ log_frequency 0.6553 0.8453 0.7385
+ vowel_ratio 0.6821 0.8201 0.7452
+ has_complex_suffix + 0.7152 0.7871 0.7494
consonant_clusters
Best model performance: Training — P: 0.8240, R: 0.8440, F1: 0.8339
Dev — P: 0.7152, R: 0.7871, F1: 0.7494
This outperforms all baselines (Section 2) and both sklearn classifiers (Section 3).
4.2 Error Analysis
[WRITEUP 4.2]

## Page 8

3 examples the model gets RIGHT:
"abdicate" — Complex (1), correctly predicted. Long (8 chars), low frequency, 4 syllables,
ends with -ate suffix. All features agree.
"sophisticated" — Complex (1), correctly predicted. Long (13 chars), low frequency, 4
syllables, sophisticated morphology.
"tin" — Simple (0), correctly predicted. Short (3 chars), extremely high frequency, 1
syllable, common word.
3 examples the model gets WRONG:
"showings" — Predicted Complex (1), actually Simple (0). False Positive. Relatively long (9
chars) and low frequency, but it is a common verb form.
"academy" — Predicted Complex (1), actually Simple (0). False Positive. Moderate length (7
chars) but recognized as a common noun.
"coup" — Predicted Simple (0), actually Complex (1). False Negative. Short (4 chars) but
rare word with specific meaning.
Error Category 1: Short but rare words (False Negatives)
Pattern: Words with length ≤ 5 characters but low frequency (rare words).
Examples: "coup", "psalm", "viz"
Why the model fails: The word-length feature (strong weight) dominates and predicts
these as simple. The model does not have enough resolution to detect rarity when length is
short.
Possible fix: Add a "short_but_rare" feature that triggers when length ≤ 5 AND
log_frequency < threshold. Alternatively, use character n-gram features to capture subtle
orthographic cues.
Error Category 2: Long but common words / Proper nouns (False Positives)
Pattern: Words with length ≥ 7 characters but high frequency or proper nouns.
Examples: "spotify", "academy", "wizards"

## Page 9

Why the model fails: The word-length threshold (≥ 7 chars) incorrectly flags these as
complex. Frequency helps but not enough for moderately frequent long words.
Possible fix: Add a "common_long_word" feature using a curated list (e.g., Oxford 3000).
Also add part-of-speech features to distinguish nouns/verbs from adjectives.

## Page 10

Section 5: Domain Generalization
5.1 External Dataset Class Distributions
Dataset % Complex
Biomedical 48.1%
News 31.7%
Note: The biomedical dataset has a higher proportion of complex words (48.1%) than the
original dataset (43.3%), while the news dataset has a lower proportion (31.7%). This
affects baseline performance when applying our model without retraining.
5.2 External Dataset Performance
Dataset Precision Recall F1
Dev (original) 0.7152 0.7871 0.7494
Biomedical 0.7040 0.6331 0.6667
News 0.5863 0.5505 0.5678
5.3 Domain Transfer Analysis
[WRITEUP 5.3]
1. Summary Table:
Domain F1 Score
Original test set 0.7494
Biomedical 0.6667
News 0.5678
2. Why does the model transfer better to one domain than the other? The model
transfers better to Biomedical (F1 = 0.67) than to News (F1 = 0.57). This is because
biomedical text contains many multisyllabic Latinate words (e.g., "phosphorylation",
"myocardial"), which our features (length, syllables, suffixes) are designed to detect. News
text, on the other hand, contains many short but rare words (e.g., "bloc", "tycoon") and
named entities, which fool our length-based features.
3. Specific error examples:

## Page 11

Biomedical: "dna" — Predicted Simple (0), actually Complex (1). "DNA" is a common
abbreviation in biomedical text but rare in general text. Our model does not have domain-
specific features.
News: "bloc" — Predicted Simple (0), actually Complex (1). Short (4 chars) but rare political
term. Same issue as Error Category 1.

## Page 12

Section 6: Reflection Questions
6.1 Class Imbalance and Metric Choice
[WRITEUP 6.1]
a) What was the accuracy of the all-complex baseline? The accuracy of the all-complex
baseline equals the proportion of simple words in the dataset, approximately 58.2% (1 -
0.418 = 0.582). Despite this seemingly decent accuracy, the classifier is completely useless
because it cannot distinguish between simple and complex words at all — it simply labels
everything as complex.
b) Why is F1 score a better metric? Accuracy would give the all-complex classifier ~43%
— worse than random (50%). F1 is the harmonic mean of precision and recall, penalizing
imbalanced performance. In the CWI task, both types of errors matter: missing a complex
word (FN) means the reader is left without simplification help; flagging a simple word as
complex (FP) wastes effort and creates noise. F1 captures both failure modes
symmetrically.
6.2 Model Interpretation
[WRITEUP 6.2]
a) Feature weights after training Logistic Regression:
Feature Weight Direction
word_frequency -8.1338 Higher frequency → simple
(largest negative weight)
syllable_count +0.7308 More syllables → complex
word_length +0.4748 Longer words → complex
Interpretation:
The word_frequency weight is the largest in magnitude and negative: after z-score normalization,
each 1-SD increase in frequency pushes the log-odds of "complex" down by 8.13 points. This matches
linguistic theory — frequent words are learned earliest and known universally.
The syllable_count weight is positive (+0.73): controlling for frequency, adding one SD of syllables
increases the log-odds of "complex" by 0.73. Even at the same frequency level, longer polysyllabic
words carry more morphological complexity.
The word_length weight is smaller (+0.47) because length is partially collinear with syllable count
after normalization.

## Page 13

b) Interpretability in NLP: With Random Forest or neural networks, direct weight
interpretation is not possible. Random Forests make decisions through ensemble voting of
many trees, while neural networks learn distributed representations across thousands of
parameters — making individual feature contributions opaque.
Interpretability is crucial in NLP applications where decisions affect people's lives. For
example, a non-interpretable model used for automated hiring might systematically
discriminate against certain demographic groups without anyone knowing why. Similarly, a
"black-box" medical diagnosis NLP could miss rare conditions or confidently hallucinate
incorrect diagnoses, and patients would have no way to understand or question the
system's reasoning. Interpretable models allow stakeholders to verify reasoning, identify
biases, and build trust.
