# -*- coding: utf-8 -*-
"""
CIS 5300 Homework 1: Text Classification
Complete implementation - NO WordNet/NLTK required
Student: Yeager Ye (QIUYU YE) | Penn ID: [STUDENT_ID_REDACTED]
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import gzip
import re
import subprocess

# Install syllables if needed (lightweight, no network data required)
try:
    import syllables
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'syllables'],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import syllables

# ============================================================
# Section 0: Data Loading
# ============================================================

########## DO NOT MODIFY ##########
def load_labeled_file(data_file):
    words, labels = [], []
    with open(data_file, 'rt', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                words.append(parts[0].lower())
                labels.append(int(parts[1]))
    return words, labels

def load_unlabeled_file(data_file):
    with open(data_file, 'rt', encoding='utf-8') as f:
        next(f)
        words = [line.strip().split('\t')[0].lower() for line in f if line.strip()]
    return words

def load_ngram_counts(ngram_counts_file='data/ngram_counts.txt.gz'):
    counts = defaultdict(int)
    with gzip.open(ngram_counts_file, 'rt') as f:
        for line in f:
            token, count = line.strip().split('\t')
            if token[0].islower():
                counts[token] = int(count)
    return counts
########## DO NOT MODIFY ##########

try:
    train_words, train_labels = load_labeled_file('data/complex_words_training.txt')
    dev_words, dev_labels = load_labeled_file('data/complex_words_development.txt')
    test_words = load_unlabeled_file('data/complex_words_test_unlabeled.txt')
    print("Loading Google NGram counts (this takes ~20 seconds)...")
    ngram_counts = load_ngram_counts()
    print(f"Loaded {len(ngram_counts):,} word frequencies.")
except FileNotFoundError as e:
    print(f"WARNING: Data files not found: {e}")
    train_words, train_labels = [], []
    dev_words, dev_labels = [], []
    test_words = []
    ngram_counts = {}

if train_words:
    n_train = len(train_words)
    n_complex_train = sum(train_labels)
    n_simple_train  = n_train - n_complex_train
    print(f"\nTraining set: {n_train} words")
    print(f"  Simple (0): {n_simple_train} ({n_simple_train/n_train:.1%})")
    print(f"  Complex (1): {n_complex_train} ({n_complex_train/n_train:.1%})")
    print(f"Dev set: {len(dev_words)} words")
    print(f"Test set: {len(test_words)} words (unlabeled)")
    simple_ex  = [w for w, l in zip(train_words[:50], train_labels[:50]) if l == 0][:5]
    complex_ex = [w for w, l in zip(train_words[:50], train_labels[:50]) if l == 1][:5]
    print(f"Example simple words:  {simple_ex}")
    print(f"Example complex words: {complex_ex}")


# ============================================================
# Section 1: Evaluation Metrics (15 points)
# ============================================================

def get_precision(y_true, y_pred):
    """Precision = TP / (TP + FP). Returns 0.0 if no positive predictions."""
    TP = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    FP = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    return TP / (TP + FP) if (TP + FP) > 0 else 0.0


def get_recall(y_true, y_pred):
    """Recall = TP / (TP + FN). Returns 0.0 if no actual positives."""
    TP = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    FN = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return TP / (TP + FN) if (TP + FN) > 0 else 0.0


def get_fscore(y_true, y_pred):
    """F1 = 2*P*R / (P+R). Returns 0.0 if P+R=0."""
    P = get_precision(y_true, y_pred)
    R = get_recall(y_true, y_pred)
    return 2 * P * R / (P + R) if (P + R) > 0 else 0.0


# Sanity checks
try:
    y_true_ex = [1, 0, 1, 1, 0, 0, 1, 0]
    y_pred_ex = [1, 0, 0, 1, 1, 0, 1, 0]
    assert abs(get_precision(y_true_ex, y_pred_ex) - 0.75) < 1e-6
    assert abs(get_recall(y_true_ex, y_pred_ex) - 0.75) < 1e-6
    assert abs(get_fscore(y_true_ex, y_pred_ex) - 0.75) < 1e-6
    assert get_precision([1, 1, 0], [0, 0, 0]) == 0.0
    assert get_recall([0, 0, 0], [1, 1, 1]) == 0.0
    print("\n✓ All Section 1 sanity checks passed!")
except Exception as e:
    print(f"WARNING: Sanity checks failed: {e}")


# ============================================================
# Section 2: Baselines (18 points)
# ============================================================

# --- 2.1 Always-Complex Baseline ---

def all_complex(words):
    """Predict every word as complex (label = 1)."""
    return [1] * len(words)


try:
    train_pred_ac = all_complex(train_words)
    dev_pred_ac   = all_complex(dev_words)
    ac_train_p = get_precision(train_labels, train_pred_ac)
    ac_train_r = get_recall(train_labels, train_pred_ac)
    ac_train_f = get_fscore(train_labels, train_pred_ac)
    ac_dev_p   = get_precision(dev_labels, dev_pred_ac)
    ac_dev_r   = get_recall(dev_labels, dev_pred_ac)
    ac_dev_f   = get_fscore(dev_labels, dev_pred_ac)
    ac_train_acc = sum(1 for t in train_labels if t == 1) / len(train_labels)
    print("\n=== Section 2.1: All-Complex Baseline ===")
    print(f"  Training — P: {ac_train_p:.4f}, R: {ac_train_r:.4f}, F1: {ac_train_f:.4f}  (Acc: {ac_train_acc:.4f})")
    print(f"  Dev      — P: {ac_dev_p:.4f}, R: {ac_dev_r:.4f}, F1: {ac_dev_f:.4f}")
except Exception as e:
    print(f"WARNING: Skipped 2.1: {e}")
    ac_train_p = ac_train_r = ac_train_f = 0.0
    ac_dev_p = ac_dev_r = ac_dev_f = 0.0


# --- 2.2 Word Length Baseline ---

def word_length_threshold(train_words, train_labels, eval_words):
    """Best word-length threshold → predict complex (1) if len(word) >= threshold."""
    best_threshold, best_f1 = 1, -1.0
    for threshold in range(1, 26):
        preds = [1 if len(w) >= threshold else 0 for w in train_words]
        f1 = get_fscore(train_labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    eval_predictions = [1 if len(w) >= best_threshold else 0 for w in eval_words]
    return eval_predictions, best_threshold


try:
    train_pred_wl, best_len_threshold = word_length_threshold(train_words, train_labels, train_words)
    dev_pred_wl, _ = word_length_threshold(train_words, train_labels, dev_words)
    wl_train_p = get_precision(train_labels, train_pred_wl)
    wl_train_r = get_recall(train_labels, train_pred_wl)
    wl_train_f = get_fscore(train_labels, train_pred_wl)
    wl_dev_p   = get_precision(dev_labels, dev_pred_wl)
    wl_dev_r   = get_recall(dev_labels, dev_pred_wl)
    wl_dev_f   = get_fscore(dev_labels, dev_pred_wl)
    print(f"\n=== Section 2.2: Word Length Baseline (best threshold = {best_len_threshold}) ===")
    print(f"  Training — P: {wl_train_p:.4f}, R: {wl_train_r:.4f}, F1: {wl_train_f:.4f}")
    print(f"  Dev      — P: {wl_dev_p:.4f}, R: {wl_dev_r:.4f}, F1: {wl_dev_f:.4f}")
except Exception as e:
    print(f"WARNING: Skipped 2.2: {e}")
    wl_train_p = wl_train_r = wl_train_f = 0.0
    wl_dev_p = wl_dev_r = wl_dev_f = 0.0
    best_len_threshold = 0

# P-R curve for word length baseline
try:
    pr_len_precisions, pr_len_recalls = [], []
    for t in range(1, 26):
        preds = [1 if len(w) >= t else 0 for w in train_words]
        pr_len_precisions.append(get_precision(train_labels, preds))
        pr_len_recalls.append(get_recall(train_labels, preds))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(pr_len_recalls, pr_len_precisions, 'b-o', markersize=5, linewidth=1.5,
            label='Word Length Baseline')
    for i, t in enumerate(range(1, 26)):
        if t in [1, 5, 8, 10, 13, 16, 20, 25]:
            ax.annotate(f't={t}', (pr_len_recalls[i], pr_len_precisions[i]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve: Word Length Baseline\n(Thresholds 1-25, Training Data)',
                 fontsize=12)
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pr_curve_word_length.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved pr_curve_word_length.png")
except Exception as e:
    print(f"WARNING: Length P-R curve skipped: {e}")


# --- 2.3 Word Frequency Baseline ---

def word_frequency_threshold(train_words, train_labels, eval_words, ngram_counts):
    """Best word-frequency threshold → predict complex (1) if freq < threshold."""
    best_threshold, best_f1 = 0, -1.0
    for threshold in range(0, 100_000_000, 100_000):
        preds = [1 if ngram_counts.get(w, 0) < threshold else 0 for w in train_words]
        f1 = get_fscore(train_labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    eval_predictions = [1 if ngram_counts.get(w, 0) < best_threshold else 0 for w in eval_words]
    return eval_predictions, best_threshold


try:
    train_pred_wf, best_freq_threshold = word_frequency_threshold(
        train_words, train_labels, train_words, ngram_counts)
    dev_pred_wf, _ = word_frequency_threshold(
        train_words, train_labels, dev_words, ngram_counts)
    wf_train_p = get_precision(train_labels, train_pred_wf)
    wf_train_r = get_recall(train_labels, train_pred_wf)
    wf_train_f = get_fscore(train_labels, train_pred_wf)
    wf_dev_p   = get_precision(dev_labels, dev_pred_wf)
    wf_dev_r   = get_recall(dev_labels, dev_pred_wf)
    wf_dev_f   = get_fscore(dev_labels, dev_pred_wf)
    print(f"\n=== Section 2.3: Word Frequency Baseline (best threshold = {best_freq_threshold:,}) ===")
    print(f"  Training — P: {wf_train_p:.4f}, R: {wf_train_r:.4f}, F1: {wf_train_f:.4f}")
    print(f"  Dev      — P: {wf_dev_p:.4f}, R: {wf_dev_r:.4f}, F1: {wf_dev_f:.4f}")
except Exception as e:
    print(f"WARNING: Skipped 2.3: {e}")
    wf_train_p = wf_train_r = wf_train_f = 0.0
    wf_dev_p = wf_dev_r = wf_dev_f = 0.0
    best_freq_threshold = 0

# P-R curve for word frequency baseline (50 sample points)
try:
    sample_thresholds = list(range(0, 100_000_000, 2_000_000))
    pr_freq_precisions, pr_freq_recalls = [], []
    for t in sample_thresholds:
        preds = [1 if ngram_counts.get(w, 0) < t else 0 for w in train_words]
        pr_freq_precisions.append(get_precision(train_labels, preds))
        pr_freq_recalls.append(get_recall(train_labels, preds))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(pr_freq_recalls, pr_freq_precisions, 'r-o', markersize=4, linewidth=1.5,
            label='Word Frequency Baseline')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve: Word Frequency Baseline\n(Training Data)', fontsize=12)
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pr_curve_word_frequency.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved pr_curve_word_frequency.png")
except Exception as e:
    print(f"WARNING: Frequency P-R curve skipped: {e}")


# ============================================================
# Section 3: Feature Engineering & Classifiers (22 points)
# ============================================================

def get_features(words, ngram_counts):
    """
    Extract 3 raw features per word:
      0: word length, 1: word frequency, 2: syllable count
    Returns np.ndarray of shape (len(words), 3)
    """
    features = []
    for word in words:
        length = len(word)
        freq   = ngram_counts.get(word, 0)
        syl    = syllables.estimate(word)
        features.append([length, freq, syl])
    return np.array(features, dtype=float)


def normalize_features(X_train, X_eval=None):
    """
    Z-score normalization using TRAINING statistics only.
    Returns (X_train_scaled, means, stds) or (X_train_scaled, X_eval_scaled, means, stds)
    """
    means = X_train.mean(axis=0)
    stds  = X_train.std(axis=0)
    stds[stds == 0] = 1.0   # avoid division by zero

    X_train_scaled = (X_train - means) / stds

    if X_eval is None:
        return X_train_scaled, means, stds
    else:
        X_eval_scaled = (X_eval - means) / stds
        return X_train_scaled, X_eval_scaled, means, stds


try:
    X_train_raw = get_features(train_words, ngram_counts)
    X_dev_raw   = get_features(dev_words, ngram_counts)

    X_train, X_dev, means, stds = normalize_features(X_train_raw, X_dev_raw)
    Y_train = np.array(train_labels)
    Y_dev   = np.array(dev_labels)

    print(f"\nTraining features shape: {X_train.shape}")
    print(f"Dev features shape:      {X_dev.shape}")
    print(f"Feature means (training): {means}")
    print(f"Feature stds  (training): {stds}")
    assert X_train.shape == (len(train_words), 3)
    assert np.allclose(X_train.mean(axis=0), 0, atol=1e-10)
    print("✓ Feature shapes and normalization look correct!")
except Exception as e:
    print(f"WARNING: Feature setup skipped: {e}")
    X_train = X_dev = Y_train = Y_dev = None
    means = stds = None


# --- 3.1 Naive Bayes ---

def naive_bayes(X_train, Y_train, X_eval):
    """Train GaussianNB; return (train_predictions, eval_predictions)."""
    clf = GaussianNB()
    clf.fit(X_train, Y_train)
    return clf.predict(X_train), clf.predict(X_eval)


try:
    train_pred_nb, dev_pred_nb = naive_bayes(X_train, Y_train, X_dev)
    nb_train_p = get_precision(train_labels, train_pred_nb)
    nb_train_r = get_recall(train_labels, train_pred_nb)
    nb_train_f = get_fscore(train_labels, train_pred_nb)
    nb_dev_p   = get_precision(dev_labels, dev_pred_nb)
    nb_dev_r   = get_recall(dev_labels, dev_pred_nb)
    nb_dev_f   = get_fscore(dev_labels, dev_pred_nb)
    print("\n=== Section 3.1: Naive Bayes ===")
    print(f"  Training — P: {nb_train_p:.4f}, R: {nb_train_r:.4f}, F1: {nb_train_f:.4f}")
    print(f"  Dev      — P: {nb_dev_p:.4f}, R: {nb_dev_r:.4f}, F1: {nb_dev_f:.4f}")
except Exception as e:
    print(f"WARNING: Skipped NB: {e}")
    nb_train_p = nb_train_r = nb_train_f = 0.0
    nb_dev_p = nb_dev_r = nb_dev_f = 0.0


# --- 3.2 Logistic Regression ---

def logistic_regression(X_train, Y_train, X_eval):
    """Train LogisticRegression; return (train_predictions, eval_predictions)."""
    clf = LogisticRegression(solver='lbfgs', max_iter=1000)
    clf.fit(X_train, Y_train)
    return clf.predict(X_train), clf.predict(X_eval)


lr_clf = None  # keep reference for Section 6 weight inspection
try:
    train_pred_lr, dev_pred_lr = logistic_regression(X_train, Y_train, X_dev)
    lr_train_p = get_precision(train_labels, train_pred_lr)
    lr_train_r = get_recall(train_labels, train_pred_lr)
    lr_train_f = get_fscore(train_labels, train_pred_lr)
    lr_dev_p   = get_precision(dev_labels, dev_pred_lr)
    lr_dev_r   = get_recall(dev_labels, dev_pred_lr)
    lr_dev_f   = get_fscore(dev_labels, dev_pred_lr)
    print("\n=== Section 3.2: Logistic Regression ===")
    print(f"  Training — P: {lr_train_p:.4f}, R: {lr_train_r:.4f}, F1: {lr_train_f:.4f}")
    print(f"  Dev      — P: {lr_dev_p:.4f}, R: {lr_dev_r:.4f}, F1: {lr_dev_f:.4f}")

    # Inspect weights (for Section 6.2)
    lr_clf = LogisticRegression(solver='lbfgs', max_iter=1000)
    lr_clf.fit(X_train, Y_train)
    feature_names_3 = ['word_length', 'word_frequency', 'syllable_count']
    coefs = lr_clf.coef_[0]
    print(f"\nLR feature weights: {dict(zip(feature_names_3, coefs))}")
    print(f"  Intercept: {lr_clf.intercept_[0]:.4f}")
    idx_max = int(np.argmax(coefs))
    idx_min = int(np.argmin(coefs))
    print(f"  Largest positive (->complex): {feature_names_3[idx_max]} = {coefs[idx_max]:.4f}")
    print(f"  Largest negative (->simple):  {feature_names_3[idx_min]} = {coefs[idx_min]:.4f}")
    lr_coefs = dict(zip(feature_names_3, coefs))
    lr_intercept = lr_clf.intercept_[0]
except Exception as e:
    print(f"WARNING: Skipped LR: {e}")
    lr_train_p = lr_train_r = lr_train_f = 0.0
    lr_dev_p = lr_dev_r = lr_dev_f = 0.0
    lr_coefs = {}
    lr_intercept = 0.0


# ============================================================
# Section 4: Build Your Own Model (25 points)
# ============================================================

# Academic/Latin/Greek complex word suffixes
COMPLEX_SUFFIXES = (
    'tion', 'sion', 'ment', 'ance', 'ence', 'ity', 'ous', 'ious',
    'eous', 'ful', 'less', 'ness', 'ize', 'ise', 'ify', 'ate',
    'ology', 'ography', 'ician', 'arium', 'ism', 'ist',
    'phy', 'sis', 'tic', 'tive', 'ary', 'ery',
    'ction', 'xion', 'ssion', 'ation'
)

VOWELS = set('aeiou')


def get_custom_features(words, ngram_counts):
    """
    Extract 6 features per word for the custom model:

      0: word length (characters)
      1: log1p(word frequency) -- log-scale to reduce outlier dominance
      2: syllable count
      3: vowel ratio (proportion of vowels in word)
      4: has complex academic/Latin suffix (binary)
      5: number of consonant clusters (sequences of 2+ consonants)

    Feature design rationale:
      - Feature 1 (log freq): Raw frequency spans 0-billions; log compresses the scale
        so that the difference between 100 and 1000 is weighted more than 1M vs 1.1M.
        This matches how human perception of word familiarity works.
      - Feature 3 (vowel ratio): Complex technical/borrowed words often have unusual
        vowel-consonant patterns; lower vowel ratios indicate consonant-heavy structure.
      - Feature 4 (complex suffix): Latin/Greek morphology (-tion, -ence, -ity) is
        a strong signal of academic/complex vocabulary rarely used in everyday speech.
      - Feature 5 (consonant clusters): Words with many consecutive consonants (e.g.,
        "strength", "transcript") tend to be less common and more complex.

    Returns np.ndarray of shape (len(words), 6)
    """
    features = []
    for word in words:
        length    = len(word)
        freq      = ngram_counts.get(word, 0)
        log_freq  = float(np.log1p(freq))
        syl       = syllables.estimate(word)

        # Vowel ratio
        n_vowels = sum(1 for c in word if c in VOWELS)
        vowel_ratio = n_vowels / length if length > 0 else 0.0

        # Has complex academic suffix
        has_suf = int(any(word.endswith(suf) for suf in COMPLEX_SUFFIXES))

        # Consonant cluster count
        clusters = re.findall(r'[^aeiou]{2,}', word)
        n_clusters = len(clusters)

        features.append([length, log_freq, syl, vowel_ratio, has_suf, n_clusters])
    return np.array(features, dtype=float)


custom_means = custom_stds = None
gb_clf = None
custom_dev_pred = []
cm_train_p = cm_train_r = cm_train_f = 0.0
cm_dev_p   = cm_dev_r   = cm_dev_f   = 0.0

try:
    print("\n=== Section 4: Custom Model (GradientBoosting + 6 features) ===")
    X_custom_train_raw = get_custom_features(train_words, ngram_counts)
    X_custom_dev_raw   = get_custom_features(dev_words, ngram_counts)

    X_custom_train, X_custom_dev, custom_means, custom_stds = normalize_features(
        X_custom_train_raw, X_custom_dev_raw)

    print(f"Custom feature matrix shape (train): {X_custom_train.shape}")

    gb_clf = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        random_state=42
    )
    gb_clf.fit(X_custom_train, Y_train)

    custom_train_pred = gb_clf.predict(X_custom_train)
    custom_dev_pred   = gb_clf.predict(X_custom_dev)

    cm_train_p = get_precision(train_labels, custom_train_pred)
    cm_train_r = get_recall(train_labels, custom_train_pred)
    cm_train_f = get_fscore(train_labels, custom_train_pred)
    cm_dev_p   = get_precision(dev_labels, custom_dev_pred)
    cm_dev_r   = get_recall(dev_labels, custom_dev_pred)
    cm_dev_f   = get_fscore(dev_labels, custom_dev_pred)

    print(f"  Training — P: {cm_train_p:.4f}, R: {cm_train_r:.4f}, F1: {cm_train_f:.4f}")
    print(f"  Dev      — P: {cm_dev_p:.4f}, R: {cm_dev_r:.4f}, F1: {cm_dev_f:.4f}")

    # Feature importances
    custom_feat_names = ['word_length', 'log_frequency', 'syllable_count',
                          'vowel_ratio', 'has_complex_suffix', 'consonant_clusters']
    importances = gb_clf.feature_importances_
    print("\nFeature importances:")
    for nm, imp in sorted(zip(custom_feat_names, importances), key=lambda x: -x[1]):
        print(f"  {nm}: {imp:.4f}")

    # Ablation study
    print("\nAblation study (features added incrementally to baseline):")
    base_feats_train = X_custom_train[:, :3]
    base_feats_dev   = X_custom_dev[:, :3]
    base_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    base_clf.fit(base_feats_train, Y_train)
    base_dev_pred = base_clf.predict(base_feats_dev)
    base_f1 = get_fscore(dev_labels, base_dev_pred)
    print(f"  Baseline (length+log_freq+syl):         Dev F1 = {base_f1:.4f}")
    for fi, fn in enumerate(custom_feat_names[3:], start=3):
        cols = list(range(3)) + list(range(3, fi+1))
        clf_tmp = GradientBoostingClassifier(n_estimators=100, random_state=42)
        clf_tmp.fit(X_custom_train[:, cols], Y_train)
        pred_tmp = clf_tmp.predict(X_custom_dev[:, cols])
        print(f"  + {fn:30s}: Dev F1 = {get_fscore(dev_labels, pred_tmp):.4f}")

except Exception as e:
    print(f"WARNING: Custom model skipped: {type(e).__name__}: {e}")
    import traceback; traceback.print_exc()

# --- 4.2 Error Analysis ---
try:
    print("\n=== Section 4.2: Error Analysis ===")
    false_positives = [(w, i) for i, (w, t, p) in
                       enumerate(zip(dev_words, dev_labels, custom_dev_pred))
                       if t == 0 and p == 1]
    false_negatives = [(w, i) for i, (w, t, p) in
                       enumerate(zip(dev_words, dev_labels, custom_dev_pred))
                       if t == 1 and p == 0]
    true_positives  = [(w, i) for i, (w, t, p) in
                       enumerate(zip(dev_words, dev_labels, custom_dev_pred))
                       if t == 1 and p == 1]
    true_negatives  = [(w, i) for i, (w, t, p) in
                       enumerate(zip(dev_words, dev_labels, custom_dev_pred))
                       if t == 0 and p == 0]

    print(f"True  Positives: {len(true_positives)}")
    print(f"True  Negatives: {len(true_negatives)}")
    print(f"False Positives: {len(false_positives)} -- predicted complex, actually simple")
    print(f"False Negatives: {len(false_negatives)} -- predicted simple, actually complex")
    print(f"\nFP examples (simple words mislabeled complex): {[w for w, _ in false_positives[:15]]}")
    print(f"FN examples (complex words missed):           {[w for w, _ in false_negatives[:15]]}")
    print(f"TP examples (correctly identified complex):    {[w for w, _ in true_positives[:10]]}")
    print(f"TN examples (correctly identified simple):     {[w for w, _ in true_negatives[:10]]}")
except Exception as e:
    print(f"WARNING: Error analysis skipped: {e}")
    false_positives = false_negatives = true_positives = true_negatives = []

# --- 4.3 Test Set Predictions ---
try:
    X_test_raw    = get_custom_features(test_words, ngram_counts)
    X_test_scaled = (X_test_raw - custom_means) / custom_stds
    test_pred     = gb_clf.predict(X_test_scaled)

    with open('test_labels.txt', 'w') as f:
        f.write('\n'.join(str(int(label)) for label in test_pred))

    print(f"\nSaved {len(test_pred)} predictions to test_labels.txt")
    print(f"  Complex: {int(sum(test_pred))} ({sum(test_pred)/len(test_pred):.1%})")
    print(f"  Simple:  {len(test_pred)-int(sum(test_pred))} ({(len(test_pred)-sum(test_pred))/len(test_pred):.1%})")
except Exception as e:
    print(f"WARNING: Test predictions skipped: {e}")


# ============================================================
# Section 5: Domain Generalization (10 points)
# ============================================================

def load_external_dataset(filepath):
    """Load external CWI dataset TSV. Returns (words, labels)."""
    words, labels = [], []
    with open(filepath, 'rt', encoding='utf-8') as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            words.append(parts[0].lower())
            labels.append(int(parts[-1]))  # binary label is last column
    return words, labels


bio_p = bio_r = bio_f = 0.0
news_p = news_r = news_f = 0.0
bio_fn_examples = bio_fp_examples = []
news_fn_examples = news_fp_examples = []
biomed_words_count = news_words_count = 0
pct_bio = pct_news = 0.0

try:
    biomed_words, biomed_labels = load_external_dataset('data/complex_biomed_test.tsv')
    news_words,   news_labels   = load_external_dataset('data/cwi2018_news_test.tsv')

    biomed_words_count = len(biomed_words)
    news_words_count   = len(news_words)
    pct_bio  = sum(biomed_labels) / len(biomed_labels)
    pct_news = sum(news_labels)   / len(news_labels)

    print(f"\n=== Section 5.1: External Dataset Distributions ===")
    print(f"  Biomedical: {biomed_words_count} words, {pct_bio:.1%} complex")
    print(f"  News:       {news_words_count} words, {pct_news:.1%} complex")

    # 5.2 Apply best model without retraining
    X_bio_raw  = get_custom_features(biomed_words, ngram_counts)
    X_news_raw = get_custom_features(news_words,   ngram_counts)
    X_bio      = (X_bio_raw  - custom_means) / custom_stds
    X_news     = (X_news_raw - custom_means) / custom_stds

    bio_pred  = gb_clf.predict(X_bio)
    news_pred = gb_clf.predict(X_news)

    bio_p  = get_precision(biomed_labels, bio_pred)
    bio_r  = get_recall(biomed_labels, bio_pred)
    bio_f  = get_fscore(biomed_labels, bio_pred)
    news_p = get_precision(news_labels, news_pred)
    news_r = get_recall(news_labels, news_pred)
    news_f = get_fscore(news_labels, news_pred)

    print(f"\n=== Section 5.2: Domain Generalization Results ===")
    print(f"{'Dataset':20s} {'P':>8s} {'R':>8s} {'F1':>8s}")
    print(f"{'HW01 Dev':20s} {cm_dev_p:>8.4f} {cm_dev_r:>8.4f} {cm_dev_f:>8.4f}")
    print(f"{'Biomedical':20s} {bio_p:>8.4f} {bio_r:>8.4f} {bio_f:>8.4f}")
    print(f"{'News':20s} {news_p:>8.4f} {news_r:>8.4f} {news_f:>8.4f}")

    bio_fn_examples  = [w for w, t, p in zip(biomed_words, biomed_labels, bio_pred)
                        if t == 1 and p == 0][:8]
    bio_fp_examples  = [w for w, t, p in zip(biomed_words, biomed_labels, bio_pred)
                        if t == 0 and p == 1][:8]
    news_fn_examples = [w for w, t, p in zip(news_words, news_labels, news_pred)
                        if t == 1 and p == 0][:8]
    news_fp_examples = [w for w, t, p in zip(news_words, news_labels, news_pred)
                        if t == 0 and p == 1][:8]

    print(f"\nBiomedical FN (missed complex): {bio_fn_examples}")
    print(f"Biomedical FP (false complex):  {bio_fp_examples}")
    print(f"News FN (missed complex):       {news_fn_examples}")
    print(f"News FP (false complex):        {news_fp_examples}")

except Exception as e:
    print(f"WARNING: Section 5 skipped: {type(e).__name__}: {e}")
    import traceback; traceback.print_exc()


# ============================================================
# Section 6 Reference Summary
# ============================================================

print("\n" + "="*60)
print("REFERENCE SUMMARY (for report writeup)")
print("="*60)
print(f"\n--- All-Complex Baseline ---")
print(f"  Train P={ac_train_p:.4f} R={ac_train_r:.4f} F1={ac_train_f:.4f}")
print(f"  Dev   P={ac_dev_p:.4f} R={ac_dev_r:.4f} F1={ac_dev_f:.4f}")

print(f"\n--- Word Length Baseline (threshold={best_len_threshold}) ---")
print(f"  Train P={wl_train_p:.4f} R={wl_train_r:.4f} F1={wl_train_f:.4f}")
print(f"  Dev   P={wl_dev_p:.4f} R={wl_dev_r:.4f} F1={wl_dev_f:.4f}")

print(f"\n--- Word Frequency Baseline (threshold={best_freq_threshold:,}) ---")
print(f"  Train P={wf_train_p:.4f} R={wf_train_r:.4f} F1={wf_train_f:.4f}")
print(f"  Dev   P={wf_dev_p:.4f} R={wf_dev_r:.4f} F1={wf_dev_f:.4f}")

print(f"\n--- Naive Bayes ---")
print(f"  Train P={nb_train_p:.4f} R={nb_train_r:.4f} F1={nb_train_f:.4f}")
print(f"  Dev   P={nb_dev_p:.4f} R={nb_dev_r:.4f} F1={nb_dev_f:.4f}")

print(f"\n--- Logistic Regression ---")
print(f"  Train P={lr_train_p:.4f} R={lr_train_r:.4f} F1={lr_train_f:.4f}")
print(f"  Dev   P={lr_dev_p:.4f} R={lr_dev_r:.4f} F1={lr_dev_f:.4f}")
print(f"  LR weights: {lr_coefs}")
print(f"  LR intercept: {lr_intercept:.4f}")

print(f"\n--- Custom Model (GradientBoosting, 6 features) ---")
print(f"  Train P={cm_train_p:.4f} R={cm_train_r:.4f} F1={cm_train_f:.4f}")
print(f"  Dev   P={cm_dev_p:.4f} R={cm_dev_r:.4f} F1={cm_dev_f:.4f}")

print(f"\n--- Domain Generalization ---")
print(f"  Biomedical  P={bio_p:.4f} R={bio_r:.4f} F1={bio_f:.4f}")
print(f"  News        P={news_p:.4f} R={news_r:.4f} F1={news_f:.4f}")

print("\n===== ALL SECTIONS COMPLETE =====")
print("Output files:")
print("  - test_labels.txt")
print("  - pr_curve_word_length.png")
print("  - pr_curve_word_frequency.png")
