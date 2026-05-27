"""
classifier.py — Active-learning email classifier

Lifecycle:
  Zone 1 (< MIN_SAMPLES_PER_CLASS per category) : always asks the user
  Zone 2 (trained but confidence < HIGH_CONF)    : asks only for uncertain emails
  Zone 3 (confidence >= HIGH_CONF)               : acts silently, logs decision

TF-IDF + Logistic Regression pipeline
"""

import json
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np

# ── scikit-learn imports──────────────────
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

CATEGORIES = ["travel", "bills", "jobs", "personal", "ads"]

# Confidence thresholds
LOW_CONF   = 0.50   
HIGH_CONF  = 0.80   

# Minimum labelled examples per category before prediction
MIN_SAMPLES_PER_CLASS = 3


class ActiveLearningClassifier:

    def __init__(self, data_dir: Path):
        self.data_dir   = Path(data_dir)
        self.model_path = self.data_dir / "model.pkl"
        self.labels_path = self.data_dir / "labelled_emails.json"
        self.stats_path  = self.data_dir / "training_stats.json"

        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.labelled: list[dict] = self._load_labels()
        self.pipeline = self._load_model()
        self.stats    = self._load_stats()

    # persistence 

    def _load_labels(self) -> list:
        if self.labels_path.exists():
            try:
                content = self.labels_path.read_text(encoding="utf-8").strip()

                if not content:
                    return []

                return json.loads(content)

            except Exception:
                return []

        return []

    def _save_labels(self):
        self.labels_path.write_text(
            json.dumps(self.labelled, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    def _load_model(self):
        if not SKLEARN_OK:
            return None
        if self.model_path.exists():
            with open(self.model_path, "rb") as f:
                return pickle.load(f)
        return None

    def _save_model(self):
        if self.pipeline is None:
            return
        with open(self.model_path, "wb") as f:
            pickle.dump(self.pipeline, f)

    def _load_stats(self) -> dict:
        if self.stats_path.exists():
            return json.loads(self.stats_path.read_text(encoding="utf-8"))
        return {
            "total_labelled":    0,
            "total_auto":        0,
            "total_asked":       0,
            "corrections":       0,
            "accuracy_estimate": None,
            "per_category":      {c: 0 for c in CATEGORIES},
        }

    def _save_stats(self):
        self.stats_path.write_text(
            json.dumps(self.stats, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    # feature engineering 

    @staticmethod
    def email_to_text(email_summary: dict) -> str:
        """
        Combine sender domain, subject, and body into a single string.
        Sender domain is repeated 3× to give it more weight than body words.
        """
        sender = email_summary.get("from", "")
        # Extract domain from  "Name <user@domain.com>"  or  "user@domain.com"
        domain_match = re.search(r"@([\w.\-]+)", sender)
        domain = domain_match.group(1).lower() if domain_match else ""
        domain_tokens = " ".join(domain.replace(".", " ").split()) + " "

        subject = email_summary.get("subject", "")
        body    = email_summary.get("body",    "")[:400]

        return f"{domain_tokens} {domain_tokens} {domain_tokens} {subject} {body}"

    # training 

    def _class_distribution(self) -> Counter:
        return Counter(e["category"] for e in self.labelled)

    def _can_train(self) -> bool:
        if not SKLEARN_OK or len(self.labelled) < MIN_SAMPLES_PER_CLASS * 2:
            return False
        dist = self._class_distribution()
        seen = [c for c in CATEGORIES if dist.get(c, 0) >= MIN_SAMPLES_PER_CLASS]
        return len(seen) >= 2   # need at least 2 classes

    def train(self, force: bool = False) -> bool:

        if not self._can_train() and not force:
            return False

        texts  = [self.email_to_text(e) for e in self.labelled]
        labels = [e["category"] for e in self.labelled]

        # Filter out categories with < MIN_SAMPLES_PER_CLASS
        dist    = Counter(labels)
        valid   = {c for c in CATEGORIES if dist.get(c, 0) >= MIN_SAMPLES_PER_CLASS}
        paired  = [(t, l) for t, l in zip(texts, labels) if l in valid]
        if not paired:
            return False

        texts_f, labels_f = zip(*paired)

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer      = "word",
                ngram_range   = (1, 2),
                max_features  = 8000,
                sublinear_tf  = True,
                min_df        = 1,
                strip_accents = "unicode",
                lowercase     = True,
            )),
            ("clf", LogisticRegression(
                max_iter  = 1000,
                C         = 2.0,
                class_weight = "balanced",   # handles imbalanced category counts
                solver    = "lbfgs",
            )),
        ])
        self.pipeline.fit(texts_f, labels_f)
        self._save_model()

        # Simple leave-one-out accuracy estimate on training set
        from sklearn.model_selection import cross_val_score

        if len(paired) >= 10:
            cv_scores = cross_val_score(self.pipeline, texts_f, labels_f, cv=min(5, len(paired)//2))
            self.stats["accuracy_estimate"] = round(float(cv_scores.mean()), 3)

        self.stats["total_labelled"] = len(self.labelled)
        self._save_stats()
        return True

    # prediction 

    def predict(self, email_summary: dict) -> dict:

        dist = self._class_distribution()
        total = sum(dist.values())

        # Zone 1
        if not self._can_train() or self.pipeline is None:
            needed = max(0, MIN_SAMPLES_PER_CLASS * 2 - total)
            return {
                "category":   "unknown",
                "confidence": 0.0,
                "zone":       1,
                "should_ask": True,
                "reason":     f"Learning phase — need ~{needed} more labelled examples",
                "top_probs":  {},
            }

        # Zone 2 / 3
        text   = self.email_to_text(email_summary)
        probs  = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_
        top_probs = {c: round(float(p), 3) for c, p in zip(classes, probs)}

        best_idx  = int(np.argmax(probs))
        category  = classes[best_idx]
        confidence = float(probs[best_idx])

        # Check if this sender's domain has ever been seen
        domain_match = re.search(r"@([\w.\-]+)", email_summary.get("from", ""))
        domain = domain_match.group(1).lower() if domain_match else ""
        seen_domains = {
            re.search(r"@([\w.\-]+)", e.get("from", "")).group(1).lower()
            for e in self.labelled
            if re.search(r"@([\w.\-]+)", e.get("from", ""))
        }
        is_new_sender = domain and domain not in seen_domains

        # Force Zone 2 for unseen senders 
        if is_new_sender:
            confidence = min(confidence, HIGH_CONF - 0.01)

        zone = 3 if confidence >= HIGH_CONF else (2 if confidence >= LOW_CONF else 1)
        should_ask = zone < 3

        return {
            "category":   category,
            "confidence": round(confidence, 3),
            "zone":       zone,
            "should_ask": should_ask,
            "reason":     f"{'New sender — ' if is_new_sender else ''}confidence {confidence:.0%}",
            "top_probs":  top_probs,
        }

    # labelling 

    def add_label(self, email_summary: dict, category: str, was_correction: bool = False):
        """Record a labelled example and retrain."""
        record = {
            "from":     email_summary.get("from", ""),
            "subject":  email_summary.get("subject", ""),
            "body":     email_summary.get("body", "")[:300],
            "category": category,
        }
        self.labelled.append(record)
        self._save_labels()

        self.stats["total_labelled"] += 1
        self.stats["per_category"][category] = self.stats["per_category"].get(category, 0) + 1
        if was_correction:
            self.stats["corrections"] += 1
        self._save_stats()

        # Retrain
        self.train()

    def record_auto_decision(self, category: str):
        self.stats["total_auto"] += 1
        self._save_stats()

    def record_asked(self):
        self.stats["total_asked"] += 1
        self._save_stats()

    # status report

    def status_report(self) -> dict:
        dist   = self._class_distribution()
        total  = sum(dist.values())
        acc    = self.stats.get("accuracy_estimate")
        needed = max(0, MIN_SAMPLES_PER_CLASS * 2 - total)

        lines = []
        if not self._can_train():
            lines.append(f"  Collecting training data — {total} labelled, ~{needed} more needed.")
        else:
            lines.append(f"  Model trained on {total} emails.")
            if acc:
                lines.append(f"  Estimated accuracy: {acc:.0%}")
            lines.append(f"  Auto-classified: {self.stats['total_auto']}  |  "
                         f"Asked you: {self.stats['total_asked']}  |  "
                         f"Corrections: {self.stats['corrections']}")
        lines.append(f"  Labels per category: " +
                     ", ".join(f"{c}={dist.get(c,0)}" for c in CATEGORIES))
        return {
            "summary": "\n".join(lines),
            "total_labelled": total,
            "can_train": self._can_train(),
            "accuracy": acc,
            "distribution": dict(dist),
        }
