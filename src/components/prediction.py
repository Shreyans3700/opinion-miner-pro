"""Class-based prediction component for single-sentence sentiment inference."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import spacy
import yaml

from src.components.data_preprocessing import DataPreprocessing
from src.config.data_preprocessing_config import DataPreprocessingConfig
from src.config.predict_config import PredictConfig
from src.utils.exception import CustomException
from src.utils.logger import logging


class Prediction:
    """Run text preprocessing, embedding, and model inference for one sentence."""

    def __init__(
        self,
        predict_config: PredictConfig,
        preprocessing_config: DataPreprocessingConfig | None = None,
    ) -> None:
        self.predict_config = predict_config
        self.preprocessing_config = preprocessing_config or DataPreprocessingConfig()
        self._preprocessor = DataPreprocessing(self.preprocessing_config)
        self._vectorizer: Any | None = None
        self._model: Any | None = None
        self._label_encoder: Any | None = None
        self._keyword_config = self._load_keyword_config()
        self._nlp = self._load_spacy_model()

    def run(self, sentence: str) -> dict[str, Any]:
        """Predict sentiment from input sentence and return observations."""
        try:
            if not isinstance(sentence, str):
                raise ValueError("Input sentence must be a string.")
            if sentence.strip() == "":
                raise ValueError("Input sentence cannot be empty.")

            self._ensure_artifacts_loaded()

            cleaned_text = self._preprocessor.preprocess_text(sentence)
            if cleaned_text.strip() == "":
                raise ValueError(
                    "Sentence became empty after preprocessing. Provide richer text."
                )
            tokens = cleaned_text.split()

            embedding = self._vectorizer.transform([cleaned_text])
            predicted = self._model.predict(embedding)[0]
            predicted_label = self._decode_label(predicted)

            observations = {
                "input_text": sentence,
                "cleaned_text": cleaned_text,
                "tokens": tokens,
                "token_count": len(tokens),
                "embedding_shape": [int(embedding.shape[0]), int(embedding.shape[1])],
                "predicted_label": predicted_label,
                "predicted_raw": self._to_python_scalar(predicted),
            }
            observations.update(self._probability_or_score_observations(embedding))
            sentiment_breakdown = self._build_sentiment_breakdown(observations)
            if sentiment_breakdown:
                observations["sentiment_breakdown"] = sentiment_breakdown
                observations["positive_score"] = float(
                    sentiment_breakdown.get("positive", 0.0)
                )
                observations["negative_score"] = float(
                    sentiment_breakdown.get("negative", 0.0)
                )
            observations["overall_sentiment"] = self._apply_neutral_threshold(
                predicted_label, observations
            )
            observations.update(self._predict_aspect_sentiments(sentence))
            observations["word_sentiments"] = self._predict_word_sentiments(tokens)
            return observations
        except Exception as error:
            raise CustomException(error, sys) from error

    def _ensure_artifacts_loaded(self) -> None:
        """Load and cache model/vectorizer/label-encoder artifacts."""
        if self._model is not None and self._vectorizer is not None:
            return

        required = [
            self.predict_config.model_path,
            self.predict_config.vectorizer_path,
        ]
        for artifact_path in required:
            if not Path(artifact_path).exists():
                raise FileNotFoundError(
                    f"Required prediction artifact not found: {artifact_path}"
                )

        self._model = joblib.load(self.predict_config.model_path)
        self._vectorizer = joblib.load(self.predict_config.vectorizer_path)

        if Path(self.predict_config.label_encoder_path).exists():
            self._label_encoder = joblib.load(self.predict_config.label_encoder_path)
        else:
            self._label_encoder = None
            logging.warning(
                "Label encoder not found at '%s'; returning raw predicted labels.",
                self.predict_config.label_encoder_path,
            )

    def _load_keyword_config(self) -> dict[str, Any]:
        """Load sentiment/aspect keyword rules from YAML with safe defaults."""
        defaults: dict[str, Any] = {
            "positive_cues": ["good", "great", "excellent", "love"],
            "negative_cues": ["bad", "worst", "poor", "terrible"],
            "positive_weights": {},
            "negative_weights": {},
            "forced_negative_any": [],
            "forced_negative_pairs": [],
            "forced_negative_score": 0.82,
            "forced_pair_negative_score": 0.72,
            "neutral_confidence_threshold": 0.6,
            "banned_aspect_tokens": [],
            "preferred_features": [],
            "bug_features": [],
        }

        config_path = (
            Path(__file__).resolve().parents[1] / "config" / "sentiment_keywords.yaml"
        )
        if not config_path.exists():
            logging.warning(
                "Keyword config YAML not found at '%s'. Using defaults.", config_path
            )
            return defaults

        try:
            parsed = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            if not isinstance(parsed, dict):
                logging.warning(
                    "Keyword config YAML is not a mapping. Using defaults: %s",
                    config_path,
                )
                return defaults
        except Exception as error:
            logging.warning(
                "Failed to parse keyword config YAML '%s': %s. Using defaults.",
                config_path,
                error,
            )
            return defaults

        merged = dict(defaults)
        for key, value in parsed.items():
            merged[key] = value
        return merged

    @staticmethod
    def _load_spacy_model():
        """Load spaCy English model, downloading if needed."""
        model_name = "en_core_web_sm"
        try:
            return spacy.load(model_name, disable=["ner", "textcat"])
        except OSError:
            from spacy.cli import download

            download(model_name)
            return spacy.load(model_name, disable=["ner", "textcat"])

    def _decode_label(self, predicted_value: Any) -> str:
        """Decode model output to original class label if encoder is available."""
        if self._label_encoder is None:
            return str(self._to_python_scalar(predicted_value))
        decoded = self._label_encoder.inverse_transform([predicted_value])[0]
        return str(decoded)

    def _probability_or_score_observations(self, embedding: Any) -> dict[str, Any]:
        """Return confidence observations from predict_proba or decision_function."""
        if hasattr(self._model, "predict_proba"):
            probabilities = self._model.predict_proba(embedding)[0]
            max_idx = int(np.argmax(probabilities))
            class_map = self._class_probability_map(probabilities)
            return {
                "confidence": float(probabilities[max_idx]),
                "class_probabilities": class_map,
            }

        if hasattr(self._model, "decision_function"):
            decision = self._model.decision_function(embedding)
            if np.ndim(decision) == 1:
                margin = float(decision[0])
            else:
                margin = float(np.max(decision[0]))
            return {
                "decision_margin": margin,
            }

        return {}

    def _class_probability_map(self, probabilities: np.ndarray) -> dict[str, float]:
        """Map probability vector to human-readable class names."""
        if self._label_encoder is not None and hasattr(self._label_encoder, "classes_"):
            class_names = [str(label) for label in self._label_encoder.classes_]
            if len(class_names) == len(probabilities):
                return {
                    class_names[idx]: float(probabilities[idx])
                    for idx in range(len(probabilities))
                }

        return {
            str(idx): float(probabilities[idx]) for idx in range(len(probabilities))
        }

    def _build_sentiment_breakdown(
        self, observations: dict[str, Any]
    ) -> dict[str, float] | None:
        """Build explicit positive/negative scores for easier frontend rendering."""
        class_probabilities = observations.get("class_probabilities")
        if isinstance(class_probabilities, dict) and class_probabilities:
            positive_key = self._match_label_key(
                class_probabilities, ("positive", "pos", "1")
            )
            negative_key = self._match_label_key(
                class_probabilities, ("negative", "neg", "0")
            )
            if positive_key is not None and negative_key is not None:
                return {
                    "positive": float(class_probabilities[positive_key]),
                    "negative": float(class_probabilities[negative_key]),
                }

        margin = observations.get("decision_margin")
        if margin is not None:
            positive_score = self._sigmoid(float(margin))
            negative_score = 1.0 - positive_score
            return {
                "positive": float(positive_score),
                "negative": float(negative_score),
            }

        return None

    @staticmethod
    def _match_label_key(
        score_map: dict[str, float], candidates: tuple[str, ...]
    ) -> str | None:
        lowered = {str(key).lower(): key for key in score_map.keys()}
        for candidate in candidates:
            for lowered_key, original_key in lowered.items():
                if lowered_key == candidate or candidate in lowered_key:
                    return original_key
        return None

    @staticmethod
    def _sigmoid(value: float) -> float:
        return float(1.0 / (1.0 + np.exp(-value)))

    def _predict_aspect_sentiments(self, sentence: str) -> dict[str, Any]:
        """Extract aspect-opinion pairs using spaCy dependency parsing."""
        doc = self._nlp(sentence)
        positive_cues, negative_cues = self._sentiment_cue_sets()
        all_cues = positive_cues | negative_cues
        banned = set(self._keyword_config.get("banned_aspect_tokens", []))
        preferred = set(self._keyword_config.get("preferred_features", []))

        # Extract (aspect_token, opinion_span) pairs from dependency tree
        aspect_opinions: list[tuple[str, str]] = []
        for token in doc:
            # Look for nouns/proper nouns that are subjects or objects
            if token.pos_ not in ("NOUN", "PROPN"):
                continue
            if token.text.lower() in banned:
                continue
            if len(token.text) < 3:
                continue

            # Find opinion modifiers attached to this noun
            opinion_tokens = []
            # Direct adjectival modifiers (amod): "great camera"
            for child in token.children:
                if child.dep_ in ("amod", "acomp") and child.pos_ == "ADJ":
                    opinion_tokens.append(child.text.lower())
            # Check if noun is subject of a copular/verb construction
            # e.g., "battery is terrible" -> battery(nsubj) -> is -> terrible(acomp)
            if token.dep_ in ("nsubj", "nsubjpass"):
                head = token.head
                for child in head.children:
                    if child.dep_ in ("acomp", "attr") and child.pos_ == "ADJ":
                        opinion_tokens.append(child.text.lower())
                    # "battery drains quickly" -> verb itself is the signal
                    if child == token:
                        continue
                if head.pos_ == "VERB" and head.lemma_.lower() in all_cues:
                    opinion_tokens.append(head.lemma_.lower())

            if opinion_tokens:
                aspect_opinions.append((token.lemma_.lower(), " ".join(opinion_tokens)))
            elif token.lemma_.lower() in preferred:
                # Preferred feature with no direct modifier — use surrounding clause
                aspect_opinions.append((token.lemma_.lower(), ""))

        # Deduplicate aspects, keeping first occurrence
        seen_aspects: set[str] = set()
        unique_pairs: list[tuple[str, str]] = []
        for aspect, opinion in aspect_opinions:
            if aspect not in seen_aspects:
                seen_aspects.add(aspect)
                unique_pairs.append((aspect, opinion))

        # If spaCy found nothing, fall back to simple clause splitting
        if not unique_pairs:
            return self._fallback_clause_aspects(sentence)

        aspect_sentiments: dict[str, str] = {}
        aspect_sentiment_scores: dict[str, dict[str, float]] = {}
        main_feature_points: list[dict[str, Any]] = []

        for aspect, opinion in unique_pairs:
            # Determine sentiment from opinion words + model
            context = opinion if opinion else aspect
            cleaned = self._preprocessor.preprocess_text(context)
            if not cleaned:
                cleaned = aspect

            clause_embedding = self._vectorizer.transform([cleaned])
            predicted = self._model.predict(clause_embedding)[0]
            label = self._decode_label(predicted)
            detail = self._probability_or_score_observations(clause_embedding)
            breakdown = self._build_sentiment_breakdown(detail)
            label, breakdown = self._apply_clause_polarity_rules(
                cleaned, label, breakdown
            )

            aspect_sentiments[aspect] = label
            if breakdown:
                aspect_sentiment_scores[aspect] = {
                    "positive": float(breakdown.get("positive", 0.0)),
                    "negative": float(breakdown.get("negative", 0.0)),
                }
            main_feature_points.append(
                {
                    "feature": aspect,
                    "sentiment": label,
                    "evidence": opinion or aspect,
                    "scores": (
                        {
                            "positive": float(breakdown.get("positive", 0.0)),
                            "negative": float(breakdown.get("negative", 0.0)),
                        }
                        if breakdown
                        else None
                    ),
                }
            )

        return {
            "aspect_sentiments": aspect_sentiments,
            "aspect_sentiment_scores": aspect_sentiment_scores,
            "main_feature_points": main_feature_points,
        }

    def _fallback_clause_aspects(self, sentence: str) -> dict[str, Any]:
        """Fallback: split on conjunctions when spaCy finds no aspect-opinion pairs."""
        clause_candidates = [
            part.strip()
            for part in re.split(
                r"\b(?:but|however|though|although|whereas|while|yet)\b|[;!?]|\.(?=\s+[A-Z])",
                sentence,
                flags=re.IGNORECASE,
            )
            if part and part.strip()
        ]
        if not clause_candidates:
            return {
                "aspect_sentiments": {},
                "aspect_sentiment_scores": {},
                "main_feature_points": [],
            }

        aspect_sentiments: dict[str, str] = {}
        aspect_sentiment_scores: dict[str, dict[str, float]] = {}
        main_feature_points: list[dict[str, Any]] = []

        for idx, clause in enumerate(clause_candidates, start=1):
            cleaned_clause = self._preprocessor.preprocess_text(clause)
            if not cleaned_clause or not self._has_sentiment_signal(cleaned_clause):
                continue

            aspect_name = self._extract_aspect_name(clause, cleaned_clause, idx)
            clause_embedding = self._vectorizer.transform([cleaned_clause])
            predicted = self._model.predict(clause_embedding)[0]
            label = self._decode_label(predicted)
            detail = self._probability_or_score_observations(clause_embedding)
            breakdown = self._build_sentiment_breakdown(detail)
            label, breakdown = self._apply_clause_polarity_rules(
                cleaned_clause, label, breakdown
            )

            aspect_sentiments[aspect_name] = label
            if breakdown:
                aspect_sentiment_scores[aspect_name] = {
                    "positive": float(breakdown.get("positive", 0.0)),
                    "negative": float(breakdown.get("negative", 0.0)),
                }
            main_feature_points.append(
                {
                    "feature": aspect_name,
                    "sentiment": label,
                    "evidence": clause.strip(),
                    "scores": (
                        {
                            "positive": float(breakdown.get("positive", 0.0)),
                            "negative": float(breakdown.get("negative", 0.0)),
                        }
                        if breakdown
                        else None
                    ),
                }
            )

        return {
            "aspect_sentiments": aspect_sentiments,
            "aspect_sentiment_scores": aspect_sentiment_scores,
            "main_feature_points": main_feature_points,
        }

    def _predict_word_sentiments(self, tokens: list[str]) -> list[dict[str, Any]]:
        """Return sentiment probabilities for each token in sequence."""
        word_sentiments: list[dict[str, Any]] = []

        for idx, token in enumerate(tokens):
            token_embedding = self._vectorizer.transform([token])
            predicted_raw = self._model.predict(token_embedding)[0]
            predicted_label = self._decode_label(predicted_raw)
            detail = self._probability_or_score_observations(token_embedding)
            breakdown = self._build_sentiment_breakdown(detail)

            if breakdown is None:
                breakdown = self._scores_from_label(predicted_label)

            word_sentiments.append(
                {
                    "position": idx,
                    "word": token,
                    "predicted_label": predicted_label,
                    "positive": float(breakdown.get("positive", 0.0)),
                    "negative": float(breakdown.get("negative", 0.0)),
                }
            )

        return word_sentiments

    def _extract_aspect_name(
        self, clause: str, cleaned_clause: str, clause_index: int
    ) -> str:
        """Pick a stable aspect token from a clause using lightweight heuristics."""
        raw_tokens = re.findall(r"[a-zA-Z]+", clause.lower())
        cleaned_tokens = [token.lower() for token in cleaned_clause.split()]
        stop_words = getattr(self._preprocessor, "_stop_words", set())
        banned_tokens = set(self._keyword_config.get("banned_aspect_tokens", []))
        preferred_features = set(self._keyword_config.get("preferred_features", []))
        bug_features = set(self._keyword_config.get("bug_features", []))

        filtered_tokens: list[str] = []
        for token in raw_tokens:
            if token in stop_words:
                continue
            if token in banned_tokens:
                continue
            if len(token) < 3:
                continue
            if token.endswith("ly"):
                continue
            filtered_tokens.append(token)

        if not filtered_tokens:
            filtered_tokens = [
                token
                for token in cleaned_tokens
                if token not in banned_tokens and len(token) >= 3
            ]

        if not filtered_tokens:
            return f"aspect_{clause_index}"

        for token in reversed(filtered_tokens):
            if token in bug_features:
                return token

        preferred_hits = [
            token for token in filtered_tokens if token in preferred_features
        ]
        if preferred_hits:
            return preferred_hits[-1]

        return filtered_tokens[-1]

    def _apply_clause_polarity_rules(
        self,
        cleaned_clause: str,
        label: str,
        breakdown: dict[str, float] | None,
    ) -> tuple[str, dict[str, float] | None]:
        """Adjust clause label using polarity cue words for aspect-level output."""
        positive_cues, negative_cues = self._sentiment_cue_sets()
        tokens = cleaned_clause.split()
        token_set = set(tokens)
        forced_negative_any = set(self._keyword_config.get("forced_negative_any", []))
        forced_negative_pairs = self._keyword_config.get("forced_negative_pairs", [])
        forced_negative_score = float(
            self._keyword_config.get("forced_negative_score", 0.82)
        )
        forced_pair_negative_score = float(
            self._keyword_config.get("forced_pair_negative_score", 0.72)
        )

        # Explicitly force common failure modes like "battery draining fast" to negative.
        if forced_negative_any.intersection(token_set):
            if breakdown:
                negative_score = max(
                    float(breakdown.get("negative", 0.0)), forced_negative_score
                )
                return "negative", {
                    "positive": float(1.0 - negative_score),
                    "negative": float(negative_score),
                }
            return "negative", {
                "positive": float(1.0 - forced_negative_score),
                "negative": forced_negative_score,
            }

        if any(set(pair).issubset(token_set) for pair in forced_negative_pairs):
            if breakdown:
                negative_score = max(
                    float(breakdown.get("negative", 0.0)),
                    forced_pair_negative_score,
                )
                return "negative", {
                    "positive": float(1.0 - negative_score),
                    "negative": float(negative_score),
                }
            return "negative", {
                "positive": float(1.0 - forced_pair_negative_score),
                "negative": forced_pair_negative_score,
            }

        # Weighted cues avoid ties where a weak positive token ("fast")
        # co-exists with stronger negative signal.
        positive_weights_raw = self._keyword_config.get("positive_weights", {})
        negative_weights_raw = self._keyword_config.get("negative_weights", {})
        positive_weights = {
            str(key): float(value) for key, value in positive_weights_raw.items()
        }
        negative_weights = {
            str(key): float(value) for key, value in negative_weights_raw.items()
        }

        positive_score_cues = sum(
            positive_weights.get(token, 1.0)
            for token in tokens
            if token in positive_cues
        )
        negative_score_cues = sum(
            negative_weights.get(token, 1.0)
            for token in tokens
            if token in negative_cues
        )

        if positive_score_cues == negative_score_cues:
            # No clear signal — check if model confidence is also low
            threshold = float(
                self._keyword_config.get("neutral_confidence_threshold", 0.6)
            )
            if breakdown:
                max_score = max(
                    breakdown.get("positive", 0), breakdown.get("negative", 0)
                )
                if max_score < threshold:
                    return "neutral", breakdown
            return label, breakdown

        if negative_score_cues > positive_score_cues:
            if breakdown:
                negative_score = max(float(breakdown.get("negative", 0.0)), 0.7)
                return "negative", {
                    "positive": float(1.0 - negative_score),
                    "negative": float(negative_score),
                }
            return "negative", {"positive": 0.3, "negative": 0.7}

        if breakdown:
            positive_score = max(float(breakdown.get("positive", 0.0)), 0.7)
            return "positive", {
                "positive": float(positive_score),
                "negative": float(1.0 - positive_score),
            }
        return "positive", {"positive": 0.7, "negative": 0.3}

    def _has_sentiment_signal(self, cleaned_clause: str) -> bool:
        """Check whether a clause contains explicit sentiment cues."""
        positive_cues, negative_cues = self._sentiment_cue_sets()
        tokens = set(cleaned_clause.split())
        if tokens.intersection(positive_cues):
            return True
        if tokens.intersection(negative_cues):
            return True
        return False

    def _sentiment_cue_sets(self) -> tuple[set[str], set[str]]:
        positive_cues = set(self._keyword_config.get("positive_cues", []))
        negative_cues = set(self._keyword_config.get("negative_cues", []))
        return positive_cues, negative_cues

    def _apply_neutral_threshold(self, label: str, observations: dict[str, Any]) -> str:
        """Return 'neutral' if confidence is below the configured threshold."""
        threshold = float(self._keyword_config.get("neutral_confidence_threshold", 0.6))
        confidence = observations.get("confidence")
        if confidence is not None and confidence < threshold:
            return "neutral"
        # Fallback: check sentiment breakdown spread
        breakdown = observations.get("sentiment_breakdown")
        if breakdown and abs(
            breakdown.get("positive", 0) - breakdown.get("negative", 0)
        ) < (1 - threshold):
            return "neutral"
        return label

    @staticmethod
    def _scores_from_label(label: str) -> dict[str, float]:
        lowered = str(label).lower()
        if lowered == "positive":
            return {"positive": 1.0, "negative": 0.0}
        if lowered == "negative":
            return {"positive": 0.0, "negative": 1.0}
        return {"positive": 0.5, "negative": 0.5}

    @staticmethod
    def _to_python_scalar(value: Any) -> Any:
        """Convert numpy scalar values to native Python types."""
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                return value
        return value
