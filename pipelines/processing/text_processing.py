from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from typing import Iterable, List, Optional, Sequence

from langdetect import DetectorFactory, LangDetectException, detect

from .schemas import Document, ProcessedDocument

DetectorFactory.seed = 0

logger = logging.getLogger(__name__)


class OptionalTranslator:
    """Lightweight Marian/transformers-based translator with graceful fallback."""

    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-mul-en"):
        self.model_name = model_name
        self._pipeline = None
        try:
            from transformers import pipeline

            self._pipeline = pipeline("translation", model=model_name, tokenizer=model_name)
            logger.info("Translation model %s loaded", model_name)
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("Translation pipeline unavailable (%s). Falling back to original text.", exc)

    def translate(self, text: str) -> str:
        if not self._pipeline:
            return text
        try:
            result = self._pipeline(text, max_length=512)
            return result[0]["translation_text"]
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.error("Translation failed: %s", exc)
            return text


class TextProcessor:
    """Detect language, translate when needed, normalize, deduplicate, and filter spam."""

    def __init__(
        self,
        target_language: str = "en",
        enable_deduplication: bool = True,
        enable_spam_filter: bool = True,
        enable_normalization: bool = True,
        translator: Optional[OptionalTranslator] = None,
    ):
        self.target_language = target_language
        self.enable_deduplication = enable_deduplication
        self.enable_spam_filter = enable_spam_filter
        self.enable_normalization = enable_normalization
        self.translator = translator or OptionalTranslator()
        self._seen_hashes: set[str] = set()

    def _detect_language(self, text: str) -> Optional[str]:
        try:
            return detect(text)
        except LangDetectException:
            return None

    def _normalize_text(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        normalized = normalized.replace("“", '"').replace("”", '"').replace("’", "'")
        return normalized

    def _is_spam(self, text: str) -> bool:
        uppercase_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        link_count = len(re.findall(r"https?://", text))
        repeated_chars = re.search(r"(.)\1{5,}", text) is not None
        return uppercase_ratio > 0.6 or link_count > 5 or repeated_chars

    def _is_duplicate(self, text: str) -> bool:
        fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if fingerprint in self._seen_hashes:
            return True
        self._seen_hashes.add(fingerprint)
        return False

    def process(self, doc: Document) -> Optional[ProcessedDocument]:
        text = doc.text or ""
        if self.enable_normalization:
            text = self._normalize_text(text)

        language = doc.language or self._detect_language(text) or "unknown"
        translated = False
        if language != self.target_language:
            translated_text = self.translator.translate(text)
            translated = translated_text != text
            text = translated_text
            language = self.target_language if translated else language

        if self.enable_spam_filter and self._is_spam(text):
            logger.info("Spam filtered from %s", doc.source)
            return None

        if self.enable_deduplication and self._is_duplicate(text):
            logger.info("Duplicate filtered from %s", doc.source)
            return None

        return ProcessedDocument(
            text=text,
            source=doc.source,
            language=language,
            translated=translated,
            metadata=doc.metadata,
        )

    def process_many(self, docs: Sequence[Document]) -> List[ProcessedDocument]:
        processed: List[ProcessedDocument] = []
        for doc in docs:
            result = self.process(doc)
            if result:
                processed.append(result)
        return processed
