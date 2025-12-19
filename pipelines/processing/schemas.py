from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from uuid import uuid4


@dataclass
class Document:
    """Raw document as produced by a collector."""

    text: str
    source: str
    language: Optional[str] = None
    id: str = field(default_factory=lambda: str(uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessedDocument:
    """Document after normalization/translation/dedup/spam filtering."""

    text: str
    source: str
    language: str
    translated: bool = False
    id: str = field(default_factory=lambda: str(uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
