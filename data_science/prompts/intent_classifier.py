"""
Intent Classifier for Dynamic Prompting

Classifies user messages into task types using weighted keyword/regex rules.
Pure Python — no LLM call, <1ms execution.
"""

import re
from dataclasses import dataclass, field

TASK_TYPES = (
    "data_query",
    "analysis",
    "visualization",
    "comparison",
    "geographic",
    "policy",
    "general",
)

INCLUSION_THRESHOLD = 0.3
GENERAL_CEILING = 0.4  # general only wins if nothing else scores above this


@dataclass
class IntentResult:
    """Result of intent classification."""

    primary: str
    detected: list[str]
    confidence: float


@dataclass
class _Rule:
    """A single keyword or regex rule with a weight."""

    pattern: re.Pattern
    weight: float


@dataclass
class _TaskRules:
    """All rules for a single task type."""

    task_type: str
    rules: list[_Rule] = field(default_factory=list)


def _kw(word: str, weight: float = 1.0) -> _Rule:
    """Create a whole-word keyword rule."""
    return _Rule(pattern=re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE), weight=weight)


def _rx(pattern: str, weight: float = 1.0) -> _Rule:
    """Create a regex rule."""
    return _Rule(pattern=re.compile(pattern, re.IGNORECASE), weight=weight)


# ---------------------------------------------------------------------------
# Rule definitions per task type
# ---------------------------------------------------------------------------

_RULES: list[_TaskRules] = [
    _TaskRules(
        "data_query",
        [
            _kw("show", 0.5),
            _kw("list", 0.8),
            _kw("count", 1.0),
            _kw("fetch", 1.0),
            _kw("get", 0.4),
            _kw("total", 0.6),
            _kw("number of", 0.8),
            _kw("data", 0.5),
            _kw("records", 0.8),
            _kw("query", 0.9),
            _kw("sql", 1.2),
            _kw("table", 0.6),
            _kw("select", 0.8),
            _rx(r"\bhow\s+many\b", 1.2),
            _rx(r"\bwhat\s+is\s+the\b", 0.6),
            _rx(r"\bgive\s+me\b", 0.5),
            _rx(r"\bwhat\s+are\b", 0.5),
            _rx(r"\bbreak\s*down\b", 0.7),
        ],
    ),
    _TaskRules(
        "analysis",
        [
            _kw("trend", 1.2),
            _kw("trends", 1.2),
            _kw("analyze", 1.2),
            _kw("analyse", 1.2),
            _kw("analysis", 1.2),
            _kw("correlate", 1.2),
            _kw("correlation", 1.2),
            _kw("statistics", 1.0),
            _kw("statistical", 1.0),
            _kw("growth", 0.9),
            _kw("decline", 0.9),
            _kw("change", 0.5),
            _kw("increase", 0.6),
            _kw("decrease", 0.6),
            _kw("pattern", 0.8),
            _kw("forecast", 1.0),
            _kw("predict", 0.9),
            _kw("regression", 1.2),
            _kw("outlier", 1.0),
            _kw("anomaly", 1.0),
            _rx(r"\bover\s+time\b", 1.0),
            _rx(r"\bmonth[\s-]+over[\s-]+month\b", 1.2),
            _rx(r"\byear[\s-]+over[\s-]+year\b", 1.2),
            _rx(r"\brate\s+of\s+change\b", 1.0),
        ],
    ),
    _TaskRules(
        "visualization",
        [
            _kw("chart", 1.2),
            _kw("graph", 1.2),
            _kw("plot", 1.0),
            _kw("visualize", 1.2),
            _kw("visualise", 1.2),
            _kw("visualization", 1.2),
            _kw("dashboard", 0.9),
            _kw("heatmap", 1.0),
            _kw("bar chart", 1.2),
            _kw("line chart", 1.2),
            _kw("pie chart", 1.2),
            _kw("kpi", 0.9),
            _rx(r"\b(show|display|draw)\s+(me\s+)?a\s+(chart|graph|plot|bar|line|pie)\b", 1.5),
            _rx(r"\bmecdm_stat\b", 1.5),
            _rx(r"\bmecdm_viz\b", 1.5),
        ],
    ),
    _TaskRules(
        "comparison",
        [
            _kw("compare", 1.2),
            _kw("comparison", 1.2),
            _kw("rank", 1.0),
            _kw("ranking", 1.0),
            _kw("top", 0.7),
            _kw("bottom", 0.7),
            _kw("best", 0.7),
            _kw("worst", 0.8),
            _kw("better", 0.6),
            _kw("worse", 0.6),
            _kw("highest", 0.8),
            _kw("lowest", 0.8),
            _kw("versus", 1.0),
            _rx(r"\bvs\.?\b", 1.0),
            _rx(r"\bdifference\s+between\b", 1.2),
            _rx(r"\bhow\s+does\s+.+\s+compare\b", 1.5),
            _rx(r"\bwhich\s+(district|block|village)s?\s+(is|are|has|have)\s+(the\s+)?(highest|lowest|most|least|best|worst)\b", 1.5),
        ],
    ),
    _TaskRules(
        "geographic",
        [
            _kw("map", 1.2),
            _kw("nearest", 1.2),
            _kw("closest", 1.2),
            _kw("distance", 1.0),
            _kw("location", 0.8),
            _kw("spatial", 1.2),
            _kw("facility", 0.7),
            _kw("facilities", 0.7),
            _kw("choropleth", 1.5),
            _kw("bubble map", 1.5),
            _kw("geographic", 1.2),
            _kw("geographical", 1.2),
            _rx(r"\bwhere\s+is\b", 0.8),
            _rx(r"\bwhere\s+are\b", 0.8),
            _rx(r"\bnear\s+(me|my|this|the)\b", 1.0),
            _rx(r"\bon\s+(a\s+)?map\b", 1.2),
            _rx(r"\bshow\s+on\s+map\b", 1.5),
            _rx(r"\bmecdm_map\b", 1.5),
            _rx(r"\bnearest\s+(phc|chc|sc|dh|awc|hospital|clinic|health\s+center)\b", 1.5),
        ],
    ),
    _TaskRules(
        "policy",
        [
            _kw("policy", 1.2),
            _kw("policies", 1.2),
            _kw("recommendation", 1.0),
            _kw("recommendations", 1.0),
            _kw("recommend", 1.0),
            _kw("guideline", 1.0),
            _kw("guidelines", 1.0),
            _kw("scheme", 0.9),
            _kw("program", 0.6),
            _kw("programme", 0.6),
            _kw("intervention", 1.0),
            _kw("nfhs", 1.2),
            _kw("sdg", 1.0),
            _kw("target", 0.5),
            _kw("benchmark", 0.8),
            _rx(r"\bwhat\s+should\s+(we|i|the)\b", 1.0),
            _rx(r"\bhow\s+to\s+improve\b", 1.0),
            _rx(r"\baction\s+plan\b", 1.2),
            _rx(r"\bbest\s+practice\b", 1.0),
        ],
    ),
    _TaskRules(
        "general",
        [
            _kw("hello", 1.0),
            _kw("hi", 0.8),
            _kw("hey", 0.8),
            _kw("help", 0.9),
            _kw("thanks", 0.8),
            _kw("thank you", 0.8),
            _rx(r"\bwhat\s+can\s+you\s+do\b", 1.5),
            _rx(r"\bwho\s+are\s+you\b", 1.5),
            _rx(r"\byour\s+capabilities\b", 1.2),
            _rx(r"\bgood\s+(morning|afternoon|evening)\b", 1.0),
        ],
    ),
]


class IntentClassifier:
    """Classifies user messages into task types using keyword/regex rules."""

    def __init__(self) -> None:
        self._task_rules = _RULES

    def classify(self, message: str) -> IntentResult:
        """Classify a user message into one or more task types.

        Returns an IntentResult with the primary (highest-scoring) type,
        all detected types above the inclusion threshold, and the
        confidence score for the primary type.
        """
        if not message or not message.strip():
            return IntentResult(primary="general", detected=["general"], confidence=1.0)

        scores: dict[str, float] = {}
        for task_rules in self._task_rules:
            score = 0.0
            for rule in task_rules.rules:
                if rule.pattern.search(message):
                    score += rule.weight
            if score > 0:
                scores[task_rules.task_type] = score

        if not scores:
            return IntentResult(primary="general", detected=["general"], confidence=1.0)

        # Normalize scores to 0-1 range using the max possible score
        max_score = max(scores.values())
        normalized = {k: v / max_score for k, v in scores.items()}

        # Filter by threshold and sort by score descending
        detected = sorted(
            [k for k, v in normalized.items() if v >= INCLUSION_THRESHOLD],
            key=lambda k: normalized[k],
            reverse=True,
        )

        if not detected:
            return IntentResult(primary="general", detected=["general"], confidence=1.0)

        primary = detected[0]

        # If general is the only strong contender but other types scored,
        # prefer the non-general type
        if primary == "general" and len(detected) > 1:
            non_general = [d for d in detected if d != "general"]
            if non_general and normalized[non_general[0]] >= GENERAL_CEILING:
                primary = non_general[0]
                detected = [primary] + [d for d in detected if d != primary]

        return IntentResult(
            primary=primary,
            detected=detected,
            confidence=normalized[primary],
        )
