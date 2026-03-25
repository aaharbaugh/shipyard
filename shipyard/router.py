from __future__ import annotations

from typing import Any


def route_message(
    current_text: str,
    interaction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    interaction = interaction or {}
    model_result = _try_tiny_router(current_text, interaction)
    if model_result is not None:
        routed = dict(model_result)
        routed["provider"] = "tiny_router"
        routed["priority_score"] = _priority_score(routed)
        return routed

    routed = _heuristic_route(current_text, interaction)
    routed["provider"] = "heuristic"
    routed["priority_score"] = _priority_score(routed)
    return routed


def _try_tiny_router(current_text: str, interaction: dict[str, Any]) -> dict[str, Any] | None:
    try:
        from tiny_router.runtime import predict_message  # type: ignore
    except Exception:
        return None

    try:
        result = predict_message({"current_text": current_text, "interaction": interaction})
    except Exception:
        return None
    if not isinstance(result, dict):
        return None
    return result


def _heuristic_route(current_text: str, interaction: dict[str, Any]) -> dict[str, Any]:
    text = (current_text or "").strip().lower()
    relation = "new"
    if any(token in text for token in ("actually", "instead", "correction", "fix that", "change that")):
        relation = "correction"
    elif any(token in text for token in ("also", "and then", "next", "follow up", "follow-up")):
        relation = "follow_up"
    elif any(token in text for token in ("yes", "ok", "looks good", "confirm")):
        relation = "confirmation"
    elif any(token in text for token in ("cancel", "never mind", "stop")):
        relation = "cancellation"
    elif any(token in text for token in ("done", "thanks", "that works")):
        relation = "closure"

    actionability = "act"
    if any(token in text for token in ("thoughts", "plan", "discuss", "what do you think")):
        actionability = "review"
    if any(token in text for token in ("thanks", "cool", "nice")) and len(text.split()) < 6:
        actionability = "none"

    retention = "useful"
    if any(token in text for token in ("tmp", "temporary", "for now", "testing")):
        retention = "ephemeral"
    if any(token in text for token in ("remember", "always", "from now on")):
        retention = "remember"

    urgency = "low"
    if any(token in text for token in ("today", "now", "urgent", "asap", "immediately")):
        urgency = "high"
    elif any(token in text for token in ("soon", "next", "when you can")):
        urgency = "medium"

    return {
        "relation_to_previous": {"label": relation, "confidence": 0.68},
        "actionability": {"label": actionability, "confidence": 0.74},
        "retention": {"label": retention, "confidence": 0.66},
        "urgency": {"label": urgency, "confidence": 0.71},
        "overall_confidence": 0.69,
    }


def _priority_score(result: dict[str, Any]) -> int:
    urgency = result.get("urgency", {}).get("label", "low")
    actionability = result.get("actionability", {}).get("label", "review")
    relation = result.get("relation_to_previous", {}).get("label", "new")
    score = {"low": 1, "medium": 2, "high": 3}.get(urgency, 1) * 10
    score += {"none": 0, "review": 2, "act": 4}.get(actionability, 1)
    if relation == "correction":
        score += 3
    elif relation == "follow_up":
        score += 1
    return score
