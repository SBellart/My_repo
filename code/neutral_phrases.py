from __future__ import annotations

DEFAULT_NEUTRAL_PHRASE = "I am a generic sentence"

LANGUAGE_NEUTRAL_PHRASES: dict[str, str] = {
    "en": DEFAULT_NEUTRAL_PHRASE,
}

DATASET_NEUTRAL_PHRASES: dict[str, str] = {
    "scifact": DEFAULT_NEUTRAL_PHRASE,
    "nfcorpus": DEFAULT_NEUTRAL_PHRASE,
    "fiqa": DEFAULT_NEUTRAL_PHRASE,
    "arguana": DEFAULT_NEUTRAL_PHRASE,
}


def resolve_neutral_phrase(
    *,
    dataset_key: str | None = None,
    language: str | None = None,
) -> str:
    normalized_dataset_key = (dataset_key or "").strip().lower()
    if normalized_dataset_key:
        dataset_phrase = DATASET_NEUTRAL_PHRASES.get(normalized_dataset_key)
        if dataset_phrase is not None:
            return dataset_phrase

    normalized_language = (language or "").strip().lower()
    if normalized_language == "en":
        return DEFAULT_NEUTRAL_PHRASE

    return DEFAULT_NEUTRAL_PHRASE
