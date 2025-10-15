def clip_text(text: str, threshold: int = 100) -> str:
    if len(text) <= threshold:
        return text
    return text[:threshold] + "..."