import re
from typing import List


def _clean_text(text: str) -> str:
    if not text:
        return ""

    cleaned = text.replace("\x00", "").replace("\ufffd", "")
    cleaned = re.sub(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F]", "", cleaned)
    cleaned = re.sub(r"[ \t\r]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 120, min_chunk_size: int = 500) -> List[str]:
    """
    Split long text into chunks that are roughly 500-1000 characters.
    Uses a sliding window with overlap so adjacent chunks keep some context.
    """
    normalized = _clean_text(text)
    if not normalized:
        return []

    if len(normalized) <= chunk_size:
        return [normalized]

    chunks: List[str] = []
    start = 0
    text_len = len(normalized)

    while start < text_len:
      end = min(start + chunk_size, text_len)
      chunk = normalized[start:end].strip()

      if chunk:
        # Prefer to avoid tiny trailing chunks unless we are at the end.
        if len(chunk) >= min_chunk_size or end == text_len or not chunks:
          chunks.append(chunk)
        elif chunks:
          chunks[-1] = f"{chunks[-1]}\n{chunk}"

      if end >= text_len:
        break

      start = max(end - overlap, start + 1)

    return chunks