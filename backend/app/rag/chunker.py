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


def _is_table_line(line: str) -> bool:
    """Return True if a line belongs to a pipe-delimited table or Markdown table."""
    stripped = line.strip()
    if not stripped:
        return False
    # Pipe-delimited content row: contains at least one '|' and some alphanumeric chars
    if '|' in stripped and re.search(r'[A-Za-z0-9\u4e00-\u9fff]', stripped):
        return True
    # Markdown separator row: |---|---|---|
    if re.match(r'^[\|\s\-:]+$', stripped) and '|' in stripped:
        return True
    return False


def _split_into_blocks(text: str) -> List[str]:
    """
    Split text into alternating prose and table blocks.
    A "table block" is a contiguous run of pipe-delimited lines.
    Returns a flat list of strings, each being one block.
    """
    lines = text.split('\n')
    blocks: List[str] = []
    current_lines: List[str] = []
    in_table = False

    for line in lines:
        is_tbl = _is_table_line(line)
        if is_tbl != in_table:
            # Transition: flush current block
            if current_lines:
                blocks.append('\n'.join(current_lines))
            current_lines = [line]
            in_table = is_tbl
        else:
            current_lines.append(line)

    if current_lines:
        blocks.append('\n'.join(current_lines))

    return blocks


def chunk_text(
    text: str,
    chunk_size: int = 2000,
    overlap: int = 120,
    min_chunk_size: int = 500,
) -> List[str]:
    """
    Split long text into chunks while preserving table structure.

    Rules:
    - Table blocks (pipe-delimited) are NEVER split mid-table.  If a table block
      exceeds `chunk_size`, it is emitted as a single oversized chunk rather than
      being sliced through the middle of rows.
    - Prose blocks use the original sliding-window algorithm.
    - Tiny trailing chunks (< min_chunk_size) are merged into the previous chunk.
    """
    normalized = _clean_text(text)
    if not normalized:
        return []

    # Fast path: short text fits in one chunk
    if len(normalized) <= chunk_size:
        return [normalized]

    blocks = _split_into_blocks(normalized)
    chunks: List[str] = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Detect if this block is a table
        first_non_empty = next((l for l in block.split('\n') if l.strip()), "")
        block_is_table = _is_table_line(first_non_empty)

        if block_is_table:
            # Keep the entire table block as ONE chunk (never split mid-table)
            if chunks and len(chunks[-1]) + len(block) + 1 < min_chunk_size * 2:
                # Small table: merge with previous chunk to avoid tiny orphans
                chunks[-1] = chunks[-1] + '\n' + block
            else:
                chunks.append(block)
        else:
            # Prose block: use original sliding-window algorithm
            block_len = len(block)
            if block_len <= chunk_size:
                # Small prose: try to merge with previous prose chunk
                if chunks and not _is_table_line(
                    next((l for l in chunks[-1].split('\n') if l.strip()), "")
                ) and len(chunks[-1]) + block_len + 1 <= chunk_size:
                    chunks[-1] = chunks[-1] + '\n' + block
                else:
                    chunks.append(block)
            else:
                # Sliding-window for long prose
                start = 0
                while start < block_len:
                    end = min(start + chunk_size, block_len)
                    chunk = block[start:end].strip()
                    if chunk:
                        if len(chunk) >= min_chunk_size or end == block_len or not chunks:
                            chunks.append(chunk)
                        elif chunks:
                            chunks[-1] = chunks[-1] + '\n' + chunk
                    if end >= block_len:
                        break
                    start = max(end - overlap, start + 1)

    return [c for c in chunks if c.strip()]