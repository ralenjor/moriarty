import sys
import re
import yaml
import tiktoken
import lxml
import json
import nltk
import chromadb

from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from lxml import etree
from nltk.tokenize import sent_tokenize

# -------------------------
# Prerequisites Check
# -------------------------
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


# -------------------------
# Load Configuration
# -------------------------
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


config = load_config()
MODEL_NAME = config['model_settings']['model_name']
MAX_TOKENS = config['model_settings']['max_tokens']
OVERLAP = config['model_settings']['overlap_tokens']

# -------------------------
# Tokenizer
# -------------------------

# Llama 3.1 uses a tiktoken-based 128k vocab.
# cl100k_base is the closest widely available local match.

try:
    encoding = tiktoken.get_encoding("cl100k_base")
except Exception as e:
    # Extreme fallback if even tiktoken fails
    encoding = None


def token_count(text: str) -> int:
    if encoding:
        return len(encoding.encode(text))
    # Rough fallback: ~4 characters per token
    return len(text) // 4


# -------------------------
# Wiki section parsing
# -------------------------
SECTION_RE = re.compile(r"^(={2,6})\s*(.*?)\s*\1$", re.MULTILINE)


def split_by_sections(text):
    # Regex to find == Section Name ==
    parts = re.split(r'==\s*(.*?)\s*==', text)

    # If there are no headers, return the whole thing as "Introduction"
    if len(parts) == 1:
        return [("Introduction", parts[0])]

    # re.split with a capture group returns: [pre-text, title1, text1, title2, text2...]
    sections = []
    # Handle the very first part (before the first header)
    if parts[0].strip():
        sections.append(("Introduction", parts[0].strip()))

    # Group the rest into (title, text) pairs
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        sections.append((title, content))

    return sections


def classify_section(section_title: str) -> str:
    s = section_title.lower()
    if "appearance" in s:
        return "appearance"
    if "background" in s or "history" in s:
        return "background"
    if "note" in s:
        return "notes"
    if "apocrypha" in s:
        return "apocrypha"
    return "core"


# -------------------------
# Paragraph + sentence chunking
# -------------------------

def synopsis(text, max_sentences=1):
    sentences = sent_tokenize(text)
    return " ".join(sentences[:max_sentences]).strip()


def chunk_text_semantically(text, max_tokens):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if token_count(current + para) <= max_tokens:
            current += "\n\n" + para
        else:
            if current:
                chunks.extend(split_by_sentence(current, max_tokens))
            current = para

    if current:
        chunks.extend(split_by_sentence(current, max_tokens))

    return chunks


def split_by_sentence(text, max_tokens):
    sentences = sent_tokenize(text)
    chunks = []
    current_sentences = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = token_count(sent)
        # If a single sentence is longer than the max, we have to force it in
        # or it would break the loop

        if sent_tokens > max_tokens:
            # optional, could sub-split this
            if current_sentences:
                chunks.append(" ".join(current_sentences).strip())
                current_sentences = []
                current_tokens = 0
                # Note: This logic forces long sentences as independent chunks
                chunks.append(sent.strip())
                continue

        if current_tokens + sent_tokens <= max_tokens:
            current_sentences.append(sent)
            current_tokens += sent_tokens
        else:
            # 1. Save the current chunk
            chunks.append(" ".join(current_sentences).strip())
            # 2. Start the next chunk with overlap
            # 3. Backtrack through previous sentences until we hit the overlap limit
            overlap_sentences = []
            overlap_tokens = 0
            for prev_sent in reversed(current_sentences):
                prev_sent_tokens = token_count(prev_sent)
                if overlap_tokens + prev_sent_tokens <= OVERLAP:
                    overlap_sentences.insert(0, prev_sent)
                    overlap_tokens += prev_sent_tokens
                else:
                    break

            # New state starts with the overlap + current sentence
            current_sentences = overlap_sentences + [sent]
            current_tokens = overlap_tokens + sent_tokens

    # last batch
    if current_sentences:
        chunks.append(" ".join(current_sentences).strip())

    return chunks


# -------------------------
# XML streaming + chunking
# -------------------------
def chunk_memory_alpha(xml_path):
    # 1. Use a wildcard namespace or resilient tag matching
    context = etree.iterparse(xml_path, events=("end",))

    for _, elem in context:
        # Check if the tag is 'page' regardless of namespace
        if elem.tag.endswith('page'):
            # Resilient tag finding for title and text
            title_elem = elem.find('.//{*}title')
            title = title_elem.text if title_elem is not None else "Unknown Title"

            # Text is nested: page -> revision -> text
            text_elem = elem.find('.//{*}revision/{*}text')
            text = text_elem.text if text_elem is not None else ""

            if not text:
                elem.clear()
                continue

            # 4. Pass the valid text to section splitter
            sections = split_by_sections(text)

            for section_title, section_text in sections:
                chunks = split_by_sentence(section_text, MAX_TOKENS)

                # 5. Pass to sentence chunker
                for i, chunk_text in enumerate(chunks):
                    prev_text = chunks[i - 1] if i > 0 else None
                    next_text = chunks[i + 1] if i < len(chunks) - 1 else None

                    prev_synopsis = synopsis(prev_text) if prev_text else ""
                    next_synopsis = synopsis(next_text) if next_text else ""

                    # This is what gets embedded
                    enriched_text = (
                        f"Previous context: {prev_synopsis}\n\n"
                        f"Current content:\n{chunk_text}\n\n"
                        f"Next context: {next_synopsis}"
                    ).strip()

                    yield {
                        "text": enriched_text,
                        "raw_text": chunk_text,  # optional but recommended
                        "metadata": {
                            "title": title,
                            "section": section_title,
                            "chunk_index": i,
                            "has_prev": bool(prev_text),
                            "has_next": bool(next_text)
                        }
                    }

            # 6. Memory Management: Clean up the element after processing
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]


# -------------------------
# Entity Extraction Regex
# -------------------------
ENTITY_RE = re.compile(
    r'\b((?:USS\s+)?[A-Z][A-Za-z0-9\-]+(?:\s+[A-Z][A-Za-z0-9\-]+)+)\b'
)


def extract_entities(text: str) -> list[str]:
    return list(set(ENTITY_RE.findall(text)))


# -------------------------
# Main Execution Logic
# -------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python database_chunker.py <path_to_xml>")
        return

    xml_path = sys.argv[1]
    print(f"--- Processing: {xml_path} ---")
    print(f"--- Model: {MODEL_NAME} | Max Tokens: {MAX_TOKENS} | Overlap: {OVERLAP} ---")

    # -------------------------
    # Initialize ChromaDB
    # -------------------------
    client = chromadb.PersistentClient(path="./star_trek_db")
    embedding_function = ONNXMiniLM_L6_V2()

    collection = client.get_or_create_collection(
        name="memory_alpha",
        embedding_function=embedding_function
    )

    # -------------------------
    # Streaming Upload (Memory Efficient)
    # -------------------------
    batch_docs = []
    batch_metadatas = []
    batch_ids = []
    batch_size = 100  # Lowered for stability on local machines
    total_count = 0

    print("Streaming data to ChromaDB...")

    for chunk in chunk_memory_alpha(xml_path):
        chunk_text = chunk.get("text", "")
        chunk_meta = chunk.get("metadata", {}).copy()

        # --- Extract entities and convert list to string for ChromaDB ---
        entities = extract_entities(chunk_text)
        chunk_meta["entities"] = ", ".join(entities)

        # --- Preserve raw text in metadata ---
        chunk_meta["raw_text"] = chunk.get("raw_text", "")

        # --- Add to batch ---
        batch_docs.append(chunk_text)
        batch_metadatas.append(chunk_meta)
        batch_ids.append(f"id_{total_count}")

        total_count += 1

        if len(batch_docs) >= batch_size:
            collection.add(
                documents=batch_docs,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            batch_docs = []
            batch_metadatas = []
            batch_ids = []
            print(f"Uploaded {total_count} chunks...")

    # Upload remaining batch
    if batch_docs:
        collection.add(
            documents=batch_docs,
            metadatas=batch_metadatas,
            ids=batch_ids
        )

    print(f"Success. Total chunks stored: {total_count}")


if __name__ == "__main__":
    main()