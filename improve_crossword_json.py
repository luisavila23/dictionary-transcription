import os
import re
import json
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI

# =========================
# Configuration
# =========================
API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-v4-pro"

INPUT_FILE = "crossword-words.json"
OUTPUT_FILE = "crossword-words.improved.json"
CHECKPOINT_FILE = "crossword-words.checkpoint.json"

BATCH_SIZE = 25
MAX_RETRIES = 5
SLEEP_BETWEEN_BATCHES = 1.0
TEMPERATURE = 0.2

FALLBACK_TO_ORIGINAL = True


# =========================
# Prompts
# =========================
SYSTEM_PROMPT = """
You write polished game-ready word content for crossword and educational word search apps.

Your job is to improve weak dictionary-style text into natural, concise, mobile-friendly Spanish.

You must always return valid JSON only.

Writing style requirements:
- Natural human wording
- Clear simple Spanish
- Short, smooth, polished phrasing
- Educational but easy to read
- Great mobile UX copy
- Never sound like raw dictionary text
- Never sound mechanical or overly literal

Field rules:
- `hint`: short crossword-style clue, ideally 2 to 6 words
- `description`: exactly 1 sentence, smooth and easy to understand
- `details`: 1 or 2 short educational sentences with useful context
- `hint`, `description`, and `details` must be written in natural Spanish.

Critical safety rules:
1. Never include the answer word in the hint.
2. Never include the answer word in the description.
3. Avoid obvious singular/plural variants of the answer word in hint and description.
4. Do not use first-letter clues, spelling clues, or letter-count clues.
5. Do not reveal the answer too directly.
6. Do not invent doubtful facts.
7. If the source text is awkward, incomplete, or poorly written, rewrite it naturally while preserving the meaning.
8. If a safe and natural hint or description cannot be produced, make the safest possible rewrite without spoilers.
9. `details` may include the answer word if needed for clarity, but should still sound natural and educational.
10. Prefer short, elegant phrasing over literal paraphrasing.

Quality standard:
- Good output should feel like professionally written app content.
- Bad output sounds like broken dictionary cleanup.

Good example:
{
  "word": "marrón",
  "hint": "Color del chocolate",
  "description": "Es un color cálido y oscuro común en la madera, la tierra y el café.",
  "details": "El marrón es frecuente en la naturaleza y puede variar desde tonos claros hasta tonos oscuros."
}

Bad example:
{
  "word": "mes",
  "hint": "Período en que el año se divide históricamente",
  "description": "Período en que se divide el año, históricamente basado en las fases de la luna."
}

The bad example is too mechanical, too literal, and not polished enough.
The good example is natural, simple, smooth, and app-friendly.
""".strip()


# =========================
# Helpers
# =========================
def load_json(path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any] | None]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}") from e

    if isinstance(data, list):
        return data, None

    if isinstance(data, dict) and isinstance(data.get("words"), list):
        return data["words"], data.get("metadata")

    raise ValueError("Unsupported JSON format. Expected a list or an object with a 'words' array.")


def save_json(path: str, entries: List[Dict[str, Any]], metadata: Dict[str, Any] | None = None) -> None:
    payload: Any
    if metadata is None:
        payload = entries
    else:
        payload = {
            "metadata": metadata,
            "words": entries,
        }

    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def load_checkpoint(path: str) -> Dict[str, Any]:
    if not Path(path).exists():
        return {"last_completed_index": -1}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"[WARN] Checkpoint file '{path}' is invalid JSON. Starting from scratch.")
        return {"last_completed_index": -1}

    if not isinstance(data, dict):
        print(f"[WARN] Checkpoint file '{path}' has invalid format. Starting from scratch.")
        return {"last_completed_index": -1}
    return data


def save_checkpoint(path: str, last_completed_index: int) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump({"last_completed_index": last_completed_index}, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def normalize_spanish_word(word: str) -> str:
    w = normalize_text(str(word)).lower()
    if not w:
        return ""
    # Deterministic Spanish-friendly ASCII normalization.
    w = w.replace("ñ", "n").replace("Ñ", "n").replace("ü", "u").replace("Ü", "u")
    w = unicodedata.normalize("NFD", w)
    w = "".join(ch for ch in w if unicodedata.category(ch) != "Mn")
    return w


def singular_plural_variants(word: str) -> List[str]:
    w = word.lower().strip()
    variants = {w}

    if w.endswith("y") and len(w) > 2:
        variants.add(w[:-1] + "ies")
    if w.endswith("ies") and len(w) > 3:
        variants.add(w[:-3] + "y")

    if w.endswith("s"):
        variants.add(w[:-1])
    else:
        variants.add(w + "s")

    if w.endswith("es"):
        variants.add(w[:-2])
    else:
        variants.add(w + "es")

    if w.endswith("f"):
        variants.add(w[:-1] + "ves")
    if w.endswith("fe"):
        variants.add(w[:-2] + "ves")

    return sorted(v for v in variants if v)


def contains_answer_leak(word: str, text: str) -> bool:
    if not text:
        return True

    lowered = text.lower()
    for variant in singular_plural_variants(word):
        if re.search(rf"\b{re.escape(variant)}\b", lowered):
            return True
    return False


def ensure_period(sentence: str) -> str:
    sentence = sentence.strip()
    if not sentence:
        return sentence
    if sentence[-1] not in ".!?":
        sentence += "."
    return sentence


def clean_details(text: str) -> str:
    text = normalize_text(text)
    if not text:
        return text

    # Limit to 2 sentences max by rough split
    parts = re.split(r"(?<=[.!?])\s+", text)
    parts = [p.strip() for p in parts if p.strip()]
    parts = parts[:2]
    cleaned = " ".join(parts)
    return ensure_period(cleaned)


def sentence_count(text: str) -> int:
    if not text:
        return 0
    parts = re.split(r"(?<=[.!?])\s+", normalize_text(text))
    return len([p for p in parts if p.strip()])


def ensure_details(value: str, fallback_description: str) -> str:
    details = clean_details(value)
    if details:
        return details

    description = ensure_period(normalize_text(fallback_description))
    if description:
        return f"{description} Este término se usa con frecuencia en juegos educativos de palabras."

    return "No hay contexto educativo adicional disponible."


def details_quality_score(text: str) -> int:
    cleaned = clean_details(text)
    if not cleaned:
        return 0

    score = len(cleaned)
    sentences = sentence_count(cleaned)
    if 1 <= sentences <= 2:
        score += 20
    else:
        score -= 20

    lowered = cleaned.lower()
    if "general educational context is not available" in lowered:
        score -= 100
    if "no hay contexto educativo adicional disponible" in lowered:
        score -= 100

    return score


def is_clearly_better_details(original_details: str, candidate_details: str) -> bool:
    original_clean = clean_details(original_details)
    candidate_clean = clean_details(candidate_details)

    if not candidate_clean:
        return False
    if not original_clean:
        return True
    if candidate_clean.lower() == original_clean.lower():
        return False

    return details_quality_score(candidate_clean) >= details_quality_score(original_clean) + 25


def fallback_details_from_entry(word: str, category: str, description: str) -> str:
    normalized_description = ensure_period(normalize_text(description))
    normalized_category = normalize_text(category)
    normalized_word = normalize_text(word)

    if normalized_description and normalized_category:
        return f"{normalized_description} Este término se usa con frecuencia en juegos educativos de palabras de la categoría {normalized_category}."
    if normalized_description:
        return f"{normalized_description} Este término se usa con frecuencia en juegos educativos de palabras."
    if normalized_category:
        return f"Este término pertenece al vocabulario de {normalized_category} y aparece en juegos educativos de palabras."
    if normalized_word:
        return f"Este término se refiere a {normalized_word} y aparece con frecuencia en juegos educativos de palabras."
    return "No hay contexto educativo adicional disponible."


def choose_details(word: str, category: str, description: str, original_details: str, candidate_details: str) -> str:
    original_clean = clean_details(original_details)
    candidate_clean = clean_details(candidate_details)

    if original_clean:
        if is_clearly_better_details(original_clean, candidate_clean):
            return candidate_clean
        return original_clean

    if candidate_clean:
        return candidate_clean

    return clean_details(fallback_details_from_entry(word, category, description))


def ensure_details_last(entry: Dict[str, Any]) -> Dict[str, Any]:
    finalized = dict(entry)
    finalized["normalizedWord"] = normalize_spanish_word(str(finalized.get("word", "")))
    existing_details = finalized.pop("details", "")
    finalized["details"] = clean_details(str(existing_details))
    if not finalized["details"]:
        finalized["details"] = clean_details(
            fallback_details_from_entry(
                word=str(finalized.get("word", "")),
                category=str(finalized.get("category", "")),
                description=str(finalized.get("description", "")),
            )
        )
    return finalized


def validate_entry(original: Dict[str, Any], candidate: Dict[str, Any]) -> bool:
    word = str(original.get("word", "")).strip().lower()
    hint = normalize_text(str(candidate.get("hint", "")))
    description = normalize_text(str(candidate.get("description", "")))
    if not word or not hint or not description:
        return False

    if contains_answer_leak(word, hint):
        return False

    if contains_answer_leak(word, description):
        return False

    hint_word_count = len(hint.split())
    if hint_word_count < 2 or hint_word_count > 8:
        return False

    if len(description) < 20:
        return False

    if sentence_count(description) != 1:
        return False

    return True


def build_user_prompt(batch: List[Dict[str, Any]]) -> str:
    batch_payload = []
    for idx, item in enumerate(batch):
        batch_payload.append(
            {
                "index": idx,
                "word": item.get("word", ""),
                "hint": item.get("hint", ""),
                "description": item.get("description", ""),
                "details": item.get("details", ""),
                "difficulty": item.get("difficulty", ""),
                "category": item.get("category", ""),
            }
        )

    return f"""
Rewrite the following entries into polished game-ready JSON.

Return JSON only with this structure:

{{
  "items": [
    {{
      "index": 0,
      "word": "string",
      "hint": "string",
      "description": "string",
      "details": "string",
      "difficulty": "string",
      "category": "string"
    }}
  ]
}}

Instructions:
- Keep `index`, `word`, `difficulty`, and `category` unchanged.
- Rewrite only `hint`, `description`, and `details`.
- Input words and source content are in Spanish.
- Final `hint`, `description`, and `details` must be in natural Spanish.
- `hint` must be short, natural, and clue-like.
- `description` must be exactly 1 sentence.
- `details` must be 1 or 2 short educational sentences.
- `hint` must not contain the answer word.
- `description` must not contain the answer word.
- Avoid awkward dictionary-style phrasing.
- Use clear, polished, human-sounding Spanish.
- Keep the meaning accurate.
- Do not invent uncertain facts.
- If the original entry is weak, improve it significantly for fluency and UX.

Input items:
{json.dumps(batch_payload, ensure_ascii=False, indent=2)}
""".strip()


def call_deepseek(client: OpenAI, batch: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    response = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(batch)},
        ],
    )

    if not response.choices or not response.choices[0].message:
        raise ValueError("Model response did not include a valid choice/message.")

    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Model response content is empty.")

    parsed = json.loads(content)

    if not isinstance(parsed, dict) or not isinstance(parsed.get("items"), list):
        raise ValueError("Model response did not contain an 'items' array.")

    return parsed["items"]


def improve_batch(client: OpenAI, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            generated_items = call_deepseek(client, batch)
            indexed: Dict[int, Dict[str, Any]] = {}
            for item in generated_items:
                if not isinstance(item, dict):
                    continue
                try:
                    idx = int(item.get("index"))
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < len(batch):
                    indexed[idx] = item

            improved: List[Dict[str, Any]] = []

            for idx, original in enumerate(batch):
                candidate = indexed.get(idx, {})
                merged = dict(original)

                original_hint = normalize_text(str(original.get("hint", "")))
                original_description = ensure_period(normalize_text(str(original.get("description", ""))))

                candidate_hint = normalize_text(str(candidate.get("hint", "")))
                if candidate_hint:
                    merged["hint"] = candidate_hint
                else:
                    merged["hint"] = original_hint

                candidate_description = ensure_period(normalize_text(str(candidate.get("description", ""))))
                if candidate_description:
                    merged["description"] = candidate_description
                else:
                    merged["description"] = original_description

                merged["details"] = choose_details(
                    word=str(original.get("word", "")),
                    category=str(original.get("category", "")),
                    description=merged["description"] or original_description,
                    original_details=str(original.get("details", "")),
                    candidate_details=str(candidate.get("details", "")),
                )
                merged["difficulty"] = original.get("difficulty", "")
                merged["category"] = original.get("category", "")
                merged["word"] = original.get("word", "")
                merged = ensure_details_last(merged)

                if validate_entry(original, merged):
                    improved.append(merged)
                else:
                    if FALLBACK_TO_ORIGINAL:
                        fallback = dict(original)
                        fallback["hint"] = original_hint
                        fallback["description"] = original_description
                        fallback["details"] = choose_details(
                            word=str(original.get("word", "")),
                            category=str(original.get("category", "")),
                            description=original_description,
                            original_details=str(original.get("details", "")),
                            candidate_details="",
                        )
                        improved.append(ensure_details_last(fallback))
                    else:
                        improved.append(ensure_details_last(original))

            return improved

        except Exception as e:
            wait_time = min(2 ** attempt, 30)
            print(f"[WARN] Batch failed on attempt {attempt}/{MAX_RETRIES}: {e}")
            if attempt == MAX_RETRIES:
                print("[WARN] Using original batch after max retries.")
                fallback_batch: List[Dict[str, Any]] = []
                for original in batch:
                    fallback = dict(original)
                    fallback["hint"] = normalize_text(str(original.get("hint", "")))
                    fallback["description"] = ensure_period(normalize_text(str(original.get("description", ""))))
                    fallback["details"] = choose_details(
                        word=str(original.get("word", "")),
                        category=str(original.get("category", "")),
                        description=fallback["description"],
                        original_details=str(original.get("details", "")),
                        candidate_details="",
                    )
                    fallback_batch.append(ensure_details_last(fallback))
                return fallback_batch
            time.sleep(wait_time)

    return [ensure_details_last(dict(item)) for item in batch]


# =========================
# Main
# =========================
def main() -> None:
    if not API_KEY:
        raise EnvironmentError("Missing DEEPSEEK_API_KEY environment variable.")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    entries, metadata = load_json(INPUT_FILE)
    checkpoint = load_checkpoint(CHECKPOINT_FILE)
    try:
        last_completed_index = int(checkpoint.get("last_completed_index", -1))
    except (TypeError, ValueError):
        last_completed_index = -1

    if last_completed_index < -1:
        last_completed_index = -1
    if last_completed_index >= len(entries):
        last_completed_index = len(entries) - 1

    print(f"Loaded {len(entries)} entries.")
    print(f"Resuming after index: {last_completed_index}")

    output_entries = list(entries)
    if Path(OUTPUT_FILE).exists() and last_completed_index >= 0:
        try:
            existing_output, _ = load_json(OUTPUT_FILE)
            if len(existing_output) == len(entries):
                output_entries = existing_output
                print(f"Loaded existing output from {OUTPUT_FILE} for safe resume.")
            else:
                print("[WARN] Existing output length mismatch. Starting from input baseline.")
        except Exception as e:
            print(f"[WARN] Failed to load existing output file: {e}. Starting from input baseline.")

    start_index = last_completed_index + 1
    if start_index >= len(entries):
        print("Nothing left to process. Writing current output.")
        output_entries = [ensure_details_last(item) for item in output_entries]
        save_json(OUTPUT_FILE, output_entries, metadata)
        return

    current = start_index
    while current < len(entries):
        batch = entries[current: current + BATCH_SIZE]
        batch_end = current + len(batch) - 1

        print(f"Processing entries {current} to {batch_end}...")
        improved_batch = improve_batch(client, batch)

        output_entries[current: current + len(batch)] = improved_batch
        output_entries = [ensure_details_last(item) for item in output_entries]
        save_json(OUTPUT_FILE, output_entries, metadata)
        save_checkpoint(CHECKPOINT_FILE, batch_end)

        current += len(batch)
        time.sleep(SLEEP_BETWEEN_BATCHES)

    print(f"Done. Saved improved data to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
