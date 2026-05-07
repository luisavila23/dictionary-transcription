#!/usr/bin/env python3
# pip install openai python-dotenv

"""
Clean and validate multilingual JSON word datasets with optional DeepSeek-assisted fixes.

Features:
- Reads one JSON file or all JSON files in an input folder
- Supports JSON array and JSON Lines input formats
- Performs local validation and flags suspicious records
- Sends only suspicious records to DeepSeek in batches
- Supports dry-run mode (no API calls)
- Preserves record order and unknown extra fields
- Never changes "word" and only updates "normalizedWord" when missing/clearly wrong
- Writes cleaned output, report JSON, and errors/skipped JSONL
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from copy import deepcopy
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

try:
    from openai import OpenAI
except Exception as exc:
    raise RuntimeError(
        "Missing dependency 'openai'. Install with: pip install openai python-dotenv"
    ) from exc


SUPPORTED_CATEGORIES = {
    "plant",
    "animal",
    "object",
    "action",
    "person",
    "place",
    "body",
    "food",
    "medicine",
    "emotion",
    "profession",
    "clothing",
    "tool",
    "other",
}

LANGUAGE_CONFIG = {
    "spanish": {
        "hint_ideal": (4, 10),
        "hint_max": 14,
        "hint_definition_prefixes": ("es ", "se refiere", "dicho de", "persona que", "planta que", "animal que", "objeto que", "accion de"),
        "description_clue_prefixes": ("adivina", "pista", "clue", "guess", "indice"),
        "risky_phrases": (
            "medicinal",
            "curativo",
            "fitoterapia",
            "simbolo",
            "simboliza",
            "historicamente",
            "etimologia",
            "nombre cientifico",
            "cientifico",
            "se usa para tratar",
            "se usa en conservas",
            "usos ornamentales",
        ),
        "stopwords": {
            "de", "la", "el", "y", "en", "un", "una", "que", "con", "por", "para", "es", "se", "del"
        },
    },
    "english": {
        "hint_ideal": (4, 9),
        "hint_max": 14,
        "hint_definition_prefixes": ("a type of", "a kind of", "it is", "this is", "refers to", "said of", "person who", "plant that", "animal that"),
        "description_clue_prefixes": ("guess", "clue", "hint"),
        "risky_phrases": (
            "medicinal",
            "healing",
            "herbal medicine",
            "symbol",
            "symbolizes",
            "historically",
            "etymology",
            "scientific name",
            "used to treat",
            "used in preserves",
            "ornamental uses",
        ),
        "stopwords": {
            "the", "a", "an", "and", "in", "on", "of", "to", "for", "is", "with", "that", "by"
        },
    },
    "french": {
        "hint_ideal": (4, 10),
        "hint_max": 14,
        "hint_definition_prefixes": ("c'est", "il s'agit", "se dit de", "personne qui", "plante qui", "animal qui", "objet qui"),
        "description_clue_prefixes": ("devine", "indice", "piste"),
        "risky_phrases": (
            "medicinal",
            "curatif",
            "phytotherapie",
            "symbole",
            "symbolise",
            "historiquement",
            "etymologie",
            "nom scientifique",
            "utilise pour traiter",
            "utilise en conserve",
            "usages ornementaux",
        ),
        "stopwords": {
            "de", "la", "le", "et", "en", "un", "une", "que", "avec", "pour", "est", "des", "du"
        },
    },
    "portuguese": {
        "hint_ideal": (4, 10),
        "hint_max": 14,
        "hint_definition_prefixes": ("e uma", "refere-se", "diz-se de", "pessoa que", "planta que", "animal que", "objeto que"),
        "description_clue_prefixes": ("dica", "adivinhe"),
        "risky_phrases": (
            "medicinal",
            "curativo",
            "fitoterapia",
            "simbolo",
            "simboliza",
            "historicamente",
            "etimologia",
            "nome cientifico",
            "usado para tratar",
            "usado em conservas",
            "usos ornamentais",
        ),
        "stopwords": {
            "de", "a", "o", "e", "em", "um", "uma", "que", "com", "para", "e", "do", "da"
        },
    },
}

MODE_CONFIG = {
    "conservative": {"hint_max": 14, "overlap_threshold": 0.50, "important_overlap": 4},
    "balanced": {"hint_max": 10, "overlap_threshold": 0.35, "important_overlap": 3},
    "aggressive": {"hint_max": 10, "overlap_threshold": 0.35, "important_overlap": 3},
}

CATEGORY_KEYWORDS = {
    "plant": ["flor", "planta", "arbol", "arbusto", "hierba", "hoja", "fruto", "semilla", "tallo", "raiz", "madera", "fibra vegetal", "vegetal", "silvestre", "flower", "plant", "tree", "shrub", "herb", "leaf", "fruit", "seed", "stem", "root", "wood", "wild", "fleur", "plante", "arbre", "arbuste", "herbe", "feuille", "graine", "tige", "bois", "sauvage", "flor", "planta", "arvore", "erva", "folha", "semente", "caule", "madeira"],
    "animal": ["animal", "caballo", "perro", "ave", "pajaro", "pez", "insecto", "mamifero", "equino", "bovino", "reptil", "horse", "dog", "bird", "fish", "insect", "mammal", "equine", "bovine", "cheval", "chien", "oiseau", "poisson", "insecte", "mammif", "cavalo", "cao", "passaro", "peixe", "mamifero"],
    "person": ["persona", "miembro", "descendiente", "habitante", "individuo", "lider", "sacerdote", "trabajador", "comerciante", "person", "member", "descendant", "inhabitant", "individual", "leader", "priest", "worker", "seller", "personne", "membre", "descendant", "habitant", "individu", "dirigeant", "pretre", "travailleur", "commercant", "pessoa", "membro", "descendente", "habitante", "individuo", "lider", "sacerdote", "trabalhador", "comerciante"],
    "profession": ["oficio", "profesion", "comerciante", "vendedor", "cultivador", "artesano", "medico", "profesor", "merchant", "seller", "grower", "farmer", "artisan", "doctor", "teacher", "occupation", "metier", "vendeur", "cultivateur", "medecin", "professeur", "profissao", "vendedor", "cultivador", "artesao", "medico", "professor"],
    "action": ["verbo", "accion", "hacer", "padecer", "sufrir", "moverse", "hablar", "trabajar", "dedicarse", "verb", "action", "to do", "to suffer", "to move", "to speak", "to work", "verbe", "faire", "souffrir", "bouger", "parler", "trabalhar", "acao", "fazer", "sofrer", "mover", "falar"],
    "medicine": ["enfermedad", "afeccion", "inflamacion", "dolor", "glandula", "articulacion", "curacion", "medicinal", "sintoma", "disease", "condition", "inflammation", "pain", "gland", "joint", "symptom", "maladie", "affection", "douleur", "glande", "articulation", "symptome", "doenca", "condicao", "inflamacao", "dor", "sintoma"],
    "body": ["cabeza", "mano", "pie", "brazo", "pierna", "ojo", "boca", "diente", "articulacion", "glandula", "babilla", "head", "hand", "foot", "arm", "leg", "eye", "mouth", "tooth", "tete", "main", "pied", "bras", "jambe", "oeil", "bouche", "dent", "cabeca", "mao", "pe", "braco", "perna", "olho", "dente"],
    "food": ["comestible", "alimento", "comida", "bebida", "fruto", "pan", "carne", "dulce", "conserva", "edible", "food", "drink", "bread", "meat", "sweet", "preserve", "aliment", "nourriture", "boisson", "viande", "confiture", "comestivel", "alimento", "bebida", "pao", "doce", "conserva"],
    "tool": ["herramienta", "instrumento", "utensilio", "aparato", "dispositivo", "tool", "utensil", "device", "outil", "ustensile", "appareil", "ferramenta", "utensilio", "aparelho", "dispositivo"],
    "object": ["objeto", "cosa", "pieza", "prenda", "material", "tela", "fibra", "cuerda", "object", "thing", "piece", "garment", "material", "fabric", "rope", "objet", "chose", "piece", "vetement", "materiau", "tissu", "corde", "objeto", "coisa", "peca", "roupa", "material", "tecido", "corda"],
    "place": ["lugar", "ciudad", "pais", "region", "city", "country", "place", "region", "lieu"],
    "emotion": ["emocion", "sentimiento", "emotion", "feeling", "emotion", "sentiment", "emocao", "sentimento"],
    "clothing": ["ropa", "vestimenta", "prenda", "clothing", "garment", "vetement", "roupa"],
}


@dataclass
class RecordIssue:
    index: int
    reasons: List[str]


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def load_json_records(path: str) -> Tuple[List[Dict[str, Any]], str]:
    """Load records and detect original format: 'array' or 'jsonl'."""
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    stripped = raw.lstrip()

    if stripped.startswith("["):
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array in {path}")
        records = [obj for obj in data if isinstance(obj, dict)]
        return records, "array"

    records: List[Dict[str, Any]] = []
    for line_no, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
        if isinstance(obj, dict):
            records.append(obj)
    return records, "jsonl"


def save_json_records(path: str, records: List[Dict[str, Any]], original_format: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        if original_format == "jsonl":
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        else:
            json.dump(records, f, ensure_ascii=False, indent=2)
            f.write("\n")


def normalize_word(word: str, language: str, keep_ene: bool = False) -> str:
    if not isinstance(word, str):
        return ""
    text = word.strip().lower()
    if keep_ene and language == "spanish":
        text = text.replace("ñ", "__enye__")
    text = strip_accents(text)
    if keep_ene and language == "spanish":
        text = text.replace("__enye__", "ñ")
    text = re.sub(r"[^a-z0-9ñ]+", "", text)
    return text


def tokenize_text(text: str) -> List[str]:
    if not text:
        return []
    clean = strip_accents(text.lower())
    return re.findall(r"[a-z0-9]+", clean)


def similarity_score(text_a: str, text_b: str) -> float:
    a = " ".join(tokenize_text(text_a))
    b = " ".join(tokenize_text(text_b))
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def important_overlap_count(text_a: str, text_b: str, stopwords: set[str]) -> int:
    ta = [t for t in tokenize_text(text_a) if t not in stopwords and len(t) > 2]
    tb = [t for t in tokenize_text(text_b) if t not in stopwords and len(t) > 2]
    return len(set(ta).intersection(tb))


def _guess_category(text: str) -> Optional[str]:
    t = strip_accents(text.lower())
    scores: Dict[str, int] = {}
    for cat, needles in CATEGORY_KEYWORDS.items():
        score = sum(1 for n in needles if n in t)
        if score:
            scores[cat] = score
    if scores:
        return max(scores.items(), key=lambda kv: kv[1])[0]
    return None


def detect_issues(record: Dict[str, Any], language: str, mode: str, keep_ene: bool = False) -> List[str]:
    lang = LANGUAGE_CONFIG.get(language, LANGUAGE_CONFIG["english"])
    mode_cfg = MODE_CONFIG[mode]
    reasons: List[str] = []

    required_fields = ["word", "hint", "description", "difficulty", "category", "normalizedWord", "details"]
    missing = [f for f in required_fields if f not in record or record.get(f) in (None, "")]
    if missing:
        reasons.append(f"missing_required_fields:{','.join(missing)}")

    hint = str(record.get("hint", "") or "")
    description = str(record.get("description", "") or "")
    details = str(record.get("details", "") or "")
    category = str(record.get("category", "") or "").strip().lower()

    sim = similarity_score(hint, description)
    overlap = important_overlap_count(hint, description, lang["stopwords"])
    if sim > 0.55:
        reasons.append("hint_description_too_similar")
    if overlap > mode_cfg["important_overlap"] - 1:
        reasons.append("hint_description_too_similar")
    token_set_hint = {t for t in tokenize_text(hint) if t not in lang["stopwords"] and len(t) > 2}
    token_set_desc = {t for t in tokenize_text(description) if t not in lang["stopwords"] and len(t) > 2}
    jaccard = (len(token_set_hint & token_set_desc) / len(token_set_hint | token_set_desc)) if (token_set_hint or token_set_desc) else 0.0
    if jaccard > mode_cfg["overlap_threshold"]:
        reasons.append("hint_description_too_similar")

    hint_tokens = tokenize_text(hint)
    hint_word_count = len(hint_tokens)
    if hint_word_count > mode_cfg["hint_max"]:
        reasons.append("hint_too_long")

    hint_lc = strip_accents(hint.lower()).strip()
    if any(hint_lc.startswith(prefix) for prefix in lang["hint_definition_prefixes"]):
        reasons.append("hint_looks_like_definition")
    if re.match(r"^(es|se refiere|dicho de|persona que|planta que|animal que|objeto que|accion de|it is|a type of|a kind of|refers to|said of|person who|plant that|animal that|c[' ]est|il s[' ]agit|se dit de|personne qui|plante qui|animal qui|objet qui|e uma|refere-se|diz-se de|pessoa que|planta que|animal que|objeto que)\b", hint_lc):
        reasons.append("hint_looks_like_definition")

    description_word_count = len(tokenize_text(description))
    if description_word_count < 8:
        reasons.append("description_too_short")

    if description and hint and similarity_score(description, hint) > 0.6:
        reasons.append("description_repeats_hint")
    if mode in ("balanced", "aggressive") and any(strip_accents(description.lower()).startswith(p) for p in lang.get("description_clue_prefixes", ())):
        reasons.append("description_looks_like_clue")
    if mode in ("balanced", "aggressive") and len(tokenize_text(description)) >= 8 and len(set(tokenize_text(description))) <= 5:
        reasons.append("description_too_vague")

    if details and (similarity_score(details, description) > 0.70 or similarity_score(details, hint) > 0.70):
        reasons.append("details_repeats_description")

    details_lc = strip_accents(details.lower())
    extra_risky = ("region", "región", "regional", "histor", "simbol", "etimolog", "scientific", "cientific")
    for phrase in lang["risky_phrases"]:
        if phrase in details_lc:
            reasons.append("risky_details")
    if any(p in details_lc for p in extra_risky):
        reasons.append("risky_details")

    if category and category not in SUPPORTED_CATEGORIES:
        reasons.append("invalid_category")

    word = str(record.get("word", "") or "")
    nword = str(record.get("normalizedWord", "") or "")
    merged_text = " ".join([word, hint, description, details])
    guess = _guess_category(merged_text)
    if category == "other":
        if guess and guess != "other":
            reasons.append("category_other_but_obvious")
        elif mode in ("balanced", "aggressive"):
            reasons.append("category_other_needs_review")
    expected = normalize_word(word, language=language, keep_ene=keep_ene)
    if not nword:
        reasons.append("normalizedWord_missing")
    elif normalize_word(nword, language=language, keep_ene=keep_ene) != expected:
        reasons.append("normalizedWord_incorrect")

    ideal_min, ideal_max = lang["hint_ideal"]
    if mode in ("balanced", "aggressive") and hint_word_count and (hint_word_count < ideal_min or hint_word_count > ideal_max):
        reasons.append("hint_outside_ideal_range")

    return list(dict.fromkeys(reasons))


def build_deepseek_prompt(
    records: List[Dict[str, Any]],
    language: str,
    mode: str,
    force_rewrite_fields: Set[str],
) -> Tuple[str, str]:
    system_message = (
        "You are cleaning a multilingual word dataset for word games. "
        "You must return valid JSON only. Do not invent facts. "
        "Make useful improvements, not just minimal risk removals. Preserve the original meaning."
    )

    user_payload = {
        "language": language,
        "mode": mode,
        "force_rewrite_fields": sorted(force_rewrite_fields),
        "instructions": [
            "Keep 'word' unchanged.",
            "Keep all unknown extra fields.",
            "Keep 'difficulty' unless there is a clear reason to change it.",
            "Validate 'normalizedWord' by removing accents and diacritics. Only change it if missing or clearly incorrect.",
            "Rewrite 'hint' as a short indirect game clue.",
            "Rewrite 'description' as a clear educational explanation.",
            "Ensure hint and description are meaningfully different.",
            "Improve 'category' when a better category is obvious.",
            "Clean 'details' by removing unsupported, overly specific, or repetitive claims.",
            "Do not be overly conservative. If hint and description are too similar, rewrite at least one of them.",
            "If category is 'other' but a better category is obvious, change it.",
            "Do not add medicinal uses, cultural meanings, regions, etymology, scientific names, symbolism, or historical facts unless explicitly present in the original record.",
            "Field purpose: hint is clue before guess; description is explanation after reveal; details are extra supported context only.",
            "Return a correction even when current record is acceptable but can be clearly improved for game quality.",
            "Do not omit records.",
            "Do not add markdown.",
            "Do not explain.",
        ],
        "required_output_schema": {
            "records": [
                {
                    "index": 0,
                    "word": "...",
                    "hint": "...",
                    "description": "...",
                    "difficulty": "...",
                    "category": "...",
                    "normalizedWord": "...",
                    "details": "...",
                }
            ]
        },
        "records": records,
    }

    user_message = f"Clean the following records in {language}. Return JSON only.\n\n" + json.dumps(
        user_payload, ensure_ascii=False
    )
    return system_message, user_message


def call_deepseek(
    records: List[Dict[str, Any]],
    language: str,
    api_key: str,
    model: str,
    temperature: float,
    mode: str,
    force_rewrite_fields: Set[str],
    timeout: int = 120,
    retries: int = 3,
) -> str:
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=timeout)
    system_message, user_message = build_deepseek_prompt(records, language, mode=mode, force_rewrite_fields=force_rewrite_fields)

    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response content from model")
            return content
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            sleep_s = 2 ** (attempt - 1)
            print(f"[warn] DeepSeek call failed (attempt {attempt}/{retries}): {exc}. Retrying in {sleep_s}s...")
            time.sleep(sleep_s)

    raise RuntimeError(f"DeepSeek API failed after {retries} attempts: {last_error}")


def validate_ai_response(
    response: str, original_records: List[Dict[str, Any]], language: str, keep_ene: bool = False
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (valid_corrections, validation_errors)."""
    errors: List[Dict[str, Any]] = []
    valid: List[Dict[str, Any]] = []

    try:
        parsed = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model response is not valid JSON: {exc}") from exc

    records = parsed.get("records")
    if not isinstance(records, list):
        raise ValueError("Model response missing 'records' list")

    index_to_original = {r["index"]: r for r in original_records}

    for item in records:
        if not isinstance(item, dict):
            errors.append({"error": "record_not_object", "record": item})
            continue

        idx = item.get("index")
        if not isinstance(idx, int) or idx not in index_to_original:
            errors.append({"error": "invalid_index", "record": item})
            continue

        source = index_to_original[idx]

        corrected = deepcopy(source)
        rejection_reasons: List[str] = []
        for key, value in item.items():
            if key == "index":
                continue
            corrected[key] = value
        corrected.pop("index", None)
        corrected.pop("_localIssues", None)

        if corrected.get("word") != source.get("word"):
            errors.append({"index": idx, "error": "word_changed_forbidden"})
            continue

        # Never accept invalid category values.
        cat = str(corrected.get("category", source.get("category", ""))).strip().lower()
        if cat and cat not in SUPPORTED_CATEGORIES:
            corrected["category"] = source.get("category")
            rejection_reasons.append("invalid_category_reverted")

        # Difficulty only if non-empty string.
        diff = corrected.get("difficulty")
        if diff is None or str(diff).strip() == "":
            corrected["difficulty"] = source.get("difficulty")
            rejection_reasons.append("invalid_difficulty_reverted")

        original_n = str(source.get("normalizedWord", "") or "")
        candidate_n = str(corrected.get("normalizedWord", "") or "")
        expected = normalize_word(str(source.get("word", "") or ""), language=language, keep_ene=keep_ene)
        original_ok = normalize_word(original_n, language=language, keep_ene=keep_ene) == expected
        candidate_ok = normalize_word(candidate_n, language=language, keep_ene=keep_ene) == expected

        if original_ok and candidate_n != original_n:
            corrected["normalizedWord"] = original_n
            rejection_reasons.append("normalizedWord_change_not_needed")
        elif not original_ok and not candidate_ok:
            corrected["normalizedWord"] = expected
            rejection_reasons.append("normalizedWord_invalid_corrected_locally")

        valid.append({"index": idx, "record": corrected, "rejection_reasons": rejection_reasons})

    return valid, errors


def merge_corrections(
    original_records: List[Dict[str, Any]],
    corrections: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged = deepcopy(original_records)
    allowed_fields = {"hint", "description", "category", "details", "normalizedWord", "difficulty"}
    by_index = {c["index"]: c["record"] for c in corrections}
    for i, rec in enumerate(merged):
        if i not in by_index:
            continue
        new_rec = by_index[i]
        out = deepcopy(rec)
        for field in allowed_fields:
            if field in new_rec:
                out[field] = new_rec[field]
        out["word"] = rec.get("word")
        for k, v in rec.items():
            if k not in out:
                out[k] = v
        merged[i] = out
    return merged


def write_report(report_path: str, report_data: Dict[str, Any]) -> None:
    p = Path(report_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _diff_fields(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    changed: Dict[str, Dict[str, Any]] = {}
    all_keys = set(old.keys()).union(new.keys())
    for k in sorted(all_keys):
        if old.get(k) != new.get(k):
            changed[k] = {"old": old.get(k), "new": new.get(k)}
    return changed


def _build_default_paths(
    input_path: Path,
    output_file: Optional[Path],
    output_folder: Optional[Path],
) -> Tuple[Path, Path, Path]:
    stem = input_path.stem
    parent = output_folder if output_folder else input_path.parent
    cleaned_path = output_file if output_file else parent / f"{stem}.cleaned.json"
    report_path = parent / f"{stem}.report.json"
    errors_path = parent / f"{stem}.errors.jsonl"
    return cleaned_path, report_path, errors_path


def process_file(
    input_path: Path,
    output_file: Optional[Path],
    output_folder: Optional[Path],
    language: str,
    batch_size: int,
    dry_run: bool,
    api_key: Optional[str],
    model: str,
    temperature: float,
    mode: str,
    force_rewrite_fields: Set[str],
    keep_ene: bool,
    force_api: bool,
) -> Dict[str, Any]:
    print(f"[info] Processing: {input_path}")
    records, original_format = load_json_records(str(input_path))
    cleaned_path, report_path, errors_path = _build_default_paths(input_path, output_file, output_folder)

    suspicious: List[RecordIssue] = []
    reason_counts: Dict[str, int] = {}
    for i, rec in enumerate(records):
        reasons = detect_issues(rec, language=language, mode=mode, keep_ene=keep_ene)
        if reasons:
            suspicious.append(RecordIssue(index=i, reasons=reasons))
            for reason in reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

    suspicious_by_index = {x.index: x.reasons for x in suspicious}
    if force_api:
        candidate_indices = list(range(len(records)))
    elif force_rewrite_fields:
        candidate_indices = list(range(len(records))) if mode == "aggressive" else [x.index for x in suspicious]
    else:
        candidate_indices = [x.index for x in suspicious]
    print(f"Loaded {len(records)} records.")
    print(f"Detected {len(suspicious)} suspicious records.")
    if reason_counts:
        print("Reasons:")
        for reason, count in sorted(reason_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"- {reason}: {count}")
    print(f"Sending {0 if dry_run else len(candidate_indices)} records to DeepSeek...")

    working = deepcopy(records)
    report_entries: List[Dict[str, Any]] = []
    error_lines: List[Dict[str, Any]] = []
    accepted_corrections: List[Dict[str, Any]] = []
    api_batches_attempted = 0
    api_batches_succeeded = 0
    records_sent_to_api = 0
    records_received_from_api = 0
    corrections_accepted = 0
    corrections_rejected = 0

    if dry_run:
        for issue in suspicious:
            report_entries.append(
                {
                    "index": issue.index,
                    "word": records[issue.index].get("word"),
                    "issue_reasons": issue.reasons,
                    "fields_requested_for_rewrite": sorted(force_rewrite_fields) if force_rewrite_fields else ["hint", "description", "category", "details"],
                    "fields_changed": [],
                    "old_values": {},
                    "new_values": {},
                    "api_called": False,
                    "correction_applied": False,
                    "skipped_reason": "dry_run",
                }
            )
    else:
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required unless --dry-run is used")

        for start in range(0, len(candidate_indices), batch_size):
            batch_indices = candidate_indices[start : start + batch_size]
            api_batches_attempted += 1
            batch_records: List[Dict[str, Any]] = []
            for idx in batch_indices:
                rec = deepcopy(working[idx])
                rec["index"] = idx
                batch_records.append(rec)
            records_sent_to_api += len(batch_records)

            print(
                f"[info] Batch {start // batch_size + 1} "
                f"({start + 1}-{min(start + len(batch_indices), len(candidate_indices))} / {len(candidate_indices)})"
            )

            try:
                raw = call_deepseek(
                    records=batch_records,
                    language=language.capitalize(),
                    api_key=api_key,
                    model=model,
                    temperature=temperature,
                    mode=mode,
                    force_rewrite_fields=force_rewrite_fields,
                )
            except Exception as exc:
                print(f"[error] Batch failed: {exc}")
                error_lines.append(
                    {
                        "type": "api_error",
                        "batch_start": start,
                        "batch_size": len(batch_indices),
                        "error": str(exc),
                    }
                )
                continue

            try:
                corrections, validation_errors = validate_ai_response(
                    raw, batch_records, language=language, keep_ene=keep_ene
                )
                api_batches_succeeded += 1
                records_received_from_api += len(corrections)
            except Exception as exc:
                print(f"[error] Invalid JSON response: {exc}")
                error_lines.append(
                    {
                        "type": "response_parse_error",
                        "batch_start": start,
                        "batch_size": len(batch_indices),
                        "error": str(exc),
                        "raw_response": raw,
                    }
                )
                continue

            if validation_errors:
                error_lines.append(
                    {
                        "type": "response_validation_errors",
                        "batch_start": start,
                        "errors": validation_errors,
                    }
                )

            returned_indexes = {c["index"] for c in corrections}
            for idx in batch_indices:
                if idx in returned_indexes:
                    continue
                requested = sorted(force_rewrite_fields) if force_rewrite_fields else ["hint", "description", "category", "details"]
                report_entries.append(
                    {
                        "index": idx,
                        "word": working[idx].get("word"),
                        "issue_reasons": suspicious_by_index.get(idx, ["forced_api_review"]),
                        "fields_requested_for_rewrite": requested,
                        "fields_changed": [],
                        "old_values": {},
                        "new_values": {},
                        "api_called": True,
                        "correction_applied": False,
                        "skipped_reason": "missing_from_api_response",
                    }
                )
                corrections_rejected += 1

            for correction in corrections:
                idx = correction["index"]
                before = working[idx]
                after = correction["record"]
                requested = sorted(force_rewrite_fields) if force_rewrite_fields else ["hint", "description", "category", "details"]
                full_changes = _diff_fields(before, after)
                fields_changed = [f for f in requested + ["normalizedWord", "difficulty"] if f in full_changes]
                changes = {k: v for k, v in full_changes.items() if k in fields_changed}
                if changes:
                    report_entries.append(
                        {
                            "index": idx,
                            "word": before.get("word"),
                            "issue_reasons": suspicious_by_index.get(idx, ["forced_api_review"]),
                            "fields_requested_for_rewrite": requested,
                            "fields_changed": list(changes.keys()),
                            "old_values": {k: v["old"] for k, v in changes.items()},
                            "new_values": {k: v["new"] for k, v in changes.items()},
                            "api_called": True,
                            "correction_applied": True,
                            "skipped_reason": None,
                            "validation_rejections": correction.get("rejection_reasons", []),
                        }
                    )
                    accepted_corrections.append(correction)
                    working[idx] = after
                    corrections_accepted += 1
                else:
                    report_entries.append(
                        {
                            "index": idx,
                            "word": before.get("word"),
                            "issue_reasons": suspicious_by_index.get(idx, ["forced_api_review"]),
                            "fields_requested_for_rewrite": requested,
                            "fields_changed": [],
                            "old_values": {},
                            "new_values": {},
                            "api_called": True,
                            "correction_applied": False,
                            "skipped_reason": "no_meaningful_change",
                            "validation_rejections": correction.get("rejection_reasons", []),
                        }
                    )
                    corrections_rejected += 1

    if not dry_run:
        final_records = merge_corrections(records, accepted_corrections)
    else:
        final_records = records

    save_json_records(str(cleaned_path), final_records, original_format=original_format)

    changed_entries = [e for e in report_entries if e.get("correction_applied")]
    changed_indexes = {e["index"] for e in changed_entries}
    api_batches_failed = api_batches_attempted - api_batches_succeeded
    report_obj = {
        "input_file": str(input_path),
        "output_file": str(cleaned_path),
        "language": language.capitalize(),
        "mode": mode,
        "total_records": len(records),
        "suspicious_records": len(suspicious),
        "records_sent_to_api": records_sent_to_api,
        "records_changed": len(changed_entries),
        "records_unchanged": len(records) - len(changed_indexes),
        "api_batches_completed": api_batches_succeeded,
        "api_batches_failed": api_batches_failed,
        "changes": sorted(report_entries, key=lambda x: x["index"]),
        "errors": error_lines,
    }
    write_report(str(report_path), report_obj)

    errors_path.parent.mkdir(parents=True, exist_ok=True)
    with errors_path.open("w", encoding="utf-8", newline="\n") as ef:
        for item in error_lines:
            ef.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[done] Cleaned: {cleaned_path}")
    print(f"[done] Report:  {report_path}")
    print(f"[done] Errors:  {errors_path}")
    print(f"DeepSeek batches completed: {api_batches_succeeded}")
    print(f"DeepSeek batches failed: {api_batches_failed}")
    print(f"Corrections returned: {records_received_from_api}")
    print(f"Corrections accepted: {corrections_accepted}")
    print(f"Corrections rejected: {corrections_rejected}")
    print(f"Final changed records: {len(changed_entries)}")

    return {
        "input": str(input_path),
        "cleaned": str(cleaned_path),
        "report": str(report_path),
        "errors": str(errors_path),
        "total": len(records),
        "suspicious": len(suspicious),
        "sent": records_sent_to_api,
        "changed": len(changed_entries),
        "api_completed": api_batches_succeeded,
        "api_failed": api_batches_failed,
    }


def infer_language_from_name(path: Path) -> str:
    name = path.name.lower()
    if "spanish" in name or re.search(r"(^|[_\-.])es([_\-.]|$)", name):
        return "spanish"
    if "english" in name or re.search(r"(^|[_\-.])en([_\-.]|$)", name):
        return "english"
    if "french" in name or re.search(r"(^|[_\-.])fr([_\-.]|$)", name):
        return "french"
    if "portuguese" in name or re.search(r"(^|[_\-.])pt([_\-.]|$)", name):
        return "portuguese"
    return "english"


def iter_input_files(input_folder: Path) -> Iterable[Path]:
    for p in sorted(input_folder.glob("*.json")):
        if p.is_file():
            yield p


def parse_force_rewrite_fields(raw: str) -> Set[str]:
    allowed = {"hint", "description", "category", "details"}
    if not raw.strip():
        return set()
    parts = {x.strip().lower() for x in raw.split(",") if x.strip()}
    invalid = parts - allowed
    if invalid:
        raise ValueError(f"Invalid --force-rewrite-fields values: {sorted(invalid)}")
    return parts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and clean multilingual JSON word datasets")
    parser.add_argument("--input", help="Single input JSON/JSONL file path")
    parser.add_argument("--input-folder", help="Folder containing .json datasets")
    parser.add_argument("--output", help="Output cleaned file path (single-file mode)")
    parser.add_argument("--output-folder", help="Output folder (folder mode or defaults)")
    parser.add_argument(
        "--language",
        default=None,
        help="Dataset language: spanish|english|french|portuguese (auto-inferred if omitted in folder mode)",
    )
    parser.add_argument("--batch-size", type=int, default=50, help="Suspicious records per API batch")
    parser.add_argument("--model", default="deepseek-chat", help="DeepSeek model name")
    parser.add_argument("--temperature", type=float, default=0.2, help="Model temperature")
    parser.add_argument(
        "--mode",
        choices=["conservative", "balanced", "aggressive"],
        default="balanced",
        help="Cleaning strictness mode",
    )
    parser.add_argument("--dry-run", action="store_true", help="Detect and report issues without API calls")
    parser.add_argument(
        "--force-api",
        action="store_true",
        help="Send records to DeepSeek even when local validation does not flag them",
    )
    parser.add_argument(
        "--keep-spanish-ene",
        action="store_true",
        help="Keep ñ in normalizedWord for Spanish (default is ASCII n)",
    )
    parser.add_argument(
        "--force-rewrite-fields",
        default="",
        help="Comma-separated fields to force rewrite: hint,description,category,details",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        force_rewrite_fields = parse_force_rewrite_fields(args.force_rewrite_fields)
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    if not args.input and not args.input_folder:
        print("[error] Provide --input or --input-folder", file=sys.stderr)
        return 2

    if args.input and args.input_folder:
        print("[error] Use only one: --input or --input-folder", file=sys.stderr)
        return 2

    if load_dotenv:
        load_dotenv()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    

    output_file = Path(args.output).resolve() if args.output else None
    output_folder = Path(args.output_folder).resolve() if args.output_folder else None

    summaries: List[Dict[str, Any]] = []

    if args.input:
        input_path = Path(args.input).resolve()
        language = (args.language or infer_language_from_name(input_path)).strip().lower()
        if language not in LANGUAGE_CONFIG:
            print(f"[error] Unsupported language: {language}", file=sys.stderr)
            return 2

        summary = process_file(
            input_path=input_path,
            output_file=output_file,
            output_folder=output_folder,
            language=language,
            batch_size=max(1, args.batch_size),
            dry_run=args.dry_run,
            api_key=api_key,
            model=args.model,
            temperature=args.temperature,
            mode=args.mode,
            force_rewrite_fields=force_rewrite_fields,
            keep_ene=args.keep_spanish_ene,
            force_api=args.force_api,
        )
        summaries.append(summary)
    else:
        input_folder = Path(args.input_folder).resolve()
        if not input_folder.exists() or not input_folder.is_dir():
            print(f"[error] Invalid input folder: {input_folder}", file=sys.stderr)
            return 2

        files = list(iter_input_files(input_folder))
        if not files:
            print(f"[warn] No .json files found in: {input_folder}")
            return 0

        for file_path in files:
            language = (args.language or infer_language_from_name(file_path)).strip().lower()
            if language not in LANGUAGE_CONFIG:
                print(f"[warn] Skipping unsupported language file: {file_path.name}")
                continue

            try:
                summary = process_file(
                    input_path=file_path,
                    output_file=None,
                    output_folder=output_folder,
                    language=language,
                    batch_size=max(1, args.batch_size),
                    dry_run=args.dry_run,
                    api_key=api_key,
                    model=args.model,
                    temperature=args.temperature,
                    mode=args.mode,
                    force_rewrite_fields=force_rewrite_fields,
                    keep_ene=args.keep_spanish_ene,
                    force_api=args.force_api,
                )
                summaries.append(summary)
            except Exception as exc:
                print(f"[error] Failed file {file_path.name}: {exc}")

    total_files = len(summaries)
    total_records = sum(s["total"] for s in summaries)
    total_suspicious = sum(s["suspicious"] for s in summaries)
    total_sent = sum(s.get("sent", 0) for s in summaries)
    total_changed = sum(s.get("changed", 0) for s in summaries)
    total_completed = sum(s.get("api_completed", 0) for s in summaries)
    total_failed = sum(s.get("api_failed", 0) for s in summaries)
    print(
        f"[summary] files={total_files} records={total_records} suspicious={total_suspicious} "
        f"sent={total_sent} changed={total_changed} batches_completed={total_completed} batches_failed={total_failed} "
        f"mode={args.mode} dry_run={args.dry_run} force_api={args.force_api} force_rewrite_fields={sorted(force_rewrite_fields)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
