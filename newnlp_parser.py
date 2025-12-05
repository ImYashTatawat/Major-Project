

import re
from datetime import datetime
from typing import List, Dict, Any, Optional

CATEGORY_MAP = {
    "animal": ["dog", "cat", "elephant", "giraffe", "zebra", "horse", "cow", "lion", "tiger", "bear", "sheep", "goat"],
    "person": ["person", "man", "woman", "boy", "girl", "human", "people"],
    "vehicle": ["car", "truck", "bus", "bike", "bicycle", "motorcycle"],
}

REVERSE_MAP = {}
for k, vs in CATEGORY_MAP.items():
    REVERSE_MAP[k] = k
    for v in vs:
        REVERSE_MAP[v] = k

DATE_PATTERNS = [
    (r"([A-Za-z]+ \d{1,2}, \d{4})", "%B %d, %Y"),
    (r"(\d{1,2} [A-Za-z]+ \d{4})", "%d %B %Y"),
    (r"(\d{1,2}/\d{1,2}/\d{2,4})", None),
    (r"(\d{4}-\d{2}-\d{2})", "%Y-%m-%d"),
]

def _parse_date_token(token: str) -> Optional[datetime]:
    token = token.strip()
    for pattern, fmt in DATE_PATTERNS:
        m = re.match(pattern, token)
        if not m: continue
        s = m.group(1)
        try:
            if fmt:
                return datetime.strptime(s, fmt)
            else:
                try: return datetime.strptime(s, "%m/%d/%Y")
                except: return datetime.strptime(s, "%m/%d/%y")
        except: continue
    return None

def _normalize_word(w: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", w.lower()).strip()

def _tokenize(prompt: str) -> List[str]:
    prompt = prompt.lower().replace(" but not ", " not ").replace(" without ", " not ")
    return [p.strip() for p in re.split(r"[;,]", prompt) if p.strip()]

def parse_natural_language(prompt: str) -> Dict[str, Any]:
    if not prompt:
        return {"mode":"SINGLE","targets":[],"not_groups":[],"start_date":None,"end_date":None}

    text = prompt.strip()
    start_date = end_date = None

    m = re.search(r"between\s+(.+?)\s+and\s+(.+?)", text, flags=re.I)
    if m:
        start_date = _parse_date_token(m.group(1).strip())
        end_date = _parse_date_token(m.group(2).strip())
    else:
        m2 = re.search(r"from\s+(.+?)\s+(?:to|-)\s+(.+?)", text, flags=re.I)
        if m2:
            start_date = _parse_date_token(m2.group(1).strip())
            end_date = _parse_date_token(m2.group(2).strip())

    if start_date and not end_date: end_date = start_date

    cleaned = re.sub(r"(between\s+.+?\s+and\s+.+?$)|(from\s+.+?\s+(?:to|-)\s+.+?$)", " ", text, flags=re.I)
    for pattern, _ in DATE_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.I)

    lower = text.lower()
    if " and " in lower and " or " not in lower: mode="AND"
    elif " or " in lower and " and " not in lower: mode="OR"
    elif " and " in lower and " or " in lower: mode="AND"
    else: mode="SINGLE"

    parts = re.split(r"\band\b" if mode=="AND" else r"\bor\b", cleaned, flags=re.I) if mode!="SINGLE" else _tokenize(cleaned)
    targets, not_groups = [], []

    for part in parts:
        part = part.strip()
        if not part: continue
        if " not " in part:
            left,right = part.split(" not ",1)
            left_terms = [REVERSE_MAP.get(_normalize_word(w), _normalize_word(w)) for w in re.findall(r"[a-z0-9]+", left)]
            right_terms = [REVERSE_MAP.get(_normalize_word(w), _normalize_word(w)) for w in re.findall(r"[a-z0-9]+", right)]
            if left_terms: targets.append(left_terms)
            if right_terms: not_groups.append(right_terms)
            continue

        words = [w for w in re.findall(r"[a-z0-9]+", part) if w not in {"the","show","images","with","containing","take","all"}]
        if words: targets.append([REVERSE_MAP.get(_normalize_word(w), _normalize_word(w)) for w in words])

    for m in re.findall(r"(?:not|without|except)\s+([a-z0-9]+)", text, flags=re.I):
        w = _normalize_word(m)
        if w: not_groups.append([REVERSE_MAP.get(w,w)])

    # Deduplicate without expanding synonyms (only expand explicit categories)
    def _norm_group(g: List[str]) -> List[str]:
        out=[]
        for w in g:
            if w in CATEGORY_MAP: out.extend([w]+CATEGORY_MAP[w])
            else: out.append(w)
        seen=set(); final=[]
        for x in out:
            if x not in seen: final.append(x); seen.add(x)
        return final

    targets = [_norm_group(g) for g in targets if g]
    not_groups = [_norm_group(g) for g in not_groups if g]

    return {"mode":mode,"targets":targets,"not_groups":not_groups,"start_date":start_date,"end_date":end_date}




