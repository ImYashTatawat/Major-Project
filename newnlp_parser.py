# """
# Advanced NLP parser for AI Image Filter
# - Converts user text into a structured dictionary:
# {
#     "objects": ["person", "dog"],
#     "start_date": datetime or None,
#     "end_date": datetime or None,
#     "location": str or None,
#     "camera_model": str or None
# }
# - Supports 600+ object synonyms (expandable)
# """

# import re
# from datetime import datetime

# # ----------------- Synonyms / Object Mapping -----------------
# SYNONYMS = {
#     "person": ["person", "man", "woman", "human", "people", "insaan", "aadmi", "aurat"],
#     "car": ["car", "auto", "vehicle", "gaadi", "suv", "sedan", "truck", "van", "jeep"],
#     "dog": ["dog", "puppy", "kutta", "canine", "pup"],
#     "cat": ["cat", "billi", "kitten", "feline"],
#     "animal": ["animal", "creature", "janwar", "wildlife"],
#     "bird": ["bird", "sparrow", "crow", "parrot", "eagle"],
#     "tree": ["tree", "plant", "ped", "bush", "shrub"],
#     "glasses": ["glasses", "spectacles", "chashma", "eyewear"],
#     "bottle": ["bottle", "water bottle", "paani ki bottle"],
#     "phone": ["phone", "mobile", "cellphone", "smartphone"],
#     "laptop": ["laptop", "computer", "notebook"],
#     "motorbike": ["motorbike", "bike", "motorcycle"],
#     "bicycle": ["bicycle", "cycle"],
#     "horse": ["horse", "ghoda"],
#     "cow": ["cow", "gai"],
#     "sheep": ["sheep", "bhed"],
#     "tiger": ["tiger", "bagh", "sher", "big cat"],
#     # --- ADD MANY MORE CLASSES HERE, up to 600+ ---
# }

# # ----------------- Utility Functions -----------------
# def normalize(text: str) -> str:
#     text = text.lower().strip()
#     text = re.sub(r"[^a-zA-Z0-9\s:-]", " ", text)
#     return text

# def match_synonyms(text: str):
#     """Return all classes detected by synonyms in text."""
#     found = []
#     for key, words in SYNONYMS.items():
#         for w in words:
#             if w in text:
#                 found.append(key)
#                 break
#     return list(set(found))

# def extract_dates(text: str):
#     """Extract dates in format YYYY-MM-DD or DD/MM/YYYY"""
#     date_patterns = [
#         r"(\d{4}[-/]\d{1,2}[-/]\d{1,2})",
#         r"(\d{1,2}[-/]\d{1,2}[-/]\d{4})",
#     ]
#     dates = []
#     for pattern in date_patterns:
#         matches = re.findall(pattern, text)
#         for m in matches:
#             try:
#                 dt = datetime.strptime(m, "%Y-%m-%d")
#             except Exception:
#                 try:
#                     dt = datetime.strptime(m, "%d/%m/%Y")
#                 except Exception:
#                     continue
#             dates.append(dt)
#     if len(dates) == 1:
#         return dates[0], None
#     elif len(dates) >= 2:
#         return min(dates), max(dates)
#     else:
#         return None, None

# def extract_location(text: str):
#     """Naive extraction of location (could improve with NLP libraries)"""
#     loc_patterns = ["in ([a-zA-Z\s]+)", "at ([a-zA-Z\s]+)"]
#     for pattern in loc_patterns:
#         match = re.search(pattern, text)
#         if match:
#             return match.group(1).strip()
#     return None

# def extract_camera(text: str):
#     """Optional: detect camera model in prompt"""
#     cam_patterns = ["camera ([a-zA-Z0-9]+)", "model ([a-zA-Z0-9]+)"]
#     for pattern in cam_patterns:
#         match = re.search(pattern, text)
#         if match:
#             return match.group(1).strip()
#     return None

# # ----------------- Main Parser -----------------
# def parse_user_prompt(prompt: str):
#     """
#     Returns:
#         {
#             "objects": ["person", "dog"],
#             "start_date": datetime or None,
#             "end_date": datetime or None,
#             "location": str or None,
#             "camera_model": str or None
#         }
#     """
#     if not prompt:
#         return {"objects": [], "start_date": None, "end_date": None, "location": None, "camera_model": None}

#     text = normalize(prompt)

#     # 1. Extract objects by synonyms
#     objects = match_synonyms(text)

#     # 2. Handle "and/or/with/near" splits
#     for sep in [" and ", " with ", " near ", " or ", ",", ";", "|"]:
#         if sep in text:
#             parts = text.split(sep)
#             for p in parts:
#                 objects += match_synonyms(p)

#     objects = list(dict.fromkeys(objects))  # remove duplicates

#     # 3. Extract metadata: dates, location, camera
#     start_date, end_date = extract_dates(prompt)
#     location = extract_location(prompt)
#     camera_model = extract_camera(prompt)

#     return {
#         "objects": objects,
#         "start_date": start_date,
#         "end_date": end_date,
#         "location": location,
#         "camera_model": camera_model
#     }

# # ----------------- Quick Test -----------------
# if __name__ == "__main__":
#     test_prompt = "Find person or dog near Mumbai in 01/01/2024 to 15/02/2024 with Canon camera"
#     result = parse_user_prompt(test_prompt)
#     print(result)


# nlp_parser.py
# """
# NLP parser that converts user text into a structured dictionary:
# {
#    "objects": ["person", "tree"],
#    "start_date": datetime_object,
#    "end_date": datetime_object
# }
# """
# import re
# from datetime import datetime

# # Example synonym mapping
# SYNONYMS = {
#     "person": ["person", "man", "woman", "human", "people", "adult", "child"],
#     "dog": ["dog", "puppy", "kutta", "canine"],
#     "cat": ["cat", "billi", "kitten", "feline"],
#     "car": ["car", "vehicle", "auto", "sedan"],
#     "tree": ["tree", "plant", "ped"],
#     "tiger": ["tiger", "big cat", "bagh", "sher"],
#     "phone": ["phone", "mobile", "cellphone", "smartphone"],
# }

# DATE_PATTERNS = [
#     r"(\w+\s\d{1,2},\s\d{4})",          # e.g. October 29, 2023
#     r"(\d{1,2}/\d{1,2}/\d{2,4})",       # e.g. 10/29/2023
#     r"(\d{4}-\d{2}-\d{2})",             # e.g. 2023-10-29
# ]

# RANGE_PATTERNS = [
#     r"from\s(.+?)\sto\s(.+?)",          # from Oct 1, 2025 to Oct 5, 2025
#     r"between\s(.+?)\sand\s(.+?)"       # between 01/10/2025 and 05/10/2025
# ]

# def normalize(text: str) -> str:
#     text = text.lower().strip()
#     text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
#     return text

# def match_synonyms(text: str):
#     found = []
#     for key, words in SYNONYMS.items():
#         for w in words:
#             if w in text:
#                 found.append(key)
#                 break
#     return list(dict.fromkeys(found))

# def parse_date_str(date_str: str) -> datetime:
#     for fmt in ("%B %d, %Y", "%m/%d/%Y", "%Y-%m-%d"):
#         try:
#             return datetime.strptime(date_str.strip(), fmt)
#         except:
#             continue
#     return None

# def extract_dates(text: str):
#     dates = []
#     for pattern in DATE_PATTERNS:
#         matches = re.findall(pattern, text)
#         for match in matches:
#             dt = parse_date_str(match)
#             if dt:
#                 dates.append(dt)
#     return dates

# def extract_date_range(text: str):
#     for pattern in RANGE_PATTERNS:
#         match = re.search(pattern, text)
#         if match:
#             start = parse_date_str(match.group(1))
#             end = parse_date_str(match.group(2))
#             if start and end:
#                 return start, end
#     return None, None

# def parse_user_prompt(prompt: str):
#     """
#     Returns:
#     {
#         "objects": ["person", "tree"],
#         "dates": [datetime1, datetime2, ...],
#         "start_date": datetime,
#         "end_date": datetime
#     }
#     """
#     if not prompt:
#         return {"objects": [], "dates": [], "start_date": None, "end_date": None}

#     text = normalize(prompt)
#     objects = match_synonyms(text)
#     dates = extract_dates(prompt)

#     # Detect date range first
#     start_date, end_date = extract_date_range(prompt)
#     if not start_date or not end_date:
#         if dates:
#             start_date, end_date = min(dates), max(dates)
#         else:
#             start_date = end_date = None

#     # Handle AND / OR logic
#     if " and " in text or " with " in text:
#         parts = re.split(r"and|with", text)
#         for p in parts:
#             objects += match_synonyms(p)
#     if " or " in text:
#         parts = text.split(" or ")
#         for p in parts:
#             objects += match_synonyms(p)

#     objects = list(dict.fromkeys(objects))
#     return {
#         "objects": objects,
#         "dates": dates,
#         "start_date": start_date,
#         "end_date": end_date
#     }




# 11111111111111111111112222222222222222222222222222111111111111111111111111111112222222222222222222222222222 may be be
 
# # nlp_parser.py
# import re
# from datetime import datetime
# from typing import List, Dict, Any, Optional

# # small synonym/category map (extend as needed)
# CATEGORY_MAP = {
#     "animal": ["dog", "cat", "elephant", "giraffe", "zebra", "horse", "cow", "lion", "tiger", "bear", "sheep", "goat"],
#     "person": ["person", "man", "woman", "boy", "girl", "human", "people"],
#     "vehicle": ["car", "truck", "bus", "bike", "bicycle", "motorcycle"],
#     # add more categories/synonyms here if you want
# }

# # reverse mapping for quick lookup (synonym -> canonical)
# REVERSE_MAP = {}
# for k, vs in CATEGORY_MAP.items():
#     REVERSE_MAP[k] = k
#     for v in vs:
#         REVERSE_MAP[v] = k

# DATE_PATTERNS = [
#     # e.g. October 29, 2023
#     (r"([A-Za-z]+ \d{1,2}, \d{4})", "%B %d, %Y"),
#     # e.g. 29 October 2023
#     (r"(\d{1,2} [A-Za-z]+ \d{4})", "%d %B %Y"),
#     # e.g. 10/29/2023 or 1/2/2023
#     (r"(\d{1,2}/\d{1,2}/\d{2,4})", None),
#     # e.g. 2023-10-29
#     (r"(\d{4}-\d{2}-\d{2})", "%Y-%m-%d"),
# ]

# def _parse_date_token(token: str) -> Optional[datetime]:
#     token = token.strip()
#     for pattern, fmt in DATE_PATTERNS:
#         m = re.match(pattern, token)
#         if not m:
#             continue
#         s = m.group(1)
#         try:
#             if fmt:
#                 return datetime.strptime(s, fmt)
#             else:
#                 # try mm/dd/YYYY or mm/dd/YY
#                 try:
#                     return datetime.strptime(s, "%m/%d/%Y")
#                 except:
#                     return datetime.strptime(s, "%m/%d/%y")
#         except Exception:
#             continue
#     return None

# def _normalize_word(w: str) -> str:
#     return re.sub(r"[^a-z0-9 ]", "", w.lower()).strip()

# def _tokenize(prompt: str) -> List[str]:
#     # crude tokenization keeping multiword sequences if separated by commas/and/or
#     # We will split by commas, ' and ', ' or ', ' but not ', ' without '
#     prompt = prompt.lower()
#     # replace connectors by spaced tokens
#     prompt = prompt.replace(" but not ", " not ")
#     prompt = prompt.replace(" without ", " not ")
#     return [p.strip() for p in re.split(r"[;,]", prompt) if p.strip()]

# def parse_natural_language(prompt: str) -> Dict[str, Any]:
#     """
#     Returns a dict with stable keys:
#     {
#       "mode": "AND" | "OR" | "SINGLE",
#       "targets": [ ['person','man'], ['dog'] ]   # groups (list of lists)
#       "not_groups": [ ['cat'] ],
#       "start_date": datetime or None,
#       "end_date": datetime or None
#     }
#     """
#     if not prompt:
#         return {"mode": "SINGLE", "targets": [], "not_groups": [], "start_date": None, "end_date": None}

#     text = prompt.strip()

#     # Detect explicit date range phrases like "from <date> to <date>" or "between <date> and <date>"
#     start_date = end_date = None
#     # between X and Y
#     m = re.search(r"between\s+(.+?)\s+and\s+(.+?)(?:\s|$)", text, flags=re.I)
#     if m:
#         d1 = _parse_date_token(m.group(1).strip())
#         d2 = _parse_date_token(m.group(2).strip())
#         if d1:
#             start_date = d1
#         if d2:
#             end_date = d2
#     else:
#         # from X to Y
#         m2 = re.search(r"from\s+(.+?)\s+(?:to|-)\s+(.+?)(?:\s|$)", text, flags=re.I)
#         if m2:
#             d1 = _parse_date_token(m2.group(1).strip())
#             d2 = _parse_date_token(m2.group(2).strip())
#             if d1:
#                 start_date = d1
#             if d2:
#                 end_date = d2

#     # If no explicit range, try to find single date mentions (use as both start and end)
#     if start_date is None and end_date is None:
#         # scan for any date tokens
#         for pattern, _ in DATE_PATTERNS:
#             for m in re.findall(pattern, text):
#                 dt = _parse_date_token(m)
#                 if dt:
#                     if start_date is None:
#                         start_date = dt
#                     else:
#                         end_date = dt
#         if start_date and not end_date:
#             end_date = start_date

#     # Remove date phrases from text to avoid confusing tokenization
#     cleaned = re.sub(r"(between\s+.+?\s+and\s+.+?$)|(from\s+.+?\s+(?:to|-)\s+.+?$)", " ", text, flags=re.I)
#     # Also remove explicit dates tokens
#     for pattern, _ in DATE_PATTERNS:
#         cleaned = re.sub(pattern, " ", cleaned, flags=re.I)

#     # Determine mode: AND if ' and ' in original prompt, OR if ' or ' present, else SINGLE
#     mode = "SINGLE"
#     lower = text.lower()
#     if " and " in lower and " or " not in lower:
#         mode = "AND"
#     elif " or " in lower and " and " not in lower:
#         mode = "OR"
#     elif " and " in lower and " or " in lower:
#         # mixed: prefer AND (users often combine), but keep SINGLE groups by commas
#         mode = "AND"

#     # Build groups according to mode
#     targets: List[List[str]] = []
#     not_groups: List[List[str]] = []

#     # Split primary parts by connectors depending on mode
#     if mode == "AND":
#         parts = re.split(r"\band\b", cleaned, flags=re.I)
#     elif mode == "OR":
#         parts = re.split(r"\bor\b", cleaned, flags=re.I)
#     else:
#         parts = _tokenize(cleaned)

#     # For each part, handle 'not' inside and synonyms
#     for part in parts:
#         part = part.strip()
#         if not part:
#             continue
#         # handle "X not Y" inside the same segment
#         if " not " in part:
#             left, right = part.split(" not ", 1)
#             left = left.strip()
#             right = right.strip()
#             # expand left into canonical words
#             left_terms = []
#             for w in re.findall(r"[a-z0-9]+", left):
#                 w = _normalize_word(w)
#                 if not w:
#                     continue
#                 if w in REVERSE_MAP:
#                     left_terms.append(REVERSE_MAP[w])
#                 else:
#                     left_terms.append(w)
#             if left_terms:
#                 targets.append(left_terms)

#             # right -> not_groups
#             right_terms = []
#             for w in re.findall(r"[a-z0-9]+", right):
#                 w = _normalize_word(w)
#                 if not w:
#                     continue
#                 if w in REVERSE_MAP:
#                     right_terms.append(REVERSE_MAP[w])
#                 else:
#                     right_terms.append(w)
#             if right_terms:
#                 not_groups.append(right_terms)
#             continue

#         # simple part (no inline not)
#         # Extract words and expand synonyms/categories
#         words = [w for w in re.findall(r"[a-z0-9]+", part) if w not in {"the", "show", "images", "imageswith", "with", "containing", "contains", "take", "takeout", "takeoutall", "all"}]
#         if not words:
#             continue

#         group = []
#         for w in words:
#             w = _normalize_word(w)
#             if not w:
#                 continue
#             if w in REVERSE_MAP:
#                 group.append(REVERSE_MAP[w])
#             else:
#                 group.append(w)
#         if group:
#             targets.append(group)

#     # also detect explicit "not X" patterns at top level
#     for m in re.findall(r"(?:not|without|except)\s+([a-z0-9]+)", text, flags=re.I):
#         w = _normalize_word(m)
#         if not w:
#             continue
#         if w in REVERSE_MAP:
#             not_groups.append([REVERSE_MAP[w]])
#         else:
#             not_groups.append([w])

#     # Deduplicate groups and normalize to canonical names
#     def _norm_group(g: List[str]) -> List[str]:
#         out = []
#         for w in g:
#             if w in CATEGORY_MAP:
#                 # expand category to its values (we keep category itself too)
#                 out.append(w)
#                 out.extend(CATEGORY_MAP[w])
#             else:
#                 out.append(w)
#         # unique keeping order
#         seen = set()
#         final = []
#         for x in out:
#             if x not in seen:
#                 final.append(x)
#                 seen.add(x)
#         return final

#     targets = [_norm_group(g) for g in targets if g]
#     not_groups = [_norm_group(g) for g in not_groups if g]

#     return {
#         "mode": mode,
#         "targets": targets,
#         "not_groups": not_groups,
#         "start_date": start_date,
#         "end_date": end_date
#     }




# 3333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333


# import re
# from datetime import datetime
# from typing import List, Dict, Any, Optional

# CATEGORY_MAP = {
#     "animal": ["dog", "cat", "elephant", "giraffe", "zebra", "horse", "cow", "lion", "tiger", "bear", "sheep", "goat"],
#     "person": ["person", "man", "woman", "boy", "girl", "human", "people"],
#     "vehicle": ["car", "truck", "bus", "bike", "bicycle", "motorcycle"],
# }

# REVERSE_MAP = {}
# for k, vs in CATEGORY_MAP.items():
#     REVERSE_MAP[k] = k
#     for v in vs:
#         REVERSE_MAP[v] = k

# DATE_PATTERNS = [
#     (r"([A-Za-z]+ \d{1,2}, \d{4})", "%B %d, %Y"),
#     (r"(\d{1,2} [A-Za-z]+ \d{4})", "%d %B %Y"),
#     (r"(\d{1,2}/\d{1,2}/\d{2,4})", None),
#     (r"(\d{4}-\d{2}-\d{2})", "%Y-%m-%d"),
# ]

# def _parse_date_token(token: str) -> Optional[datetime]:
#     token = token.strip()
#     for pattern, fmt in DATE_PATTERNS:
#         m = re.match(pattern, token)
#         if not m: continue
#         s = m.group(1)
#         try:
#             if fmt:
#                 return datetime.strptime(s, fmt)
#             else:
#                 try: return datetime.strptime(s, "%m/%d/%Y")
#                 except: return datetime.strptime(s, "%m/%d/%y")
#         except: continue
#     return None

# def _normalize_word(w: str) -> str:
#     return re.sub(r"[^a-z0-9 ]", "", w.lower()).strip()

# def _tokenize(prompt: str) -> List[str]:
#     prompt = prompt.lower().replace(" but not ", " not ").replace(" without ", " not ")
#     return [p.strip() for p in re.split(r"[;,]", prompt) if p.strip()]

# def parse_natural_language(prompt: str) -> Dict[str, Any]:
#     if not prompt:
#         return {"mode":"SINGLE","targets":[],"not_groups":[],"start_date":None,"end_date":None}

#     text = prompt.strip()
#     start_date = end_date = None

#     m = re.search(r"between\s+(.+?)\s+and\s+(.+?)", text, flags=re.I)
#     if m:
#         start_date = _parse_date_token(m.group(1).strip())
#         end_date = _parse_date_token(m.group(2).strip())
#     else:
#         m2 = re.search(r"from\s+(.+?)\s+(?:to|-)\s+(.+?)", text, flags=re.I)
#         if m2:
#             start_date = _parse_date_token(m2.group(1).strip())
#             end_date = _parse_date_token(m2.group(2).strip())

#     if start_date and not end_date: end_date = start_date

#     cleaned = re.sub(r"(between\s+.+?\s+and\s+.+?$)|(from\s+.+?\s+(?:to|-)\s+.+?$)", " ", text, flags=re.I)
#     for pattern, _ in DATE_PATTERNS:
#         cleaned = re.sub(pattern, " ", cleaned, flags=re.I)

#     lower = text.lower()
#     if " and " in lower and " or " not in lower: mode="AND"
#     elif " or " in lower and " and " not in lower: mode="OR"
#     elif " and " in lower and " or " in lower: mode="AND"
#     else: mode="SINGLE"

#     parts = re.split(r"\band\b" if mode=="AND" else r"\bor\b", cleaned, flags=re.I) if mode!="SINGLE" else _tokenize(cleaned)
#     targets, not_groups = [], []

#     for part in parts:
#         part = part.strip()
#         if not part: continue
#         if " not " in part:
#             left,right = part.split(" not ",1)
#             left_terms = [REVERSE_MAP.get(_normalize_word(w), _normalize_word(w)) for w in re.findall(r"[a-z0-9]+", left)]
#             right_terms = [REVERSE_MAP.get(_normalize_word(w), _normalize_word(w)) for w in re.findall(r"[a-z0-9]+", right)]
#             if left_terms: targets.append(left_terms)
#             if right_terms: not_groups.append(right_terms)
#             continue

#         words = [w for w in re.findall(r"[a-z0-9]+", part) if w not in {"the","show","images","with","containing","take","all"}]
#         if words: targets.append([REVERSE_MAP.get(_normalize_word(w), _normalize_word(w)) for w in words])

#     for m in re.findall(r"(?:not|without|except)\s+([a-z0-9]+)", text, flags=re.I):
#         w = _normalize_word(m)
#         if w: not_groups.append([REVERSE_MAP.get(w,w)])

#     # Deduplicate without expanding synonyms (only expand explicit categories)
#     def _norm_group(g: List[str]) -> List[str]:
#         out=[]
#         for w in g:
#             if w in CATEGORY_MAP: out.extend([w]+CATEGORY_MAP[w])
#             else: out.append(w)
#         seen=set(); final=[]
#         for x in out:
#             if x not in seen: final.append(x); seen.add(x)
#         return final

#     targets = [_norm_group(g) for g in targets if g]
#     not_groups = [_norm_group(g) for g in not_groups if g]

#     return {"mode":mode,"targets":targets,"not_groups":not_groups,"start_date":start_date,"end_date":end_date}


import re
from datetime import datetime
from typing import List, Dict, Any, Optional

# ---- IMPORT YOUR 600-CLASS SYNONYM MAP ----
from synonyms import SYNONYM_MAP


DATE_PATTERNS = [
    (r"([A-Za-z]+ \d{1,2}, \d{4})", "%B %d, %Y"),
    (r"(\d{1,2} [A-Za-z]+ \d{4})", "%d %B %Y"),
    (r"(\d{1,2}/\d{1,2}/\d{2,4})", None),
    (r"(\d{4}-\d{2}-\d{2})", "%Y-%m-%d"),
]

# --------- NORMALIZATION ---------
def _normalize(w: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", w.lower()).strip()


# --------- MATCH SYNONYM USING SYNONYM_MAP ONLY ---------
def match_synonym(word: str) -> str:
    word = _normalize(word)

    # Exact canonical match
    if word in SYNONYM_MAP:
        return word

    # Search synonyms
    for canonical, syns in SYNONYM_MAP.items():
        for s in syns:
            if _normalize(s) == word:
                return canonical

    # If not found, return raw normalized word
    return word


# --------- DATE PARSER ---------
def _parse_date_token(token: str) -> Optional[datetime]:
    token = token.strip()
    for pattern, fmt in DATE_PATTERNS:
        m = re.match(pattern, token)
        if not m:
            continue
        s = m.group(1)
        try:
            if fmt:
                return datetime.strptime(s, fmt)
            else:
                try:
                    return datetime.strptime(s, "%m/%d/%Y")
                except:
                    return datetime.strptime(s, "%m/%d/%y")
        except:
            continue
    return None


# --------- MAIN NATURAL LANGUAGE PARSER ---------
def parse_natural_language(prompt: str) -> Dict[str, Any]:
    if not prompt:
        return {
            "mode": "SINGLE",
            "targets": [],
            "not_groups": [],
            "start_date": None,
            "end_date": None,
        }

    text = prompt.strip()

    # ===== DATE EXTRACTION =====
    start_date = end_date = None
    m = re.search(r"between\s+(.+?)\s+and\s+(.+?)", text, flags=re.I)
    if m:
        start_date = _parse_date_token(m.group(1))
        end_date   = _parse_date_token(m.group(2))

    # ===== MODE DETECTION =====
    # lower = text.lower()
    # if " and " in lower and " or " not in lower:
    #     mode = "AND"
    # elif " or " in lower and " and " not in lower:
    #     mode = "OR"
    # else:
    #     mode = "SINGLE"


    lower = text.lower()
    if " and " in lower and " or " not in lower:
        mode = "AND"
    elif " or " in lower and " and " not in lower:
        mode = "OR"
    elif " and " in lower and " or " in lower:
        # default to AND for mixed cases
        mode = "AND"
    else:
        mode = "SINGLE"


    # ===== WORD EXTRACTION =====
    words = re.findall(r"[a-z0-9]+", lower)

    # Resolve each word with your synonym dictionary
    resolved = [match_synonym(w) for w in words]

    # Single mode → each word separate
    # AND/OR mode → also treat each as separate filter
    targets = [[w] for w in resolved]

    return {
        "mode": mode,
        "targets": targets,
        "not_groups": [],
        "start_date": start_date,
        "end_date": end_date
    }
