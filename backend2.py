# import os
# from PIL import Image
# import numpy as np

# # YOLO
# try:
#     from ultralytics import YOLO
# except ImportError:
#     YOLO = None
#     print("❌ ultralytics not installed.")

# # Synonyms
# from openimages_synonyms import SYNONYM_MAP

# class SimplePromptTranslator:
#     def __init__(self, class_names):
#         self.class_names = class_names
#         self.class_word_sets = [set(name.lower().split()) for name in class_names]
#         self.synonym_map = SYNONYM_MAP

#     def expand_query(self, query):
#         query_lower = query.lower()
#         expanded_terms = [query_lower]
#         for word in query_lower.split():
#             for category, synonyms in self.synonym_map.items():
#                 if word in synonyms:
#                     expanded_terms.extend(synonyms)
#                     break
#         return list(set(expanded_terms))

#     def translate_prompt(self, user_query, top_k=5, threshold=0.1):
#         expanded_queries = self.expand_query(user_query)
#         scores = []
#         for i, class_word_set in enumerate(self.class_word_sets):
#             max_similarity = 0
#             for query in expanded_queries:
#                 query_words = set(query.split())
#                 if query_words and class_word_set:
#                     intersection = len(query_words.intersection(class_word_set))
#                     union = len(query_words.union(class_word_set))
#                     similarity = intersection / union if union > 0 else 0
#                 else:
#                     similarity = 0
#                 if query in self.class_names[i].lower():
#                     similarity += 0.8
#                 elif any(word in self.class_names[i].lower() for word in query_words):
#                     similarity += 0.3
#                 max_similarity = max(max_similarity, similarity)
#             scores.append((max_similarity, i))
#         scores.sort(reverse=True)
#         top_results = []
#         for score, idx in scores:
#             if score > threshold and len(top_results) < top_k:
#                 top_results.append({'class': self.class_names[idx], 'score': round(score, 3)})
#         return top_results

# class AdvancedImageFilter:
#     def __init__(self, model_path=None, confidence=0.2, language='en'):
#         self.model_path = model_path or "yolov8s-oiv7.pt"
#         self.confidence = confidence
#         self.language = language
#         self.model = None
#         self.load_model()
#         self.coco_classes = self.load_class_names()
#         self.translator = SimplePromptTranslator(self.coco_classes)

#     def load_class_names(self):
#         if self.model and hasattr(self.model, 'names'):
#             return list(self.model.names.values())
#         # fallback classes
#         return ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat']

#     def load_model(self):
#         if YOLO is None:
#             raise ImportError("YOLO not available")
#         if not os.path.exists(self.model_path):
#             raise FileNotFoundError(f"Model not found: {self.model_path}")
#         self.model = YOLO(self.model_path)
#         self.model.conf = self.confidence

#     def filter_folder(self, folder_path, prompt):
#         parsed_objects = [r['class'] for r in self.translator.translate_prompt(prompt, top_k=10)]
#         image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
#         all_files = os.listdir(folder_path)
#         image_files = [f for f in all_files if f.lower().endswith(image_extensions)]
#         matching_images = []
#         for image_file in image_files:
#             image_path = os.path.join(folder_path, image_file)
#             results = self.model(image_path, conf=self.confidence, verbose=False)
#             detected_objects = []
#             if results and len(results) > 0 and results[0].boxes is not None:
#                 for box in results[0].boxes:
#                     class_id = int(box.cls.item())
#                     name = self.model.names[class_id]
#                     detected_objects.append(name)
#             if any(obj in detected_objects for obj in parsed_objects):
#                 matching_images.append({'path': image_path, 'objects': detected_objects})
#         return matching_images













# import os
# import shutil
# from collections import Counter
# from PIL import Image

# # YOLO
# try:
#     from ultralytics import YOLO
# except ImportError:
#     YOLO = None
#     print("❌ ultralytics not installed.")

# # Synonyms
# from openimages_synonyms import SYNONYM_MAP


# # -------------------------------------------------------
# # FIXED TRANSLATOR – NOW FULLY ACCURATE
# # -------------------------------------------------------
# class SimplePromptTranslator:
#     def __init__(self, class_names):
#         self.class_names = class_names
#         self.class_lower = [c.lower() for c in class_names]
#         self.synonym_map = SYNONYM_MAP

#     def expand_query(self, query):
#         q = query.lower().strip()
#         words = q.split()

#         expanded = set([q])

#         for w in words:
#             for key, synonyms in self.synonym_map.items():
#                 if w in synonyms:
#                     expanded.update(synonyms)
#                     expanded.add(key)

#         return list(expanded)

#     def translate_prompt(self, user_query, top_k=10):
#         expanded = self.expand_query(user_query)

#         matched = []

#         for cls in self.class_lower:
#             for q in expanded:
#                 if q in cls:
#                     matched.append(cls)
#                     break

#         matched = list(dict.fromkeys(matched))
#         if len(matched) > top_k:
#             matched = matched[:top_k]

#         # return original class names, not lowercase
#         final = []
#         for m in matched:
#             idx = self.class_lower.index(m)
#             final.append(self.class_names[idx])

#         return final


# # -------------------------------------------------------
# # MAIN BACKEND
# # -------------------------------------------------------
# class AdvancedImageFilter:
#     def __init__(self, model_path=None, confidence=0.25, language='en'):
#         self.model_path = model_path or "yolov8m-oiv7.pt"
#         self.confidence = confidence
#         self.language = language
#         self.model = None
#         self.last_results = []

#         self.load_model()
#         self.coco_classes = self.load_class_names()
#         self.translator = SimplePromptTranslator(self.coco_classes)

#     def load_class_names(self):
#         if self.model and hasattr(self.model, 'names'):
#             return list(self.model.names.values())
#         return []

#     def load_model(self):
#         if YOLO is None:
#             raise ImportError("YOLO not available")
#         if not os.path.exists(self.model_path):
#             raise FileNotFoundError(f"Model not found: {self.model_path}")

#         self.model = YOLO(self.model_path)
#         self.model.conf = self.confidence

#     # -------------------------------------------------------
#     # PROMPT PARSER — TRUE AND/OR BEHAVIOR
#     # -------------------------------------------------------
#     def parse_prompt(self, prompt):
#         p = prompt.lower().strip()

#         # AND logic
#         if " and " in p:
#             parts = [x.strip() for x in p.split(" and ")]
#             groups = [self.translator.translate_prompt(x, top_k=7) for x in parts]
#             return {"type": "AND", "groups": groups}

#         # OR logic
#         if " or " in p:
#             parts = [x.strip() for x in p.split(" or ")]
#             groups = [self.translator.translate_prompt(x, top_k=7) for x in parts]
#             return {"type": "OR", "groups": groups}

#         # Single
#         g = self.translator.translate_prompt(p, top_k=10)
#         return {"type": "SINGLE", "groups": [g]}

#     # -------------------------------------------------------
#     # IMAGE ANALYZER — FIXED AND LOGIC
#     # -------------------------------------------------------
#     def analyze_image(self, image_path, parsed_prompt):
#         res = self.model(image_path, conf=self.confidence, verbose=False)

#         detected = []
#         if res and res[0].boxes is not None:
#             for box in res[0].boxes:
#                 cls = int(box.cls.item())
#                 detected.append(self.model.names[cls].lower())

#         detected_set = set(detected)

#         logic = parsed_prompt["type"]
#         groups = parsed_prompt["groups"]

#         # AND logic ⇒ EACH required group must be present
#         if logic == "AND":
#             for group in groups:
#                 group_lower = [g.lower() for g in group]
#                 if not any(g in detected_set for g in group_lower):
#                     return False, detected
#             return True, detected

#         # OR logic ⇒ ANY group match
#         if logic == "OR":
#             for group in groups:
#                 group_lower = [g.lower() for g in group]
#                 if any(g in detected_set for g in group_lower):
#                     return True, detected
#             return False, detected

#         # Single logic
#         group = groups[0]
#         group_lower = [g.lower() for g in group]
#         return any(g in detected_set for g in group_lower), detected

#     # -------------------------------------------------------
#     # FOLDER FILTER
#     # -------------------------------------------------------
#     def filter_folder(self, folder_path, prompt):
#         parsed = self.parse_prompt(prompt)

#         exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

#         files = [f for f in os.listdir(folder_path) if f.lower().endswith(exts)]

#         results = []

#         for f in files:
#             p = os.path.join(folder_path, f)
#             matched, det = self.analyze_image(p, parsed)
#             if matched:
#                 results.append({
#                     "path": p,
#                     "objects": det
#                 })

#         self.last_results = results
#         return results

#     # -------------------------------------------------------
#     # EXPORT SELECTED IMAGES
#     # -------------------------------------------------------
#     def export_results(self, destination_folder, selected_images=None, move=False):
#         if not self.last_results:
#             raise ValueError("No results to export.")

#         os.makedirs(destination_folder, exist_ok=True)

#         if selected_images is None:
#             items = self.last_results
#         else:
#             items = [x for x in self.last_results if x["path"] in selected_images]

#         for item in items:
#             src = item["path"]
#             dst = os.path.join(destination_folder, os.path.basename(src))
#             if move:
#                 shutil.move(src, dst)
#             else:
#                 shutil.copy2(src, dst)

#         return len(items)




# # backend2.py
# import os
# import shutil
# from collections import Counter
# from datetime import datetime
# from typing import List, Dict, Any, Optional, Tuple

# # YOLO
# try:
#     from ultralytics import YOLO
# except ImportError:
#     YOLO = None

# # NLP parser placeholder (replace with your actual parser)
# def parse_natural_language(prompt: str) -> Dict[str, Any]:
#     return {"mode": "SINGLE", "targets": [[prompt]], "not_groups": [], "start_date": None, "end_date": None}

# # Placeholder datetime extractor (replace with EXIF or real logic)
# def extract_image_datetime(image_path: str) -> Optional[datetime]:
#     try:
#         return datetime.fromtimestamp(os.path.getmtime(image_path))
#     except Exception:
#         return None

# # backend2.py
# import os
# import shutil
# from typing import List, Dict, Any, Optional, Tuple
# from collections import Counter
# from datetime import datetime
# from ultralytics import YOLO
# from newnlp_parser import parse_natural_language, CATEGORY_MAP

# class AdvancedImageFilter:
#     def __init__(self, model_path: Optional[str] = None, confidence: float = 0.25):
#         self.model_path = model_path or "yolov8m-oiv7.pt"
#         self.confidence = float(confidence)
#         self.model = None
#         self.last_results: List[Dict[str, Any]] = []

#         self._load_model_if_possible()

#     def _load_model_if_possible(self):
#         if YOLO is None:
#             self.model = None
#             return
#         if not os.path.exists(self.model_path):
#             raise FileNotFoundError(f"Model weights not found at: {self.model_path}")
#         self.model = YOLO(self.model_path)
#         # set default confidence
#         try:
#             setattr(self.model, "conf", float(self.confidence))
#         except Exception:
#             pass

#     # ---------- Detect a single image ----------
#     def detect_image(self, image_path: str) -> Dict[str, Any]:
#         if self.model is None:
#             return {"objects": [], "raw": {}}
#         try:
#             results = self.model(image_path, conf=self.confidence, verbose=False)
#             detected = []
#             if results and len(results) > 0 and getattr(results[0], "boxes", None) is not None:
#                 for box in results[0].boxes:
#                     cls_id = int(getattr(box, "cls", 0))
#                     name = str(self.model.names.get(cls_id, cls_id)).lower() if hasattr(self.model, "names") else str(cls_id)
#                     detected.append(name)
#             return {"objects": detected, "raw": {}}
#         except Exception:
#             return {"objects": [], "raw": {}}

#     # ---------- Analyze prompt ----------
#     def analyze_prompt(self, prompt: str) -> Dict[str, Any]:
#         parsed = parse_natural_language(prompt)
#         # ensure keys exist
#         parsed.setdefault("mode", "SINGLE")
#         parsed.setdefault("targets", [])
#         parsed.setdefault("not_groups", [])
#         parsed.setdefault("start_date", None)
#         parsed.setdefault("end_date", None)
#         return parsed

#     # ---------- Check if image matches ----------
#     def _image_matches(self, image_path: str, parsed_prompt: Dict[str, Any], use_date: bool = False) -> Tuple[bool, List[str]]:
#         # Date filter
#         img_dt = datetime.fromtimestamp(os.path.getmtime(image_path))
#         if use_date:
#             start_dt = parsed_prompt.get("start_date")
#             end_dt = parsed_prompt.get("end_date")
#             if start_dt and img_dt < start_dt:
#                 return False, []
#             if end_dt and img_dt > end_dt:
#                 return False, []

#         # Object detection
#         det_result = self.detect_image(image_path)
#         detected = [d.lower() for d in det_result.get("objects", [])]
#         detected_set = set(detected)

#         targets = parsed_prompt.get("targets", []) or []
#         not_groups = parsed_prompt.get("not_groups", []) or []
#         mode = parsed_prompt.get("mode", "SINGLE")

#         # Expand targets using CATEGORY_MAP
#         expanded_targets = []
#         for group in targets:
#             exp_group = set()
#             for t in group:
#                 if t in CATEGORY_MAP:
#                     exp_group.add(t)
#                     exp_group.update(CATEGORY_MAP[t])
#                 else:
#                     exp_group.add(t)
#             expanded_targets.append(exp_group)

#         # Expand not_groups similarly
#         expanded_not_groups = []
#         for group in not_groups:
#             exp_group = set()
#             for t in group:
#                 if t in CATEGORY_MAP:
#                     exp_group.add(t)
#                     exp_group.update(CATEGORY_MAP[t])
#                 else:
#                     exp_group.add(t)
#             expanded_not_groups.append(exp_group)

#         # Reject if any not_group member present
#         for ng in expanded_not_groups:
#             if detected_set.intersection(ng):
#                 return False, detected

#         # Mode logic
#         if not expanded_targets:
#             return True, detected  # no object filtering requested

#         if mode == "AND":
#             for group in expanded_targets:
#                 if not detected_set.intersection(group):
#                     return False, detected
#             return True, detected
#         elif mode == "OR" or mode == "SINGLE":
#             for group in expanded_targets:
#                 if detected_set.intersection(group):
#                     return True, detected
#             return False, detected
#         return False, detected

#     # ---------- Filter folder ----------
#     def filter_folder(self, folder_path: str, prompt: str, use_date_filter: bool = False) -> List[dict]:
#         parsed_prompt = self.analyze_prompt(prompt)
#         image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
#         all_files = os.listdir(folder_path)
#         image_files = [f for f in all_files if f.lower().endswith(image_extensions)]
#         matching_images = []

#         for image_file in image_files:
#             img_path = os.path.join(folder_path, image_file)
#             matched, counts = self._image_matches(img_path, parsed_prompt, use_date_filter)
#             if matched:
#                 matching_images.append({
#                     "path": img_path,
#                     "objects": counts
#                 })

#         self.last_results = matching_images
#         return matching_images

#     # ---------- Export ----------
#     def export_results(self, destination_folder: str, selected_images: Optional[List[str]] = None, move: bool = False) -> int:
#         if not self.last_results:
#             return 0
#         os.makedirs(destination_folder, exist_ok=True)
#         if selected_images is None:
#             to_export = self.last_results
#         else:
#             path_set = set(selected_images)
#             to_export = [r for r in self.last_results if r["path"] in path_set]

#         count = 0
#         for item in to_export:
#             src = item["path"]
#             dst = os.path.join(destination_folder, os.path.basename(src))
#             try:
#                 if move:
#                     shutil.move(src, dst)
#                 else:
#                     shutil.copy2(src, dst)
#                 count += 1
#             except Exception:
#                 continue
#         return count




# import os
# import shutil
# from collections import Counter
# from datetime import datetime
# from typing import List, Dict, Any, Optional, Tuple

# # YOLO
# try:
#     from ultralytics import YOLO
# except ImportError:
#     YOLO = None

# # Import your parser
# from newnlp_parser import parse_natural_language

# # Helper: extract image datetime (modification time)
# def extract_image_datetime(image_path: str) -> Optional[datetime]:
#     try:
#         return datetime.fromtimestamp(os.path.getmtime(image_path))
#     except Exception:
#         return None

# class AdvancedImageFilter:
#     def __init__(self, model_path: Optional[str] = None, confidence: float = 0.25):
#         self.model_path = model_path or "yolov8m-oiv7.pt"
#         self.confidence = float(confidence)
#         self.model = None
#         self.last_results: List[Dict[str, Any]] = []
#         self._load_model_if_possible()

#     def _load_model_if_possible(self):
#         if YOLO is None:
#             self.model = None
#             return
#         if not os.path.exists(self.model_path):
#             raise FileNotFoundError(f"Model weights not found at: {self.model_path}")
#         try:
#             self.model = YOLO(self.model_path)
#             try:
#                 setattr(self.model, "conf", float(self.confidence))
#             except Exception:
#                 pass
#         except Exception as e:
#             raise RuntimeError(f"Failed to load YOLO model: {e}")

#     # ---------- Detection ----------
#     def detect_image(self, image_path: str) -> Dict[str, Any]:
#         if self.model is None:
#             return {"objects": []}
#         try:
#             results = self.model(image_path, conf=self.confidence, verbose=False)
#             if not results or len(results) == 0:
#                 return {"objects": []}
#             r0 = results[0]
#             objects = []
#             if getattr(r0, "boxes", None) is not None:
#                 for box in r0.boxes:
#                     try:
#                         cls_id = int(box.cls.item())
#                     except Exception:
#                         cls_id = int(getattr(box, "cls", 0))
#                     name = None
#                     if hasattr(self.model, "names"):
#                         try:
#                             name = self.model.names.get(cls_id, str(cls_id)) if isinstance(self.model.names, dict) else self.model.names[cls_id]
#                         except Exception:
#                             name = str(cls_id)
#                     else:
#                         name = str(cls_id)
#                     objects.append(name.lower() if isinstance(name, str) else str(name))
#             return {"objects": objects}
#         except Exception:
#             return {"objects": []}

#     # ---------- Analyze prompt ----------
#     def analyze_prompt(self, prompt: str) -> Dict[str, Any]:
#         parsed = {}
#         try:
#             parsed = parse_natural_language(prompt) or {}
#         except Exception:
#             parsed = {}
#         mode = parsed.get("mode", "SINGLE")
#         targets = parsed.get("targets", []) or []
#         not_groups = parsed.get("not_groups", []) or []
#         start_date = parsed.get("start_date", None)
#         end_date = parsed.get("end_date", None)
#         return {
#             "mode": mode,
#             "targets": targets,
#             "not_groups": not_groups,
#             "start_date": start_date,
#             "end_date": end_date
#         }

#     # ---------- Core matching logic ----------
#     def _image_matches(
#         self,
#         image_path: str,
#         parsed_prompt: Dict[str, Any],
#         require_date: bool = False
#     ) -> Tuple[bool, List[str]]:
#         # Date filter
#         img_dt = extract_image_datetime(image_path)
#         if require_date:
#             start_dt = parsed_prompt.get("start_date")
#             end_dt = parsed_prompt.get("end_date")
#             if img_dt is None:
#                 return False, []
#             if start_dt and img_dt < start_dt:
#                 return False, []
#             if end_dt and img_dt > end_dt:
#                 return False, []

#         # Object detection
#         targets = parsed_prompt.get("targets", [])
#         not_groups = parsed_prompt.get("not_groups", [])
#         mode = parsed_prompt.get("mode", "SINGLE")

#         if not targets:
#             det_result = self.detect_image(image_path)
#             return True, det_result.get("objects", [])

#         det_result = self.detect_image(image_path)
#         detected = [d.lower() for d in det_result.get("objects", [])]
#         detected_set = set(detected)

#         # NOT groups
#         for ng in not_groups:
#             ng_lower = [g.lower() for g in ng]
#             if any(g in detected_set for g in ng_lower):
#                 return False, detected

#         # AND / OR / SINGLE logic
#         if mode == "AND":
#             for group in targets:
#                 group_lower = [g.lower() for g in group]
#                 if not any(g in detected_set for g in group_lower):
#                     return False, detected
#             return True, detected

#         if mode == "OR":
#             for group in targets:
#                 group_lower = [g.lower() for g in group]
#                 if any(g in detected_set for g in group_lower):
#                     return True, detected
#             return False, detected

#         # SINGLE
#         group = targets[0] if targets else []
#         group_lower = [g.lower() for g in group]
#         return (any(g in detected_set for g in group_lower), detected)

#     # ---------- Filter folder ----------
#     def filter_folder(
#         self,
#         folder_path: str,
#         prompt: str,
#         use_date_filter: bool = False
#     ) -> List[dict]:
#         parsed_prompt = self.analyze_prompt(prompt)
#         image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
#         all_files = os.listdir(folder_path)
#         image_files = [f for f in all_files if f.lower().endswith(image_extensions)]
#         matching_images = []

#         for image_file in image_files:
#             img_path = os.path.join(folder_path, image_file)
#             matched, counts = self._image_matches(img_path, parsed_prompt, require_date=use_date_filter)
#             if matched:
#                 matching_images.append({
#                     "path": img_path,
#                     "objects": counts
#                 })

#         self.last_results = matching_images
#         return matching_images

#     # ---------- Export ----------
#     def export_results(self, destination_folder: str, selected_images: Optional[List[str]] = None, move: bool = False) -> int:
#         if not self.last_results:
#             return 0
#         os.makedirs(destination_folder, exist_ok=True)
#         if selected_images is None:
#             to_export = self.last_results
#         else:
#             path_set = set(selected_images)
#             to_export = [r for r in self.last_results if r["path"] in path_set]

#         count = 0
#         for item in to_export:
#             src = item["path"]
#             dst = os.path.join(destination_folder, os.path.basename(src))
#             try:
#                 if move:
#                     shutil.move(src, dst)
#                 else:
#                     shutil.copy2(src, dst)
#                 count += 1
#             except Exception:
#                 continue
#         return count









# import os
# import shutil
# from collections import Counter
# from datetime import datetime
# from typing import List, Dict, Any, Optional, Tuple

# # YOLO
# try:
#     from ultralytics import YOLO
# except ImportError:
#     YOLO = None

# # Import your parser
# from newnlp_parser import parse_natural_language

# # Helper: extract image datetime (modification time)
# def extract_image_datetime(image_path: str) -> Optional[datetime]:
#     try:
#         return datetime.fromtimestamp(os.path.getmtime(image_path))
#     except Exception:
#         return None

# class AdvancedImageFilter:
#     def __init__(self, model_path: Optional[str] = None, confidence: float = 0.25):
#         self.model_path = model_path or "yolov8m-oiv7.pt"
#         self.confidence = float(confidence)
#         self.model = None
#         self.last_results: List[Dict[str, Any]] = []
#         self._load_model_if_possible()

#     def _load_model_if_possible(self):
#         if YOLO is None:
#             self.model = None
#             return
#         if not os.path.exists(self.model_path):
#             raise FileNotFoundError(f"Model weights not found at: {self.model_path}")
#         try:
#             self.model = YOLO(self.model_path)
#             try:
#                 setattr(self.model, "conf", float(self.confidence))
#             except Exception:
#                 pass
#         except Exception as e:
#             raise RuntimeError(f"Failed to load YOLO model: {e}")

#     # ---------- Detection ----------
#     def detect_image(self, image_path: str) -> Dict[str, Any]:
#         if self.model is None:
#             return {"objects": []}
#         try:
#             results = self.model(image_path, conf=self.confidence, verbose=False)
#             if not results or len(results) == 0:
#                 return {"objects": []}
#             r0 = results[0]
#             objects = []
#             if getattr(r0, "boxes", None) is not None:
#                 for box in r0.boxes:
#                     try:
#                         cls_id = int(box.cls.item())
#                     except Exception:
#                         cls_id = int(getattr(box, "cls", 0))
#                     name = None
#                     if hasattr(self.model, "names"):
#                         try:
#                             name = self.model.names.get(cls_id, str(cls_id)) if isinstance(self.model.names, dict) else self.model.names[cls_id]
#                         except Exception:
#                             name = str(cls_id)
#                     else:
#                         name = str(cls_id)
#                     objects.append(name.lower() if isinstance(name, str) else str(name))
#             return {"objects": objects}
#         except Exception:
#             return {"objects": []}

#     # ---------- Analyze prompt ----------
#     def analyze_prompt(self, prompt: str) -> Dict[str, Any]:
#         parsed = {}
#         try:
#             parsed = parse_natural_language(prompt) or {}
#         except Exception:
#             parsed = {}
#         mode = parsed.get("mode", "SINGLE")
#         targets = parsed.get("targets", []) or []
#         not_groups = parsed.get("not_groups", []) or []
#         start_date = parsed.get("start_date", None)
#         end_date = parsed.get("end_date", None)
#         return {
#             "mode": mode,
#             "targets": targets,
#             "not_groups": not_groups,
#             "start_date": start_date,
#             "end_date": end_date
#         }

#     # ---------- Core matching logic ----------
#     def _image_matches(
#         self,
#         image_path: str,
#         parsed_prompt: Dict[str, Any],
#         require_date: bool = False
#     ) -> Tuple[bool, List[str]]:
#         # Date filter
#         img_dt = extract_image_datetime(image_path)
#         if require_date:
#             start_dt = parsed_prompt.get("start_date")
#             end_dt = parsed_prompt.get("end_date")
#             if img_dt is None:
#                 return False, []
#             if start_dt and img_dt < start_dt:
#                 return False, []
#             if end_dt and img_dt > end_dt:
#                 return False, []

#         # Object detection
#         targets = parsed_prompt.get("targets", [])
#         not_groups = parsed_prompt.get("not_groups", [])
#         mode = parsed_prompt.get("mode", "SINGLE")

#         if not targets:
#             det_result = self.detect_image(image_path)
#             return True, det_result.get("objects", [])

#         det_result = self.detect_image(image_path)
#         detected = [d.lower() for d in det_result.get("objects", [])]
#         detected_set = set(detected)

#         # NOT groups
#         for ng in not_groups:
#             ng_lower = [g.lower() for g in ng]
#             if any(g in detected_set for g in ng_lower):
#                 return False, detected

#         # AND / OR / SINGLE logic
#         if mode == "AND":
#             for group in targets:
#                 group_lower = [g.lower() for g in group]
#                 if not any(g in detected_set for g in group_lower):
#                     return False, detected
#             return True, detected

#         if mode == "OR":
#             for group in targets:
#                 group_lower = [g.lower() for g in group]
#                 if any(g in detected_set for g in group_lower):
#                     return True, detected
#             return False, detected

#         # SINGLE
#         group = targets[0] if targets else []
#         group_lower = [g.lower() for g in group]
#         return (any(g in detected_set for g in group_lower), detected)

#     # ---------- Filter folder ----------
#     def filter_folder(
#         self,
#         folder_path: str,
#         prompt: str,
#         use_date_filter: bool = False
#     ) -> List[dict]:
#         parsed_prompt = self.analyze_prompt(prompt)
#         image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
#         all_files = os.listdir(folder_path)
#         image_files = [f for f in all_files if f.lower().endswith(image_extensions)]
#         matching_images = []

#         for image_file in image_files:
#             img_path = os.path.join(folder_path, image_file)
#             matched, counts = self._image_matches(img_path, parsed_prompt, require_date=use_date_filter)
#             if matched:
#                 matching_images.append({
#                     "path": img_path,
#                     "objects": counts
#                 })

#         self.last_results = matching_images
#         return matching_images

#     # ---------- Export ----------
#     def export_results(self, destination_folder: str, selected_images: Optional[List[str]] = None, move: bool = False) -> int:
#         if not self.last_results:
#             return 0
#         os.makedirs(destination_folder, exist_ok=True)
#         if selected_images is None:
#             to_export = self.last_results
#         else:
#             path_set = set(selected_images)
#             to_export = [r for r in self.last_results if r["path"] in path_set]

#         count = 0
#         for item in to_export:
#             src = item["path"]
#             dst = os.path.join(destination_folder, os.path.basename(src))
#             try:
#                 if move:
#                     shutil.move(src, dst)
#                 else:
#                     shutil.copy2(src, dst)
#                 count += 1
#             except Exception:
#                 continue
#         return count



























# import os
# import shutil
# from collections import Counter
# from datetime import datetime
# from typing import List, Dict, Any, Optional, Tuple

# # YOLO
# try:
#     from ultralytics import YOLO
# except ImportError:
#     YOLO = None

# # Import your parser
# from newnlp_parser import parse_natural_language

# # Helper: extract image datetime (modification time)
# def extract_image_datetime(image_path: str) -> Optional[datetime]:
#     try:
#         return datetime.fromtimestamp(os.path.getmtime(image_path))
#     except Exception:
#         return None

# class AdvancedImageFilter:
#     def __init__(self, model_path: Optional[str] = None, confidence: float = 0.25):
#         self.model_path = model_path or "yolov8m-oiv7.pt"
#         self.confidence = float(confidence)
#         self.model = None
#         self.last_results: List[Dict[str, Any]] = []
#         self._load_model_if_possible()

#     def _load_model_if_possible(self):
#         if YOLO is None:
#             self.model = None
#             return
#         if not os.path.exists(self.model_path):
#             raise FileNotFoundError(f"Model weights not found at: {self.model_path}")
#         try:
#             self.model = YOLO(self.model_path)
#             try:
#                 setattr(self.model, "conf", float(self.confidence))
#             except Exception:
#                 pass
#         except Exception as e:
#             raise RuntimeError(f"Failed to load YOLO model: {e}")

#     # ---------- Detection ----------
#     def detect_image(self, image_path: str) -> Dict[str, Any]:
#         if self.model is None:
#             return {"objects": []}
#         try:
#             results = self.model(image_path, conf=self.confidence, verbose=False)
#             if not results or len(results) == 0:
#                 return {"objects": []}
#             r0 = results[0]
#             objects = []
#             if getattr(r0, "boxes", None) is not None:
#                 for box in r0.boxes:
#                     try:
#                         cls_id = int(box.cls.item())
#                     except Exception:
#                         cls_id = int(getattr(box, "cls", 0))
#                     name = None
#                     if hasattr(self.model, "names"):
#                         try:
#                             name = self.model.names.get(cls_id, str(cls_id)) if isinstance(self.model.names, dict) else self.model.names[cls_id]
#                         except Exception:
#                             name = str(cls_id)
#                     else:
#                         name = str(cls_id)
#                     objects.append(name.lower() if isinstance(name, str) else str(name))
#             return {"objects": objects}
#         except Exception:
#             return {"objects": []}

#     # ---------- Analyze prompt ----------
#     def analyze_prompt(self, prompt: str) -> Dict[str, Any]:
#         parsed = {}
#         try:
#             parsed = parse_natural_language(prompt) or {}
#         except Exception:
#             parsed = {}
#         mode = parsed.get("mode", "SINGLE")
#         targets = parsed.get("targets", []) or []
#         not_groups = parsed.get("not_groups", []) or []
#         start_date = parsed.get("start_date", None)
#         end_date = parsed.get("end_date", None)
#         return {
#             "mode": mode,
#             "targets": targets,
#             "not_groups": not_groups,
#             "start_date": start_date,
#             "end_date": end_date
#         }

#     # ---------- Core matching logic ----------
#     def _image_matches(
#         self,
#         image_path: str,
#         parsed_prompt: Dict[str, Any],
#         require_date: bool = False
#     ) -> Tuple[bool, List[str]]:
#         # Date filter
#         img_dt = extract_image_datetime(image_path)
#         if require_date:
#             start_dt = parsed_prompt.get("start_date")
#             end_dt = parsed_prompt.get("end_date")
#             if img_dt is None:
#                 return False, []
#             if start_dt and img_dt < start_dt:
#                 return False, []
#             if end_dt and img_dt > end_dt:
#                 return False, []

#         # Object detection
#         targets = parsed_prompt.get("targets", [])
#         not_groups = parsed_prompt.get("not_groups", [])
#         mode = parsed_prompt.get("mode", "SINGLE")

#         if not targets:
#             det_result = self.detect_image(image_path)
#             return True, det_result.get("objects", [])

#         det_result = self.detect_image(image_path)
#         detected = [d.lower() for d in det_result.get("objects", [])]
#         detected_set = set(detected)

#         # NOT groups
#         for ng in not_groups:
#             ng_lower = [g.lower() for g in ng]
#             if any(g in detected_set for g in ng_lower):
#                 return False, detected

#         # AND / OR / SINGLE logic
#         if mode == "AND":
#             for group in targets:
#                 group_lower = [g.lower() for g in group]
#                 if not any(g in detected_set for g in group_lower):
#                     return False, detected
#             return True, detected

#         if mode == "OR":
#             for group in targets:
#                 group_lower = [g.lower() for g in group]
#                 if any(g in detected_set for g in group_lower):
#                     return True, detected
#             return False, detected

#         # SINGLE
#         group = targets[0] if targets else []
#         group_lower = [g.lower() for g in group]
#         return (any(g in detected_set for g in group_lower), detected)

#     # ---------- Filter folder ----------
#     def filter_folder(
#         self,
#         folder_path: str,
#         prompt: str,
#         use_date_filter: bool = False
#     ) -> List[dict]:
#         parsed_prompt = self.analyze_prompt(prompt)
#         image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
#         all_files = os.listdir(folder_path)
#         image_files = [f for f in all_files if f.lower().endswith(image_extensions)]
#         matching_images = []

#         for image_file in image_files:
#             img_path = os.path.join(folder_path, image_file)
#             matched, counts = self._image_matches(img_path, parsed_prompt, require_date=use_date_filter)
#             if matched:
#                 matching_images.append({
#                     "path": img_path,
#                     "objects": counts
#                 })

#         self.last_results = matching_images
#         return matching_images

#     # ---------- Export ----------
#     def export_results(self, destination_folder: str, selected_images: Optional[List[str]] = None, move: bool = False) -> int:
#         if not self.last_results:
#             return 0
#         os.makedirs(destination_folder, exist_ok=True)
#         if selected_images is None:
#             to_export = self.last_results
#         else:
#             path_set = set(selected_images)
#             to_export = [r for r in self.last_results if r["path"] in path_set]

#         count = 0
#         for item in to_export:
#             src = item["path"]
#             dst = os.path.join(destination_folder, os.path.basename(src))
#             try:
#                 if move:
#                     shutil.move(src, dst)
#                 else:
#                     shutil.copy2(src, dst)
#                 count += 1
#             except Exception:
#                 continue
#         return count



# this above code is baseline it detects okay but not good enough   111111111111111111111111111111111111














# # below is second last code 2


# # backend2.py
# import os
# import shutil
# from datetime import datetime
# from typing import List, Dict, Any, Optional, Tuple

# # YOLO
# try:
#     from ultralytics import YOLO
# except ImportError:
#     YOLO = None

# # Import your parser
# from newnlp_parser import parse_natural_language

# # Helper: extract image datetime (modification time)
# def extract_image_datetime(image_path: str) -> Optional[datetime]:
#     try:
#         return datetime.fromtimestamp(os.path.getmtime(image_path))
#     except Exception:
#         return None

# class AdvancedImageFilter:
#     def __init__(self, model_path: Optional[str] = None, confidence: float = 0.25):
#         self.model_path = model_path or "yolov8m-oiv7.pt"
#         self.confidence = float(confidence)
#         self.model = None
#         self.last_results: List[Dict[str, Any]] = []
#         self._load_model_if_possible()

#     def _load_model_if_possible(self):
#         if YOLO is None:
#             self.model = None
#             return
#         if not os.path.exists(self.model_path):
#             raise FileNotFoundError(f"Model weights not found at: {self.model_path}")
#         try:
#             self.model = YOLO(self.model_path)
#             try:
#                 setattr(self.model, "conf", float(self.confidence))
#             except Exception:
#                 pass
#         except Exception as e:
#             raise RuntimeError(f"Failed to load YOLO model: {e}")

#     # ---------- Detection ----------
#     def detect_image(self, image_path: str) -> Dict[str, Any]:
#         if self.model is None:
#             return {"objects": []}
#         try:
#             results = self.model(image_path, conf=self.confidence, verbose=False)
#             if not results or len(results) == 0:
#                 return {"objects": []}
#             r0 = results[0]
#             objects = []
#             if getattr(r0, "boxes", None) is not None:
#                 for box in r0.boxes:
#                     try:
#                         cls_id = int(box.cls.item())
#                     except Exception:
#                         cls_id = int(getattr(box, "cls", 0))
#                     name = None
#                     if hasattr(self.model, "names"):
#                         try:
#                             name = self.model.names.get(cls_id, str(cls_id)) if isinstance(self.model.names, dict) else self.model.names[cls_id]
#                         except Exception:
#                             name = str(cls_id)
#                     else:
#                         name = str(cls_id)
#                     objects.append(name.lower() if isinstance(name, str) else str(name))
#             return {"objects": objects}
#         except Exception:
#             return {"objects": []}

#     # ---------- Analyze prompt ----------
#     def analyze_prompt(self, prompt: str) -> Dict[str, Any]:
#         parsed = {}
#         try:
#             parsed = parse_natural_language(prompt) or {}
#         except Exception:
#             parsed = {}
#         mode = parsed.get("mode", "SINGLE")
#         targets = parsed.get("targets", []) or []
#         not_groups = parsed.get("not_groups", []) or []
#         start_date = parsed.get("start_date", None)
#         end_date = parsed.get("end_date", None)
#         return {
#             "mode": mode,
#             "targets": targets,
#             "not_groups": not_groups,
#             "start_date": start_date,
#             "end_date": end_date
#         }

#     # ---------- Core matching logic ----------
#     def _image_matches(
#         self,
#         image_path: str,
#         parsed_prompt: Dict[str, Any],
#         require_date: bool = False
#     ) -> Tuple[bool, List[str]]:
#         # Date filter
#         img_dt = extract_image_datetime(image_path)
#         if require_date:
#             start_dt = parsed_prompt.get("start_date")
#             end_dt = parsed_prompt.get("end_date")
#             if img_dt is None:
#                 return False, []
#             if start_dt and img_dt < start_dt:
#                 return False, []
#             if end_dt and img_dt > end_dt:
#                 return False, []

#         # Object detection
#         targets = parsed_prompt.get("targets", [])
#         not_groups = parsed_prompt.get("not_groups", [])
#         mode = parsed_prompt.get("mode", "SINGLE")

#         det_result = self.detect_image(image_path)
#         detected = [d.lower() for d in det_result.get("objects", [])]
#         detected_set = set(detected)

#         # NOT groups: any detected in not_groups -> reject
#         for ng in not_groups:
#             ng_lower = [g.lower() for g in ng]
#             if any(g in detected_set for g in ng_lower):
#                 return False, detected

#         # AND logic: all groups must match at least one object
#         if mode == "AND":
#             for group in targets:
#                 group_lower = [g.lower() for g in group]
#                 if not any(g in detected_set for g in group_lower):
#                     return False, detected
#             return True, detected

#         # OR logic: any group matches -> accept
#         if mode == "OR":
#             for group in targets:
#                 group_lower = [g.lower() for g in group]
#                 if any(g in detected_set for g in group_lower):
#                     return True, detected
#             return False, detected

#         # SINGLE mode: only first target group
#         group = targets[0] if targets else []
#         group_lower = [g.lower() for g in group]
#         return (any(g in detected_set for g in group_lower), detected)

#     # ---------- Filter folder ----------
#     def filter_folder(
#         self,
#         folder_path: str,
#         prompt: str,
#         use_date_filter: bool = False
#     ) -> List[dict]:
#         parsed_prompt = self.analyze_prompt(prompt)
#         image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
#         all_files = os.listdir(folder_path)
#         image_files = [f for f in all_files if f.lower().endswith(image_extensions)]
#         matching_images = []

#         for image_file in image_files:
#             img_path = os.path.join(folder_path, image_file)
#             matched, counts = self._image_matches(img_path, parsed_prompt, require_date=use_date_filter)
#             if matched:
#                 matching_images.append({
#                     "path": img_path,
#                     "objects": counts
#                 })

#         self.last_results = matching_images
#         return matching_images
# # ---------- Export ----------
# def export_results(self, destination_folder: str, selected_images: Optional[List[str]] = None, move: bool = False) -> int:
#     if not self.last_results:
#         print("No images to export.")
#         return 0

#     os.makedirs(destination_folder, exist_ok=True)

#     if selected_images is None:
#         to_export = self.last_results
#     else:
#         # Convert all paths to absolute to ensure matching
#         selected_set = set(os.path.abspath(p) for p in selected_images)
#         to_export = [r for r in self.last_results if os.path.abspath(r["path"]) in selected_set]

#     count = 0
#     for item in to_export:
#         src = item["path"]
#         dst = os.path.join(destination_folder, os.path.basename(src))
#         try:
#             if move:
#                 shutil.move(src, dst)
#                 print(f"Moved: {src} -> {dst}")
#             else:
#                 shutil.copy2(src, dst)
#                 print(f"Copied: {src} -> {dst}")
#             count += 1
#         except Exception as e:
#             print(f"Failed to copy/move {src}: {e}")

#     print(f"Total exported: {count}")
#     return count




















import os, shutil
from typing import List, Dict, Any, Optional, Tuple
from ultralytics import YOLO
from newnlp_parser import parse_natural_language
from synonyms import SYNONYM_MAP
from datetime import datetime
from PIL import Image, ExifTags

# -----------------------------------
# BUILD REVERSE MAP (synonyms → canonical)
# -----------------------------------
REVERSE_MAP = {}
for canonical, syns in SYNONYM_MAP.items():
    c = canonical.lower().strip()
    REVERSE_MAP[c] = c
    for s in syns:
        REVERSE_MAP[s.lower().strip()] = c

print(f"[Synonyms] Loaded {len(REVERSE_MAP)} entries")

# -----------------------------------
# DATE EXTRACTOR
# -----------------------------------
def extract_image_datetime(img_path: str) -> Optional[datetime]:
    try:
        img = Image.open(img_path)
        exif = img._getexif()
        if exif:
            for tag, val in exif.items():
                decoded = ExifTags.TAGS.get(tag, tag)
                if decoded == "DateTimeOriginal":
                    return datetime.strptime(val, "%Y:%m:%d %H:%M:%S")
    except:
        pass
    try:
        return datetime.fromtimestamp(os.path.getmtime(img_path))
    except:
        return None


# ============================================================
#                     ADVANCED IMAGE FILTER
# ============================================================
class AdvancedImageFilter:
    def __init__(self, model_path: str, confidence: float = 0.25):
        self.model_path = model_path
        self.confidence = float(confidence)
        self.model = YOLO(self.model_path)
        self.last_results: List[Dict[str, Any]] = []

    # ----------------------------------------------------------
    # PROMPT PARSER WRAPPER
    # ----------------------------------------------------------
    def analyze_prompt(self, text: str):
        return parse_natural_language(text)

    # ----------------------------------------------------------
    # YOLO DETECTION
    # ----------------------------------------------------------
        # -------------------
    def detect_image(self, image_path: str) -> Dict[str, Any]:
        """
        Run YOLO on a single image and return:
          { "objects": [name1, name2, ...], "detections": [ (name, conf), ... ] }
        Keeps names normalized (lowercase).
        """
        try:
            # Run inference at configured confidence (model may filter below that)
            results = self.model(image_path, conf=self.confidence, verbose=False)
            if not results:
                return {"objects": [], "detections": []}
            r0 = results[0]
            detections = []
            objects = []
            if getattr(r0, "boxes", None) is not None:
                for box in r0.boxes:
                    try:
                        cls_id = int(box.cls)
                    except Exception:
                        cls_id = int(getattr(box, "cls", 0))
                    # box.conf may be a tensor -- get float
                    try:
                        conf = float(box.conf) if hasattr(box, "conf") else None
                    except:
                        conf = None

                    # robust name lookup
                    name = None
                    if hasattr(self.model, "names"):
                        try:
                            name = (self.model.names.get(cls_id, str(cls_id))
                                    if isinstance(self.model.names, dict)
                                    else self.model.names[cls_id])
                        except Exception:
                            name = str(cls_id)
                    else:
                        name = str(cls_id)

                    name = str(name).lower()
                    detections.append((name, conf))
                    objects.append(name)
            return {"objects": objects, "detections": detections}
        except Exception as e:
            print(f"[detect_image] inference error for {image_path}: {e}")
            return {"objects": [], "detections": []}


    # -------------------
    def _image_matches(self, image_path: str, parsed_prompt: Dict[str, Any], require_date: bool) -> Tuple[bool, List[str]]:
        """
        Per-image check used by OR/SINGLE and fallback for AND.
        Returns (matched_bool, list_of_detected_normalized_names)
        """
        # Date filter
        if require_date:
            img_dt = extract_image_datetime(image_path) if "extract_image_datetime" in globals() else None
            if img_dt is None:
                return False, []
            if parsed_prompt.get("start_date") and img_dt < parsed_prompt["start_date"]:
                return False, []
            if parsed_prompt.get("end_date") and img_dt > parsed_prompt["end_date"]:
                return False, []

        det = self.detect_image(image_path)
        # normalize detected names via REVERSE_MAP
        detected_raw = det.get("objects", [])
        detected_norm = [REVERSE_MAP.get(x, x) for x in detected_raw]
        detected_set = set(detected_norm)

        targets = parsed_prompt.get("targets", []) or []
        not_groups = parsed_prompt.get("not_groups", []) or []
        mode = parsed_prompt.get("mode", "SINGLE")

        # NOT: if any not-group member present => reject
        for ng in not_groups:
            ng_norm = [REVERSE_MAP.get(x.lower(), x.lower()) for x in ng]
            if any(x in detected_set for x in ng_norm):
                # debug:
                # print(f"[NOT] image {os.path.basename(image_path)} rejected because {detected_set & set(ng_norm)}")
                return False, detected_norm

        # OR mode: union of all target groups
        if mode == "OR":
            flat = []
            for g in targets:
                for t in g:
                    flat.append(REVERSE_MAP.get(t.lower(), t.lower()))
            if any(t in detected_set for t in flat):
                return True, detected_norm
            return False, detected_norm

        # SINGLE mode => only first group matters
        if mode == "SINGLE":
            if not targets:
                return True, detected_norm
            first = [REVERSE_MAP.get(t.lower(), t.lower()) for t in targets[0]]
            if any(t in detected_set for t in first):
                return True, detected_norm
            return False, detected_norm

        # AND fallback: require all flattened targets (less efficient)
        if mode == "AND":
            flat = []
            for g in targets:
                for t in g:
                    flat.append(REVERSE_MAP.get(t.lower(), t.lower()))
            if all(t in detected_set for t in flat):
                return True, detected_norm
            return False, detected_norm

        return False, detected_norm


    # -------------------
    def filter_folder(
        self,
        folder_path: str,
        prompt: str,
        use_date_filter: bool = False,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
   
    # Parse prompt
            parsed = self.analyze_prompt(prompt)
            if start_date:
                parsed["start_date"] = start_date
            if end_date:
                parsed["end_date"] = end_date

            require_date = bool(use_date_filter or parsed.get("start_date") or parsed.get("end_date"))
            mode = parsed.get("mode", "SINGLE")
            targets = parsed.get("targets", []) or []

            # List all images
            exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
            files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(exts)]

            # Case 1: No targets → return all (filtered by date)
            if not targets:
                out = []
                for p in files:
                    ok, objs = self._image_matches(p, parsed, require_date)
                    if ok:
                        out.append({"path": p, "objects": objs})
                self.last_results = out
                return out

            # Case 2: AND logic
            if mode == "AND":
                candidates = files[:]  # start with all images
                for group in targets:
                    # Normalize group names
                    group_norm = [REVERSE_MAP.get(t.lower(), t.lower()) for t in group]
                    next_candidates = []

                    for p in candidates:
                        # Date filter
                        if require_date:
                            dt = extract_image_datetime(p)
                            if not dt:
                                continue
                            if parsed.get("start_date") and dt < parsed["start_date"]:
                                continue
                            if parsed.get("end_date") and dt > parsed["end_date"]:
                                continue

                        # Detect objects
                        det = self.detect_image(p)
                        detected_norm = [REVERSE_MAP.get(x, x) for x in det.get("objects", [])]
                        detected_set = set(detected_norm)

                        # Keep only images that have ANY object from this group
                        if any(t in detected_set for t in group_norm):
                            # Keep only objects relevant to current group
                            filtered_objects = [obj for obj in detected_norm if obj in group_norm]
                            next_candidates.append({"path": p, "objects": filtered_objects})

                    # Update candidates for next group
                    candidates = [c["path"] for c in next_candidates]
                    if not candidates:
                        self.last_results = []
                        return []

                # Final collection: include only objects in the prompt
                final_results = []
                prompt_words = set()
                for g in targets:
                    for t in g:
                        prompt_words.add(REVERSE_MAP.get(t.lower(), t.lower()))

                for p in candidates:
                    det = self.detect_image(p)
                    detected_norm = [REVERSE_MAP.get(x, x) for x in det.get("objects", [])]
                    final_objects = [obj for obj in detected_norm if obj in prompt_words]
                    final_results.append({"path": p, "objects": final_objects})

                self.last_results = final_results
                return final_results

            # Case 3: OR / SINGLE
            out = []
            for p in files:
                ok, objs = self._image_matches(p, parsed, require_date)
                if ok:
                    out.append({"path": p, "objects": objs})

            self.last_results = out
            return out
    def export_results(self, destination_folder: str, selected_images: Optional[List[str]] = None, move: bool = False) -> int:
        """
        Export (copy or move) images from the last query results.
        If `selected_images` is None, export all images in last_results.
        """
        if not hasattr(self, "last_results") or not self.last_results:
            print("[export_results] No last_results to export.")
            return 0

        os.makedirs(destination_folder, exist_ok=True)

        # determine which images to export
        if selected_images is None:
            to_export = self.last_results
        else:
            selected_set = set(os.path.abspath(p) for p in selected_images)
            to_export = [r for r in self.last_results if os.path.abspath(r["path"]) in selected_set]

        count = 0
        for item in to_export:
            src = item["path"]
            dst = os.path.join(destination_folder, os.path.basename(src))
            try:
                if move:
                    shutil.move(src, dst)
                else:
                    shutil.copy2(src, dst)
                count += 1
            except Exception as e:
                print(f"[export_results] Failed for {src}: {e}")

        return count