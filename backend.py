# backend.py
"""
Advanced backend for YOLOv8 detection + simple prompt-to-class translation.
Designed to be used with the provided Streamlit frontend (app.py).
"""

from typing import List, Dict, Any
import os
import traceback

try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    print("ultralytics not installed or failed to import. Install with `pip install ultralytics`.")
    print(e)

from PIL import Image
import numpy as np

# Try to import richer prompt mappings if user has promptmodeling.py
try:
    import promptmodeling as pm
    _HAS_PROMPTMODELING = True
except Exception:
    _HAS_PROMPTMODELING = False


class SimplePromptTranslator:
    """
    Translate a user prompt (like 'kutta' or 'dog' or 'dog or cat') into a list of
    target class names (as strings) using synonyms and simple parsing.
    """
    def __init__(self, class_names: List[str], language: str = "en", threshold: float = 0.2):
        self.class_names = class_names or []
        self.language = language
        self.threshold = threshold

        # base synonym map (expand as needed)
        self.synonym_map = {
            "person": ["person", "man", "woman", "people", "insaan", "aadmi", "aurat"],
            "dog": ["dog", "kutta", "pooch"],
            "cat": ["cat", "billi"],
            "bird": ["bird", "parrot", "crow", "sparrow"],
            "car": ["car", "vehicle", "gaadi"],
            "bicycle": ["bicycle", "cycle", "bike"],
            "motorbike": ["motorbike", "motorcycle", "bike"],
            "horse": ["horse", "ghoda"],
            "cow": ["cow", "gai"],
            "sheep": ["sheep", "bhed"],
            # add more as needed...
        }

        # If promptmodeling.py exists, attempt to merge its mappings
        if _HAS_PROMPTMODELING:
            try:
                if hasattr(pm, "synonym_map"):
                    self.synonym_map.update(pm.synonym_map)
            except Exception:
                pass

        # add language-specific synonyms (small examples)
        if language == "hi":
            # Hindi additions (expand as you like)
            self.synonym_map.update({
                "dog": ["kutta", "dog", "pooch"],
                "cat": ["billi", "cat"],
                "person": ["insaan", "aadmi", "aurat", "vyakti"]
            })

    def expand_prompt(self, prompt: str) -> List[str]:
        """
        Given a prompt string, return a list of class names (from model class list)
        that are relevant to the prompt according to synonyms and simple matching.
        """
        if not prompt:
            return []

        prompt = prompt.lower().strip()

        # split by common separators (and/or/,)
        tokens = [t.strip() for t in re_split_separators(prompt) if t.strip()]
        target_names = set()

        # For each token, find synonyms, then match to class_names via substring match or exact match
        for token in tokens:
            # match direct class name
            for cname in self.class_names:
                if token == cname.lower() or token in cname.lower():
                    target_names.add(cname)
            # match synonyms
            for key, syns in self.synonym_map.items():
                if token in syns:
                    # map key -> model class if present
                    for cname in self.class_names:
                        if key.lower() == cname.lower() or key.lower() in cname.lower():
                            target_names.add(cname)
            # fallback: substring match against class names
            for cname in self.class_names:
                if token in cname.lower():
                    target_names.add(cname)

        return sorted(list(target_names))


def re_split_separators(text: str) -> List[str]:
    # simple split by common separators
    for sep in [" or ", " OR ", "/", ",", ";", " and ", "+", "|"]:
        text = text.replace(sep, "||")
    return text.split("||")


class AdvancedImageFilter:
    """
    Loads YOLOv8 model and runs detection on provided images (PIL or file paths).
    Exposes methods for:
      - loading/reloading model
      - running detect on a list of images
    """

    def __init__(self, model_path: str = "yolov8m.pt", confidence: float = 0.1, language: str = "en"):
        self.model_path = model_path or "yolov8m.pt"
        self.confidence = float(confidence)
        self.language = language
        self.model = None
        self.names = []
        self.translator = None
        # attempt initial load
        self.load_model()
        # initialize translator after we have class names
        self.translator = SimplePromptTranslator(self.names, language=self.language)

    def load_model(self) -> None:
        """
        Load YOLO model. Safe to call multiple times (reload).
        """
        if YOLO is None:
            raise ImportError("ultralytics YOLO is not available. Install ultralytics package.")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model weights not found at: {self.model_path}")

        try:
            self.model = YOLO(self.model_path)
            # Try to fetch names from model if available
            try:
                # ultralytics results usually expose .names
                self.names = list(self.model.model.names.values()) if hasattr(self.model, "model") and hasattr(self.model.model, "names") else []
            except Exception:
                # fallback: empty
                self.names = []

            # set default conf if attribute exists
            try:
                # some ultralytics versions support setting model.conf
                setattr(self.model, "conf", float(self.confidence))
            except Exception:
                pass

            print(f"Loaded model from {self.model_path} with confidence={self.confidence}")
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Failed to load model from {self.model_path}: {e}")

    def set_confidence(self, c: float):
        self.confidence = float(c)
        try:
            setattr(self.model, "conf", float(self.confidence))
        except Exception:
            pass

    def set_language(self, language: str):
        self.language = language
        if self.translator:
            self.translator.language = language

    def detect_image(self, image_input: Any) -> Dict[str, Any]:
        """
        Detect objects in a single image.
        image_input may be:
          - PIL.Image
          - numpy array (HWC)
          - path to image file
          - bytes-like (handled by PIL)
        Returns a dict:
          {
            "objects": [ {"object": name, "confidence": float, "bbox": [x1,y1,x2,y2]}, ... ],
            "raw": raw_results_object (safe repr or summary)
          }
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Prepare image for model
        pil_img = None
        img_path_used = None
        try:
            if isinstance(image_input, str):
                # path
                img_path_used = image_input
                pil_img = Image.open(image_input).convert("RGB")
                img_for_model = np.asarray(pil_img)
            elif isinstance(image_input, Image.Image):
                pil_img = image_input.convert("RGB")
                img_for_model = np.asarray(pil_img)
            else:
                # try bytes-like or numpy
                try:
                    pil_img = Image.open(image_input).convert("RGB")
                    img_for_model = np.asarray(pil_img)
                except Exception:
                    # if it's already numpy array
                    if isinstance(image_input, np.ndarray):
                        img_for_model = image_input
                        pil_img = Image.fromarray(img_for_model)
                    else:
                        raise ValueError("Unsupported image input type for detect_image")
        except Exception as e:
            return {"error": f"Failed to load image: {e}"}

        detections = []
        raw_summary = {}
        try:
            # Run model; specify conf via kwargs where possible.
            # We attempt a few ways to call model to be robust across ultralytics versions.
            results = None
            try:
                # preferred: use predict API if present
                results = self.model.predict(source=img_for_model, conf=float(self.confidence), verbose=False)
            except Exception:
                try:
                    results = self.model(img_for_model)
                except Exception as e:
                    # As last resort attempt model.predict with no conf
                    results = self.model.predict(source=img_for_model, verbose=False)

            if results is None or len(results) == 0:
                raw_summary = {"note": "no results returned by model"}
            else:
                # results[0] is typical
                r0 = results[0]
                # try to get names mapping
                names_map = {}
                try:
                    if hasattr(r0, "names") and isinstance(r0.names, dict):
                        names_map = r0.names
                    elif isinstance(self.names, list) and len(self.names) > 0:
                        names_map = {i: n for i, n in enumerate(self.names)}
                except Exception:
                    names_map = {}

                raw_summary['names_len'] = len(names_map)

                # iterate boxes
                boxes = []
                try:
                    # r0.boxes is common and iterable
                    for b in getattr(r0, "boxes", []):
                        # each b typically has .xyxy, .conf, .cls
                        try:
                            cls_id = None
                            conf = None
                            xyxy = None
                            # get class id
                            if hasattr(b, "cls"):
                                cls_attr = b.cls
                                # cls could be a tensor/array
                                try:
                                    cls_id = int(cls_attr[0]) if hasattr(cls_attr, "__len__") else int(cls_attr)
                                except Exception:
                                    cls_id = int(cls_attr)
                            # get conf
                            if hasattr(b, "conf"):
                                conf_attr = b.conf
                                try:
                                    conf = float(conf_attr[0]) if hasattr(conf_attr, "__len__") else float(conf_attr)
                                except Exception:
                                    conf = float(conf_attr)
                            # bbox
                            if hasattr(b, "xyxy"):
                                xyxy_attr = b.xyxy
                                try:
                                    xyxy = [float(x) for x in xyxy_attr[0]] if hasattr(xyxy_attr, "__len__") and len(xyxy_attr) and hasattr(xyxy_attr[0], "__len__") else [float(x) for x in xyxy_attr]
                                except Exception:
                                    try:
                                        xyxy = [float(x) for x in xyxy_attr]
                                    except Exception:
                                        xyxy = None

                            # name resolution
                            obj_name = names_map.get(cls_id, str(cls_id)) if cls_id is not None else "unknown"
                            boxes.append({"object": obj_name, "confidence": conf or 0.0, "bbox": xyxy})
                        except Exception:
                            # fallback to safe repr
                            boxes.append({"object": "unknown", "confidence": 0.0, "bbox": None})
                except Exception:
                    # if r0.boxes isn't available, try r0.boxes.data or r0.boxes.xyxy
                    try:
                        data = getattr(r0, "boxes", None)
                        raw_summary['fallback_box_repr'] = str(data)
                    except Exception:
                        raw_summary['fallback_box_repr'] = "no fallback"

                detections = boxes
                raw_summary['detection_count'] = len(detections)

        except Exception as e:
            traceback.print_exc()
            return {"error": f"Model inference failed: {e}", "trace": traceback.format_exc()}

        return {"objects": detections, "raw": raw_summary}


# If run directly, quick local test (not needed for Streamlit use)
if __name__ == "__main__":
    print("This is backend.py. Use from app.py.")
