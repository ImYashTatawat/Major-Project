# build_synonyms.py
import os
from ultralytics import YOLO

# --- CONFIG ---
MODEL_PATH = "yolov8m-oiv7.pt"  # Path to your YOLOv8 Open Images model
OUTPUT_FILE = "synonyms.py"

# --- Load YOLO model ---
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"YOLO model not found: {MODEL_PATH}")

model = YOLO(MODEL_PATH)

# --- Extract class names ---
classes = list(model.names.values())
print(f"✅ Found {len(classes)} classes")

# --- Generate SYNONYM_MAP ---
# You can customize the synonyms manually later
synonym_map = {}
for cls in classes:
    cls_lower = cls.lower().replace('_', ' ')
    # Simple synonym list: the class itself + lowercase version
    synonym_map[cls_lower] = [cls_lower]

# --- Write to Python file ---
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write("# Auto-generated Open Images V7 synonyms dictionary\n")
    f.write("SYNONYM_MAP = {\n")
    for key, syns in synonym_map.items():
        syns_str = ", ".join(f"'{s}'" for s in syns)
        f.write(f"    '{key}': [{syns_str}],\n")
    f.write("}\n")

print(f"✅ Synonym file generated: {OUTPUT_FILE}")

