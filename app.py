# app.py
"""
Improved Streamlit UI for YOLOv8 detection with prompt-based filtering and Pillow textbbox fix.
Features:
 - Attractive header + tips
 - Prompt sample buttons
 - Toggle to draw bounding boxes on images
 - Results shown as cards with detection count badges
 - Download CSV of results
 - Tabs to switch between Results and Raw debug
 - Uses strict matching for prompts (no substring matches)
"""

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import tempfile, os, time, io, csv, string
from backend import AdvancedImageFilter
import pandas as pd

st.set_page_config(page_title="Animal Detector — Clean UI", layout="wide", initial_sidebar_state="expanded")

# ---------------- Header ----------------
col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.title("🐾 Animal/Object Detector")
    st.markdown(
        "Upload images, enter a prompt (e.g., `dog and person`, `kutta`, `dog inside car`) "
        "and get filtered detection results. Use sidebar to tune model/confidence/language."
    )
with col2:
    # small decorative image (replace or remove if you prefer)
    try:
        st.image("https://static.streamlit.io/examples/dice.jpg", width=80)
    except Exception:
        pass

st.markdown("---")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Model & UI settings")
    model_path = st.text_input("Model path (weights)", value="yolov8m.pt")
    confidence = st.slider(
        "Confidence threshold", min_value=0.01, max_value=0.9, value=0.10, step=0.01,
        help="Lower = more detections (but more false positives). Increase to be stricter."
    )
    language = st.selectbox("Language for synonyms", options=["en", "hi"], index=0)
    show_boxes = st.checkbox("Draw bounding boxes on images", value=True)
    box_thickness = st.slider("Box thickness", 1, 6, 2)
    reload_btn = st.button("🔄 Reload model")

    st.markdown("---")
    st.subheader("Quick prompts")
    st.caption("Click to fill the prompt input quickly:")
    if st.button("dog or cat"):
        st.session_state["_quick_prompt"] = "dog or cat"
    if st.button("person and dog"):
        st.session_state["_quick_prompt"] = "person and dog"
    if st.button("dog inside car"):
        st.session_state["_quick_prompt"] = "dog inside car"
    if st.button("kutta"):
        st.session_state["_quick_prompt"] = "kutta"

    st.markdown("---")
    st.caption("Tip: Use 'Reload model' after changing model path or weights file.")

# ---------------- Model init (session-state) ----------------
if "filter" not in st.session_state:
    try:
        st.session_state["filter"] = AdvancedImageFilter(model_path=model_path, confidence=confidence, language=language)
        st.success("Model loaded.")
    except Exception as e:
        st.session_state["filter"] = None
        st.error(f"Failed to load model: {e}")

# Reload if requested
if reload_btn:
    try:
        st.info("Reloading model — this may take a few seconds...")
        st.session_state["filter"] = AdvancedImageFilter(model_path=model_path, confidence=confidence, language=language)
        st.success("Model reloaded successfully.")
    except Exception as e:
        st.error(f"Reload failed: {e}")

# Sync small settings
if st.session_state.get("filter") is not None:
    try:
        st.session_state["filter"].set_confidence(confidence)
        st.session_state["filter"].set_language(language)
    except Exception:
        pass

# ---------------- Upload + Prompt area ----------------
st.subheader("Upload Images")
uploaded_files = st.file_uploader("Upload one or more images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

st.subheader("Prompt (search/filter)")
# populate quick-prompt if clicked
prompt_text = st.text_input("Enter prompt (e.g., 'dog or cat', 'kutta')", value=st.session_state.get("_quick_prompt", ""))
match_mode = st.radio("Match mode", ("Any (show if any target found)", "All (show only if all targets present)"))

run_btn = st.button("Run detection & filter")

if not uploaded_files:
    st.info("Upload images (jpg/png) to run detection.")
    st.stop()

if st.session_state.get("filter") is None:
    st.error("Model not loaded. Fix model path in sidebar and press Reload model.")
    st.stop()

# ---------------- Save uploaded files ----------------
tmp_dir = tempfile.mkdtemp(prefix="yolo_uploads_")
saved_paths = []
for up in uploaded_files:
    out_path = os.path.join(tmp_dir, up.name)
    with open(out_path, "wb") as f:
        f.write(up.read())
    saved_paths.append(out_path)

if not saved_paths:
    st.error("No images saved.")
    st.stop()

# ---------------- Utilities for strict matching and drawing ----------------
STOPWORDS = {"inside", "with", "on", "the", "in", "a", "an", "at", "of", "and", "or", "by", "near", "next"}
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_token(tok: str) -> str:
    tok = tok.lower().strip().translate(PUNCT_TABLE)
    if tok.endswith("s") and len(tok) > 3:
        tok = tok[:-1]
    return tok


def image_matches_targets_strict(detected_objects, targets, mode="Any"):
    if not targets:
        return True
    detected_norms = [normalize_token(str(d.get("object", "")).lower().strip()) for d in (detected_objects or []) if d.get("object")]
    if mode == "Any":
        return any(t in detected_norms for t in targets)
    else:
        return all(t in detected_norms for t in targets)


def draw_boxes_on_image(image_path: str, detections: list, thickness: int = 2):
    """
    Draws bounding boxes and labels on the image and returns a PIL.Image object.
    Expects detections like: {'object': name, 'confidence': 0.x, 'bbox': [x1,y1,x2,y2]}
    This function uses textbbox when available (Pillow 10+) and falls back to font.getsize.
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return None
    draw = ImageDraw.Draw(img)
    # Attempt to use a truetype-ish font if available; fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", size=14)
    except Exception:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

    w, h = img.size
    for d in (detections or []):
        bbox = d.get("bbox")
        name = str(d.get("object", "unknown"))
        conf = d.get("confidence", 0.0) or 0.0
        if not bbox or len(bbox) < 4:
            continue
        # Convert coords to safe ints and clip
        try:
            x1 = int(max(0, min(bbox[0], w)))
            y1 = int(max(0, min(bbox[1], h)))
            x2 = int(max(0, min(bbox[2], w)))
            y2 = int(max(0, min(bbox[3], h)))
        except Exception:
            # fallback if bbox structure is unusual
            try:
                coords = [int(float(x)) for x in bbox[:4]]
                x1, y1, x2, y2 = coords
            except Exception:
                continue

        # draw rectangle (thickness)
        for t in range(thickness):
            rect = [x1 - t, y1 - t, x2 + t, y2 + t]
            draw.rectangle(rect, outline="red")

        # label background + text
        label = f"{name} {conf*100:.0f}%"

        # Measure text size safely (Pillow 10+ has textbbox; older versions may not)
        try:
            bbox_text = draw.textbbox((0, 0), label, font=font)  # (x0,y0,x1,y1)
            text_w = bbox_text[2] - bbox_text[0]
            text_h = bbox_text[3] - bbox_text[1]
        except Exception:
            # fallback to font.getsize if available
            try:
                text_w, text_h = font.getsize(label)
            except Exception:
                # final fallback estimates
                text_w = len(label) * 6
                text_h = 12

        # background rectangle behind text (clip to image)
        text_bg = [x1, max(0, y1 - text_h - 4), min(w, x1 + text_w + 8), y1]
        draw.rectangle(text_bg, fill="red")

        # draw text on top
        draw.text((x1 + 4, max(0, y1 - text_h - 2)), label, fill="white", font=font)

    return img

# ---------------- Run detection ----------------
if run_btn:
    filter_obj = st.session_state["filter"]
    # progress UI
    with st.spinner("Running detection on uploaded images..."):
        results = []
        for idx, path in enumerate(saved_paths):
            res = filter_obj.detect_image(path)
            results.append({"path": path, "filename": os.path.basename(path), "result": res})

    # Expand prompt to targets via translator
    target_classes = []
    if prompt_text and hasattr(filter_obj, "translator") and filter_obj.translator is not None:
        try:
            expanded = filter_obj.translator.expand_prompt(prompt_text)
            target_classes = [normalize_token(t) for t in expanded if t and normalize_token(t) not in STOPWORDS]
        except Exception:
            # fallback split
            toks = [t.strip() for t in prompt_text.replace("/", ",").split(",")]
            target_classes = [normalize_token(t) for t in toks if normalize_token(t) not in STOPWORDS]

    # fallback derive if empty
    if not target_classes and prompt_text:
        toks = [t.strip() for t in prompt_text.replace("/", ",").split(",")]
        derived = []
        for tok in toks:
            for sub in tok.replace(" or ", ",").replace(" and ", ",").split(","):
                subn = normalize_token(sub)
                if subn and subn not in STOPWORDS:
                    derived.append(subn)
        target_classes = derived

    # Show sidebar info
    st.sidebar.markdown("**Targets used**")
    if target_classes:
        for t in target_classes:
            st.sidebar.write(f"- {t}")
    else:
        st.sidebar.write("- (none)")

    # Build a DataFrame for downloadable CSV
    rows_for_csv = []
    # Results tab layout
    tab1, tab2 = st.tabs(["Results", "Raw Debug"])
    with tab1:
        st.caption(f"Showing images that match prompt (match mode = {match_mode}). Use checkbox in sidebar to toggle boxes.")
        # show cards in responsive grid
        cols = st.columns(3)
        shown = 0
        for item in results:
            r = item["result"] or {}
            detected = r.get("objects", []) if isinstance(r, dict) else []
            mode_key = "Any" if "Any" in match_mode else "All"
            show = image_matches_targets_strict(detected, target_classes, mode=mode_key)
            if not show:
                continue

            col = cols[shown % 3]
            with col:
                # small card: image (possibly boxed), count badge, filename, top detections
                # image with bounding boxes
                if show_boxes and detected:
                    boxed_img = draw_boxes_on_image(item["path"], detected, thickness=box_thickness)
                    if boxed_img:
                        st.image(boxed_img, use_column_width=True)
                    else:
                        st.image(item["path"], use_column_width=True)
                else:
                    st.image(item["path"], use_column_width=True)

                # header row inside card
                det_count = len(detected) if detected else 0
                st.markdown(f"**{item['filename']}**  •  Detected: `{det_count}`")
                # show top 4 detections
                if detected:
                    det_sorted = sorted(detected, key=lambda x: x.get("confidence", 0.0), reverse=True)[:4]
                    for d in det_sorted:
                        nm = d.get("object", "unknown")
                        cf = d.get("confidence", 0.0) or 0.0
                        st.write(f"- {nm} — {cf*100:.0f}%")
                else:
                    st.write("_No objects detected_")
                st.markdown("---")

            # append CSV rows for each detection
            if detected:
                for d in detected:
                    rows_for_csv.append({
                        "filename": item["filename"],
                        "object": d.get("object", ""),
                        "confidence": d.get("confidence", 0.0),
                        "bbox": d.get("bbox", None)
                    })
            else:
                rows_for_csv.append({"filename": item["filename"], "object": "", "confidence": "", "bbox": ""})

            shown += 1

        if shown == 0:
            st.info("No images matched the prompt. Try changing prompt, match mode, or adjust confidence.")

        # Download CSV if data exists
        if rows_for_csv:
            df = pd.DataFrame(rows_for_csv)
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download results CSV", data=csv_bytes, file_name="detection_results.csv", mime="text/csv")

    with tab2:
        st.subheader("Raw model outputs (for debugging)")
        for item in results:
            st.markdown(f"**{item['filename']}**")
            st.json(item["result"].get("raw", {}) if isinstance(item["result"], dict) else item["result"])

else:
    st.info("Click 'Run detection & filter' to process images and apply the prompt filter.")
