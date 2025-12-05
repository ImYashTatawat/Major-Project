# app.py
"""
Prompt-based Image Finder — full updated app.py
Features:
 - Prompt parsing with AND / OR (AND higher precedence)
 - Strict class matching (no substring matches)
 - Persist detection results and parsed clauses in session_state
 - Per-image export (download) and global export (folder + ZIP)
 - Bounding boxes drawing with Pillow textbbox fallback
 - CSV download of detections
"""

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import tempfile, os, time, io, string, shutil, zipfile, re
from backend import AdvancedImageFilter
import pandas as pd

st.set_page_config(page_title="Prompt-based Image Finder", layout="wide", initial_sidebar_state="expanded")

# ---------------- Header ----------------
col1, col2 = st.columns([0.85, 0.15])
with col1:
    st.title("🔎 Prompt-based Image Finder — AI powered")
    st.markdown(
        "Upload images and search by natural-language prompts (e.g., `dog and person`, `kutta`, `dog inside car`). "
        "AND / OR supported (AND has higher precedence). Tune detection in the sidebar and export results."
    )
with col2:
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
    if st.button("dog or cat"):
        st.session_state["_quick_prompt"] = "dog or cat"
    if st.button("person and dog"):
        st.session_state["_quick_prompt"] = "person and dog"
    if st.button("dog inside car"):
        st.session_state["_quick_prompt"] = "dog inside car"
    if st.button("kutta"):
        st.session_state["_quick_prompt"] = "kutta"

    st.markdown("---")
    st.subheader("Export settings")
    default_export_dir = os.path.join(os.getcwd(), "exported_images")
    export_dir = st.text_input("Export folder path", value=default_export_dir, help="Folder where exported images are saved.")
    export_with_boxes = st.checkbox("Export images WITH bounding boxes (global)", value=True)
    export_box_thickness = st.slider("Export box thickness", 1, 6, value=box_thickness, help="Thickness of boxes on exported images.")
    st.caption("Per-image export buttons are available on each result card.")
    st.markdown("---")
    st.caption("Tip: Use 'Reload model' after changing model path or weights file.")

# ---------------- Model init ----------------
if "filter" not in st.session_state:
    try:
        st.session_state["filter"] = AdvancedImageFilter(model_path=model_path, confidence=confidence, language=language)
        st.success("Model loaded.")
    except Exception as e:
        st.session_state["filter"] = None
        st.error(f"Failed to load model: {e}")

# reload model
if reload_btn:
    try:
        st.info("Reloading model — this may take a few seconds...")
        st.session_state["filter"] = AdvancedImageFilter(model_path=model_path, confidence=confidence, language=language)
        st.success("Model reloaded successfully.")
    except Exception as e:
        st.error(f"Reload failed: {e}")

# sync small settings
if st.session_state.get("filter") is not None:
    try:
        st.session_state["filter"].set_confidence(confidence)
        st.session_state["filter"].set_language(language)
    except Exception:
        pass

# ---------------- Upload + Prompt UI ----------------
st.subheader("Upload Images")
uploaded_files = st.file_uploader("Upload one or more images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

st.subheader("Prompt (supports AND / OR)")
prompt_text = st.text_input("Enter prompt (e.g., 'dog or cat', 'dog and man')", value=st.session_state.get("_quick_prompt", ""))
# match_mode no longer needed because prompt supports boolean; we keep an option to force All/Any over whole clause groups
global_match_mode = st.selectbox("Global match mode for top-level clauses", options=["OR (default)", "AND"], index=0,
                                help="OR: any clause matching makes the image show. AND: require all clauses (advanced).")

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

# ---------------- Utilities ----------------
STOPWORDS = {"inside", "with", "on", "the", "in", "a", "an", "at", "of", "by", "near", "next"}
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_token(tok: str) -> str:
    tok = tok.lower().strip().translate(PUNCT_TABLE)
    if tok.endswith("s") and len(tok) > 3:
        tok = tok[:-1]
    return tok


def draw_boxes_on_image(image_path: str, detections: list, thickness: int = 2):
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return None
    draw = ImageDraw.Draw(img)
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
        try:
            x1 = int(max(0, min(bbox[0], w)))
            y1 = int(max(0, min(bbox[1], h)))
            x2 = int(max(0, min(bbox[2], w)))
            y2 = int(max(0, min(bbox[3], h)))
        except Exception:
            try:
                coords = [int(float(x)) for x in bbox[:4]]
                x1, y1, x2, y2 = coords
            except Exception:
                continue

        for t in range(thickness):
            rect = [x1 - t, y1 - t, x2 + t, y2 + t]
            draw.rectangle(rect, outline="red")

        label = f"{name} {conf*100:.0f}%"
        try:
            bbox_text = draw.textbbox((0, 0), label, font=font)
            text_w = bbox_text[2] - bbox_text[0]
            text_h = bbox_text[3] - bbox_text[1]
        except Exception:
            try:
                text_w, text_h = font.getsize(label)
            except Exception:
                text_w = len(label) * 6
                text_h = 12

        text_bg = [x1, max(0, y1 - text_h - 4), min(w, x1 + text_w + 8), y1]
        draw.rectangle(text_bg, fill="red")
        draw.text((x1 + 4, max(0, y1 - text_h - 2)), label, fill="white", font=font)

    return img


# ---------------- Prompt parsing with AND/OR ----------------
def parse_prompt_to_clauses(prompt: str, translator) -> list:
    """
    Returns clauses, where each clause is a list of term-sets.
    Each term-set is a set of normalized acceptable model class names for that term.
    Example: "dog and man or cat" -> [ [ {'dog'}, {'person'} ], [ {'cat'} ] ]
    """
    if not prompt or not prompt.strip():
        return []

    p = prompt.strip()
    # split on top-level OR (case-insensitive)
    or_parts = re.split(r'\s+or\s+', p, flags=re.I)

    clauses = []
    for part in or_parts:
        and_parts = re.split(r'\s+and\s+', part, flags=re.I)
        term_sets = []
        for term in and_parts:
            t = term.strip()
            if not t:
                continue
            # try translator
            candidates = set()
            try:
                if translator is not None:
                    cand = translator.expand_prompt(t)
                    for c in cand:
                        nc = normalize_token(c)
                        if nc and nc not in STOPWORDS:
                            candidates.add(nc)
            except Exception:
                pass
            # fallback: token itself (split by comma/slash)
            if not candidates:
                parts = re.split(r'[,/|]+', t)
                for ptoken in parts:
                    nc = normalize_token(ptoken)
                    if nc and nc not in STOPWORDS:
                        candidates.add(nc)
            if candidates:
                term_sets.append(candidates)
        if term_sets:
            clauses.append(term_sets)
    return clauses


def image_matches_clauses(detected_objects: list, clauses: list, top_mode_or=True) -> bool:
    """
    detected_objects: list of {'object': name, ...}
    clauses: list of clauses; clause = list of term-sets
    top_mode_or: if True (default), overall match = OR over clauses (any clause satisfies)
                  if False, overall match = AND over clauses (all clauses must satisfy)
    """
    if not clauses:
        return True

    detected_norms = set([normalize_token(str(d.get("object", "")).lower().strip()) for d in (detected_objects or []) if d.get("object")])

    def clause_matches(clause):
        # clause is list of term_sets; each term_set must be satisfied (AND within clause)
        for term_set in clause:
            # term_set is set of acceptable names; require intersection non-empty
            if not (term_set & detected_norms):
                return False
        return True

    clause_results = [clause_matches(c) for c in clauses]

    if top_mode_or:
        return any(clause_results)
    else:
        return all(clause_results)


# ---------------- Run detection (persist results) ----------------
if run_btn:
    filter_obj = st.session_state["filter"]
    with st.spinner("Running detection on uploaded images..."):
        computed_results = []
        for idx, path in enumerate(saved_paths):
            res = filter_obj.detect_image(path)
            computed_results.append({"path": path, "filename": os.path.basename(path), "result": res})
        st.session_state["last_results"] = computed_results

    # parse prompt to clauses and persist
    clauses = parse_prompt_to_clauses(prompt_text, st.session_state["filter"].translator if st.session_state.get("filter") else None)
    st.session_state["last_clauses"] = clauses

# ---------------- Use persisted results for rendering ----------------
results = st.session_state.get("last_results", [])
clauses = st.session_state.get("last_clauses", [])

if not results:
    st.info("No results available yet. Click 'Run detection & filter' to detect and persist results.")
    st.stop()

# show parsed clauses in sidebar
st.sidebar.markdown("**Parsed prompt clauses**")
if clauses:
    for ci, clause in enumerate(clauses, start=1):
        readable = []
        for term_set in clause:
            readable.append("(" + " | ".join(sorted(term_set)) + ")")
        st.sidebar.write(f"Clause {ci}: " + " AND ".join(readable))
else:
    st.sidebar.write("- (none)")

# ---------------- Display results & exports ----------------
rows_for_csv = []
tab1, tab2 = st.tabs(["Results", "Raw Debug"])
with tab1:
    st.caption("Results matching your prompt. Use per-image export buttons or global export below.")
    cols = st.columns(3)
    shown = 0
    for item in results:
        r = item.get("result") or {}
        detected = r.get("objects", []) if isinstance(r, dict) else []
        top_mode_or = True if "OR" in global_match_mode.upper() else False
        show = image_matches_clauses(detected, clauses, top_mode_or)
        if not show:
            continue

        col = cols[shown % 3]
        with col:
            # preview image
            if show_boxes and detected:
                boxed_img = draw_boxes_on_image(item["path"], detected, thickness=box_thickness)
                if boxed_img:
                    st.image(boxed_img, use_column_width=True)
                else:
                    st.image(item["path"], use_column_width=True)
            else:
                st.image(item["path"], use_column_width=True)

            st.markdown(f"**{item['filename']}**  •  Detected: `{len(detected)}`")

            # per-image export (in-memory download)
            ecol1, ecol2 = st.columns([1, 1])
            # original
            with ecol1:
                key_o = f"exp_orig_{item['filename']}_{shown}"
                if st.button("Export original", key=key_o):
                    try:
                        with open(item["path"], "rb") as f:
                            b = f.read()
                        # save to export dir optionally
                        try:
                            os.makedirs(export_dir, exist_ok=True)
                            out_path = os.path.join(export_dir, item["filename"])
                            with open(out_path, "wb") as fo:
                                fo.write(b)
                        except Exception:
                            out_path = None
                        st.download_button(label="Download original", data=b, file_name=item["filename"], mime="image/jpeg")
                        if out_path:
                            st.success(f"Saved original to {out_path}")
                    except Exception as e:
                        st.error(f"Export failed: {e}")

            # boxed
            with ecol2:
                key_b = f"exp_box_{item['filename']}_{shown}"
                if st.button("Export boxed", key=key_b):
                    try:
                        if detected and export_with_boxes:
                            boxed_img2 = draw_boxes_on_image(item["path"], detected, thickness=export_box_thickness)
                            if boxed_img2:
                                buf = io.BytesIO()
                                boxed_img2.save(buf, format="JPEG")
                                buf.seek(0)
                                try:
                                    os.makedirs(export_dir, exist_ok=True)
                                    out_name = f"boxed_{item['filename']}"
                                    out_path = os.path.join(export_dir, out_name)
                                    with open(out_path, "wb") as fo:
                                        fo.write(buf.getvalue())
                                except Exception:
                                    out_path = None
                                st.download_button(label="Download boxed", data=buf.getvalue(), file_name=f"boxed_{item['filename']}", mime="image/jpeg")
                                if out_path:
                                    st.success(f"Saved boxed image to {out_path}")
                            else:
                                st.warning("Could not create boxed image; no bbox data.")
                        else:
                            with open(item["path"], "rb") as f:
                                b = f.read()
                            st.download_button(label="Download original", data=b, file_name=item["filename"], mime="image/jpeg")
                    except Exception as e:
                        st.error(f"Export failed: {e}")

            # show top detections
            if detected:
                det_sorted = sorted(detected, key=lambda x: x.get("confidence", 0.0), reverse=True)[:4]
                for d in det_sorted:
                    nm = d.get("object", "unknown")
                    cf = d.get("confidence", 0.0) or 0.0
                    st.write(f"- {nm} — {cf*100:.0f}%")
            else:
                st.write("_No objects detected_")
            st.markdown("---")

        # append CSV rows
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
        st.info("No images matched the prompt. Try changing the prompt or adjust confidence/model.")

    # CSV download
    if rows_for_csv:
        df = pd.DataFrame(rows_for_csv)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download results CSV", data=csv_bytes, file_name="detection_results.csv", mime="text/csv")

    # Global export (folder + ZIP)
    st.markdown("---")
    st.subheader("Export all matched images (global)")
    if st.button("📁 Export all matched images to folder and ZIP"):
        try:
            os.makedirs(export_dir, exist_ok=True)
        except Exception as e:
            st.error(f"Could not create export folder: {e}")
            export_dir = None

        if export_dir:
            exported_files = []
            for item in results:
                r = item.get("result") or {}
                detected = r.get("objects", []) if isinstance(r, dict) else []
                top_mode_or = True if "OR" in global_match_mode.upper() else False
                show = image_matches_clauses(detected, clauses, top_mode_or)
                if not show:
                    continue
                src_path = item["path"]
                filename = os.path.basename(src_path)
                try:
                    if export_with_boxes and detected:
                        boxed_img3 = draw_boxes_on_image(src_path, detected, thickness=export_box_thickness)
                        if boxed_img3:
                            out_name = f"boxed_{filename}"
                            out_path = os.path.join(export_dir, out_name)
                            boxed_img3.save(out_path)
                        else:
                            out_path = os.path.join(export_dir, filename)
                            shutil.copy2(src_path, out_path)
                    else:
                        out_path = os.path.join(export_dir, filename)
                        shutil.copy2(src_path, out_path)
                    exported_files.append(out_path)
                except Exception as e:
                    st.warning(f"Failed to export {filename}: {e}")

            if not exported_files:
                st.info("No matched images to export.")
            else:
                st.success(f"Exported {len(exported_files)} images to:\n`{export_dir}`")
                try:
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                        for fpath in exported_files:
                            arcname = os.path.basename(fpath)
                            zf.write(fpath, arcname=arcname)
                    zip_buffer.seek(0)
                    st.download_button(
                        label="⬇️ Download exported images (ZIP)",
                        data=zip_buffer.getvalue(),
                        file_name="exported_images.zip",
                        mime="application/zip",
                    )
                except Exception as e:
                    st.warning(f"Could not create ZIP for download: {e}")

with tab2:
    st.subheader("Raw model outputs (for debugging)")
    for item in results:
        st.markdown(f"**{item['filename']}**")
        st.json(item["result"].get("raw", {}) if isinstance(item["result"], dict) else item["result"])

# ---------------- End ----------------
