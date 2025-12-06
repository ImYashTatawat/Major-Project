# import streamlit as st
# from backend2 import AdvancedImageFilter
# from PIL import Image
# import os

# st.set_page_config(page_title="AI Image Filter", layout="wide")
# st.title("📸 AI Image Filter (Open Images V7)")

# # Sidebar
# model_path = st.sidebar.text_input("YOLO Model Path", "yolov8m-oiv7.pt")
# confidence = st.sidebar.slider("Detection Confidence", 0.0, 1.0, 0.2)
# language = st.sidebar.selectbox("Language", ["en", "hi"])

# # Inputs
# folder_path = st.text_input("Folder path with images:")
# prompt = st.text_input("Search prompt (e.g., 'person or car'):")

# if st.button("Run Filter"):
#     if not os.path.exists(folder_path):
#         st.error("❌ Folder not found")
#     elif not prompt.strip():
#         st.error("❌ Enter a prompt")
#     else:
#         with st.spinner("Processing images..."):
#             fif = AdvancedImageFilter(model_path=model_path, confidence=confidence, language=language)
#             results = fif.filter_folder(folder_path, prompt)
#         st.success(f"Found {len(results)} matching images")
#         cols = st.columns(4)
#         for i, meta in enumerate(results[:20]):
#             with cols[i % 4]:
#                 try:
#                     img = Image.open(meta["path"])
#                     st.image(img, caption=", ".join(meta["objects"]), use_column_width=True)
#                 except:
#                     st.write("Error displaying image")









# import streamlit as st
# from backend2 import AdvancedImageFilter
# from PIL import Image
# import os

# st.set_page_config(page_title="AI Image Filter", layout="wide")
# st.title("📸 AI Image Filter with AND/OR Logic")

# # Sidebar settings
# st.sidebar.header("⚙ Settings")
# model_path = st.sidebar.text_input("YOLO Model Path", "yolov8m-oiv7.pt")
# confidence = st.sidebar.slider("Detection Confidence", 0.0, 1.0, 0.2)

# # Main inputs
# folder_path = st.text_input("Enter folder path containing images:")
# prompt = st.text_input("Enter prompt (example: 'man and dog' or 'cat or dog')")

# if st.button("Run Filter"):
#     if not os.path.exists(folder_path):
#         st.error("❌ Folder not found")
#     elif not prompt.strip():
#         st.error("❌ Please enter a prompt")
#     else:
#         with st.spinner("Processing images..."):
#             fif = AdvancedImageFilter(
#                 model_path=model_path,
#                 confidence=confidence
#             )
#             results = fif.filter_folder(folder_path, prompt)

#         if results:
#             st.success(f"✅ Found {len(results)} matching images")
            
#             # Prepare selection
#             selected_paths = st.multiselect(
#                 "Select images to export (or select all):",
#                 options=[img['path'] for img in results],
#                 default=[img['path'] for img in results]
#             )

#             # Show preview images in grid
#             cols = st.columns(4)
#             for i, img_info in enumerate(results[:20]):
#                 with cols[i % 4]:
#                     try:
#                         img = Image.open(img_info['path'])
#                         objects_list = ", ".join(img_info['objects']) if img_info['objects'] else "None"
#                         st.image(
#                             img,
#                             caption=f"{os.path.basename(img_info['path'])}\nObjects: {objects_list}",
#                             use_column_width=True
#                         )
#                     except Exception as e:
#                         st.write(f"Error displaying image: {e}")

#                     else:
#                         st.warning("No images matched the prompt.")

# # Export section
# dest_folder = st.text_input("Destination folder to copy/move images:")
# move_images = st.checkbox("Move instead of copy")

# if st.button("Export Selected Images"):
#     if not dest_folder:
#         st.error("❌ Provide destination folder")
#     elif 'results' not in locals() or not results:
#         st.error("❌ Run filter first to select images")
#     else:
#         fif = AdvancedImageFilter(model_path=model_path, confidence=confidence)
#         count = fif.export_results(dest_folder, selected_images=selected_paths, move=move_images)
#         st.success(f"✅ Exported {count} images to: {dest_folder}")


# this above code is baseline it detects okay  1







# below is second last code  2

# import streamlit as st
# from backend2 import AdvancedImageFilter
# from PIL import Image
# import os
# from datetime import datetime

# st.set_page_config(page_title="AI Image Filter", layout="wide")
# st.title("📸 AI Image Filter with AND/OR Logic + Date Filter")

# # Sidebar settings
# st.sidebar.header("⚙ Settings")
# model_path = st.sidebar.text_input("YOLO Model Path", "yolov8m-oiv7.pt")
# confidence = st.sidebar.slider("Detection Confidence", 0.0, 1.0, 0.25)

# # Main inputs
# folder_path = st.text_input("Enter folder path containing images:")
# prompt = st.text_input("Enter prompt (example: 'man and dog' or 'cat or dog')")

# # Date filter options
# use_date_filter = st.radio("Use date filter?", ("No", "Yes")) == "Yes"
# start_dt = end_dt = None
# if use_date_filter:
#     col1, col2 = st.columns(2)
#     with col1:
#         start_dt = st.date_input("Start Date")
#     with col2:
#         end_dt = st.date_input("End Date")
#     # convert to datetime
#     start_dt = datetime.combine(start_dt, datetime.min.time())
#     end_dt = datetime.combine(end_dt, datetime.max.time())

# # Session state to keep selections persistent
# if "selected_paths" not in st.session_state:
#     st.session_state.selected_paths = []

# results = []

# if st.button("Run Filter"):
#     if not os.path.exists(folder_path):
#         st.error("❌ Folder not found")
#     elif not prompt.strip():
#         st.error("❌ Please enter a prompt")
#     else:
#         with st.spinner("Processing images..."):
#             fif = AdvancedImageFilter(model_path=model_path, confidence=confidence)
#             results = fif.filter_folder(folder_path, prompt, use_date_filter=use_date_filter)
        
#         if results:
#             st.success(f"✅ Found {len(results)} matching images")
#             # update session_state selected_paths
#             st.session_state.selected_paths = [img['path'] for img in results]

#             # Show preview images in grid
#             cols = st.columns(4)
#             for i, img_info in enumerate(results[:20]):
#                 with cols[i % 4]:
#                     try:
#                         img = Image.open(img_info['path'])
#                         objects_list = ", ".join(img_info['objects']) if img_info['objects'] else "None"
#                         checked = img_info['path'] in st.session_state.selected_paths
#                         # Checkbox for selection
#                         if st.checkbox(os.path.basename(img_info['path']), value=checked, key=f"chk_{i}"):
#                             if img_info['path'] not in st.session_state.selected_paths:
#                                 st.session_state.selected_paths.append(img_info['path'])
#                         else:
#                             if img_info['path'] in st.session_state.selected_paths:
#                                 st.session_state.selected_paths.remove(img_info['path'])

#                         st.image(img, caption=f"Objects: {objects_list}", use_column_width=True)
#                     except Exception as e:
#                         st.write(f"Error displaying image: {e}")
#         else:
#             st.warning("No images matched the prompt.")

# # Export section
# dest_folder = st.text_input("Destination folder to copy/move images:")
# move_images = st.checkbox("Move instead of copy")

# if st.button("Export Selected Images"):
#     if not dest_folder:
#         st.error("❌ Provide destination folder")
#     elif not st.session_state.selected_paths:
#         st.error("❌ No images selected for export")
#     else:
#         fif = AdvancedImageFilter(model_path=model_path, confidence=confidence)
#         count = fif.export_results(dest_folder, selected_images=st.session_state.selected_paths, move=move_images)
#         st.success(f"✅ Exported {count} images to: {dest_folder}")




# import streamlit as st
# from backend2 import AdvancedImageFilter
# from PIL import Image
# import os
# from datetime import datetime


# # ------------------ Page Config ------------------
# st.set_page_config(page_title="AI Image Filter", layout="wide")
# st.title("📸 AI Image Filter with AND/OR + Date Filter ")


# # ------------------ Sidebar ------------------
# st.sidebar.header("⚙ Settings")
# model_path = st.sidebar.text_input("YOLO Model Path", "yolov8m-oiv7.pt")
# confidence = st.sidebar.slider("Detection Confidence", 0.0, 1.0, 0.25)


# # ------------------ Inputs ------------------
# folder_path = st.text_input("Enter folder path containing images:")
# prompt = st.text_input("Enter prompt (example: 'man and dog', 'cat or dog', 'animal')")

# use_date_filter = st.radio("Use date filter?", ("No", "Yes")) == "Yes"

# start_dt = end_dt = None
# if use_date_filter:
#     c1, c2 = st.columns(2)
#     with c1:
#         s = st.date_input("Start Date")
#     with c2:
#         e = st.date_input("End Date")

#     start_dt = datetime.combine(s, datetime.min.time())
#     end_dt = datetime.combine(e, datetime.max.time())


# # ------------------ Session State ------------------
# if "selected_paths" not in st.session_state:
#     st.session_state.selected_paths = []

# if "fif_instance" not in st.session_state:
#     st.session_state.fif_instance = None


# results = []


# # ------------------ Run Filter ------------------
# # if st.button("Run Filter"):
# #     if not os.path.exists(folder_path):
# #         st.error("❌ Folder not found")

# #     elif not prompt.strip():
# #         st.error("❌ Please enter a prompt")

# #     else:
# #         with st.spinner("🔍 Scanning images..."):
# #             fif = AdvancedImageFilter(model_path=model_path, confidence=confidence)
# #             results = fif.filter_folder(
# #                 folder_path,
# #                 prompt,
# #                 use_date_filter=use_date_filter,
# #                 start_date=start_dt,
# #                 end_date=end_dt
# #             )
# #             st.session_state.fif_instance = fif

# #         if results:
# #             st.success(f"✅ Found {len(results)} matching images")

# #             # Normalize & store selected paths
# #             normalized = []
# #             for r in results:
# #                 p = os.path.normpath(os.path.abspath(r["path"]))
# #                 normalized.append(p)
# #             st.session_state.selected_paths = normalized
# # ------------------ Run Filter ------------------



# # ------------------ Run Filter ------------------
# # ------------------ Run Filter ------------------
# if st.button("Run Filter"):
#     if not os.path.exists(folder_path):
#         st.error("❌ Folder not found")
#     elif not prompt.strip():
#         st.error("❌ Please enter a prompt")
#     else:
#         with st.spinner("🔍 Scanning images..."):
#             fif = AdvancedImageFilter(model_path=model_path, confidence=confidence)
#             results = fif.filter_folder(
#                 folder_path,
#                 prompt,
#                 use_date_filter=use_date_filter,
#                 start_date=start_dt,
#                 end_date=end_dt
#             )
#             # Save results & instance in session_state
#             st.session_state.fif_instance = fif
#             st.session_state.results = results
#             # By default, all images are selected
#             st.session_state.selected_paths = [
#                 os.path.normpath(os.path.abspath(r["path"])) for r in results
#             ]











#             # ---- DISPLAY IMAGES ----
#         #     cols = st.columns(4)
#         #     for i, img_info in enumerate(results[:40]):  # limit preview
#         #         with cols[i % 4]:
#         #             try:
#         #                 p = os.path.normpath(os.path.abspath(img_info["path"]))
#         #                 img = Image.open(p)

#         #                 objects_list = ", ".join(img_info["objects"]) if img_info["objects"] else "None"

#         #                 # Checkbox key MUST be unique
#         #                 key = f"chk_select_{i}"

#         #                 checked = p in st.session_state.selected_paths

#         #                 # Checkbox logic
#         #                 if st.checkbox(os.path.basename(p), value=checked, key=key):
#         #                     if p not in st.session_state.selected_paths:
#         #                         st.session_state.selected_paths.append(p)
#         #                 else:
#         #                     if p in st.session_state.selected_paths:
#         #                         st.session_state.selected_paths.remove(p)

#         #                 st.image(img, caption=f"Objects: {objects_list}", use_column_width=True)
#         #             except Exception as e:
#         #                 st.write(f"⚠ Error showing image: {e}")

#         # else:
#         #     st.warning("⚠ No images matched the prompt.")
# # ------------------ Display Images ------------------
#             # ------------------ Display Images ------------------
#             # ------------------ Display Images ------------------
# results_to_show = st.session_state.get("results", [])

# if results_to_show:
#     st.success(f"✅ Found {len(results_to_show)} matching images")
#     cols = st.columns(4)
#     for i, img_info in enumerate(results_to_show[:40]):
#         with cols[i % 4]:
#             try:
#                 img_path = os.path.normpath(os.path.abspath(img_info["path"]))
#                 img = Image.open(img_path)
#                 objects_list = ", ".join(img_info["objects"]) if img_info["objects"] else "None"
#                 key = f"chk_select_{i}"

#                 checked = img_path in st.session_state.selected_paths

#                 if st.checkbox(os.path.basename(img_path), value=checked, key=key):
#                     if img_path not in st.session_state.selected_paths:
#                         st.session_state.selected_paths.append(img_path)
#                 else:
#                     if img_path in st.session_state.selected_paths:
#                         st.session_state.selected_paths.remove(img_path)

#                 st.image(img, caption=f"Objects: {objects_list}", use_column_width=True)
#             except Exception as e:
#                 st.write(f"⚠ Error showing image: {e}")
# else:
#     st.info("⚠ No images to display. Run filter first.")


# # ------------------ Export Section ------------------
# st.subheader("📤 Export Selected Images")

# dest_folder = st.text_input("Destination folder to copy/move images:")
# move_images = st.checkbox("Move instead of copy")

# if st.button("Export Selected Images"):
#     fif = st.session_state.fif_instance

#     if fif is None:
#         st.error("❌ Please run filter first")

#     elif not st.session_state.selected_paths:
#         st.error("❌ No images selected to export")

#     else:
#         count = fif.export_results(
#             dest_folder,
#             selected_images=st.session_state.selected_paths,
#             move=move_images
#         )

#         st.success(f"✅ Exported {count} images to: {dest_folder}")

import streamlit as st
from backend2 import AdvancedImageFilter
from PIL import Image
import os
from datetime import datetime


# --------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------
st.set_page_config(page_title="AI Image Filter", layout="wide")
st.title("📸 Prompt Based AI Image Finder")


# --------------------------------------------------------
# SIDEBAR SETTINGS
# --------------------------------------------------------
st.sidebar.header("⚙ Settings")
model_path = st.sidebar.text_input("YOLO Model Path", "yolov8m-oiv7.pt")
confidence = st.sidebar.slider("For Details and Precision set 0.15 ", 0.0, 1.0, 0.25)


# --------------------------------------------------------
# INPUT FIELDS
# --------------------------------------------------------
folder_path = st.text_input("📁 Folder path containing images:")

prompt = st.text_input(
    "🧠 Search Prompt (Examples: 'man and dog', 'dog or cat', 'person not car')"
)

use_date_filter = st.radio(
    "Filter by date?", ["No", "Yes"], index=0
) == "Yes"

start_dt = end_dt = None
if use_date_filter:
    col1, col2 = st.columns(2)
    with col1:
        s = st.date_input("Start Date")
    with col2:
        e = st.date_input("End Date")

    start_dt = datetime.combine(s, datetime.min.time())
    end_dt = datetime.combine(e, datetime.max.time())


# --------------------------------------------------------
# SESSION STATE
# --------------------------------------------------------
if "results" not in st.session_state:
    st.session_state.results = []

if "selected" not in st.session_state:
    st.session_state.selected = []

if "fif" not in st.session_state:
    st.session_state.fif = None


# --------------------------------------------------------
# RUN FILTER BUTTON
# --------------------------------------------------------
if st.button("🔍 Run Image Filter"):
    if not folder_path.strip():
        st.error("❌ Please enter a folder path")
    elif not os.path.exists(folder_path):
        st.error("❌ Folder does not exist")
    elif not prompt.strip():
        st.error("❌ Please enter a search prompt")
    else:
        with st.spinner("Processing images..."):
            fif = AdvancedImageFilter(model_path=model_path, confidence=confidence)

            results = fif.filter_folder(
                folder_path=folder_path,
                prompt=prompt,
                use_date_filter=use_date_filter,
                start_date=start_dt,
                end_date=end_dt
            )

            st.session_state.fif = fif
            st.session_state.results = results
            st.session_state.selected = [
                os.path.normpath(os.path.abspath(r["path"])) for r in results
            ]

        if results:
            st.success(f"✅ Found {len(results)} images")
        else:
            st.warning("⚠ No images matched your prompt.")


# --------------------------------------------------------
# SHOW RESULTS
# --------------------------------------------------------
results = st.session_state.get("results", [])

if results:
    st.subheader("📷 Matching Images")
    cols = st.columns(4)

    for i, item in enumerate(results[:40]):  # limit preview to 40
        img_path = os.path.normpath(os.path.abspath(item["path"]))
        objects = ", ".join(item["objects"]) if item["objects"] else "None"

        try:
            img = Image.open(img_path)
        except:
            continue

        with cols[i % 4]:
            key = f"imgchk_{i}"
            checked = img_path in st.session_state.selected

            if st.checkbox(os.path.basename(img_path), value=checked, key=key):
                if img_path not in st.session_state.selected:
                    st.session_state.selected.append(img_path)
            else:
                if img_path in st.session_state.selected:
                    st.session_state.selected.remove(img_path)

            st.image(img, caption=f"Objects: {objects}", use_column_width=True)


# --------------------------------------------------------
# EXPORT SECTION
# --------------------------------------------------------
st.subheader("📤 Export Selected Images")

dest_folder = st.text_input("Destination Folder:")
move_opt = st.checkbox("Move instead of copy")

if st.button("Export"):
    fif = st.session_state.get("fif")

    if not fif:
        st.error("❌ Run filter first.")
    elif not st.session_state.selected:
        st.error("❌ No images selected.")
    else:
        count = fif.export_results(
            destination_folder=dest_folder,
            selected_images=st.session_state.selected,
            move=move_opt
        )
        st.success(f"✅ Exported {count} images!")
