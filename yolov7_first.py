import torch
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import os
import threading
import queue
import shutil
import sys
import re
import cv2
import numpy as np


# Try to import ultralytics for YOLOv8
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("❌ ultralytics not installed. Please install: pip install ultralytics")

class AdvancedImageFilter:
    def __init__(self, model_path=None):
        self.model = None
        self.model_path = model_path or "yolov8m-oiv7.pt"  # Default path
        self.load_model()
        
        # Load class names FIRST before keyword mappings
        self.coco_classes = self.load_class_names()
        
        # Enhanced keyword mappings based on ACTUAL class names
        self.keyword_mappings = self.create_smart_keyword_mappings()
    
    def create_smart_keyword_mappings(self):
        """Create keyword mappings based on actual class names in the model"""
        mappings = {
            'person': ['person', 'people', 'human', 'man', 'woman', 'child', 'baby', 'face', 'boy', 'girl', 'adult'],
            'animal': ['animal', 'pet', 'dog', 'cat', 'bird', 'horse', 'cow', 'sheep', 'elephant', 
                      'bear', 'zebra', 'giraffe', 'puppy', 'kitten', 'lion', 'tiger', 'wolf', 'deer',
                      'monkey', 'rabbit', 'squirrel', 'fox', 'raccoon', 'kangaroo', 'panda',
                      'leopard', 'cheetah', 'hippo', 'rhino', 'camel', 'gorilla'],
            'vehicle': ['vehicle', 'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'bike', 'auto', 'transport',
                       'van', 'suv', 'sedan', 'jeep', 'minivan', 'pickup', 'ambulance', 'firetruck', 'policecar'],
            'food': ['food', 'fruit', 'apple', 'banana', 'orange', 'pizza', 'sandwich', 'meal', 'eating',
                    'vegetable', 'burger', 'fries', 'pasta', 'rice', 'bread', 'cake', 'cookie', 'icecream',
                    'drink', 'beverage', 'coffee', 'tea', 'juice', 'water', 'wine', 'beer'],
            'electronic': ['electronic', 'laptop', 'computer', 'phone', 'cellphone', 'tv', 'television', 'device',
                          'tablet', 'ipad', 'smartphone', 'monitor', 'screen', 'keyboard', 'mouse', 'camera',
                          'headphones', 'speaker', 'microphone', 'printer', 'scanner', 'macbook', 'notebook'],
            'furniture': ['furniture', 'chair', 'table', 'couch', 'sofa', 'bed', 'desk', 'cabinet', 'shelf',
                         'wardrobe', 'dresser', 'bookshelf', 'stool', 'bench', 'ottoman'],
            'sports': ['sports', 'ball', 'baseball', 'tennis', 'skateboard', 'game', 'football', 'basketball',
                      'soccer', 'volleyball', 'golf', 'hockey', 'cricket', 'rugby', 'badminton', 'pool'],
            'outdoor': ['outdoor', 'tree', 'sky', 'building', 'house', 'nature', 'mountain', 'river', 'lake',
                       'ocean', 'beach', 'forest', 'park', 'garden', 'flower', 'grass', 'road', 'street'],
            'clothing': ['clothing', 'shirt', 'pants', 'dress', 'jacket', 'coat', 'hat', 'cap', 'shoes',
                        'sneakers', 'boots', 'sandals', 'glasses', 'sunglasses', 'watch', 'jewelry'],
            'kitchen': ['kitchen', 'utensil', 'knife', 'fork', 'spoon', 'plate', 'bowl', 'cup', 'glass',
                       'pot', 'pan', 'microwave', 'oven', 'refrigerator', 'stove', 'sink']
        }
        
        # Auto-expand mappings based on actual class names
        for category, keywords in mappings.items():
            expanded_keywords = keywords.copy()
            for class_name in self.coco_classes:
                class_lower = class_name.lower()
                # Add class names that match category keywords
                if any(keyword in class_lower for keyword in keywords):
                    if class_name not in expanded_keywords:
                        expanded_keywords.append(class_name)
            mappings[category] = expanded_keywords
            
        return mappings
    
    def debug_model_classes(self):
        """Debug function to see what classes are actually available"""
        if hasattr(self.model, 'names'):
            print("🔍 DEBUG: ACTUAL CLASSES IN MODEL:")
            all_classes = list(self.model.names.values())
            
            # Search for specific problematic classes
            search_terms = ['tiger', 'laptop', 'computer', 'macbook', 'cat', 'animal']
            for term in search_terms:
                matches = [c for c in all_classes if term in c.lower()]
                print(f"   '{term}': {matches}")
            
            # Show first 20 classes to understand naming convention
            print(f"📋 First 20 classes: {all_classes[:20]}")
    
    def load_class_names(self):
        """Load the 600 class names for Open Images v7"""
        # If model is loaded, get classes from model
        if self.model and hasattr(self.model, 'names'):
            classes = list(self.model.names.values())
            print(f"✅ Loaded {len(classes)} classes from model")
            # Run debug to see what we have
            self.debug_model_classes()
            return classes
        
        # Try to load from classes.txt as fallback
        class_files = ["classes.txt", "coco.names", "openimages.names"]
        for class_file in class_files:
            if os.path.exists(class_file):
                try:
                    with open(class_file, "r") as f:
                        classes = [line.strip() for line in f.readlines() if line.strip()]
                    print(f"✅ Loaded {len(classes)} classes from {class_file}")
                    return classes
                except Exception as e:
                    print(f"❌ Error loading {class_file}: {e}")
        
        # Final fallback
        print("⚠️ No class file found, using default classes")
        return ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat']
    
    def load_model(self):
        """Load YOLOv8 model with Open Images v7 classes"""
        try:
            if not ULTRALYTICS_AVAILABLE:
                raise ImportError("ultralytics package not available")
                
            print("🔄 Loading YOLOv8 model...")
            print(f"📁 Model path: {self.model_path}")
            
            if not os.path.exists(self.model_path):
                print(f"❌ Model file not found: {self.model_path}")
                possible_paths = [
                    "yolov8s-oiv7.pt", "yolov8s.pt", "yolov8.pt", "model.pt", 
                    "weights/yolov8s-oiv7.pt", "models/yolov8s-oiv7.pt"
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        self.model_path = path
                        print(f"✅ Found model at: {path}")
                        break
                else:
                    raise FileNotFoundError(f"Could not find YOLOv8 model file")
            
            # Load YOLOv8 model with lower confidence
            self.model = YOLO(self.model_path)
            self.model.conf = 0.1  # LOWER confidence for better detection
            
            print(f"✅ YOLOv8 model loaded successfully!")
            if hasattr(self.model, 'names'):
                print(f"📊 Model has {len(self.model.names)} classes")
            
            return True
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False
    
    def analyze_image_advanced(self, image_path, parsed_prompt):
        """Advanced image analysis with logic support for YOLOv8"""
        try:
            print(f"   Analyzing: {os.path.basename(image_path)}")
            
            if self.model is None:
                print("     ❌ Model not loaded")
                return False, {}
            
            # Run YOLOv8 inference with LOW confidence
            results = self.model(image_path, conf=0.1, verbose=False)  # LOW confidence
            
            # Process results
            detected_objects = {}
            
            if len(results) > 0:
                result = results[0]
                
                if result.boxes is not None and len(result.boxes) > 0:
                    for i, box in enumerate(result.boxes):
                        confidence = box.conf.item()
                        class_id = int(box.cls.item())
                        
                        # Get class name
                        if hasattr(self.model, 'names') and class_id in self.model.names:
                            obj_name = self.model.names[class_id]
                        elif class_id < len(self.coco_classes):
                            obj_name = self.coco_classes[class_id]
                        else:
                            obj_name = f"class_{class_id}"
                        
                        if obj_name not in detected_objects:
                            detected_objects[obj_name] = []
                        detected_objects[obj_name].append(confidence)
            
            print(f"     Detected: {list(detected_objects.keys())}")
            
            # Apply logic based on prompt type
            return self.apply_prompt_logic(parsed_prompt, detected_objects)
            
        except Exception as e:
            print(f"     ❌ Error analyzing image: {e}")
            return False, {}
    
    def apply_prompt_logic(self, parsed_prompt, detected_objects):
        """Apply prompt logic to detections"""
        if parsed_prompt['type'] == 'SINGLE':
            for target_obj in parsed_prompt['objects']:
                if target_obj in detected_objects:
                    print(f"     ✅ Found {target_obj}")
                    return True, detected_objects
        
        elif parsed_prompt['type'] == 'OR':
            for target_obj in parsed_prompt['objects']:
                if target_obj in detected_objects:
                    print(f"     ✅ Found {target_obj} (OR condition)")
                    return True, detected_objects
        
        elif parsed_prompt['type'] == 'AND':
            all_groups_satisfied = True
            satisfied_groups = []
            
            for object_group in parsed_prompt['objects_groups']:
                group_satisfied = False
                for target_obj in object_group:
                    if target_obj in detected_objects:
                        group_satisfied = True
                        satisfied_groups.append(target_obj)
                        break
                
                if not group_satisfied:
                    all_groups_satisfied = False
                    break
            
            if all_groups_satisfied:
                print(f"     ✅ Found all required objects: {satisfied_groups}")
                return True, detected_objects
            else:
                print(f"     ❌ Missing some required objects")
        
        print(f"     ❌ No matching objects found")
        return False, detected_objects
    
    def parse_advanced_prompt(self, prompt):
        """Parse complex prompts with AND/OR logic"""
        prompt_lower = prompt.lower().strip()
        print(f"🔍 Parsing advanced prompt: '{prompt}'")
        
        if ' and ' in prompt_lower:
            parts = [part.strip() for part in prompt_lower.split(' and ')]
            required_groups = []
            for part in parts:
                objects = self.parse_single_prompt(part)
                required_groups.append(objects)
                print(f"   - AND condition: {part} -> {objects}")
            return {'type': 'AND', 'objects_groups': required_groups}
        
        elif ' or ' in prompt_lower:
            parts = [part.strip() for part in prompt_lower.split(' or ')]
            all_objects = []
            for part in parts:
                objects = self.parse_single_prompt(part)
                all_objects.extend(objects)
            all_objects = list(set(all_objects))
            print(f"   - OR condition: {parts} -> {all_objects}")
            return {'type': 'OR', 'objects': all_objects}
        
        elif ' with ' in prompt_lower:
            parts = [part.strip() for part in prompt_lower.split(' with ')]
            required_groups = []
            for part in parts:
                objects = self.parse_single_prompt(part)
                required_groups.append(objects)
            return {'type': 'AND', 'objects_groups': required_groups}
        
        else:
            objects = self.parse_single_prompt(prompt)
            print(f"   - Single condition: {prompt} -> {objects}")
            return {'type': 'SINGLE', 'objects': objects}
    
    def parse_single_prompt(self, prompt):
        prompt_lower = prompt.lower().strip()
        target_objects = []
        
        # Common synonyms mapping
        synonym_map = {
            'glass': ['glass', 'drinking glass', 'wine glass', 'cup', 'tumbler'],
            'glasses': ['glasses', 'eyeglasses', 'spectacles', 'sunglasses'],
            'water': ['water', 'bottle', 'drink', 'beverage', 'liquid'],
            'person': ['person', 'man', 'woman', 'human', 'people'],
            'car': ['car', 'vehicle', 'automobile', 'sedan'],
        }
        
        # Expand search terms using synonyms
        search_terms = [prompt_lower]
        if prompt_lower in synonym_map:
            search_terms.extend(synonym_map[prompt_lower])
        
        print(f"🔍 Searching for: {search_terms}")
        
        # Search for all expanded terms
        for search_term in search_terms:
            for class_name in self.coco_classes:
                class_lower = class_name.lower()
                
                # Exact match or whole word match
                if (search_term == class_lower or 
                    any(search_term == word for word in class_lower.split())):
                    if class_name not in target_objects:
                        target_objects.append(class_name)
        
        print(f"✅ Found: {target_objects}")
        return target_objects
        
    def filter_images_advanced(self, folder_path, prompt, progress_callback=None):
        """Advanced image filtering with logic support"""
        if not self.model:
            print("❌ Model not loaded!")
            return []
        
        print(f"🚀 Starting ADVANCED image filtering with YOLOv8...")
        print(f"📁 Folder: {folder_path}")
        print(f"🔍 Prompt: {prompt}")
        
        parsed_prompt = self.parse_advanced_prompt(prompt)
        
        # Get all images
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        all_files = os.listdir(folder_path)
        image_files = [f for f in all_files if f.lower().endswith(image_extensions)]
        
        print(f"📊 Found {len(image_files)} image files")
        
        matching_images = []
        total_images = len(image_files)
        
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(folder_path, image_file)
            
            if progress_callback:
                progress = (i + 1) / total_images * 100
                progress_callback(progress, f"Processing {i+1}/{total_images}")
            
            has_target, all_detections = self.analyze_image_advanced(image_path, parsed_prompt)
            
            if has_target:
                # Get only the target objects that were detected
                target_detections = []
                if parsed_prompt['type'] == 'SINGLE':
                    target_objs = parsed_prompt['objects']
                elif parsed_prompt['type'] == 'OR':
                    target_objs = parsed_prompt['objects']
                else:  # AND
                    target_objs = []
                    for group in parsed_prompt['objects_groups']:
                        target_objs.extend(group)
                
                for obj_name in target_objs:
                    if obj_name in all_detections:
                        avg_confidence = sum(all_detections[obj_name]) / len(all_detections[obj_name])
                        target_detections.append({
                            'object': obj_name,
                            'confidence': avg_confidence
                        })
                
                matching_images.append({
                    'filename': image_file,
                    'path': image_path,
                    'objects': target_detections,
                    'all_detections': all_detections
                })
                print(f"🎯 ADDED TO RESULTS: {image_file}")
        
        print(f"✅ Advanced filtering complete. Found {len(matching_images)} matching images")
        return matching_images

class AdvancedImageFilterGUI:
    def __init__(self, root, model_path=None):
        self.root = root
        self.root.title("🎯 YOLOv8s-oiv7 ADVANCED - Smart Image Filter (600 Classes)")
        self.root.geometry("1200x800")
        
        self.filter = AdvancedImageFilter(model_path)
        self.current_folder = ""
        self.matching_images = []
        
        self.setup_gui()
        self.queue = queue.Queue()
        self.check_queue()
    
    def setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🎯 YOLOv8s-oiv7 ADVANCED Image Filter - 600 Open Images Classes", 
                               font=('Arial', 16, 'bold'), foreground='blue')
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Model info
        model_label = ttk.Label(main_frame, text=f"Model: {os.path.basename(self.filter.model_path) if self.filter.model_path else 'Not loaded'}", 
                               font=('Arial', 10), foreground='darkgreen')
        model_label.grid(row=0, column=2, sticky=tk.E, pady=(0, 20))
        
        # Folder selection
        ttk.Label(main_frame, text="Image Folder:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.folder_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.folder_var, width=60).grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_folder).grid(row=1, column=2, padx=5)
        
        # Search prompt with examples
        ttk.Label(main_frame, text="Advanced Search:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.prompt_var = tk.StringVar()
        self.prompt_var.set("person and car")  # Default advanced prompt
        prompt_entry = ttk.Entry(main_frame, textvariable=self.prompt_var, width=60)
        prompt_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=5)
        
        # Quick prompt buttons
        quick_frame = ttk.Frame(main_frame)
        quick_frame.grid(row=3, column=0, columnspan=3, pady=5, sticky=tk.W)
        
        ttk.Label(quick_frame, text="Quick searches:").pack(side=tk.LEFT, padx=5)
        
        quick_buttons = [
            ("👤+🚗", "person and car"),
            ("👤+🐕", "person and dog"),
            ("🚗+🏠", "car and building"),
            ("👤+💻", "person and laptop"),
            ("🍕+🥤", "pizza and bottle"),
            ("🦁+🌴", "lion and tree")
        ]
        
        for text, prompt in quick_buttons:
            ttk.Button(quick_frame, text=text, 
                      command=lambda p=prompt: self.prompt_var.set(p)).pack(side=tk.LEFT, padx=2)
        
        # Control buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=4, column=0, columnspan=3, pady=10, sticky=tk.W)
        
        ttk.Button(control_frame, text="🔍 Advanced Search", 
                  command=self.start_filtering, style='Accent.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="📁 Export Results", 
                  command=self.export_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="📋 List Folder", 
                  command=self.list_folder_contents).pack(side=tk.LEFT, padx=5)
        
        # Progress
        self.progress = ttk.Progressbar(main_frame, mode='determinate')
        self.progress.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        self.progress_label = ttk.Label(main_frame, text="Ready for advanced searches with YOLOv8!")
        self.progress_label.grid(row=6, column=0, columnspan=3, pady=(0, 10))
        
        # Results console
        console_frame = ttk.LabelFrame(main_frame, text="YOLOv8s-oiv7 ADVANCED SEARCH RESULTS", padding="10")
        console_frame.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        console_frame.columnconfigure(0, weight=1)
        console_frame.rowconfigure(0, weight=1)
        
        self.console_text = scrolledtext.ScrolledText(console_frame, height=15, width=100, wrap=tk.WORD)
        self.console_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Instructions
        instructions = """
🎯 YOLOv8s-oiv7 ADVANCED SEARCH (600 Open Images Classes):
• "person and car"        - Finds images with BOTH person AND car
• "dog or cat"            - Finds images with EITHER dog OR cat  
• "person with laptop"    - Finds images with person AND laptop
• "car and building"      - Finds images with car AND building
• "person and dog and car" - Finds images with ALL THREE objects

💡 TIP: YOLOv8s-oiv7 detects 600 different object types from Open Images v7
       including rare animals, specific foods, electronics, and many more!
        """
        
        instructions_label = ttk.Label(main_frame, text=instructions, justify=tk.LEFT, 
                                      font=('Arial', 9), foreground='darkgreen')
        instructions_label.grid(row=8, column=0, columnspan=3, pady=10, sticky=tk.W)
        
        # Redirect print to console
        self.original_stdout = sys.stdout
        class TextRedirector:
            def __init__(self, text_widget):
                self.text_widget = text_widget
            def write(self, string):
                self.text_widget.insert(tk.END, string)
                self.text_widget.see(tk.END)
            def flush(self):
                pass
        
        sys.stdout = TextRedirector(self.console_text)
    
    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.folder_var.set(folder)
            self.current_folder = folder
            print(f"📁 Selected folder: {folder}")
    
    def list_folder_contents(self):
        folder = self.folder_var.get()
        if not folder or not os.path.exists(folder):
            print("❌ No valid folder selected!")
            return
        
        print(f"\n📁 FOLDER CONTENTS: {folder}")
        print("=" * 50)
        all_files = os.listdir(folder)
        image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Total files: {len(all_files)}")
        print(f"Image files: {len(image_files)}")
        print("\nFirst 10 images:")
        for img_file in image_files[:10]:
            print(f"  - {img_file}")
        if len(image_files) > 10:
            print(f"  ... and {len(image_files) - 10} more")
        print("=" * 50)
    
    def update_progress(self, value, text):
        self.progress['value'] = value
        self.progress_label['text'] = text
    
    def start_filtering(self):
        if not self.folder_var.get():
            print("❌ Please select a folder first!")
            return
        
        if not self.prompt_var.get():
            print("❌ Please enter a search prompt!")
            return
        
        print(f"\n🚀 STARTING YOLOv8s-oiv7 ADVANCED SEARCH...")
        print(f"Folder: {self.folder_var.get()}")
        print(f"Advanced Prompt: '{self.prompt_var.get()}'")
        
        self.update_progress(0, "Starting YOLOv8 advanced search...")
        
        thread = threading.Thread(target=self.filter_images_thread)
        thread.daemon = True
        thread.start()
    
    def filter_images_thread(self):
        try:
            folder_path = self.folder_var.get()
            prompt = self.prompt_var.get()
            
            def progress_callback(progress, text):
                self.queue.put(('progress', progress, text))
            
            self.matching_images = self.filter.filter_images_advanced(folder_path, prompt, progress_callback)
            self.queue.put(('results', self.matching_images, prompt))
            
        except Exception as e:
            self.queue.put(('error', str(e)))
    
    def show_results(self, images, prompt):
        self.console_text.delete(1.0, tk.END)
        
        if not images:
            self.console_text.insert(tk.END, f"❌ No images found matching: '{prompt}'\n\n")
            self.console_text.insert(tk.END, "💡 Try these advanced searches:\n")
            self.console_text.insert(tk.END, "• 'person and car'\n")
            self.console_text.insert(tk.END, "• 'dog or cat'\n") 
            self.console_text.insert(tk.END, "• 'person with laptop'\n")
            self.console_text.insert(tk.END, "• 'food and bottle'\n")
            self.console_text.insert(tk.END, "• 'lion and tree' (Open Images special!)\n")
            return
        
        self.console_text.insert(tk.END, f"✅ Found {len(images)} images matching: '{prompt}'\n\n")
        
        for i, image_info in enumerate(images, 1):
            self.console_text.insert(tk.END, f"{i}. {image_info['filename']}\n")
            
            # Show target objects found
            objects_text = ", ".join([f"{obj['object']} ({(obj['confidence']*100):.1f}%)" 
                                    for obj in image_info['objects']])
            self.console_text.insert(tk.END, f"   Target objects: {objects_text}\n")
            
            # Show all detected objects
            all_objects = list(image_info['all_detections'].keys())
            if all_objects:
                self.console_text.insert(tk.END, f"   All detected: {', '.join(all_objects[:10])}")  # Show first 10
                if len(all_objects) > 10:
                    self.console_text.insert(tk.END, f" ... and {len(all_objects) - 10} more")
                self.console_text.insert(tk.END, "\n")
            
            self.console_text.insert(tk.END, "\n")
    
    def export_results(self):
        if not self.matching_images:
            messagebox.showinfo("Info", "No results to export!")
            return
        
        export_folder = filedialog.askdirectory(title="Select export folder")
        if not export_folder:
            return
        
        prompt_clean = "".join(c for c in self.prompt_var.get() if c.isalnum() or c in (' ', '-', '_')).rstrip()
        results_folder = os.path.join(export_folder, f"yolov8_advanced_{prompt_clean}")
        os.makedirs(results_folder, exist_ok=True)
        
        copied_count = 0
        for image_info in self.matching_images:
            try:
                dest_path = os.path.join(results_folder, image_info['filename'])
                shutil.copy2(image_info['path'], dest_path)
                copied_count += 1
            except Exception as e:
                print(f"Error copying {image_info['filename']}: {e}")
        
        messagebox.showinfo("Success", f"Exported {copied_count} images to:\n{results_folder}")
    
    def check_queue(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                if msg[0] == 'progress':
                    self.update_progress(msg[1], msg[2])
                elif msg[0] == 'results':
                    self.update_progress(100, "YOLOv8 advanced search completed!")
                    self.show_results(msg[1], msg[2])
                elif msg[0] == 'error':
                    self.update_progress(0, "Error occurred")
                    print(f"❌ ERROR: {msg[1]}")
        except queue.Empty:
            pass
        
        self.root.after(100, self.check_queue)
    
    def __del__(self):
        if hasattr(self, 'original_stdout'):
            sys.stdout = self.original_stdout

def main():
    # Hardcode your model path here
    model_path = "C:/Users/ASUS/Downloads/yolov8s-oiv7.pt"  # ← CHANGE THIS TO YOUR ACTUAL PATH
    
    # Use command line argument if provided
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    print(f"🔧 Using model path: {model_path}")
    
    # Check if ultralytics is installed
    if not ULTRALYTICS_AVAILABLE:
        print("❌ ERROR: ultralytics package not installed!")
        print("💡 Please install it with: pip install ultralytics")
        input("Press Enter to exit...")
        return
    
    root = tk.Tk()
    app = AdvancedImageFilterGUI(root, model_path)
    
    def on_closing():
        if hasattr(app, '__del__'):
            app.__del__()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
