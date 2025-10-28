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

class AdvancedImageFilter:
    def __init__(self):
        self.model = None
        self.load_model()
        
        self.keyword_mappings = {
            'person': ['person', 'people', 'human', 'man', 'woman', 'child', 'baby', 'face'],
            'animal': ['animal', 'pet', 'dog', 'cat', 'bird', 'horse', 'cow', 'sheep', 'elephant', 
                      'bear', 'zebra', 'giraffe', 'puppy', 'kitten'],
            'vehicle': ['vehicle', 'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'bike', 'auto', 'transport'],
            'food': ['food', 'fruit', 'apple', 'banana', 'orange', 'pizza', 'sandwich', 'meal', 'eating'],
            'electronic': ['electronic', 'laptop', 'computer', 'phone', 'cellphone', 'tv', 'television', 'device'],
            'furniture': ['furniture', 'chair', 'table', 'couch', 'sofa', 'bed', 'desk'],
            'sports': ['sports', 'ball', 'baseball', 'tennis', 'skateboard', 'game'],
            'outdoor': ['outdoor', 'tree', 'sky', 'building', 'house', 'nature']
        }
        
        self.coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
            'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
            'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear'
        ]
    
    def load_model(self):
        """Load YOLOv5 model"""
        try:
            print("🔄 Loading YOLOv5 model...")
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            self.model.conf = 0.3  # Lower confidence for better detection
            print("✅ Model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False
    
    def parse_advanced_prompt(self, prompt):
        """Parse complex prompts with AND/OR logic"""
        prompt_lower = prompt.lower().strip()
        print(f"🔍 Parsing advanced prompt: '{prompt}'")
        
        # Handle AND conditions (person AND car)
        if ' and ' in prompt_lower:
            parts = [part.strip() for part in prompt_lower.split(' and ')]
            required_groups = []
            for part in parts:
                objects = self.parse_single_prompt(part)
                required_groups.append(objects)
                print(f"   - AND condition: {part} -> {objects}")
            return {'type': 'AND', 'objects_groups': required_groups}
        
        # Handle OR conditions (person OR car)
        elif ' or ' in prompt_lower:
            parts = [part.strip() for part in prompt_lower.split(' or ')]
            all_objects = []
            for part in parts:
                objects = self.parse_single_prompt(part)
                all_objects.extend(objects)
            all_objects = list(set(all_objects))
            print(f"   - OR condition: {parts} -> {all_objects}")
            return {'type': 'OR', 'objects': all_objects}
        
        # Handle WITH conditions (person with car)
        elif ' with ' in prompt_lower:
            parts = [part.strip() for part in prompt_lower.split(' with ')]
            required_groups = []
            for part in parts:
                objects = self.parse_single_prompt(part)
                required_groups.append(objects)
            return {'type': 'AND', 'objects_groups': required_groups}
        
        # Single condition
        else:
            objects = self.parse_single_prompt(prompt)
            print(f"   - Single condition: {prompt} -> {objects}")
            return {'type': 'SINGLE', 'objects': objects}
    
    def parse_single_prompt(self, prompt):
        """Parse single prompt part"""
        prompt_lower = prompt.lower()
        target_objects = []
        
        # Direct COCO class matching
        for coco_class in self.coco_classes:
            if coco_class in prompt_lower:
                target_objects.append(coco_class)
        
        # Keyword mapping
        for coco_class, keywords in self.keyword_mappings.items():
            for keyword in keywords:
                if keyword in prompt_lower and coco_class not in target_objects:
                    target_objects.append(coco_class)
        
        # Special cases
        if 'animal' in prompt_lower:
            animal_classes = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']
            target_objects.extend([a for a in animal_classes if a not in target_objects])
        
        if 'vehicle' in prompt_lower:
            vehicle_classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'airplane', 'train', 'boat']
            target_objects.extend([v for v in vehicle_classes if v not in target_objects])
        
        return list(set(target_objects))
    
    def analyze_image_advanced(self, image_path, parsed_prompt):
        """Advanced image analysis with logic support"""
        try:
            print(f"   Analyzing: {os.path.basename(image_path)}")
            results = self.model(image_path)
            detections = results.pandas().xyxy[0]
            
            # Get all detected objects with confidence > 0.3
            detected_objects = {}
            for _, detection in detections.iterrows():
                if detection['confidence'] > 0.3:
                    obj_name = detection['name']
                    if obj_name not in detected_objects:
                        detected_objects[obj_name] = []
                    detected_objects[obj_name].append(detection['confidence'])
            
            print(f"     Detected: {list(detected_objects.keys())}")
            
            # Apply logic based on prompt type
            if parsed_prompt['type'] == 'SINGLE':
                # Any of the target objects found
                for target_obj in parsed_prompt['objects']:
                    if target_obj in detected_objects:
                        print(f"     ✅ Found {target_obj}")
                        return True, detected_objects
            
            elif parsed_prompt['type'] == 'OR':
                # Any of the target objects found
                for target_obj in parsed_prompt['objects']:
                    if target_obj in detected_objects:
                        print(f"     ✅ Found {target_obj} (OR condition)")
                        return True, detected_objects
            
            elif parsed_prompt['type'] == 'AND':
                # ALL object groups must be satisfied
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
            
        except Exception as e:
            print(f"     ❌ Error analyzing image: {e}")
            return False, {}
    
    def filter_images_advanced(self, folder_path, prompt, progress_callback=None):
        """Advanced image filtering with logic support"""
        if not self.model:
            print("❌ Model not loaded!")
            return []
        
        print(f"🚀 Starting ADVANCED image filtering...")
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
    def __init__(self, root):
        self.root = root
        self.root.title("🎯 ADVANCED - Smart Image Filter")
        self.root.geometry("1200x800")
        
        self.filter = AdvancedImageFilter()
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
        title_label = ttk.Label(main_frame, text="🎯 ADVANCED Image Filter - Multi-Class Search", 
                               font=('Arial', 16, 'bold'), foreground='blue')
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
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
            ("🍕+🥤", "pizza and bottle")
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
        self.progress_label = ttk.Label(main_frame, text="Ready for advanced searches!")
        self.progress_label.grid(row=6, column=0, columnspan=3, pady=(0, 10))
        
        # Results console
        console_frame = ttk.LabelFrame(main_frame, text="ADVANCED SEARCH RESULTS", padding="10")
        console_frame.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        console_frame.columnconfigure(0, weight=1)
        console_frame.rowconfigure(0, weight=1)
        
        self.console_text = scrolledtext.ScrolledText(console_frame, height=15, width=100, wrap=tk.WORD)
        self.console_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Instructions
        instructions = """
🎯 ADVANCED SEARCH EXAMPLES:
• "person and car"        - Finds images with BOTH person AND car
• "dog or cat"            - Finds images with EITHER dog OR cat  
• "person with laptop"    - Finds images with person AND laptop
• "car and building"      - Finds images with car AND building
• "person and dog and car" - Finds images with ALL THREE objects

💡 TIP: Use 'and' for multiple objects, 'or' for either object
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
        
        print(f"\n🚀 STARTING ADVANCED SEARCH...")
        print(f"Folder: {self.folder_var.get()}")
        print(f"Advanced Prompt: '{self.prompt_var.get()}'")
        
        self.update_progress(0, "Starting advanced search...")
        
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
                self.console_text.insert(tk.END, f"   All detected: {', '.join(all_objects)}\n")
            
            self.console_text.insert(tk.END, "\n")
    
    def export_results(self):
        if not self.matching_images:
            messagebox.showinfo("Info", "No results to export!")
            return
        
        export_folder = filedialog.askdirectory(title="Select export folder")
        if not export_folder:
            return
        
        prompt_clean = "".join(c for c in self.prompt_var.get() if c.isalnum() or c in (' ', '-', '_')).rstrip()
        results_folder = os.path.join(export_folder, f"advanced_{prompt_clean}")
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
                    self.update_progress(100, "Advanced search completed!")
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
    root = tk.Tk()
    app = AdvancedImageFilterGUI(root)
    
    def on_closing():
        if hasattr(app, '__del__'):
            app.__del__()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()