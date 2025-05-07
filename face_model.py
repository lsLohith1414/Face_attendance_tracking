import os
import cv2
import numpy as np
import pickle
import logging
import time
from pathlib import Path
from collections import defaultdict
import torch
import torchvision.transforms as transforms
import torchvision.models as models  # Keep this for fallback
from concurrent.futures import ThreadPoolExecutor

# Import YOLOv8 and FaceNet with graceful fallback
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("YOLOv8 not available, falling back to OpenCV's face detector")

try:
    from facenet_pytorch import InceptionResnetV1
    FACENET_AVAILABLE = True
except ImportError:
    FACENET_AVAILABLE = False
    logging.warning("FaceNet not available, falling back to ResNet for face recognition")

class FaceRecognitionSystem:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Create necessary directories
        self.data_dir = Path("face_data")
        self.data_dir.mkdir(exist_ok=True)
        
        self.temp_faces_dir = self.data_dir / "temp_faces"
        self.temp_faces_dir.mkdir(exist_ok=True)
        
        self.embeddings_file = self.data_dir / "face_embeddings.pkl"
        
        # Load face embeddings if file exists
        self.face_embeddings = {}
        if self.embeddings_file.exists():
            try:
                with open(self.embeddings_file, 'rb') as f:
                    self.face_embeddings = pickle.load(f)
                self.logger.info(f"Loaded {len(self.face_embeddings)} face embeddings")
            except Exception as e:
                self.logger.error(f"Error loading face embeddings: {e}")
        
        # Initialize face alignment model for better recognition
        try:
            self.face_aligner = self._create_face_aligner()
            self.logger.info("Face alignment model loaded")
        except Exception as e:
            self.logger.error(f"Error loading face alignment model: {e}")
            self.face_aligner = None
        
        # Initialize models
        self.initialize_models()
        
        # Face collection variables
        self.current_collection = defaultdict(list)
        self.required_face_count = 5  # Number of faces to collect per person
        self.last_detection_time = {}  # To track when a face was last detected
        
    def initialize_models(self):
        # Load YOLOv8 for face detection (if available)
        if YOLO_AVAILABLE:
            try:
                # Load a YOLOv8 model trained for face detection
                # Using the small model for better performance
                self.face_detector = YOLO('yolov8n.pt')
                
                # If YOLOv8 model doesn't exist, try to download it
                if self.face_detector is None:
                    self.logger.info("Downloading YOLOv8 model...")
                    # Use the pretrained model and update it
                    self.face_detector = YOLO('yolov8n.pt')
                    
                self.detector_type = "yolo"
                self.logger.info("YOLOv8 model loaded")
            except Exception as e:
                self.logger.warning(f"Error loading YOLOv8 face detector: {e}, falling back to OpenCV")
                self._load_fallback_detector()
        else:
            # YOLOv8 not available, use OpenCV's face detectors
            self.logger.info("YOLOv8 not available, using OpenCV face detection")
            self._load_fallback_detector()
        
        # Load FaceNet for face recognition (if available)
        if FACENET_AVAILABLE:
            try:
                # Load a pretrained FaceNet model for face embedding extraction
                self.feature_extractor = InceptionResnetV1(pretrained='vggface2').eval()
                self.logger.info("FaceNet feature extractor loaded")
                
                # Define transformations for FaceNet - it expects specific preprocessing
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((160, 160)),  # FaceNet expects 160x160
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
            except Exception as e:
                self.logger.warning(f"Error loading FaceNet: {e}, falling back to ResNet")
                self._load_fallback_feature_extractor()
        else:
            # FaceNet not available, use ResNet as fallback
            self.logger.info("FaceNet not available, using ResNet for face recognition")
            self._load_fallback_feature_extractor()
            
        # Create a thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def _load_fallback_detector(self):
        """Load a fallback face detector using OpenCV"""
        try:
            # Try to load OpenCV's DNN face detector first (more accurate)
            prototxt_path = 'deploy.prototxt'
            caffemodel_path = 'res10_300x300_ssd_iter_140000.caffemodel'
            
            # Check if model files exist
            if os.path.exists(prototxt_path) and os.path.exists(caffemodel_path):
                # Use DNN face detector
                self.face_detector = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
                self.detector_type = "dnn"
                self.logger.info("OpenCV DNN face detection model loaded as fallback")
            else:
                # Fall back to Haar Cascade
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_detector = cv2.CascadeClassifier(cascade_path)
                self.detector_type = "haar"
                self.logger.info("OpenCV Haar Cascade face detection model loaded as fallback")
        except Exception as e:
            self.logger.error(f"Error loading fallback face detector: {e}")
            self.face_detector = None
            self.detector_type = None
    
    def _load_fallback_feature_extractor(self):
        """Load a fallback feature extractor using ResNet"""
        try:
            # Load a pre-trained ResNet model with higher capacity
            self.feature_extractor = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            # Remove the last fully-connected layer (we only need features)
            self.feature_extractor = torch.nn.Sequential(*list(self.feature_extractor.children())[:-1])
            self.feature_extractor.eval()
            self.logger.info("ResNet50 feature extractor loaded as fallback")
            
            # Enhanced transformations with better preprocessing for ResNet
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        except Exception as e:
            self.logger.error(f"Error loading fallback feature extractor: {e}")
            self.feature_extractor = None
            
    def _create_face_aligner(self):
        """Create a simple face alignment helper"""
        try:
            # This is a simple implementation - in a production system you might use a dedicated library
            # This aligns faces based on eye positions
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            
            if face_cascade.empty() or eye_cascade.empty():
                self.logger.error("Failed to load cascade classifiers")
                return None
                
            return (face_cascade, eye_cascade)
        except Exception as e:
            self.logger.error(f"Error creating face aligner: {e}")
            return None
        
    def get_face_embedding(self, face_img):
        """Generate embedding for a face using ResNet"""
        if self.feature_extractor is None:
            self.logger.error("Feature extractor not loaded")
            return None
        
        try:
            # Convert BGR to RGB
            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Apply transformations
            face_tensor = self.transform(face_img_rgb).unsqueeze(0)
            
            # Get embedding
            with torch.no_grad():
                embedding = self.feature_extractor(face_tensor)
                embedding = embedding.squeeze()
            
            # Convert to numpy and flatten
            embedding = embedding.numpy().flatten()
            
            return embedding
        
        except Exception as e:
            self.logger.error(f"Error generating face embedding: {e}")
            return None
    
    def match_face(self, embedding, threshold=0.75):
        """Match a face embedding against stored embeddings using enhanced algorithm"""
        if not self.face_embeddings:
            return None, 0.0
        
        # Store similarities per person for aggregate analysis
        person_similarities = {}
        
        # Calculate similarity scores for each person's embeddings
        for person_name, stored_embeddings in self.face_embeddings.items():
            # Skip if no embeddings for this person
            if not stored_embeddings:
                continue
                
            similarities = []
            
            # Compare with all stored embeddings for this person
            for stored_embedding in stored_embeddings:
                # Calculate cosine similarity (1 is perfect match, -1 is completely different)
                similarity = np.dot(embedding, stored_embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(stored_embedding)
                )
                similarities.append(similarity)
            
            # Robust similarity calculation:
            # - Use top 3 matches (if available) to reduce impact of outliers
            # - Calculate weighted average (more weight to better matches)
            similarities.sort(reverse=True)
            top_matches = similarities[:min(3, len(similarities))]
            
            if top_matches:
                # Calculate weighted average with more weight to better matches
                weights = np.linspace(1.0, 0.6, len(top_matches))
                avg_similarity = np.average(top_matches, weights=weights)
                person_similarities[person_name] = avg_similarity
        
        # Find the person with highest average similarity
        if person_similarities:
            best_match = max(person_similarities.items(), key=lambda x: x[1])
            best_match_name, best_similarity = best_match
            
            # More strict threshold for unknown faces (was 0.8, now 0.75)
            # This improves false positive rejection while maintaining sensitivity
            if best_similarity > threshold:
                return best_match_name, best_similarity
        
        return None, max(person_similarities.values()) if person_similarities else 0.0
    
    def process_frame(self, frame):
        """Process a video frame for face detection and recognition with improved algorithms"""
        try:
            if self.face_detector is None:
                self.logger.error("Face detector not loaded")
                cv2.putText(frame, "Model not loaded", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                return frame
            
            # Make a copy of the frame for processing
            frame_for_display = frame.copy()
            
            # Handle different detector types
            if self.detector_type == "yolo":
                # Use YOLOv8 for face detection (most accurate)
                try:
                    # YOLOv8 needs RGB format
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Run YOLOv8 inference
                    results = self.face_detector(frame_rgb, conf=0.5)  # Confidence threshold 0.5
                    
                    # Process detected faces
                    faces = []
                    if len(results) > 0:
                        # Get bounding boxes from the first result
                        boxes = results[0].boxes
                        for box in boxes:
                            # Get coordinates (YOLOv8 returns normalized xyxy)
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            w, h = x2 - x1, y2 - y1
                            faces.append([x1, y1, w, h])
                except Exception as e:
                    self.logger.error(f"Error in YOLOv8 face detection: {e}")
                    # Fall back to Haar Cascade if YOLOv8 fails
                    faces = self._fallback_detect_faces(frame)
            elif self.detector_type == "dnn":
                # Use DNN face detector (more accurate than Haar but less than YOLO)
                faces = self._detect_faces_dnn(frame)
            else:
                # Fall back to Haar Cascade detector
                faces = self._fallback_detect_faces(frame)
            
            # Process each detected face
            for face_rect in faces:
                try:
                    x, y, w, h = face_rect
                    
                    # Extract the face
                    face_img = frame[y:y+h, x:x+w]
                    
                    if face_img.size == 0:
                        continue
                        
                    # Apply face alignment if available for better recognition
                    if hasattr(self, 'face_aligner') and self.face_aligner is not None:
                        aligned_face = self._align_face(face_img)
                        if aligned_face is not None:
                            face_img = aligned_face
                    
                    # Generate embedding for this face
                    embedding = self.get_face_embedding(face_img)
                    
                    if embedding is None:
                        continue
                    
                    # Match face against stored embeddings using our improved algorithm
                    person_name, similarity = self.match_face(embedding)
                    
                    # Draw bounding box with color based on confidence
                    # High confidence: green, Medium: yellow, Low: red
                    if person_name:
                        if similarity > 0.9:
                            color = (0, 255, 0)  # Green for high confidence
                        elif similarity > 0.8:
                            color = (0, 255, 255)  # Yellow for medium confidence
                        else:
                            color = (0, 165, 255)  # Orange for lower confidence
                    else:
                        color = (0, 0, 255)  # Red for unknown
                        
                    cv2.rectangle(frame_for_display, (x, y), (x+w, y+h), color, 2)
                    
                    # Display name and confidence if recognized
                    if person_name:
                        confidence_text = f"{similarity:.2f}"
                        label = f"{person_name} ({confidence_text})"
                        
                        # Update last detection time for this person
                        current_time = time.time()
                        self.last_detection_time[person_name] = current_time
                        
                        # If person wasn't detected recently, show "New detection" for a short time
                        time_since_last = current_time - self.last_detection_time.get(person_name, 0)
                        if time_since_last > 10:  # If not seen in 10 seconds
                            self.logger.info(f"New detection: {person_name}")
                    else:
                        label = "Unknown"
                    
                    # Add label with better positioning and background for readability
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(frame_for_display, (x, y - 25), (x + text_size[0] + 10, y), color, -1)
                    cv2.putText(frame_for_display, label, (x + 5, y - 8), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                except Exception as e:
                    self.logger.error(f"Error processing face: {e}")
                    continue
            
            return frame_for_display
            
        except Exception as e:
            self.logger.error(f"Error in process_frame: {e}")
            return frame
        
    def _fallback_detect_faces(self, frame):
        """Fallback method for face detection using Haar Cascade"""
        try:
            # Create a new Haar Cascade classifier for fallback
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            cascade = cv2.CascadeClassifier(cascade_path)
            
            if cascade.empty():
                self.logger.error("Failed to load Haar Cascade classifier")
                return []
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            # Convert to format: [[x, y, w, h], ...]
            faces = [[x, y, w, h] for (x, y, w, h) in faces]
            return faces
        except Exception as e:
            self.logger.error(f"Error in fallback face detection: {e}")
            return []
        
    def _detect_faces_dnn(self, frame):
        """Detect faces using DNN detector for better accuracy"""
        # Convert the frame to blob
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
        
        # Set the blob as input and get detections
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()
        
        # Process detections
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Filter by confidence
            if confidence > 0.5:
                # Get coordinates and convert to correct format
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype("int")
                
                # Ensure bounding box is within frame
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Add face rect [x, y, width, height]
                faces.append([x1, y1, x2-x1, y2-y1])
        
        return faces
    
    def _align_face(self, face_img):
        """Align face for better recognition using eye positions"""
        if not hasattr(self, 'face_aligner') or self.face_aligner is None:
            return None
            
        try:
            face_cascade, eye_cascade = self.face_aligner
            
            # Convert to grayscale for detection
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            # Detect eyes within the face
            eyes = eye_cascade.detectMultiScale(gray)
            
            # Need at least 2 eyes for alignment
            if len(eyes) >= 2:
                # Get the two most prominent eyes
                eyes = sorted(eyes, key=lambda e: e[2]*e[3], reverse=True)[:2]
                
                # Sort eyes by x-coordinate (left to right)
                eyes = sorted(eyes, key=lambda e: e[0])
                
                # Get eye centers
                left_eye = (int(eyes[0][0] + eyes[0][2]//2), int(eyes[0][1] + eyes[0][3]//2))
                right_eye = (int(eyes[1][0] + eyes[1][2]//2), int(eyes[1][1] + eyes[1][3]//2))
                
                # Calculate angle for rotation
                delta_x = right_eye[0] - left_eye[0]
                delta_y = right_eye[1] - left_eye[1]
                angle = np.degrees(np.arctan2(delta_y, delta_x))
                
                # Rotate to align eyes horizontally
                center = (int((left_eye[0] + right_eye[0]) // 2), int((left_eye[1] + right_eye[1]) // 2))
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                aligned_face = cv2.warpAffine(face_img, M, (face_img.shape[1], face_img.shape[0]), 
                                             flags=cv2.INTER_CUBIC)
                
                return aligned_face
            
        except Exception as e:
            self.logger.warning(f"Error in face alignment: {e}")
            
        return None
    
    def add_new_person(self, person_name):
        """Start the process of adding a new person"""
        try:
            # Create directory for temporary face storage
            person_dir = self.temp_faces_dir / person_name
            person_dir.mkdir(exist_ok=True)
            
            # Clear any existing face data for this person
            for file in person_dir.glob("*.jpg"):
                file.unlink()
            
            # Reset collection for this person
            self.current_collection[person_name] = []
            
            return True
        except Exception as e:
            self.logger.error(f"Error adding new person: {e}")
            return False
            
    def process_uploaded_image(self, person_name, image_file):
        """Process an uploaded image file for a person with enhanced face detection using YOLOv8"""
        try:
            if self.face_detector is None:
                self.logger.error("Face detection model not loaded")
                return {"success": False, "message": "Face detection model not loaded"}
            
            # Read the uploaded image file
            try:
                # Read the file content
                file_content = image_file.read()
                
                # Validate file content
                if not file_content:
                    self.logger.error("Empty file content")
                    return {"success": False, "message": "Empty image file"}
                
                # Convert to numpy array
                image_array = np.frombuffer(file_content, np.uint8)
                
                # Validate image array
                if image_array.size == 0:
                    self.logger.error("Empty image array")
                    return {"success": False, "message": "Invalid image data"}
                
                # Decode the image
                frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                
                if frame is None:
                    self.logger.error("Failed to decode image")
                    return {"success": False, "message": "Failed to decode image. Please ensure it's a valid image file"}
                
                # Validate frame dimensions
                if frame.size == 0:
                    self.logger.error("Empty frame")
                    return {"success": False, "message": "Invalid image dimensions"}
                
                # Log image dimensions
                self.logger.info(f"Processing image with dimensions: {frame.shape}")
                
                # Handle different detector types for better accuracy
                if self.detector_type == "yolo":
                    # Use YOLOv8 for face detection (most accurate)
                    try:
                        # YOLOv8 needs RGB format
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # Run YOLOv8 inference with lower confidence threshold
                        results = self.face_detector(frame_rgb, conf=0.3)  # Lower threshold for more detections
                        
                        # Process detected faces
                        face_rects = []
                        if len(results) > 0:
                            # Get bounding boxes from the first result
                            boxes = results[0].boxes
                            for box in boxes:
                                # Get coordinates (YOLOv8 returns normalized xyxy)
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                w, h = x2 - x1, y2 - y1
                                face_rects.append([x1, y1, w, h])
                        
                        self.logger.info(f"YOLOv8 detected {len(face_rects)} faces")
                        
                    except Exception as e:
                        self.logger.error(f"Error in YOLOv8 face detection: {e}")
                        # Fall back to alternatives if YOLOv8 fails
                        if self.detector_type == "dnn":
                            face_rects = self._detect_faces_dnn(frame)
                        else:
                            face_rects = self._fallback_detect_faces(frame)
                elif self.detector_type == "dnn":
                    # Use DNN face detector (more accurate than Haar but less than YOLO)
                    face_rects = self._detect_faces_dnn(frame)
                else:
                    # Fall back to Haar Cascade detector
                    face_rects = self._fallback_detect_faces(frame)
                
                faces_found = len(face_rects)
                self.logger.info(f"Total faces detected: {faces_found}")
                
                if faces_found == 0:
                    self.logger.warning("No faces detected in the uploaded image")
                    return {"success": False, "message": "No face detected in the uploaded image"}
                
                # Create directory for this person if it doesn't exist
                person_dir = self.temp_faces_dir / person_name
                person_dir.mkdir(exist_ok=True)
                
                processed_faces = 0
                
                # Process all detected faces (we might want multiple faces from a group photo)
                for idx, face_rect in enumerate(face_rects):
                    try:
                        # Extract the face coordinates (handle both formats)
                        if isinstance(face_rect, np.ndarray) and len(face_rect) == 4:
                            x, y, w, h = face_rect
                        else:
                            x, y, w, h = face_rect
                        
                        # Skip extremely small faces (likely false positives)
                        if w < 20 or h < 20:
                            self.logger.warning(f"Skipping face {idx} - too small ({w}x{h})")
                            continue
                        
                        # Extract the face
                        face_img = frame[y:y+h, x:x+w]
                        
                        if face_img.size == 0:
                            self.logger.warning(f"Skipping face {idx} - empty region")
                            continue
                        
                        # Apply face alignment for better recognition if available
                        if hasattr(self, 'face_aligner') and self.face_aligner is not None:
                            aligned_face = self._align_face(face_img)
                            if aligned_face is not None:
                                # Use the aligned face instead
                                face_img = aligned_face
                        
                        # Apply pre-processing (histogram equalization for better features) 
                        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                        equalized_face = cv2.equalizeHist(gray_face)
                        equalized_face_color = cv2.cvtColor(equalized_face, cv2.COLOR_GRAY2BGR)
                        
                        # Blend with original for better results (maintain color information)
                        enhanced_face = cv2.addWeighted(face_img, 0.7, equalized_face_color, 0.3, 0)
                        
                        # Save face image with the enhanced version
                        img_count = len(list(person_dir.glob("*.jpg")))
                        face_path = person_dir / f"face_{img_count + 1}.jpg"
                        cv2.imwrite(str(face_path), enhanced_face)
                        
                        # Add to current collection
                        self.current_collection[person_name].append(str(face_path))
                        processed_faces += 1
                        self.logger.info(f"Successfully processed face {idx + 1}/{faces_found}")
                        
                    except Exception as e:
                        self.logger.error(f"Error processing face {idx}: {e}")
                        continue
                
                current_count = len(self.current_collection[person_name])
                remaining = max(0, self.required_face_count - current_count)
                
                # Provide improved feedback
                completion_status = "complete" if current_count >= self.required_face_count else "in progress"
                message = f"Processed {processed_faces} faces from image ({current_count}/{self.required_face_count} total)"
                
                if current_count >= self.required_face_count:
                    message += ". You can now finish training."
                else:
                    message += f". Need {remaining} more faces."
                
                return {
                    "success": True, 
                    "message": message,
                    "current_count": current_count,
                    "required_count": self.required_face_count,
                    "remaining": remaining,
                    "complete": current_count >= self.required_face_count
                }
                
            except Exception as e:
                self.logger.error(f"Error reading image file: {e}")
                return {"success": False, "message": f"Error reading image file: {str(e)}"}
            
        except Exception as e:
            self.logger.error(f"Error processing uploaded image: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}
    
    def capture_face(self, frame, person_name):
        """Capture a face for the given person using enhanced processing"""
        try:
            if self.face_detector is None:
                return {"success": False, "message": "Face detection model not loaded"}
            
            # Handle different detector types for better accuracy
            if self.detector_type == "yolo":
                # Use YOLOv8 for face detection (most accurate)
                try:
                    # YOLOv8 needs RGB format
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Run YOLOv8 inference
                    results = self.face_detector(frame_rgb, conf=0.5)  # Confidence threshold 0.5
                    
                    # Process detected faces
                    face_rects = []
                    if len(results) > 0:
                        # Get bounding boxes from the first result
                        boxes = results[0].boxes
                        for box in boxes:
                            # Get coordinates (YOLOv8 returns normalized xyxy)
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            w, h = x2 - x1, y2 - y1
                            face_rects.append([x1, y1, w, h])
                except Exception as e:
                    self.logger.error(f"Error in YOLOv8 face detection during capture: {e}")
                    # Fall back to DNN or Haar Cascade if YOLOv8 fails
                    if self.detector_type == "dnn":
                        face_rects = self._detect_faces_dnn(frame)
                    else:
                        face_rects = self._fallback_detect_faces(frame)
            elif self.detector_type == "dnn":
                # Use DNN face detector (more accurate than Haar but less than YOLO)
                face_rects = self._detect_faces_dnn(frame)
            else:
                # Fall back to Haar Cascade detector
                face_rects = self._fallback_detect_faces(frame)
            
            faces_found = len(face_rects)
            
            if faces_found == 0:
                return {"success": False, "message": "No face detected in frame"}
            elif faces_found > 1:
                return {"success": False, "message": "Multiple faces detected, please ensure only one face is visible"}
            
            # Process the single detected face with our improved pipeline
            face_rect = face_rects[0]
            
            # Extract the face coordinates (handle both formats)
            if isinstance(face_rect, np.ndarray) and len(face_rect) == 4:
                x, y, w, h = face_rect
            else:
                x, y, w, h = face_rect
            
            # Extract the face
            face_img = frame[y:y+h, x:x+w]
            
            if face_img.size == 0:
                return {"success": False, "message": "Invalid face region detected"}
            
            # Apply face alignment for better recognition if available
            if self.face_aligner is not None:
                aligned_face = self._align_face(face_img)
                if aligned_face is not None:
                    # Use the aligned face instead
                    face_img = aligned_face
            
            # Apply pre-processing (histogram equalization for better features)
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            equalized_face = cv2.equalizeHist(gray_face)
            equalized_face_color = cv2.cvtColor(equalized_face, cv2.COLOR_GRAY2BGR)
            
            # Blend with original for better results (maintain color information)
            enhanced_face = cv2.addWeighted(face_img, 0.7, equalized_face_color, 0.3, 0)
            
            # Create directory for this person if it doesn't exist
            person_dir = self.temp_faces_dir / person_name
            person_dir.mkdir(exist_ok=True)
            
            # Save the enhanced face image
            img_count = len(list(person_dir.glob("*.jpg")))
            face_path = person_dir / f"face_{img_count + 1}.jpg"
            cv2.imwrite(str(face_path), enhanced_face)
            
            # Add to current collection
            self.current_collection[person_name].append(str(face_path))
            
            # Calculate progress
            current_count = len(self.current_collection[person_name])
            remaining = max(0, self.required_face_count - current_count)
            
            # Provide improved feedback
            message = f"Face captured ({current_count}/{self.required_face_count})"
            
            if current_count >= self.required_face_count:
                message += ". You can now finish training!"
            else:
                message += f". Need {remaining} more faces."
            
            return {
                "success": True, 
                "message": message,
                "current_count": current_count,
                "required_count": self.required_face_count,
                "remaining": remaining,
                "complete": current_count >= self.required_face_count
            }
            
        except Exception as e:
            self.logger.error(f"Error capturing face: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}
    
    def process_and_save_embeddings(self, person_name):
        """Process collected faces and save embeddings with improved quality"""
        try:
            person_dir = self.temp_faces_dir / person_name
            
            if not person_dir.exists():
                self.logger.error(f"Directory for {person_name} doesn't exist")
                return False
            
            face_files = list(person_dir.glob("*.jpg"))
            if not face_files:
                self.logger.error(f"No face images found for {person_name}")
                return False
            
            # Generate embeddings for each face - use parallel processing if available
            embeddings = []
            embedding_qualities = []  # To track embedding quality for filtering
            
            if hasattr(self, 'executor') and self.executor:
                # Process embeddings in parallel for better performance
                try:
                    # Submit all face processing tasks to thread pool
                    future_embeddings = [self.executor.submit(self._process_face_file, face_file) 
                                        for face_file in face_files]
                    
                    # Collect results as they complete
                    for future in future_embeddings:
                        result = future.result()
                        if result is not None:
                            embedding, quality = result
                            embeddings.append(embedding)
                            embedding_qualities.append(quality)
                            
                except Exception as e:
                    self.logger.error(f"Error in parallel processing: {e}, falling back to sequential")
                    # Clear partial results
                    embeddings = []
                    embedding_qualities = []
                    
                    # Fall back to sequential processing
                    for face_file in face_files:
                        result = self._process_face_file(face_file)
                        if result is not None:
                            embedding, quality = result
                            embeddings.append(embedding)
                            embedding_qualities.append(quality)
            else:
                # Process sequentially if thread pool not available
                for face_file in face_files:
                    result = self._process_face_file(face_file)
                    if result is not None:
                        embedding, quality = result
                        embeddings.append(embedding)
                        embedding_qualities.append(quality)
            
            if not embeddings:
                self.logger.error(f"No valid embeddings generated for {person_name}")
                return False
            
            # Filter out low-quality embeddings if we have enough good ones
            if len(embeddings) > 3:
                # Sort by quality score (higher is better)
                embedding_data = list(zip(embeddings, embedding_qualities))
                embedding_data.sort(key=lambda x: x[1], reverse=True)
                
                # Take the best quality embeddings
                best_embeddings = [e for e, q in embedding_data]
                embeddings = best_embeddings
                
                self.logger.info(f"Selected {len(embeddings)} highest quality embeddings")
            
            # Add to face embeddings dictionary
            self.face_embeddings[person_name] = embeddings
            
            # Save updated embeddings to disk
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(self.face_embeddings, f)
            
            self.logger.info(f"Saved {len(embeddings)} embeddings for {person_name}")
            
            # Clear temporary face collection
            self.current_collection[person_name] = []
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing faces: {e}")
            return False
            
    def _process_face_file(self, face_file):
        """Process a single face file to generate embeddings
        Returns tuple of (embedding, quality_score) or None if failed
        """
        try:
            # Load the face image
            face_img = cv2.imread(str(face_file))
            
            if face_img is None:
                self.logger.warning(f"Could not read image: {face_file}")
                return None
            
            # Calculate quality score based on image features
            quality_score = self._calculate_face_quality(face_img)
            
            # Generate embedding
            embedding = self.get_face_embedding(face_img)
            
            if embedding is not None:
                return (embedding, quality_score)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error processing face {face_file}: {e}")
            return None
            
    def _calculate_face_quality(self, face_img):
        """Calculate a quality score for a face image (higher is better)
        This helps identify the best quality face images for better recognition
        """
        try:
            # Convert to grayscale for analysis
            if len(face_img.shape) > 2:
                gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_img.copy()
            
            # Calculate factors that contribute to quality
            
            # 1. Sharpness (using variance of Laplacian)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            
            # 2. Contrast (using histogram range)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            contrast = max(1.0, (np.max(hist) - np.min(hist)) / 256.0)
            
            # 3. Brightness (mean pixel value normalized)
            brightness = np.mean(gray) / 255.0
            
            # 4. Face size relative to typical good size
            size_factor = (face_img.shape[0] * face_img.shape[1]) / (100.0 * 100.0)
            size_score = min(1.0, size_factor)  # Normalize to max of 1.0
            
            # Combine factors with weights
            quality_score = (
                0.4 * sharpness / 100.0 +  # Normalize by typical values
                0.2 * contrast +
                0.2 * (1.0 - abs(brightness - 0.5) * 2) +  # Penalize too bright/dark
                0.2 * size_score
            )
            
            return quality_score
            
        except Exception as e:
            self.logger.warning(f"Error calculating face quality: {e}")
            return 0.5  # Default middle value
    
    def get_all_people(self):
        """Get list of all people in the system"""
        people = list(self.face_embeddings.keys())
        
        # Add metadata: number of face embeddings per person
        people_with_metadata = []
        for person in people:
            embedding_count = len(self.face_embeddings.get(person, []))
            people_with_metadata.append({
                "name": person,
                "embedding_count": embedding_count
            })
        
        return people_with_metadata
    
    def delete_person(self, person_name):
        """Delete a person from the system"""
        try:
            # Remove from embeddings dictionary
            if person_name in self.face_embeddings:
                del self.face_embeddings[person_name]
                
                # Save updated embeddings to disk
                with open(self.embeddings_file, 'wb') as f:
                    pickle.dump(self.face_embeddings, f)
                
                # Delete temporary face directory if it exists
                person_dir = self.temp_faces_dir / person_name
                if person_dir.exists():
                    for file in person_dir.glob("*.jpg"):
                        file.unlink()
                    person_dir.rmdir()
                
                self.logger.info(f"Deleted person: {person_name}")
                return True
            else:
                self.logger.warning(f"Person {person_name} not found in embeddings")
                return False
                
        except Exception as e:
            self.logger.error(f"Error deleting person: {e}")
            return False