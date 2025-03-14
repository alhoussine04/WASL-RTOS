import cv2
import numpy as np
import time
import json
import os
import openvino as ov
from collections import Counter, deque
import threading
import argparse

class ASLInterpreter:
    def __init__(self, model_xml="asl-recognition-0004.xml", 
                 model_bin="asl-recognition-0004.bin", 
                 classes_file="MSASL_classes.json",
                 device="CPU",
                 confidence_threshold=0.80,
                 stability_threshold=3,
                 cooldown_threshold=45,
                 motion_threshold=50):
        # Initialize parameters
        self.confidence_threshold = confidence_threshold
        self.stability_threshold = stability_threshold
        self.cooldown_threshold = cooldown_threshold
        self.motion_threshold = motion_threshold
        
        # Initialize model
        self.compiled_model, self.input_node, self.output_node = self._load_asl_model(model_xml, model_bin, device)
        
        # Get model input shape
        input_shape = self.input_node.shape
        _, _, self.clip_length, self.height, self.width = input_shape
        
        # Load classes
        self.classes = self._load_classes(classes_file)
        
        # Initialize buffers
        self.frame_buffer = deque(maxlen=self.clip_length)
        self.prediction_buffer = deque(maxlen=8)
        
        # Initialize variables for prediction tracking
        self.transcript = []
        self.last_prediction = ""
        self.prediction_cooldown = 0
        self.stability_counter = {}
        
        # Motion detection variables
        self.prev_gray = None
        self.no_motion_frames = 0
        
        # Thread for inference
        self.inference_thread = None
        self.inference_running = False
        self.latest_result = None
        self.result_lock = threading.Lock()

    def _load_asl_model(self, model_xml, model_bin, device):
        """Load the ASL recognition model using OpenVINO"""
        try:
            # Create OpenVINO Core object
            core = ov.Core()
            
            # Check if model files exist
            if not os.path.exists(model_xml) or not os.path.exists(model_bin):
                raise FileNotFoundError(f"Model files not found: {model_xml} or {model_bin}")
            
            # Load the model
            model = core.read_model(model_xml, model_bin)
            
            # Compile the model for specified device
            compiled_model = core.compile_model(model, device)
            
            # Get input and output nodes
            input_node = compiled_model.input(0)
            output_node = compiled_model.output(0)
            
            print(f"Model loaded successfully on {device}")
            return compiled_model, input_node, output_node
        
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def _load_classes(self, classes_file):
        """Load class labels from JSON file"""
        try:
            if not os.path.exists(classes_file):
                raise FileNotFoundError(f"Classes file not found: {classes_file}")
                
            with open(classes_file, "r") as f:
                classes = json.load(f)
            return classes
        
        except Exception as e:
            print(f"Error loading classes: {e}")
            raise

    def _process_frames(self, frames):
        """Process video frames for ASL recognition"""
        # Resize frames to match model input shape (1, 3, 16, 224, 224)
        processed_frames = []
        for frame in frames:
            # Resize frame to model dimensions
            resized = cv2.resize(frame, (self.width, self.height))
            # Convert to RGB
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            # Add to processed frames
            processed_frames.append(rgb)
        
        # Stack frames to create a clip
        clip = np.array(processed_frames)
        
        # Transpose from (16, 224, 224, 3) to (3, 16, 224, 224)
        clip = np.transpose(clip, (3, 0, 1, 2))
        
        # Add batch dimension
        clip = np.expand_dims(clip, axis=0)
        
        return clip

    def _run_inference_thread(self):
        """Run inference in a separate thread to improve performance"""
        while self.inference_running:
            if len(self.frame_buffer) == self.clip_length:
                # Make a copy of the frame buffer to prevent race conditions
                frames = list(self.frame_buffer)
                
                # Process frames for model input
                input_data = self._process_frames(frames)
                
                # Run inference
                result = self.compiled_model(input_data)
                output = result[self.output_node]
                
                # Update latest result with lock to prevent race conditions
                with self.result_lock:
                    self.latest_result = output
            
            # Sleep to reduce CPU usage
            time.sleep(0.01)

    def start_inference_thread(self):
        """Start the inference thread"""
        self.inference_running = True
        self.inference_thread = threading.Thread(target=self._run_inference_thread)
        self.inference_thread.daemon = True
        self.inference_thread.start()

    def stop_inference_thread(self):
        """Stop the inference thread"""
        self.inference_running = False
        if self.inference_thread:
            self.inference_thread.join(timeout=1.0)

    def process_frame(self, frame):
        """Process a single frame and return visualization"""
        # Create a copy for visualization
        vis_frame = frame.copy()
        
        # Motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        motion_score = 0
        if self.prev_gray is not None:
            # Calculate difference between current and previous frame
            frame_diff = cv2.absdiff(self.prev_gray, gray)
            thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
            motion_score = np.sum(thresh) / 255
        
        # Update previous frame
        self.prev_gray = gray
        
        # Add frame to buffer
        self.frame_buffer.append(frame)
        
        # Handle motion detection
        if motion_score < self.motion_threshold:
            self.no_motion_frames += 1
            if self.no_motion_frames > 10:
                self.stability_counter.clear()
            
            # Display "No motion detected" if no significant movement
            if self.no_motion_frames > 20:
                cv2.putText(vis_frame, "No motion detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            self.no_motion_frames = 0
            
            # Get the latest inference result
            with self.result_lock:
                output = self.latest_result
            
            if output is not None:
                # Get top 3 predictions for more robust decision making
                top_indices = np.argsort(output[0])[-3:][::-1]
                top_confidences = output[0][top_indices]
                
                # Only consider prediction if confidence is above threshold
                if top_confidences[0] >= self.confidence_threshold:
                    prediction = self.classes[top_indices[0]]
                    confidence = top_confidences[0]
                    
                    # Add prediction to buffer
                    self.prediction_buffer.append(prediction)
                    
                    # Update stability counter
                    if prediction in self.stability_counter:
                        self.stability_counter[prediction] += 1
                    else:
                        self.stability_counter = {prediction: 1}  # Reset counter for new prediction
                    
                    # Get most common prediction in buffer
                    counter = Counter(self.prediction_buffer)
                    
                    # Only consider a prediction stable if it appears consistently
                    stable_prediction = None
                    for pred, count in self.stability_counter.items():
                        if count >= self.stability_threshold and counter[pred] >= 3:
                            stable_prediction = pred
                            break
                    
                    # Add to transcript if it's a stable prediction
                    if stable_prediction and (stable_prediction != self.last_prediction or 
                                             self.prediction_cooldown > self.cooldown_threshold):
                        self.transcript.append(f"{stable_prediction} ({confidence*100:.1f}%)")
                        self.last_prediction = stable_prediction
                        self.prediction_cooldown = 0
                        # Reset stability counter after adding to transcript
                        self.stability_counter = {stable_prediction: 0}
                    else:
                        self.prediction_cooldown += 1
                    
                    # Display prediction and confidence
                    prediction_text = f"Prediction: {prediction} ({confidence:.2f})"
                    cv2.putText(vis_frame, prediction_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Display stability information
                    stability_text = f"Stability: {self.stability_counter.get(prediction, 0)}/{self.stability_threshold}"
                    cv2.putText(vis_frame, stability_text, (10, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                else:
                    # If confidence is low, show "Uncertain"
                    cv2.putText(vis_frame, "Uncertain", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display motion score
        cv2.putText(vis_frame, f"Motion: {motion_score:.0f}/{self.motion_threshold}", (10, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Display transcript (last 5 items)
        transcript_text = " ".join(self.transcript[-5:])
        cv2.putText(vis_frame, "Transcript:", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Split transcript text into lines if too long
        max_width = 60
        words = transcript_text.split()
        lines = []
        current_line = []
        
        for word in words:
            if len(" ".join(current_line + [word])) <= max_width:
                current_line.append(word)
            else:
                lines.append(" ".join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(" ".join(current_line))
        
        for i, line in enumerate(lines):
            cv2.putText(vis_frame, line, (10, 120 + i * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return vis_frame, motion_score

    def save_transcript(self, filename="asl_transcript.txt"):
        """Save transcript to a file"""
        with open(filename, "w") as f:
            f.write(" ".join(self.transcript))
        print(f"Transcript saved to {filename}")
        
        # Also save a timestamped version
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        timestamped_filename = f"asl_transcript_{timestamp}.txt"
        with open(timestamped_filename, "w") as f:
            f.write(" ".join(self.transcript))
        print(f"Transcript also saved to {timestamped_filename}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ASL Interpreter')
    parser.add_argument('--model-xml', type=str, default="asl-recognition-0004.xml", 
                        help='Path to OpenVINO model XML file')
    parser.add_argument('--model-bin', type=str, default="asl-recognition-0004.bin", 
                        help='Path to OpenVINO model BIN file')
    parser.add_argument('--classes', type=str, default="MSASL_classes.json", 
                        help='Path to classes JSON file')
    parser.add_argument('--device', type=str, default="CPU", 
                        help='Device for inference: CPU, GPU, MYRIAD, etc.')
    parser.add_argument('--confidence', type=float, default=0.80, 
                        help='Confidence threshold for predictions')
    parser.add_argument('--stability', type=int, default=3, 
                        help='Stability threshold for predictions')
    parser.add_argument('--motion', type=int, default=50, 
                        help='Motion threshold for triggering recognition')
    parser.add_argument('--camera', type=int, default=0, 
                        help='Camera index to use')
    parser.add_argument('--output', type=str, default="asl_transcript.txt", 
                        help='Output file for transcript')
    parser.add_argument('--record', action='store_true', 
                        help='Record video output')
    args = parser.parse_args()

    # Initialize video capture
    cap = cv2.VideoCapture(args.camera)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Initialize ASL interpreter
    interpreter = ASLInterpreter(
        model_xml=args.model_xml,
        model_bin=args.model_bin,
        classes_file=args.classes,
        device=args.device,
        confidence_threshold=args.confidence,
        stability_threshold=args.stability,
        motion_threshold=args.motion
    )
    
    # Start inference thread
    interpreter.start_inference_thread()
    
    # Initialize video writer if recording
    video_writer = None
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        video_writer = cv2.VideoWriter(f'asl_recording_{timestamp}.avi', 
                                      fourcc, 20.0, (640, 480))
    
    # FPS calculation
    prev_time = time.time()
    fps = 0
    fps_buffer = deque(maxlen=30)  # Average over 30 frames
    
    print("ASL Interpreter Started - Press 'q' to quit, 's' to save transcript")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame (stream end?). Exiting ...")
                break
            
            # Calculate FPS
            current_time = time.time()
            elapsed = current_time - prev_time
            prev_time = current_time
            
            if elapsed > 0:
                current_fps = 1 / elapsed
                fps_buffer.append(current_fps)
                fps = sum(fps_buffer) / len(fps_buffer)
            
            # Process frame
            vis_frame, motion_score = interpreter.process_frame(frame)
            
            # Display FPS
            cv2.putText(vis_frame, f"FPS: {fps:.1f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display help instructions
            cv2.putText(vis_frame, "Press 'q' to quit, 's' to save transcript", 
                       (10, vis_frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow("ASL Interpreter", vis_frame)
            
            # Record frame if enabled
            if video_writer is not None:
                video_writer.write(vis_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                interpreter.save_transcript(args.output)
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        # Stop inference thread
        interpreter.stop_inference_thread()
        
        # Save final transcript
        interpreter.save_transcript(args.output)
        
        # Release resources
        if video_writer is not None:
            video_writer.release()
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final transcript
        print("\nFinal Transcript:")
        print(" ".join(interpreter.transcript))


if __name__ == "__main__":
    main()
