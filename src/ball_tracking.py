from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque

class BallTracker:
    def __init__(self, ball_model_path="models/yolo11n-volleyball.pt"):
        # Ball detection model
        self.ball_model = YOLO(ball_model_path)
        
        # Ball tracking
        self.ball_history = {
            'positions': deque(maxlen=30),  # last 30 positions
            'velocities': deque(maxlen=10),  # last 10 velocities
            'trajectory': deque(maxlen=50),  # full trajectory
            'smoothed_positions': deque(maxlen=30),  # smoothed positions
            'detection_confidence': deque(maxlen=10),  # confidence history
            'color': (0, 255, 0),  # green for ball
            'is_tracking': False,
            'missed_frames': 0,
            'predicted_position': None
        }
        
        # Kalman filter parameters (simplified)
        self.kalman_state = np.array([0, 0, 0, 0], dtype=np.float32)  # x, y, vx, vy
        self.kalman_covariance = np.eye(4) * 10
        
        # Statistics
        self.frame_count = 0
        self.total_detections = 0
        self.total_missed = 0
        
        # Thresholds
        self.MIN_CONFIDENCE = 0.3
        self.MAX_MISSED_FRAMES = 10  # how many frames without detection before we stop tracking
        self.SMOOTHING_FACTOR = 0.3  # for smoothing positions
        
    def predict_next_position(self):
        if len(self.ball_history['positions']) < 3:
            return None
            
        positions = list(self.ball_history['positions'])
        
        # Linear extrapolation
        dx = positions[-1][0] - positions[-2][0]
        dy = positions[-1][1] - positions[-2][1]
        
        predicted_x = positions[-1][0] + dx
        predicted_y = positions[-1][1] + dy
        
        return (int(predicted_x), int(predicted_y))
    
    def smooth_position(self, current_pos, alpha=0.3):
        if not self.ball_history['smoothed_positions']:
            return current_pos
        
        last_smoothed = self.ball_history['smoothed_positions'][-1]
        smoothed_x = alpha * current_pos[0] + (1 - alpha) * last_smoothed[0]
        smoothed_y = alpha * current_pos[1] + (1 - alpha) * last_smoothed[1]
        
        return (int(smoothed_x), int(smoothed_y))
    
    def calculate_ball_statistics(self):
        stats = {
            'speed': 0,
            'acceleration': 0,
            'direction': 0,
            'is_moving': False
        }
        
        if len(self.ball_history['velocities']) > 1:
            velocities = list(self.ball_history['velocities'])
            
            # Instantaneous speed (last)
            stats['speed'] = velocities[-1] if velocities else 0
            
            # Average speed
            avg_speed = np.mean(velocities) if velocities else 0
            
            # Direction (if we have enough positions)
            if len(self.ball_history['positions']) > 2:
                positions = list(self.ball_history['positions'])
                dx = positions[-1][0] - positions[-2][0]
                dy = positions[-1][1] - positions[-2][1]
                
                if dx != 0 or dy != 0:
                    stats['direction'] = np.arctan2(dy, dx) * 180 / np.pi
                    stats['is_moving'] = stats['speed'] > 2.0  # threshold in pixels/frame
            
            if len(velocities) > 2:
                stats['acceleration'] = velocities[-1] - velocities[-2]
        
        return stats
    
    def detect_ball_in_frame(self, frame):
        """Wykryj piłkę w klatce"""
        results = self.ball_model(frame, verbose=False, conf=self.MIN_CONFIDENCE)
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            # Take detection with highest confidence
            best_idx = np.argmax([box.conf[0].cpu().item() for box in results[0].boxes])
            best_box = results[0].boxes[best_idx]
            
            bbox = list(map(int, best_box.xyxy[0].cpu().numpy()))
            confidence = best_box.conf[0].cpu().item()
            
            # Ball center
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2
            
            # Size of bbox (for later analysis)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            
            self.total_detections += 1
            self.ball_history['detection_confidence'].append(confidence)
            
            return {
                'bbox': bbox,
                'center': (center_x, center_y),
                'confidence': confidence,
                'size': (width, height),
                'is_detected': True
            }
        
        return {'is_detected': False}
    
    def update_tracking(self, detection_result):
        if detection_result['is_detected']:
            # Reset missed frames counter
            self.ball_history['missed_frames'] = 0
            self.ball_history['is_tracking'] = True
            
            # Add position to history
            current_pos = detection_result['center']
            smoothed_pos = self.smooth_position(current_pos, self.SMOOTHING_FACTOR)
            
            self.ball_history['positions'].append(current_pos)
            self.ball_history['smoothed_positions'].append(smoothed_pos)
            self.ball_history['predicted_position'] = None
            
            # Calculate speed if we have enough positions
            if len(self.ball_history['positions']) > 1:
                last_pos = self.ball_history['positions'][-2]
                current_pos = self.ball_history['positions'][-1]
                
                distance = np.sqrt((current_pos[0] - last_pos[0])**2 + 
                                  (current_pos[1] - last_pos[1])**2)
                self.ball_history['velocities'].append(distance)
            
            # Update trajectory
            self.ball_history['trajectory'].append(smoothed_pos)
            
        else:
            # No detection in this frame
            self.ball_history['missed_frames'] += 1
            self.total_missed += 1
            
            # If predicting and no detection for X frames
            if self.ball_history['is_tracking']:
                if self.ball_history['missed_frames'] <= self.MAX_MISSED_FRAMES:
                    predicted = self.predict_next_position()
                    if predicted:
                        self.ball_history['predicted_position'] = predicted
                        self.ball_history['positions'].append(predicted)
                        
                        # Add to trajectory as prediction
                        self.ball_history['trajectory'].append(predicted)
                else:
                    # Too long without detection - reset tracking
                    self.reset_tracking()
    
    def reset_tracking(self):
        self.ball_history['positions'].clear()
        self.ball_history['velocities'].clear()
        self.ball_history['trajectory'].clear()
        self.ball_history['smoothed_positions'].clear()
        self.ball_history['detection_confidence'].clear()
        self.ball_history['is_tracking'] = False
        self.ball_history['missed_frames'] = 0
        self.ball_history['predicted_position'] = None
    
    def draw_tracking_info(self, frame, detection_result):
        height, width = frame.shape[:2]

        stats_bg = np.zeros((150, 300, 3), dtype=np.uint8)
        stats_bg[:] = (40, 40, 40)

        stats_x = 10
        stats_y = 10

        frame[stats_y:stats_y+150, stats_x:stats_x+300] = cv2.addWeighted(
            frame[stats_y:stats_y+150, stats_x:stats_x+300], 0.7,
            stats_bg, 0.3, 0
        )
        
        # Retrieve statistics
        stats = self.calculate_ball_statistics()
        
        # Display statistics
        y_offset = 30
        line_height = 25
        
        # Tracking status
        status_color = (0, 255, 0) if self.ball_history['is_tracking'] else (0, 0, 255)
        status_text = "TRACKING" if self.ball_history['is_tracking'] else "NOT TRACKING"
        cv2.putText(frame, f"Status: {status_text}", 
                   (stats_x + 10, stats_y + y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        y_offset += line_height
        
        # Statistics only if tracking
        if self.ball_history['is_tracking']:
            # Speed
            cv2.putText(frame, f"Speed: {stats['speed']:.1f} px/frame", 
                       (stats_x + 10, stats_y + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height
            
            # Acceleration
            cv2.putText(frame, f"Acceleration: {stats['acceleration']:.1f}", 
                       (stats_x + 10, stats_y + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height
            
            # Direction
            cv2.putText(frame, f"Direction: {stats['direction']:.0f}°", 
                       (stats_x + 10, stats_y + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height
            
            # Confidence
            if self.ball_history['detection_confidence']:
                avg_conf = np.mean(list(self.ball_history['detection_confidence'])[-5:])
                conf_color = (0, 255, 0) if avg_conf > 0.5 else (0, 165, 255) if avg_conf > 0.3 else (0, 0, 255)
                cv2.putText(frame, f"Confidence: {avg_conf:.2f}", 
                           (stats_x + 10, stats_y + y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, conf_color, 1)
                y_offset += line_height
            
            # Missed frames
            missed_color = (0, 165, 255) if self.ball_history['missed_frames'] > 0 else (0, 255, 0)
            cv2.putText(frame, f"Missed frames: {self.ball_history['missed_frames']}/{self.MAX_MISSED_FRAMES}", 
                       (stats_x + 10, stats_y + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, missed_color, 1)
            y_offset += line_height
        
        # Draw detection/bbox if it exists
        if detection_result['is_detected']:
            bbox = detection_result['bbox']
            center = detection_result['center']
            confidence = detection_result['confidence']
            
            # Draw bbox
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                         self.ball_history['color'], 2)
            
            # Draw center
            cv2.circle(frame, center, 5, self.ball_history['color'], -1)
            cv2.circle(frame, center, 8, (255, 255, 255), 2)
            
            # Label with confidence
            cv2.putText(frame, f"Ball: {confidence:.2f}", 
                       (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.ball_history['color'], 2)
        
        # Draw predicted position if it exists
        if self.ball_history['predicted_position']:
            pred_pos = self.ball_history['predicted_position']
            cv2.circle(frame, pred_pos, 10, (0, 165, 255), 2)
            cv2.putText(frame, "PREDICTED", 
                       (pred_pos[0] - 40, pred_pos[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        
        # Draw trajectory
        if len(self.ball_history['trajectory']) > 1:
            trajectory = list(self.ball_history['trajectory'])
            
            # Draw trajectory line
            for i in range(1, len(trajectory)):
                # Transparency based on the "age" of the point
                alpha = i / len(trajectory)
                color = (
                    int(self.ball_history['color'][0] * alpha),
                    int(self.ball_history['color'][1] * alpha),
                    int(self.ball_history['color'][2] * alpha)
                )
                
                thickness = max(1, int(3 * alpha))
                cv2.line(frame, trajectory[i-1], trajectory[i], 
                        color, thickness, cv2.LINE_AA)
            
            # Draw points on the trajectory every few steps
            for i in range(0, len(trajectory), 5):
                if i < len(trajectory):
                    point = trajectory[i]
                    cv2.circle(frame, point, 2, (255, 255, 255), -1)
        
        global_stats = [
            f"Frame: {self.frame_count}",
            f"Detections: {self.total_detections}",
            f"Missed: {self.total_missed}",
            f"Detection rate: {self.total_detections/max(1, self.frame_count)*100:.1f}%"
        ]
        
        y_offset_bottom = height - 120
        for stat in global_stats:
            cv2.putText(frame, stat, (width - 250, y_offset_bottom),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset_bottom += 20
        
        legend = [
            ("Ball detection", self.ball_history['color']),
            ("Predicted position", (0, 165, 255)),
            ("Trajectory", (0, 200, 0))
        ]
        
        y_offset_legend = height - 180
        for text, color in legend:
            cv2.putText(frame, text, (width - 250, y_offset_legend),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset_legend += 20
        
        return frame
    
    def process_frame(self, frame):
        self.frame_count += 1
        
        detection_result = self.detect_ball_in_frame(frame)
        self.update_tracking(detection_result)
        processed_frame = self.draw_tracking_info(frame.copy(), detection_result)
        
        return processed_frame
    
    def run_tracking(self, video_path, output_path=None):
        """Uruchom śledzenie piłki na wideo"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Nie można otworzyć wideo: {video_path}")
            return
        
        # Get video parameters
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"=== BALL TRACKING ===")
        print(f"Video: {video_path}")
        print(f"Size: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        print(f"Ball model: {self.ball_model.__class__.__name__}")
        print("Keys:")
        print("  'q' - quit")
        print("  'p' - pause")
        print("  'r' - reset tracking")
        print("  's' - save frame")
        print("  '+'/- - increase/decrease confidence threshold")
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        paused = False
        
        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame = self.process_frame(frame)

                if output_path:
                    out.write(processed_frame)
                
                cv2.imshow("Ball Tracking - Advanced", processed_frame)
            
            # Key handling
            key = cv2.waitKey(1 if not paused else 0) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print(f"{'Paused' if paused else 'Resumed'}")
            elif key == ord('r'):
                self.reset_tracking()
                print("Tracking reset")
            elif key == ord('s'):
                # Save current frame
                filename = f"ball_tracking_frame_{self.frame_count}.jpg"
                cv2.imwrite(filename, processed_frame)
                print(f"Saved: {filename}")
            elif key == ord('+'):
                self.MIN_CONFIDENCE = min(0.9, self.MIN_CONFIDENCE + 0.05)
                print(f"Confidence threshold: {self.MIN_CONFIDENCE:.2f}")
            elif key == ord('-'):
                self.MIN_CONFIDENCE = max(0.1, self.MIN_CONFIDENCE - 0.05)
                print(f"Confidence threshold: {self.MIN_CONFIDENCE:.2f}")
        
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        # Summary
        print("\n=== TRACKING SUMMARY ===")
        print(f"Processed frames: {self.frame_count}")
        print(f"Ball detections: {self.total_detections}")
        print(f"Missed detections: {self.total_missed}")
        print(f"Detection rate: {self.total_detections/max(1, self.frame_count)*100:.1f}%")
        
        if self.ball_history['velocities']:
            velocities = list(self.ball_history['velocities'])
            print(f"Average speed: {np.mean(velocities):.2f} px/frame")
            print(f"Maximum speed: {np.max(velocities):.2f} px/frame")
        
        if output_path:
            print(f"\nOutput saved to: {output_path}")

# RUN
if __name__ == "__main__":
    tracker = BallTracker(
        ball_model_path="models/yolo11n-volleyball.pt"
    )
    
    tracker.run_tracking(
        video_path="tests/ex1.mp4",
        output_path="examples/ball_tracking_output.mp4"  # optional
    )