import cv2
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions import face_mesh
import os

class FaceExpressionSticker:
    def __init__(self, emoji_folder):
        """Initialize the face expression detector with emoji stickers."""
        self.emoji_folder = emoji_folder
        
        # Load all emoji images with transparency
        self.emojis = {}
        emoji_files = {
            'angry': 'angry.png',
            'happy': 'happy.png',
            'neutral': 'Neutral.png',
            'surprise': 'Surprise.png',
            'tease': 'tease.png',
            'thumbs': 'Thumbs.png',
            'wink': 'wink.png'
        }
        
        for name, filename in emoji_files.items():
            path = os.path.join(emoji_folder, filename)
            if os.path.exists(path):
                # Load with alpha channel
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    self.emojis[name] = img
                    print(f"✓ Loaded: {name}")
                else:
                    print(f"✗ Failed to load: {path}")
            else:
                print(f"✗ File not found: {path}")
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize MediaPipe Hands for thumbs detection
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Landmark indices for expression detection
        # Mouth landmarks
        self.UPPER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
        self.LOWER_LIP = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
        self.MOUTH_TOP = 13
        self.MOUTH_BOTTOM = 14
        self.MOUTH_LEFT = 61
        self.MOUTH_RIGHT = 291
        
        # Eye landmarks
        self.LEFT_EYE_TOP = 159
        self.LEFT_EYE_BOTTOM = 145
        self.RIGHT_EYE_TOP = 386
        self.RIGHT_EYE_BOTTOM = 374
        
        # Eyebrow landmarks
        self.LEFT_EYEBROW = [70, 63, 105, 66, 107]
        self.RIGHT_EYEBROW = [336, 296, 334, 293, 300]
        self.LEFT_EYE_CENTER = 159
        self.RIGHT_EYE_CENTER = 386
        
        # Tongue detection - mouth inner landmarks
        self.INNER_MOUTH_TOP = 13
        self.INNER_MOUTH_BOTTOM = 14
        
        self.current_expression = 'neutral'
        self.expression_history = []
        self.history_size = 5  # Smooth expression changes
        
    def calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two points."""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def get_landmark_point(self, landmarks, idx, w, h):
        """Get x, y coordinates of a landmark."""
        return (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
    
    def detect_expression(self, landmarks, w, h, hand_results):
        """Detect facial expression based on landmarks."""
        
        # Check for thumbs up first (hand gesture)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                if self.is_thumbs_up(hand_landmarks):
                    return 'thumbs'
        
        # Get key points
        mouth_top = self.get_landmark_point(landmarks, self.MOUTH_TOP, w, h)
        mouth_bottom = self.get_landmark_point(landmarks, self.MOUTH_BOTTOM, w, h)
        mouth_left = self.get_landmark_point(landmarks, self.MOUTH_LEFT, w, h)
        mouth_right = self.get_landmark_point(landmarks, self.MOUTH_RIGHT, w, h)
        
        left_eye_top = self.get_landmark_point(landmarks, self.LEFT_EYE_TOP, w, h)
        left_eye_bottom = self.get_landmark_point(landmarks, self.LEFT_EYE_BOTTOM, w, h)
        right_eye_top = self.get_landmark_point(landmarks, self.RIGHT_EYE_TOP, w, h)
        right_eye_bottom = self.get_landmark_point(landmarks, self.RIGHT_EYE_BOTTOM, w, h)
        
        # Calculate ratios
        mouth_height = self.calculate_distance(mouth_top, mouth_bottom)
        mouth_width = self.calculate_distance(mouth_left, mouth_right)
        mouth_ratio = mouth_height / (mouth_width + 0.001)  # Avoid division by zero
        
        left_eye_height = self.calculate_distance(left_eye_top, left_eye_bottom)
        right_eye_height = self.calculate_distance(right_eye_top, right_eye_bottom)
        
        # Calculate face height for normalization
        face_top = self.get_landmark_point(landmarks, 10, w, h)  # Forehead
        face_bottom = self.get_landmark_point(landmarks, 152, w, h)  # Chin
        face_height = self.calculate_distance(face_top, face_bottom)
        
        # Normalize eye heights
        left_eye_ratio = left_eye_height / (face_height + 0.001)
        right_eye_ratio = right_eye_height / (face_height + 0.001)
        
        # Calculate eyebrow position (for angry detection)
        left_eyebrow_y = np.mean([landmarks[i].y for i in self.LEFT_EYEBROW])
        right_eyebrow_y = np.mean([landmarks[i].y for i in self.RIGHT_EYEBROW])
        left_eye_y = landmarks[self.LEFT_EYE_CENTER].y
        right_eye_y = landmarks[self.RIGHT_EYE_CENTER].y
        
        eyebrow_eye_dist_left = left_eye_y - left_eyebrow_y
        eyebrow_eye_dist_right = right_eye_y - right_eyebrow_y
        avg_eyebrow_dist = (eyebrow_eye_dist_left + eyebrow_eye_dist_right) / 2
        
        # Mouth corner positions for smile detection
        mouth_corner_left = self.get_landmark_point(landmarks, 61, w, h)
        mouth_corner_right = self.get_landmark_point(landmarks, 291, w, h)
        mouth_center = self.get_landmark_point(landmarks, 13, w, h)
        
        # Check if mouth corners are higher than center (smile)
        smile_factor = (mouth_center[1] - (mouth_corner_left[1] + mouth_corner_right[1]) / 2) / (face_height + 0.001)
        
        # Expression detection logic
        
        # 1. Wink detection - one eye significantly more closed than the other
        eye_diff = abs(left_eye_ratio - right_eye_ratio)
        min_eye = min(left_eye_ratio, right_eye_ratio)
        if eye_diff > 0.008 and min_eye < 0.02:
            return 'wink'
        
        # 2. Surprise - mouth wide open, eyes wide
        if mouth_ratio > 0.4 and left_eye_ratio > 0.025 and right_eye_ratio > 0.025:
            return 'surprise'
        
        # 3. Tease - tongue out (mouth open with specific shape)
        # Detect tongue by checking if mouth is open in a specific way
        if mouth_ratio > 0.25 and mouth_ratio < 0.5:
            # Check mouth shape - tongue out usually has specific proportions
            inner_mouth_ratio = mouth_height / (mouth_width + 0.001)
            if inner_mouth_ratio > 0.2 and smile_factor < 0:
                return 'tease'
        
        # 4. Angry - furrowed brows, tight mouth
        if avg_eyebrow_dist < 0.04 and mouth_ratio < 0.15 and smile_factor < -0.005:
            return 'angry'
        
        # 5. Happy - smile detected
        if smile_factor > 0.01 or (mouth_ratio > 0.15 and smile_factor > 0):
            return 'happy'
        
        # 6. Default to neutral
        return 'neutral'
    
    def is_thumbs_up(self, hand_landmarks):
        """Detect thumbs up gesture."""
        # Get landmark positions
        thumb_tip = hand_landmarks.landmark[4]
        thumb_ip = hand_landmarks.landmark[3]
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        ring_tip = hand_landmarks.landmark[16]
        pinky_tip = hand_landmarks.landmark[20]
        
        index_mcp = hand_landmarks.landmark[5]
        middle_mcp = hand_landmarks.landmark[9]
        ring_mcp = hand_landmarks.landmark[13]
        pinky_mcp = hand_landmarks.landmark[17]
        
        # Check if thumb is extended upward
        thumb_up = thumb_tip.y < thumb_ip.y
        
        # Check if other fingers are curled
        fingers_curled = (
            index_tip.y > index_mcp.y and
            middle_tip.y > middle_mcp.y and
            ring_tip.y > ring_mcp.y and
            pinky_tip.y > pinky_mcp.y
        )
        
        return thumb_up and fingers_curled
    
    def smooth_expression(self, expression):
        """Smooth expression changes to avoid flickering."""
        self.expression_history.append(expression)
        if len(self.expression_history) > self.history_size:
            self.expression_history.pop(0)
        
        # Return most common expression in history
        from collections import Counter
        counts = Counter(self.expression_history)
        return counts.most_common(1)[0][0]
    
    def overlay_emoji(self, frame, emoji_name, x, y, size):
        """Overlay emoji on frame with transparency."""
        if emoji_name not in self.emojis:
            return frame
        
        emoji = self.emojis[emoji_name]
        emoji_resized = cv2.resize(emoji, (size, size))
        
        # Handle positioning
        y1, y2 = y, y + size
        x1, x2 = x, x + size
        
        # Boundary checks
        if y1 < 0:
            emoji_resized = emoji_resized[-y1:, :]
            y1 = 0
        if x1 < 0:
            emoji_resized = emoji_resized[:, -x1:]
            x1 = 0
        if y2 > frame.shape[0]:
            emoji_resized = emoji_resized[:frame.shape[0]-y2, :]
            y2 = frame.shape[0]
        if x2 > frame.shape[1]:
            emoji_resized = emoji_resized[:, :frame.shape[1]-x2]
            x2 = frame.shape[1]
        
        if emoji_resized.shape[0] == 0 or emoji_resized.shape[1] == 0:
            return frame
        
        # Handle alpha channel
        if emoji_resized.shape[2] == 4:
            alpha = emoji_resized[:, :, 3] / 255.0
            alpha = np.stack([alpha] * 3, axis=-1)
            
            roi = frame[y1:y2, x1:x2]
            if roi.shape[:2] == emoji_resized.shape[:2]:
                blended = (1 - alpha) * roi + alpha * emoji_resized[:, :, :3]
                frame[y1:y2, x1:x2] = blended.astype(np.uint8)
        else:
            frame[y1:y2, x1:x2] = emoji_resized
        
        return frame
    
    def run(self):
        """Main loop to run the face expression detector."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera!")
            return
        
        print("\n🎭 Face Expression Sticker Detector Started!")
        print("━" * 40)
        print("Expressions detected:")
        print("  😊 Happy - Smile")
        print("  😐 Neutral - Normal face")
        print("  😮 Surprise - Open mouth wide")
        print("  😠 Angry - Furrowed brows")
        print("  😜 Tease - Stick tongue out")
        print("  😉 Wink - Close one eye")
        print("  👍 Thumbs - Show thumbs up")
        print("━" * 40)
        print("Press 'Q' to quit\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame!")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process face mesh
            face_results = self.face_mesh.process(rgb_frame)
            hand_results = self.hands.process(rgb_frame)
            
            expression = 'neutral'
            face_x, face_y, face_w = w // 2, h // 2, 100
            
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    landmarks = face_landmarks.landmark
                    
                    # Get face bounding box
                    x_coords = [lm.x * w for lm in landmarks]
                    y_coords = [lm.y * h for lm in landmarks]
                    
                    face_x = int(min(x_coords))
                    face_y = int(min(y_coords))
                    face_w = int(max(x_coords) - min(x_coords))
                    face_h = int(max(y_coords) - min(y_coords))
                    
                    # Detect expression
                    raw_expression = self.detect_expression(landmarks, w, h, hand_results)
                    expression = self.smooth_expression(raw_expression)
                    
                    # Draw face rectangle
                    cv2.rectangle(frame, (face_x, face_y), 
                                (face_x + face_w, face_y + face_h), 
                                (0, 255, 0), 2)
            
            # Calculate emoji position and size
            emoji_size = max(80, face_w // 2)
            emoji_x = face_x + face_w + 20
            emoji_y = face_y
            
            # If emoji would go off screen, place it on the left
            if emoji_x + emoji_size > w:
                emoji_x = face_x - emoji_size - 20
            
            # Overlay the emoji
            frame = self.overlay_emoji(frame, expression, emoji_x, emoji_y, emoji_size)
            
            # Display current expression text
            cv2.putText(frame, f"Expression: {expression.upper()}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 255), 2)
            
            # Add instructions
            cv2.putText(frame, "Press 'Q' to quit", 
                       (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (200, 200, 200), 1)
            
            # Show frame
            cv2.imshow('Face Expression Sticker Detector', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    # Your emoji folder path
    EMOJI_FOLDER = r"C:\Users\arman\OneDrive\Desktop\Project Amrin\FaceExpressionDetector\emojis"
    
    # Create and run the detector
    detector = FaceExpressionSticker(EMOJI_FOLDER)
    detector.run()
