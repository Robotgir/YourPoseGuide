import mediapipe as mp
import cv2

class holistic():
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils

    def findHolistic(self, video_source=0):
        cap = cv2.VideoCapture(video_source)
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()

                # Recolor Feed
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Make Detections
                results = holistic.process(image)
                # print(results.face_landmarks)

                # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks

                # Recolor image back to BGR for rendering
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # Draw face landmarks
                self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS)

                # Right hand
                self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)

                # Left Hand
                self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)

                # Pose Detections
                self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)

                # cv2.imshow('Raw Webcam Feed', image)
                # Flip the image horizontally for a selfie-view display.
                cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()
        return results

    def poseLandmarks(self):
        self.results = self.findHolistic(video_source=0)
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
        return self.lmList

    def faceLandmarks(self):
        self.results = self.findHolistic(video_source=0)
        self.lmList = []
        if self.results.face_landmarks:
            for id, lm in enumerate(self.results.face_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
        return self.lmList

    def leftHandLandmarks(self):
        self.results = self.findHolistic(video_source=0)
        self.lmList = []
        if self.results.left_hand_landmarks:
            for id, lm in enumerate(self.results.left_hand_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
        return self.lmList

    def rightHandLandmarks(self):
        self.results = self.findHolistic(video_source=0)
        self.lmList = []
        if self.results.right_hand_landmarks:
            for id, lm in enumerate(self.results.right_hand_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
        return self.lmList