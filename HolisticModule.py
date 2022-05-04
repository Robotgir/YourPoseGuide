import mediapipe as mp
import cv2
import time
import math
import numpy as np
import screeninfo
from mediapipe.framework.formats import landmark_pb2

class holisticDetector():
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils

    def fitPosekeypinScreenCheck(self):
        videodource = 0
        # videodource='demovideos/squat.mp4'
        cap = cv2.VideoCapture(videodource)
        #create object
        detector = holisticDetector()
        size = 100
        # Read logo and resize
        logo1 = cv2.imread('images/correct.png')
        logo2 = cv2.imread('images/wrong.png')
        logo1 = cv2.resize(logo1, (size, size))
        logo2 = cv2.resize(logo2, (size, size))
        # Create a mask of logo
        img2gray1 = cv2.cvtColor(logo1, cv2.COLOR_BGR2GRAY)
        img2gray2 = cv2.cvtColor(logo2, cv2.COLOR_BGR2GRAY)
        ret1, mask1 = cv2.threshold(img2gray1, 1, 255, cv2.THRESH_BINARY)
        ret2, mask2 = cv2.threshold(img2gray2, 1, 255, cv2.THRESH_BINARY)
        # Initiate holistic model
        with detector.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

            while cap.isOpened():
                ret, frame = cap.read()

                # Recolor Feed
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Make Detections
                results = holistic.process(image)
                # print(results.face_landmarks)

                # Recolor image back to BGR for rendering
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                #print('hereee')
                #print(len( results.pose_landmarks.landmark))
                #print(results.pose_landmarks.landmark)
                list = []
                istrue = False
                if results.pose_landmarks:

                    for id, lm in enumerate(results.pose_landmarks.landmark):
                        #print(lm.visibility)
                        list.append([id,lm.visibility])

                    #print(np.shape(list))
                    #print(list)

                    if all((0.5 <= j <= 1.0) for i, j in list):
                        # Region of Interest (ROI), where we want
                        # to insert logo
                        ret1, frame = cap.read()
                        roi = frame[-size - 10:-10, -size - 10:-10]

                        # Set an index of where the mask is
                        roi[np.where(mask1)] = 0
                        roi += logo1
                        istrue=True
                if not istrue:
                    #print('hereee2')

                    ret2, frame = cap.read()
                    # Region of Interest (ROI), where we want
                    # to insert logo
                    roi = frame[-size - 10:-10, -size - 10:-10]

                    # Set an index of where the mask is
                    roi[np.where(mask2)] = 0
                    roi += logo2

                cv2.imshow('YourPoseGuide', frame)
                #cv2.resizeWindow('Resized Window', 1920, 1080)
                #cv2.imshow('Raw Webcam Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()


def main():

    videodource=0
    #videodource='demovideos/squat.mp4'
    cap = cv2.VideoCapture(videodource)
    detector = holisticDetector()
    # Initiate holistic model
    with detector.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Make Detections
            results = holistic.process(image)
            # print(results.face_landmarks)
            #print(len(results.left_hand_landmarks.landmark))
            #print(results.pose_landmarks.landmark)
            #numKeypoints = [len(results.pose_landmarks.landmark), len(results.face_landmarks.landmark), len(results.left_hand_landmarks), len(results.right_hand_landmarks) ]
            #print(numKeypoints)
            # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks

            # Recolor image back to BGR for rendering
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 1. Draw face landmarks
            detector.mp_drawing.draw_landmarks(image, results.face_landmarks, detector.mp_holistic.FACEMESH_CONTOURS,
                                      detector.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                      detector.mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                      )

            # 2. Right hand
            detector.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, detector.mp_holistic.HAND_CONNECTIONS,
                                      detector.mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                      detector.mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                      )

            # 3. Left Hand
            detector.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, detector.mp_holistic.HAND_CONNECTIONS,
                                      detector.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                      detector.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                      )

            # 4. Pose Detections
            detector.mp_drawing.draw_landmarks(image, results.pose_landmarks, detector.mp_holistic.POSE_CONNECTIONS,
                                      detector.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                      detector.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )

            cv2.imshow('Raw Webcam Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

