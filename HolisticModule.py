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

    def rescale_frame(self, frame, percent=75):
        width = int(frame.shape[1] * percent / 100)
        height = int(frame.shape[0] * percent / 100)
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    def midpoint(self, p1, p2):
        return (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))

    def levelIndicator(self, image, count):
        # Start coordinate, here (100, 50)
        # represents the top left corner of rectangle
        start_point = (600, 50)
        # Ending coordinate, here (125, 80)
        # represents the bottom right corner of rectangle
        end_point = (625, 350)
        # Black color in BGR
        color = (255, 255, 255)
        # Line thickness of -1 px
        # Thickness of -1 will fill the entire shape
        thickness = 4
        # Using cv2.rectangle() method
        # Draw a rectangle of black color of thickness -1 px
        cv2.rectangle(image, start_point, end_point, color, thickness)

        def level(spoint, epoint, color_, thickness_):
            return cv2.rectangle(image, spoint, epoint, color_, thickness_)

        if count == 0:
            level((600, 300), (625, 350), (0, 0, 255), -1)
        elif count == 1:
            level((600, 300), (625, 350), (255, 0, 0), -1)
        elif count == 2:
            level((600, 250), (625, 350), (255, 0, 0), -1)
        elif count == 3:
            level((600, 200), (625, 350), (255, 0, 0), -1)
        elif count == 4:
            level((600, 150), (625, 350), (255, 0, 0), -1)
        elif count == 5:
            level((600, 100), (625, 350), (255, 0, 0), -1)
        elif count == 6:
            level((600, 50), (625, 350), (255, 0, 0), -1)

        '''
                def level(spoint, epoint, color_, thickness_):
            return cv2.rectangle(image, spoint, epoint, color_, thickness_)

        default = level((600, 300), (625, 350), (0, 0, 255), -1)
        level1 = level((600, 300), (625, 350), (255, 0, 0), -1)
        level2 = level((600, 250), (625, 350), (255, 0, 0), -1)
        level3 = level((600, 200), (625, 350), (255, 0, 0), -1)
        level4 = level((600, 150), (625, 350), (255, 0, 0), -1)
        level5 = level((600, 100), (625, 350), (255, 0, 0), -1)
        level6 = level((600, 50), (625, 350), (255, 0, 0), -1)

        level_dict = {
            1: default,
            2: level1,
            4: level2,
            6: level3,
            8: level4,
            10: level5,
            12: level6,

        }
        level_dict.get(count)
        '''

        return

    # start and stop recording video on command
    # squat counter
    def squatCounter(self, results, image, count, draw):
        global a
        global b
        # distance between kepoints of elbows and knees should reach a threshold
        # keypoint of elbows and knees
        # right_elbow 14; left_elbow 13; right_knee 26; left_knee 25
        # draw line between 14 and 13, l1, draw line between 26 and 25, l2, find mid-points,m1, m2 of two line segments
        # if the distance between the two midpoints crosses the threshold then count a squat
        # get coordinates of these points.
        if results.pose_landmarks:
            kp_info = self.poseKeypoints_info(results, image)
            # print(np.shape(kp_info))
            # print(kp_info[14][2])

            up1 = (kp_info[14][2], kp_info[14][3])
            up2 = (kp_info[13][2], kp_info[13][3])
            lp1 = (kp_info[26][2], kp_info[26][3])
            lp2 = (kp_info[25][2], kp_info[25][3])

            m1 = self.midpoint(up1, up2)
            m2 = self.midpoint(lp1, lp2)
            # print(m1[0],m2[1])
            # print(np.shape(m1))
            dist = math.dist(m1, m2)
            print(dist)
            a = False

            if dist < 110:
                a = True
                print('atrue')

            if dist > 200:
                b = True
                print('btrue')

            if a and b:
                count = count + 1
                print('ab__true')
                a = False
                b = False

            # self.levelIndicator(image,count)
            if draw:
                cv2.circle(image, (up1[0], up1[1]), 5, (255, 0, 0), cv2.FILLED)
                cv2.circle(image, (up2[0], up2[1]), 5, (255, 0, 0), cv2.FILLED)
                cv2.circle(image, (lp1[0], lp1[1]), 5, (255, 0, 0), cv2.FILLED)
                cv2.circle(image, (lp2[0], lp2[1]), 5, (255, 0, 0), cv2.FILLED)
                cv2.circle(image, m1, 5, (255, 0, 0), cv2.FILLED)
                cv2.circle(image, m2, 5, (255, 0, 0), cv2.FILLED)
                cv2.line(image, m1, m2, (255, 0, 0), 5)
                cv2.putText(image, "Squat count : {}".format(str(count)), (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 2, cv2.LINE_AA)

        return count

    def poseKeypoints_info(self, results, image):
        list = []
        if results.pose_landmarks:

            for id, lm in enumerate(results.pose_landmarks.landmark):
                # print(lm.visibility)
                h, w, c = image.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                list.append([id, lm.visibility, cx, cy])

            # print(np.shape(list))
            # print(list)
        return list

    def fitPosekeypinScreenCheck2(self, size, mask1, logo1, mask2, logo2, results, cap, image):

        istrue = False
        if results.pose_landmarks:
            list = self.poseKeypoints_info(results, image)
            if all((0.5 <= j <= 1.0) for i, j, k, l in list):
                # Region of Interest (ROI), where we want
                # to insert logo
                ret1, frame = cap.read()
                roi = frame[-size - 10:-10, -size - 10:-10]

                # Set an index of where the mask is
                roi[np.where(mask1)] = 0
                roi += logo1
                istrue = True
        if not istrue:
            # print('hereee2')

            ret2, frame = cap.read()
            # Region of Interest (ROI), where we want
            # to insert logo
            roi = frame[-size - 10:-10, -size - 10:-10]

            # Set an index of where the mask is
            roi[np.where(mask2)] = 0
            roi += logo2

        return frame, istrue

    def fitPosekeypinScreenCheck(self, count):
        videodource = 0
        # videodource='demovideos/squat.mp4'
        cap = cv2.VideoCapture(videodource)
        # create object
        detector = holisticDetector()
        # constants for correct and wrong symbols :START
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
        # constants for correct and wrong symbols END
        # count=0
        # variables for video capturing
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter('frontvideo.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (width, height))

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

                frame, screenfit = self.fitPosekeypinScreenCheck2(size, mask1, logo1, mask2, logo2, results, cap, image)
                self.levelIndicator(frame, count)
                if screenfit:
                    count = self.squatCounter(results, frame, count, draw=True)
                    if count == 0:
                        # adding note on the live screen before starting squats: START
                        # cv2.putText(frame, "Hello World!!!", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))
                        cv2.putText(frame, "Start", (180, 150), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 16,
                                    cv2.LINE_AA)
                        cv2.putText(frame, "doing", (180, 230), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 16,
                                    cv2.LINE_AA)
                        cv2.putText(frame, "squats...", (180, 310), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 16,
                                    cv2.LINE_AA)

                        cv2.putText(frame, "Start", (180, 150), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 4,
                                    cv2.LINE_AA)
                        cv2.putText(frame, "doing", (180, 230), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 4,
                                    cv2.LINE_AA)
                        cv2.putText(frame, "squats...", (180, 310), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 4,
                                    cv2.LINE_AA)
                        # adding note on the live screen before starting squats: END

                if 1 <= count <= 6:
                    writer.write(frame)
                    frame = self.rescale_frame(frame, percent=150)

                cv2.imshow('YourPoseGuide', frame)
                # cv2.resize(frame, (1920, 1080))
                # cv2.resizeWindow('Resized Window', 1920, 1080)
                # cv2.imshow('Raw Webcam Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()


def main():
    videodource = 0
    # videodource='demovideos/squat.mp4'
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
            # print(len(results.left_hand_landmarks.landmark))
            # print(results.pose_landmarks.landmark)
            # numKeypoints = [len(results.pose_landmarks.landmark), len(results.face_landmarks.landmark), len(results.left_hand_landmarks), len(results.right_hand_landmarks) ]
            # print(numKeypoints)
            # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks

            # Recolor image back to BGR for rendering
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 1. Draw face landmarks
            detector.mp_drawing.draw_landmarks(image, results.face_landmarks, detector.mp_holistic.FACEMESH_CONTOURS,
                                               detector.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1,
                                                                               circle_radius=1),
                                               detector.mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1,
                                                                               circle_radius=1)
                                               )

            # 2. Right hand
            detector.mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                                               detector.mp_holistic.HAND_CONNECTIONS,
                                               detector.mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2,
                                                                               circle_radius=4),
                                               detector.mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2,
                                                                               circle_radius=2)
                                               )

            # 3. Left Hand
            detector.mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                                               detector.mp_holistic.HAND_CONNECTIONS,
                                               detector.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2,
                                                                               circle_radius=4),
                                               detector.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2,
                                                                               circle_radius=2)
                                               )

            # 4. Pose Detections
            detector.mp_drawing.draw_landmarks(image, results.pose_landmarks, detector.mp_holistic.POSE_CONNECTIONS,
                                               detector.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2,
                                                                               circle_radius=4),
                                               detector.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2,
                                                                               circle_radius=2)
                                               )
            frame = detector.rescale_frame(image, percent=150)
            cv2.imshow('Raw Webcam Feed', frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
