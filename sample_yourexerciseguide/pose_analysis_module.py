import mediapipe as mp
import cv2
import math
import numpy as np

# initializing arguments globally for front and side squat counter functions.
a = False
b = False


# google mediapipe pose detection holistic solutions
class HolisticDetector:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils


class PoseAnalysisOperations(HolisticDetector):
    # scaling output function

    def mid_point(self, p1, p2):
        return int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)

    def rescale_frame(self, frame, percent=75):
        width = int(frame.shape[1] * percent / 100)
        height = int(frame.shape[0] * percent / 100)
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    def level_indicator(self, image, count):
        is_full = False
        final_count = 0

        def level(start_point, end_point, color_, thickness_):
            return cv2.rectangle(image, start_point, end_point, color_, thickness_)

        # background for level indicator
        level((1150, 50), (1230, 500), (0, 0, 0), -1)
        if count == 0:
            level((1150, 450), (1230, 500), (0, 0, 255), -1)
        elif count == 1:
            level((1150, 350), (1230, 500), (255, 0, 0), -1)
        elif count == 2:
            level((1150, 250), (1230, 500), (255, 0, 0), -1)
        elif count == 3:
            level((1150, 100), (1230, 500), (255, 0, 0), -1)
        elif count == 4:
            level((1150, 50), (1230, 500), (255, 0, 0), -1)
            is_full = True
            final_count = count

        ''' # to increase number of squats for analysis
        elif count == 5:
            level((600, 100), (625, 350), (255, 0, 0), -1)
        elif count == 6:
            level((600, 50), (625, 350), (255, 0, 0), -1)
        '''
        # border for level indicator
        level((1150, 50), (1230, 500), (255, 255, 255), 4)
        return is_full, final_count

    # front squat counter Start
    def front_squat_counter(self, results, image, count, draw):
        global a
        global b

        # distance between key-points of elbows and knees should reach a threshold
        # keypoint of elbows and knees
        # right_elbow 14; left_elbow 13; right_knee 26; left_knee 25; right_ear 8; left_ear 7
        # draw line between 14 and 13, l1, draw line between 26 and 25, l2, find mid-points,m1, m2 of two line segments
        # if the distance between the two midpoints crosses the threshold then count a squat
        # get coordinates of these points.
        if results.pose_landmarks:
            kp_info = self.pose_keypoints_info(results, image)
            # print(np.shape(kp_info))
            # print(kp_info[14][2])
            '''
            # if considering midpoint of elbows
            up1 = (kp_info[14][2], kp_info[14][3])
            up2 = (kp_info[13][2], kp_info[13][3])
            '''
            # if considering midpoint of eyes
            up1 = (kp_info[8][2], kp_info[8][3])
            up2 = (kp_info[7][2], kp_info[7][3])
            lp1 = (kp_info[26][2], kp_info[26][3])
            lp2 = (kp_info[25][2], kp_info[25][3])

            m1 = self.mid_point(up1, up2)
            m2 = self.mid_point(lp1, lp2)
            # print(m1[0],m2[1])
            # print(np.shape(m1))
            dist = math.dist(m1, m2)
            print(dist)
            a = False

            if dist < 250:
                a = True
                print('atrue')

            if dist > 250:
                b = True
                print('btrue')

            if a and b:
                count = count + 1
                print('ab__true')
                a = False
                b = False

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

    # front squat counter end
    def side_squat_counter(self, results, image, side_count, draw=True):
        global a
        global b

        # angle between connection = line segment with keypoints at ankle and knee, knee and hip should reach a
        # threshold
        # key-points of ankles, knees, hips
        # right_ankle 28; right_knee 26; right_hip 24; |  left_ankle 27; left_knee 25; left_hip 23
        # draw line between 28 and 26, lr1, draw line between 26 and 24, lr2
        # find angle ar between lr1 and lr2
        # draw line between 27 and 25, ll1, draw line between 25 and 23, ll2
        # find angle al between ll1 and ll2
        # if the angle between the two lines reaches the threshold then count a squat
        # get coordinates of these points. opencv has coord sys of format x positive right and y positive downward
        # regular geometry has x positive right and y positive upward, to use standard geometric operations we consider
        # opencv coord sys as forth quad of traditional coord sys.

        if results.pose_landmarks:
            kp_info = self.pose_keypoints_info(results, image)
            # print(np.shape(kp_info))
            # print(kp_info[14][2])
            # coordinates of key-points
            right_ankle_cd = (kp_info[28][2], kp_info[28][3])
            right_knee_cd = (kp_info[26][2], kp_info[26][3])
            right_hip_cd = (kp_info[24][2], kp_info[24][3])
            # defining vectors
            right_ankle = (kp_info[28][2], -kp_info[28][3])
            right_knee = (kp_info[26][2], -kp_info[26][3])
            right_hip = (kp_info[24][2], -kp_info[24][3])
            lr1_vector = [right_knee[0] - right_ankle[0], right_knee[1] - right_ankle[1]]
            lr2_vector = [right_knee[0] - right_hip[0], right_knee[1] - right_hip[1]]
            unit_vector_lr1 = lr1_vector / np.linalg.norm(lr1_vector)
            unit_vector_lr2 = lr2_vector / np.linalg.norm(lr2_vector)
            dot_product = np.dot(unit_vector_lr1, unit_vector_lr2)
            ar = math.degrees(np.arccos(dot_product))

            print(ar)
            a = False

            if ar < 140:
                a = True
                print('atrue')

            if ar > 170:
                b = True
                print('btrue')

            if a and b:
                side_count = side_count + 1
                print('ab__true')
                a = False
                b = False

            if draw:
                cv2.circle(image, (right_ankle_cd[0], right_ankle_cd[1]), 5, (255, 0, 0), cv2.FILLED)
                cv2.circle(image, (right_knee_cd[0], right_knee_cd[1]), 5, (255, 0, 0), cv2.FILLED)
                cv2.circle(image, (right_hip_cd[0], right_hip_cd[1]), 5, (255, 0, 0), cv2.FILLED)
                cv2.arrowedLine(image, right_knee_cd, right_ankle_cd,
                                (255, 0, 0), 5)
                cv2.arrowedLine(image, right_knee_cd, right_hip_cd,
                                (255, 0, 0), 5)
                cv2.putText(image, "Squat count : {}".format(str(side_count)), (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 2, cv2.LINE_AA)

        return side_count

    # side squat counter start

    # side squat counter end
    def pose_keypoints_info(self, results, image):
        list_kp = []
        if results.pose_landmarks:

            for id, lm in enumerate(results.pose_landmarks.landmark):
                # print(lm.visibility)
                h, w, c = image.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                list_kp.append([id, lm.visibility, cx, cy])

            # print(np.shape(list_kp))
            # print(list_kp)
        return list_kp

    def fit_posekeyp_inscreen_check_side(self, size, mask1, logo1, mask2, logo2, results, cap, image):
        frame = image
        is_true = False
        if results.pose_landmarks:
            list_kp = self.pose_keypoints_info(results, image)
            list_right_side = [list_kp[12][:], list_kp[14][:], list_kp[16][:], list_kp[24][:], list_kp[26][:],
                               list_kp[28][:],
                               list_kp[32][:], list_kp[8][:]]
            # list_left_side = [list_kp[11][:], list_kp[13][:], list_kp[15][:], list_kp[23][:], list_kp[25][:],
            # list_kp[27][:]]
            if all((0.5 <= j <= 1.0) for i, j, k, l in list_right_side):
                # Region of Interest (ROI), where we want
                # to insert logo
                ret1, frame = cap.read()
                roi = frame[-size - 10:-10, -size - 10:-10]

                # Set an index of where the mask is
                roi[np.where(mask1)] = 0
                roi += logo1
                is_true = True
        if not is_true:
            ret2, frame = cap.read()
            # Region of Interest (ROI), where we want
            # to insert logo
            roi = frame[-size - 10:-10, -size - 10:-10]

            # Set an index of where the mask is
            roi[np.where(mask2)] = 0
            roi += logo2

        return frame, is_true

    def fit_posekeyp_inscreen_check_front(self, size, mask1, logo1, mask2, logo2, results, cap, image):
        frame = image
        is_true = False
        if results.pose_landmarks:
            list_kp = self.pose_keypoints_info(results, image)
            if all((0.5 <= j <= 1.0) for i, j, k, l in list_kp):
                # Region of Interest (ROI), where we want
                # to insert logo
                ret1, frame = cap.read()
                roi = frame[-size - 10:-10, -size - 10:-10]
                # Set an index of where the mask is
                roi[np.where(mask1)] = 0
                roi += logo1
                is_true = True
        if not is_true:
            ret2, frame = cap.read()
            # Region of Interest (ROI), where we want
            # to insert logo
            roi = frame[-size - 10:-10, -size - 10:-10]
            # Set an index of where the mask is
            roi[np.where(mask2)] = 0
            roi += logo2

        return frame, is_true

    def fit_posekeyp_inscreen_check(self, count, side_count, front):
        # video_source = 0

        if front:
            video_source = 'demovideos/frontsquatref4.mp4'
        else:
            video_source = 'demovideos/sidesquatref4.mp4'

        cap = cv2.VideoCapture(video_source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # set new dimensions to cam object (not cap)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # create object
        detector = HolisticDetector()
        # constants for correct and wrong symbols :START
        size = 200
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
        if front:
            writer_front = cv2.VideoWriter('frontvideo.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (width, height))
        else:
            writer_side = cv2.VideoWriter('sidevideo.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (width, height))

        # Initiate holistic model
        with detector.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

            while cap.isOpened():
                ret, frame = cap.read()
                # frame = cv2.resize(frame_, (1280, 720), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)

                # Recolor Feed
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Make Detections
                results = holistic.process(image)
                # print(results.face_landmarks)

                # Recolor image back to BGR for rendering
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if front:
                    frame, screen_fit = self.fit_posekeyp_inscreen_check_front(size, mask1, logo1, mask2, logo2,
                                                                               results, cap, image)
                    is_full, final_count = self.level_indicator(frame, count)
                    if screen_fit:
                        count = self.front_squat_counter(results, frame, count, draw=True)

                        if count == 0:
                            # adding note on the live screen before starting squats: START
                            cv2.putText(frame, "FACE TOWARDS CAMERA !", (60, 100), cv2.FONT_HERSHEY_COMPLEX, 1.3,
                                        (255, 255, 255), 16,
                                        cv2.LINE_AA)

                            cv2.putText(frame, "Fill up the Bar ->", (80, 200), cv2.FONT_HERSHEY_COMPLEX, 1.5,
                                        (255, 255, 255), 16,
                                        cv2.LINE_AA)
                            cv2.putText(frame, "By doing", (80, 280), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255),
                                        16,
                                        cv2.LINE_AA)
                            cv2.putText(frame, "squats...", (80, 360), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255),
                                        16,
                                        cv2.LINE_AA)

                            cv2.putText(frame, "FACE TOWARDS CAMERA !", (60, 100), cv2.FONT_HERSHEY_COMPLEX, 1.3,
                                        (0, 0, 255), 4,
                                        cv2.LINE_AA)
                            cv2.putText(frame, "Fill up the Bar ->", (80, 200), cv2.FONT_HERSHEY_COMPLEX, 1.5,
                                        (255, 0, 0), 4,
                                        cv2.LINE_AA)
                            cv2.putText(frame, "By doing", (80, 280), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 0, 0), 4,
                                        cv2.LINE_AA)
                            cv2.putText(frame, "squats...", (80, 360), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 0, 0), 4,
                                        cv2.LINE_AA)
                            # adding note on the live screen before starting squats: END

                    if 1 <= count >= final_count:
                        writer_front.write(frame)
                        # frame = self.rescale_frame(frame, percent=150)
                        if count == final_count:
                            cap.release()
                            cv2.destroyAllWindows()
                else:
                    frame, screen_fit = self.fit_posekeyp_inscreen_check_side(size, mask1, logo1, mask2, logo2, results,
                                                                              cap, image)
                    is_full, final_count = self.level_indicator(frame, side_count)
                    if screen_fit:
                        side_count = self.side_squat_counter(results, frame, side_count, draw=True)

                        if side_count == 0:
                            # adding note on the live screen before starting squats: START
                            cv2.putText(frame, "FACE SIDEWAYS --> TO CAMERA !", (50, 100), cv2.FONT_HERSHEY_COMPLEX,
                                        0.9,
                                        (255, 255, 255), 16,
                                        cv2.LINE_AA)

                            cv2.putText(frame, "Fill up the Bar ->", (80, 200), cv2.FONT_HERSHEY_COMPLEX, 1.5,
                                        (255, 255, 255), 16,
                                        cv2.LINE_AA)
                            cv2.putText(frame, "By doing", (80, 280), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255),
                                        16,
                                        cv2.LINE_AA)
                            cv2.putText(frame, "squats...", (80, 360), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255),
                                        16,
                                        cv2.LINE_AA)

                            cv2.putText(frame, "FACE SIDEWAYS --> TO CAMERA !", (50, 100), cv2.FONT_HERSHEY_COMPLEX,
                                        0.9,
                                        (0, 0, 255), 4,
                                        cv2.LINE_AA)
                            cv2.putText(frame, "Fill up the Bar ->", (80, 200), cv2.FONT_HERSHEY_COMPLEX, 1.5,
                                        (255, 0, 0), 4,
                                        cv2.LINE_AA)
                            cv2.putText(frame, "By doing", (80, 280), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 0, 0), 4,
                                        cv2.LINE_AA)
                            cv2.putText(frame, "squats...", (80, 360), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 0, 0), 4,
                                        cv2.LINE_AA)
                            # adding note on the live screen before starting squats: END

                    if 1 <= side_count >= final_count:
                        writer_side.write(frame)
                        # frame = self.rescale_frame(frame, percent=150)
                        if side_count == final_count:
                            cap.release()
                            cv2.destroyAllWindows()

                cv2.imshow('YourPoseGuide', frame)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()


def main():
    videodource = 0
    # videodource='demovideos/squat.mp4'
    cap = cv2.VideoCapture(videodource)
    detector = HolisticDetector()
    # Initiate holistic model
    with detector.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Make Detections
            results = holistic.process(image)
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
            cv2.imshow('Holistic pose detection results', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
