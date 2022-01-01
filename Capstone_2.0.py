#  General imports  #
from __future__ import division
import numpy as np
import cv2
import os
from math import hypot
import datetime
import time
import pandas
import matplotlib.pyplot as plt

# Image processing #
from scipy.ndimage import zoom
from scipy.spatial import distance
import dlib
from tensorflow.keras.models import load_model
from imutils import face_utils
from pygame import mixer

global shape_x
global shape_y
global input_shape
global nClasses


# Import Module
from win32com import client

#----------------final veriables -----------------


def show_webcam():
    total_eye_blinking = 0

    left_eye_gaze_count = 0
    right_eye_gaze_count = 0
    center_eye_gaze_count = 0

    left_head_direction_count = 0
    right_head_direction_count = 0
    center_head_direction_count = 0
    up_head_direction_count = 0
    down_head_direction_count = 0
    head_total = 0

    happy_emotion_count = 0
    neutral_emotion_count = 0
    sad_emotion_count = 0
    fear_emotion_count = 0
    surprise_emotion_count = 0

    sleepy_state_count = 0


    now = datetime.datetime.now()

    df = pandas.DataFrame(columns=["Eye Contact", "Confidence", "Calm", "Smiled", "Excited", "Focused", "NotStressed",
                                   "NotAwkward", "Friendly"])

    eye_contact = []
    confidence = []
    calm = []
    smiled = []
    excited = []
    focused = []
    notstressed = []
    notawkward = []
    friendly = []







    shape_x = 48
    shape_y = 48
    Known_distance = 76.2
    Known_width = 14.3
    blink_count = 0
    mixer.init()
    sound = mixer.Sound('alarm.wav')
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    leye = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
    reye = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
    lbl = ['Close', 'Open']
    d_model = load_model('cnncat2.h5')
    path = os.getcwd()
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    score = 0
    thicc = 2
    rpred = [99]
    lpred = [99]


    #eye_contact calculating

    def eye_contact_detection(time, total_blink,max_gaze_direction) -> int:
        if(max_gaze_direction != "center"):
            return 4
        else:
            temp = time/5       #thresold is 12 blink per 60 seconds
            if(abs(total_blink-temp)<4):
                return 9
            elif(abs(total_blink-temp)<9):
                return 8
            elif (abs(total_blink - temp) < 12):
                return 7
            elif (abs(total_blink - temp) < 15):
                return 6
            else:
                return 5

    def calm_detection(time, total_head,max_head_direction):
        temp = time/6         #threshold is 10 head movement per 60 seconds
        if (max_head_direction != "center"):
            return 4
        else:
            temp = time / 5  # thresold is 12 blink per 60 seconds
            if (abs(total_head - temp) < 2):
                return 9
            elif (abs(total_head - temp) < 5):
                return 8
            elif (abs(total_head - temp) < 8):
                return 7
            elif (abs(total_head - temp) < 10):
                return 6
            else:
                return 5

    def confidance_detection(time, avg_head_direction, max_eye_direction, happy_emotion,e):
        if(avg_head_direction != "center" and max_eye_direction != "center"):
            return 4
        elif(avg_head_direction != "center" or max_eye_direction != "center"):
            return 5
        else:
            if(happy_emotion=="smiled" and e == 9 ):
                return 9
            elif(happy_emotion=="smiled" and e == 8 ):
                return 8
            elif(happy_emotion == "smiled" and e == 7):
                return 7
            elif(happy_emotion == "smiled" and e == 6):
                return 6


    def smiled_detection( happy_emotion, hec,nec):
        if(happy_emotion!="smiled"):
            return 4
        elif(hec > nec):
            return 10
        elif(1 < abs(nec-hec) < 15):
            return 8
        elif(15 <= abs(nec-hec) < 50):
            return 7
        elif(50 <= abs(nec-hec) < 70):
            return 6
        else:
            return 5




    def excited_detection(time, hec):
        temp = time/4     # happy emotion threshhold = 20 times per min
        if(abs(hec-temp) > 30):
            return 10
        elif(abs(hec-temp) > 25):
            return 9
        elif (abs(hec - temp) > 20):
            return 8
        elif (abs(hec - temp) > 15):
            return 7
        elif (abs(hec - temp) < 10):
            return 6
        else:
            return 5

    def focused_detection(head_direction,time, total_blink,max_gaze_direction):
        e = eye_contact_detection(time, total_blink,max_gaze_direction)
        if(head_direction == "center"):
            return e
        else:
            return 4

    def notawkward_detection(time, total_blink,max_gaze_direction,fc,sad_emotion_count,fear_emotion_count):
        print("Focus value")
        print(fc)
        e = eye_contact_detection(time, total_blink,max_gaze_direction)
        if((e == 9) and (fc == 9) and (sad_emotion_count == 0 and fear_emotion_count == 0)):
            return 9
        elif((e<5) and (fc<5) and (sad_emotion_count>1 or fear_emotion_count>1)):
            return 5
        elif ((e == 6) and (fc == 6) and (sad_emotion_count > 0 and fear_emotion_count > 0)):
            return 6
        elif ((e == 7) and (fc == 7) and (sad_emotion_count > 0 or fear_emotion_count > 0)):
            return 7
        elif ((e == 8) and (fc == 8) and (sad_emotion_count == 0 or fear_emotion_count == 0)):
            return 8
        else:
            return 4


    def notstressed_detection(total_time,happy_emotion_count,total_eye_blinking,eye_gaze_direction,t_head):
        t1 = total_time/6      # If happy emotion count is minimum 10 per minute then NOT Stressed
        t2 = total_time/2

        if(abs(t1-happy_emotion_count) < 5  and abs(t1-total_eye_blinking) < 5  and abs(t2-t_head) < 10):
             return 9
        elif(abs(t1-happy_emotion_count) < 7  and abs(t1-total_eye_blinking) < 10  and abs(t2-t_head) < 15):
            return 8
        elif (abs(t1 - happy_emotion_count) < 10 and abs(t1 - total_eye_blinking) < 15 and abs(t2 - t_head) < 20):
            return 7
        elif (abs(t1 - happy_emotion_count) < 15 or abs(t1 - total_eye_blinking) < 20 and abs(t2 - t_head) < 30):
            return 6
        elif (abs(t1 - happy_emotion_count) < 20 and abs(t1 - total_eye_blinking) < 25 and abs(t2 - t_head) < 40):
            return 5
        else:
            return 4






    def friendly_detection(eye_gaze_direction,happy_emotion):
        if(eye_gaze_direction=="center" and happy_emotion!="smiled"):
            return 7
        elif(eye_gaze_direction!="center" and happy_emotion=="smiled"):
            return 8
        elif(eye_gaze_direction =="center" and happy_emotion =="smiled"):
            return 9
        else:
            return 6



    #Head and Eye gaze

    def midpoint(p1, p2):
        return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

    def get_blinking_ratio(eye_points, facial_landmarks):
        left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
        right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
        center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
        center_bottom = midpoint(facial_landmarks.part(eye_points[4]), facial_landmarks.part(eye_points[5]))
        ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
        hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
        ratio = hor_line_length / ver_line_length
        return ratio

    def get_gaze_ratio(eye_points, facial_landmarks):
        eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                               (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                               (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                               (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                               (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                               (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)],
                              np.int32)
        height, width, _ = frame.shape
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, [eye_region], True, 255, 1)
        cv2.fillPoly(mask, [eye_region], 255)
        eyes = cv2.bitwise_and(gray, gray, mask=mask)
        min_x = np.min(eye_region[:, 0])
        max_x = np.max(eye_region[:, 0])
        min_y = np.min(eye_region[:, 1])
        max_y = np.max(eye_region[:, 1])

        gray_eye = eyes[min_y: max_y, min_x: max_x]
        _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
        h, w = threshold_eye.shape
        left_side_threshold = threshold_eye[0: h, 0: int(w / 2)]
        left_side_white = cv2.countNonZero(left_side_threshold)
        right_side_threshold = threshold_eye[0: h, int(w / 2): w]
        right_side_white = cv2.countNonZero(right_side_threshold)

        if left_side_white == 0:
            gaze_ratio = 1
        elif right_side_white == 0:
            gaze_ratio = 5
        else:
            gaze_ratio = left_side_white / right_side_white
        return gaze_ratio

    def focal_length_finder(measured_distance, real_width, width_in_rf_image):   # finding the focal length
        focal_length = (width_in_rf_image * measured_distance) / real_width
        return focal_length

    def distance_from_camera(Focal_Length, real_face_width, face_width_in_frame):
        distance = (real_face_width * Focal_Length) / face_width_in_frame
        return distance

    def face_data(image):
        face_width = 0  # making face width to zero
        # converting color image ot gray scale image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # detecting face in the image
        faces = face_detector.detectMultiScale(gray_image, 1.3, 5)

        # looping through the faces detect in the image
        # getting coordinates x, y , width and height
        for (x, y, h, w) in faces:
            # getting face width in the pixels
            face_width = w
        # return the face width in pixel
        return face_width

    # reading reference_image from directory
    ref_image = cv2.imread("Ref_image.png")

    # find the face width(pixels) in the reference_image
    ref_image_face_width = face_data(ref_image)

    Focal_length_found = focal_length_finder(Known_distance, Known_width, ref_image_face_width)

    #   head gaze detection ----------
    face_landmark_path = 'shape_predictor_68_face_landmarks.dat'
    K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
         0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
         0.0, 0.0, 1.0]
    D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

    cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
    dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)
    object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                             [1.330353, 7.122144, 6.903745],
                             [-1.330353, 7.122144, 6.903745],
                             [-6.825897, 6.760612, 4.402142],
                             [5.311432, 5.485328, 3.987654],
                             [1.789930, 5.393625, 4.413414],
                             [-1.789930, 5.393625, 4.413414],
                             [-5.311432, 5.485328, 3.987654],
                             [2.005628, 1.409845, 6.165652],
                             [-2.005628, 1.409845, 6.165652],
                             [2.774015, -2.080775, 5.048531],
                             [-2.774015, -2.080775, 5.048531],
                             [0.000000, -3.116408, 6.097667],
                             [0.000000, -7.415691, 4.070434]])

    reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                               [10.0, 10.0, -10.0],
                               [10.0, -10.0, -10.0],
                               [10.0, -10.0, 10.0],
                               [-10.0, 10.0, 10.0],
                               [-10.0, 10.0, -10.0],
                               [-10.0, -10.0, -10.0],
                               [-10.0, -10.0, 10.0]])

    line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
                  [4, 5], [5, 6], [6, 7], [7, 4],
                  [0, 4], [1, 5], [2, 6], [3, 7]]

    def get_head_pose(shape):
        image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                                shape[39], shape[42], shape[45], shape[31], shape[35],
                                shape[48], shape[54], shape[57], shape[8]])

        _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

        reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,
                                            dist_coeffs)

        reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

        # calc euler angle
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))
        _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

        return reprojectdst, euler_angle

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_landmark_path)
    sum1, sum2, sum3 = 0, 0, 0
    d = {'left': 0, 'right': 0, 'up': 0, 'down': 0, 'center': 0}
    count = 0
    h = datetime.datetime.now()
    h = h.strftime("%M")
    h = int(h) + 1

    #Head and Eye gaze



    def eye_aspect_ratio(eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    (nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    (jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]

    (eblStart, eblEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
    (ebrStart, ebrEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]

    model = load_model('video.h5')
    face_detect = dlib.get_frontal_face_detector()
    predictor_landmarks = dlib.shape_predictor("face_landmarks.dat")

    # Lancer la capture video
    video_capture = cv2.VideoCapture(0)
    start_time = now.strftime("%H:%M:%S")
    time1 = time.time()

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        frame=cv2.flip(frame, 1)

        face_index = 0

        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        new_frame = np.zeros((500, 500, 3), np.uint8)
        left_eye = leye.detectMultiScale(gray)
        right_eye = reye.detectMultiScale(gray)
        cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)
        cv2.imshow("Assesment", frame)
        rects = face_detect(gray, 1)
        # gray, detected_faces, coord = detect_face(frame)

        #head & eye gaze
        faces = detector(gray)
        for face in faces:
            x, y = face.left(), face.top()
            x1, y1 = face.right(), face.bottom()
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
            landmarks = predictor(gray, face)
            x = landmarks.part(36).x  # landmark positions
            y = landmarks.part(36).y
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 2)

            # face_width = face.width
            face_width_in_frame = face_data(frame)
            if face_width_in_frame != 0:
                Distance = distance_from_camera(Focal_length_found, Known_width, face_width_in_frame)
                # draw line as background of text
                cv2.line(frame, (290, 50), (500, 50), (0, 0, 255), 32)
                cv2.line(frame, (290, 50), (500, 50), (0, 0, 0), 28)


                # Drawing Text on the screen
                cv2.putText(frame, f"Distance: {round(Distance, 2)} CM", (300, 55), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                            (0, 255, 0), 2)

            # Eye blinking detection
            left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
            right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
            # ratio = hor_line_length/ver_line_length
            avg_ratio = (left_eye_ratio + right_eye_ratio) / 2
            if avg_ratio > 5.5:
                total_eye_blinking = total_eye_blinking + 1
                cv2.putText(frame, "BLINKING", (100, 150), cv2.FONT_HERSHEY_PLAIN, 7, (255, 100, 20), 3)

            # Gaze detection
            left_eye_gaze_ratio = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
            right_eye_gaze_ratio = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
            gaze_ratios = (left_eye_gaze_ratio + right_eye_gaze_ratio) / 2

            if gaze_ratios <= 0.7:
                new_frame[:] = (255, 0, 0)
                left_eye_gaze_count = left_eye_gaze_count + 1
                cv2.putText(frame, "Gaze Direction : Left", (230, 420), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
            elif 0.7 < gaze_ratios <= 2:
                center_eye_gaze_count = center_eye_gaze_count + 1
                cv2.putText(frame, "Gaze Direction : Center", (230, 420), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
            else:
                new_frame[:] = (0, 0, 255)
                right_eye_gaze_count = right_eye_gaze_count + 1
                cv2.putText(frame, "Gaze Direction : Right", (230, 420), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)



        #head & eye gaze

        for (i, rect) in enumerate(rects):
            #Head & Eye gaze

            if len(rects) > 0:
                shape = predictor(frame, rects[0])
                shape = face_utils.shape_to_np(shape)

                reprojectdst, euler_angle = get_head_pose(shape)

                for (x, y) in shape:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

                for start, end in line_pairs:
                    cv2.line(frame, (int(reprojectdst[start][0]), int(reprojectdst[start][1])),
                             (int(reprojectdst[end][0]), int(reprojectdst[end][1])), (0, 0, 255), 2)

                x = euler_angle[0, 0]

                y = euler_angle[1, 0]

                z = euler_angle[2, 0]
                if x < 0 and y < 0 and z > 0:
                    d['up'] += 1
                    head_total += 1;
                elif y < 0 and z < 0 and x > 0:
                    d['left'] += 1
                    head_total += 1;
                elif x < 0 and z < 0 and y > 0:
                    d['right'] += 1
                    head_total += 1;
                elif x > 0 and y > 0 and z > 0:
                    d['down'] += 1
                    head_total += 1;
                else:
                    d['center'] += 1

                if ((y > -15 and y < 15) and (x > -10 and x < 15)):
                    center_head_direction_count = center_head_direction_count + 1
                    cv2.putText(frame, " Center ".format(euler_angle[0, 0]), (200, 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (0, 0, 0), thickness=2)
                elif y < -15:
                    right_head_direction_count = right_head_direction_count + 1
                    cv2.putText(frame, " Right ".format(euler_angle[0, 0]), (200, 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (0, 0, 0), thickness=2)
                elif y > 15:
                    left_head_direction_count = left_head_direction_count + 1
                    cv2.putText(frame, " Left ".format(euler_angle[0, 0]), (200, 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (0, 0, 0), thickness=2)
                elif x > 20:
                    down_head_direction_count = down_head_direction_count + 1
                    cv2.putText(frame, " Down ".format(euler_angle[0, 0]), (200, 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (0, 0, 0), thickness=2)
                elif x < -5:
                    up_head_direction_count = up_head_direction_count + 1
                    cv2.putText(frame, " Up ".format(euler_angle[0, 0]), (200, 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (0, 0, 0), thickness=2)






            #Head & Eye gaze


            for (x, y, w, h) in right_eye:
                r_eye = frame[y:y + h, x:x + w]
                r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
                r_eye = cv2.resize(r_eye, (24, 24))
                r_eye = r_eye / 255
                r_eye = r_eye.reshape(24, 24, -1)
                r_eye = np.expand_dims(r_eye, axis=0)
                rpred = d_model.predict_classes(r_eye)
                if rpred[0] == 1:
                    lbl = 'Open'
                if rpred[0] == 0:
                    lbl = 'Closed'
                break

            for (x, y, w, h) in left_eye:
                l_eye = frame[y:y + h, x:x + w]
                l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
                l_eye = cv2.resize(l_eye, (24, 24))
                l_eye = l_eye / 255
                l_eye = l_eye.reshape(24, 24, -1)
                l_eye = np.expand_dims(l_eye, axis=0)
                lpred = d_model.predict_classes(l_eye)
                if lpred[0] == 1:
                    lbl = 'Open'
                if lpred[0] == 0:
                    lbl = 'Closed'
                break

            if rpred[0] == 0 and lpred[0] == 0:
                score = score + 1
                blink_count = blink_count + 1
                cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                score = score - 1
                cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

            if score < 0:
                score = 0
            cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            if score > 15:
                # person is feeling sleepy so we beep the alarm
                cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
                try:
                    sound.play()
                    sleepy_state_count = sleepy_state_count + 1

                except:
                    pass
                if thicc < 16:
                    thicc = thicc + 2
                else:
                    thicc = thicc - 2
                    if thicc < 2:
                        thicc = 2
                cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
            else:
                sound.stop()



            shape = predictor_landmarks(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # Identify face coordinates
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            face = gray[y:y + h, x:x + w]

            # Zoom on extracted face ---------------------------------need to handle exception
            face = zoom(face, (shape_x / face.shape[0], shape_y / face.shape[1]))

            # Cast type float
            face = face.astype(np.float32)

            # Scale
            face /= float(face.max())
            face = np.reshape(face.flatten(), (1, 48, 48, 1))

            # Make Prediction
            prediction = model.predict(face)
            prediction_result = np.argmax(prediction)

            # Rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            for (j, k) in shape:
                cv2.circle(frame, (j, k), 1, (0, 0, 255), -1)

            # 2. Annotate main image with a label
            if prediction_result == 0:
                cv2.putText(frame, "Angry", (x + w - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif prediction_result == 1:
                cv2.putText(frame, "Disgust", (x + w - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif prediction_result == 2:
                fear_emotion_count = fear_emotion_count + 1
                cv2.putText(frame, "Fear", (x + w - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif prediction_result == 3:
                happy_emotion_count = happy_emotion_count + 1
                cv2.putText(frame, "Happy", (x + w - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif prediction_result == 4:
                sad_emotion_count = sad_emotion_count + 1
                cv2.putText(frame, "Sad", (x + w - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif prediction_result == 5:
                surprise_emotion_count = surprise_emotion_count + 1
                cv2.putText(frame, "Surprise", (x + w - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                neutral_emotion_count = neutral_emotion_count + 1
                cv2.putText(frame, "Neutral", (x + w - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 3. Eye Detection and Blink Count
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            # Compute Eye Aspect Ratio
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            time2 = time.time()
            total_time = time2 - time1

            print(total_time)                    #total_time in seconds
            total_minutes = total_time / 60      #no_of_blinking (avg) = 12 to 15 per minute
            print(total_minutes)

            eye_gaze_direction="noncenter"
            if(center_eye_gaze_count > (left_eye_gaze_count+right_eye_gaze_count)):
                eye_gaze_direction = "center"

            head_direction = "noncenter"
            if (center_head_direction_count > (left_head_direction_count+right_head_direction_count+up_head_direction_count+down_head_direction_count)):
                eye_gaze_direction = "center"

            happy_emotion= "Notsmiled"
            if((happy_emotion_count+neutral_emotion_count) > (sad_emotion_count+surprise_emotion_count+fear_emotion_count)):
                happy_emotion= "smiled"

            t_head = left_head_direction_count + right_head_direction_count + up_head_direction_count


            e = eye_contact_detection(total_time,total_eye_blinking,eye_gaze_direction)
            c = calm_detection(total_time,head_total,head_direction)
            con = confidance_detection(total_time,head_direction,eye_gaze_direction,happy_emotion,e)
            sm = smiled_detection(happy_emotion,happy_emotion_count,neutral_emotion_count)
            ex = excited_detection(total_time, happy_emotion_count)
            fc = focused_detection(head_direction,total_time,total_eye_blinking,eye_gaze_direction)
            na = notawkward_detection(total_time,total_eye_blinking,eye_gaze_direction,fc,sad_emotion_count,fear_emotion_count)
            ns = notstressed_detection(total_time,happy_emotion_count,total_eye_blinking,eye_gaze_direction,t_head)
            fd = friendly_detection(eye_gaze_direction,happy_emotion)

            # eye_contact.append(total_eye_blinking)

            eye_contact.append(e)
            calm.append(c)
            confidence.append(con)
            smiled.append(sm)
            excited.append(ex)
            focused.append(fc)
            notawkward.append(na)
            notstressed.append(ns)
            friendly.append(fd)

            for i in range(0, len(eye_contact)):
               df = df.append({"Eye Contact":eye_contact[i], "Confidence":confidence[i], "Calm":calm[i], "Smiled":smiled[i], "Excited":excited[i], "Focused":focused[i], "NotStressed":notstressed[i],
                                   "NotAwkward":notawkward[i], "Friendly":friendly[i]},ignore_index=True)



            df.to_csv("Final_report.csv")

            # Reading the csv file
            df_new = pandas.read_csv('Final_report.csv')

            # saving xlsx file
            GFG = pandas.ExcelWriter('Final_report.xlsx')
            df_new.to_excel(GFG, index=False)

            GFG.save()

            # graph printing
            data = {"Eye Contact":e, "Confidence":con, "Calm":c, "Smiled":sm, "Excited":ex, "Focused":fc, "NotStressed":ns,
                                   "NotAwkward":na, "Friendly":fd }
            attributes = list(data.keys())
            values = list(data.values())

            fig = plt.figure(figsize=(10, 5))

            # creating the bar plot
            plt.bar(attributes, values, color='maroon',
                    width=0.4)

            plt.xlabel("Attributes")
            plt.ylabel("Out of 10 Rating")
            plt.title("Assessment Report")
            plt.show()

            # # # Open Microsoft Excel
            # # excel = client.Dispatch("Excel.Application")
            # #
            # # # Read Excel File
            # # sheets = excel.Workbooks.Open('Final_report.xlsx')
            # # work_sheets = sheets.Worksheets[0]
            # #
            # # # Convert into PDF File
            # # work_sheets.ExportAsFixedFormat(0, 'Final_report.pdf')

            for k, v in d.items():
                print(k)
                print(v)
            if head_total/4 > 50:
                print("Candidate is unstable State")
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


def main():
    show_webcam()

if __name__ == "__main__":
    main()
