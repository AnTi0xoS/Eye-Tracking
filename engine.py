import time

import cv2.cv2
import mediapipe.python.solutions as mediapipe_solutions
import numpy

import eye_tracking
import utils


frame_counter = 0
FONTS = cv2.FONT_ITALIC
LEFT_EYE_POINTS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_POINTS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

mp_face_mesh = mediapipe_solutions.face_mesh
cap = cv2.VideoCapture(0)

face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
start_time = time.time()

while cap.isOpened():
    frame_counter += 1
    success, frame = cap.read()
    if not success:
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        mesh_coordinates = eye_tracking.landmarks_detection(frame, results)
        blink_coefficient = eye_tracking.get_blink_coefficient(mesh_coordinates, RIGHT_EYE_POINTS, LEFT_EYE_POINTS)
        if blink_coefficient < 0.15:
            cv2.putText(frame, 'BLINK', (20, 260), FONTS, 1.3, (128, 0, 128), 2)
            closed_eyes = True
        else:
            closed_eyes = False

        cv2.polylines(frame, [numpy.array([mesh_coordinates[point] for point in LEFT_EYE_POINTS],
                                          dtype=numpy.int32)], True, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.polylines(frame, [numpy.array([mesh_coordinates[point] for point in RIGHT_EYE_POINTS],
                                          dtype=numpy.int32)], True, (0, 255, 0), 1, cv2.LINE_AA)

        right_eye_coordinates = [mesh_coordinates[p] for p in RIGHT_EYE_POINTS]
        left_eye_coordinates = [mesh_coordinates[p] for p in LEFT_EYE_POINTS]
        right_eye_cut_image, left_eye_cut_image = eye_tracking.get_cut_eyes_images(frame, right_eye_coordinates,
                                                                                   left_eye_coordinates)
        right_eye_position, right_color = eye_tracking.get_eye_position(right_eye_cut_image)
        left_eye_position, left_color = eye_tracking.get_eye_position(left_eye_cut_image)

        utils.draw_text(frame, f'Right eye: {right_eye_position}', FONTS, 0.7, (20, 120), 2, right_color[0], right_color[1], 8, 8)
        utils.draw_text(frame, f'Left eye: {left_eye_position}', FONTS, 0.7, (20, 190), 2, left_color[0], left_color[1], 8, 8)

    finish_time = time.time() - start_time
    fps = frame_counter / finish_time
    frame = utils.draw_text(frame, f'FPS: {round(fps, 1)}', FONTS, 1.0, (20, 50), text_thickness=2)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if (key == ord('q')) or (key == ord('Q')):
        break

cap.release()
cv2.destroyAllWindows()
