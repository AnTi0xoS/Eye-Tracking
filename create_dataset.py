import cv2.cv2
import mediapipe.python.solutions as mediapipe_solutions

import os
import shutil
from itertools import islice
from pathlib import Path

import eye_tracking


def save_dataset_result(dataset_directory_paths, dataset_result):
    global counter
    for i in range(len(dataset_directory_paths)):
        image = dataset_directory_paths[i]
        print(str(counter) + ' ' + image)
        current_cap = cv2.VideoCapture(image)
        success, frame = current_cap.read()
        if not success:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        mp_face_mesh = mediapipe_solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        results = face_mesh.process(rgb_frame)
        mesh_coordinates = eye_tracking.landmarks_detection(frame, results)

        right_eye_coordinates = [mesh_coordinates[p] for p in RIGHT_EYE_POINTS]
        left_eye_coordinates = [mesh_coordinates[p] for p in LEFT_EYE_POINTS]
        right_eye_cut_image, left_eye_cut_image = eye_tracking.get_cut_eyes_images(frame, right_eye_coordinates,
                                                                               left_eye_coordinates)
        right_eye_position, right_color = eye_tracking.get_eye_position(right_eye_cut_image)
        left_eye_position, left_color = eye_tracking.get_eye_position(left_eye_cut_image)

        if right_eye_position == "LEFT":
            left_eye_position = "LEFT"
        if left_eye_position == "RIGHT":
            right_eye_position = "RIGHT"

        current_result = right_eye_position + '\n'
        print(current_result + '\n')
        dataset_result.write(current_result)
        counter += 1
        if counter == 500:
            break
    return


def sort_dataset(dataset_directory_paths, dataset_result_list, sorted_datasets):
    for i in range(len(dataset_directory_paths)):
        if dataset_result_list[i] == "CENTER":
            shutil.copy(dataset_directory_train_paths[i], sorted_datasets["CENTER"])
        elif dataset_result_list[i] == "LEFT":
            shutil.copy(dataset_directory_train_paths[i], sorted_datasets["LEFT"])
        elif dataset_result_list[i] == "RIGHT":
            shutil.copy(dataset_directory_train_paths[i], sorted_datasets["RIGHT"])


LEFT_EYE_POINTS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_POINTS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

dataset_directory_train = Path("D:/DGW_data_save/train")
dataset_directory_val = Path("D:/DGW_data_save/val")
dataset_result_train = open("D:/DGW_data_save/result_train.txt", 'a')
dataset_result_val = open("D:/DGW_data_save/result_val.txt", 'a')

counter = 0
dataset_directory_train_paths = [i.path for i in islice(os.scandir(dataset_directory_train), 26496)]
print(len(dataset_directory_train_paths))
save_dataset_result(dataset_directory_train_paths, dataset_result_train)

print("----- NEXT DATASET -----" + '\n')

counter = 0
dataset_directory_val_paths = [i.path for i in islice(os.scandir(dataset_directory_val), 9382)]
print(len(dataset_directory_val_paths))
save_dataset_result(dataset_directory_val_paths, dataset_result_val)

dataset_result_train = open("D:/DGW_data_save/result_train.txt", 'r')
dataset_result_val = open("D:/DGW_data_save/result_val.txt", 'r')
dataset_result_train_list = list(map(lambda line: line.strip(), dataset_result_train.readlines()))
dataset_result_val_list = list(map(lambda line: line.strip(), dataset_result_val.readlines()))
dataset_result_train.close()
dataset_result_val.close()

sorted_train_datasets = {
    "CENTER": Path("D:/DGW_dataset/train/center"),
    "RIGHT": Path("D:/DGW_dataset/train/right"),
    "LEFT": Path("D:/DGW_dataset/train/left")
}
sorted_val_datasets = {
    "CENTER": Path("D:/DGW_dataset/val/center"),
    "RIGHT": Path("D:/DGW_dataset/val/left"),
    "LEFT": Path("D:/DGW_dataset/val/right")
}

dataset_directory_train_paths = [i.path for i in islice(os.scandir(dataset_directory_train), 26496)]
dataset_directory_val_paths = [i.path for i in islice(os.scandir(dataset_directory_val), 9382)]

sort_dataset(dataset_directory_train_paths, dataset_result_train_list, sorted_train_datasets)
sort_dataset(dataset_directory_val_paths, dataset_result_val_list, sorted_val_datasets)
