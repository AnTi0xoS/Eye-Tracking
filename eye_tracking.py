import cv2.cv2
import numpy
import utils


def landmarks_detection(image, results):
    image_height, image_width = image.shape[:2]
    mesh_coords = [(int(coord.x * image_width), int(coord.y * image_height))
                   for coord in results.multi_face_landmarks[0].landmark]
    return mesh_coords


def get_blink_coefficient(landmarks, right_points, left_points):
    right_eye_horizontal_right = landmarks[right_points[0]]
    right_eye_horizontal_left = landmarks[right_points[8]]
    right_eye_vertical_top_1 = landmarks[right_points[13]]
    right_eye_vertical_bottom_1 = landmarks[right_points[3]]
    right_eye_vertical_top_2 = landmarks[right_points[11]]
    right_eye_vertical_bottom_2 = landmarks[right_points[5]]

    left_eye_horizontal_right = landmarks[left_points[0]]
    left_eye_horizontal_left = landmarks[left_points[8]]
    left_eye_vertical_top_1 = landmarks[left_points[13]]
    left_eye_vertical_bottom_1 = landmarks[left_points[3]]
    left_eye_vertical_top_2 = landmarks[left_points[11]]
    left_eye_vertical_bottom_2 = landmarks[left_points[5]]

    right_eye_horizontal_distance = utils.coordinates_distance(right_eye_horizontal_right, right_eye_horizontal_left)
    right_eye_vertical_distance_1 = utils.coordinates_distance(right_eye_vertical_top_1, right_eye_vertical_bottom_1)
    right_eye_vertical_distance_2 = utils.coordinates_distance(right_eye_vertical_top_2, right_eye_vertical_bottom_2)

    left_eye_horizontal_distance = utils.coordinates_distance(left_eye_horizontal_right, left_eye_horizontal_left)
    left_eye_vertical_distance_1 = utils.coordinates_distance(left_eye_vertical_top_1, left_eye_vertical_bottom_1)
    left_eye_vertical_distance_2 = utils.coordinates_distance(left_eye_vertical_top_2, left_eye_vertical_bottom_2)

    right_eye_blink_coefficient = (right_eye_vertical_distance_1 + right_eye_vertical_distance_2) / \
                                  (2 * right_eye_horizontal_distance)
    left_eye_blink_coefficient = (left_eye_vertical_distance_1 + left_eye_vertical_distance_2) / \
                                 (2 * left_eye_horizontal_distance)

    total_blink_coefficient = (right_eye_blink_coefficient + left_eye_blink_coefficient) / 2
    return total_blink_coefficient


def get_cut_eyes_images(image, left_eye_coordinates, right_eye_coordinates):
    black_white_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_size = black_white_image.shape
    mask = numpy.zeros(image_size, dtype=numpy.uint8)

    cv2.fillPoly(mask, [numpy.array(right_eye_coordinates, dtype=numpy.int32)], 255)
    cv2.fillPoly(mask, [numpy.array(left_eye_coordinates, dtype=numpy.int32)], 255)

    eyes = cv2.bitwise_and(black_white_image, black_white_image, mask=mask)
    eyes[mask == 0] = 155

    left_eye_max_x = (max(left_eye_coordinates, key=lambda coordinate: coordinate[0]))[0]
    left_eye_min_x = (min(left_eye_coordinates, key=lambda coordinate: coordinate[0]))[0]
    left_eye_max_y = (max(left_eye_coordinates, key=lambda coordinate: coordinate[1]))[1]
    left_eye_min_y = (min(left_eye_coordinates, key=lambda coordinate: coordinate[1]))[1]

    right_eye_max_x = (max(right_eye_coordinates, key=lambda coordinate: coordinate[0]))[0]
    right_eye_min_x = (min(right_eye_coordinates, key=lambda coordinate: coordinate[0]))[0]
    right_eye_max_y = (max(right_eye_coordinates, key=lambda coordinate: coordinate[1]))[1]
    right_eye_min_y = (min(right_eye_coordinates, key=lambda coordinate: coordinate[1]))[1]

    left_eye_cut_image = eyes[left_eye_min_y: left_eye_max_y, left_eye_min_x: left_eye_max_x]
    right_eye_cut_image = eyes[right_eye_min_y: right_eye_max_y, right_eye_min_x: right_eye_max_x]

    return right_eye_cut_image, left_eye_cut_image


def get_eye_position(eye_cut_image):
    eye_height, eye_width = eye_cut_image.shape
    eye_cut_image = utils.remove_noise(eye_cut_image)
    ret, threshold_eye_image = cv2.threshold(eye_cut_image, 130, 255, cv2.THRESH_BINARY)
    horizontal_separation = int(eye_width / 3)

    right_image_part = threshold_eye_image[0: eye_height, 0: horizontal_separation]
    center_image_part = threshold_eye_image[0: eye_height, horizontal_separation: horizontal_separation * 2]
    left_image_part = threshold_eye_image[0: eye_height, horizontal_separation * 2: eye_width]

    eye_position, color = pixel_counter(right_image_part, center_image_part, left_image_part)
    return eye_position, color


def pixel_counter(right_image_part, center_image_part, left_image_part):
    right_image_part = numpy.sum(right_image_part == 0)
    center_image_part = numpy.sum(center_image_part == 0)
    left_image_part = numpy.sum(left_image_part == 0)
    eye_parts = [right_image_part, center_image_part, left_image_part]
    eye_position = ""
    color = list()

    max_index = eye_parts.index(max(eye_parts))
    if max_index == 0:
        eye_position = "RIGHT"
        color = [(0, 255, 255), (128, 0, 128)]
    elif max_index == 1:
        eye_position = "CENTER"
        color = [(0, 0, 0), (0, 255, 0)]
    elif max_index == 2:
        eye_position = "LEFT"
        color = [(0, 255, 255), (128, 0, 128)]

    return eye_position, color
