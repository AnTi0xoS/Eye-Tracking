import cv2.cv2
import math


def coordinates_distance(coordinate_1, coordinate_2):
    x_1, y_1 = coordinate_1
    x_2, y_2 = coordinate_2
    distance = math.sqrt((x_2 - x_1) ** 2 + (y_2 - y_1) ** 2)
    return distance


def remove_noise(image):
    gauss_blur = cv2.GaussianBlur(image, (9, 9), 0)
    median_blur = cv2.medianBlur(gauss_blur, 3)
    return median_blur


def draw_text(image, text, font, font_scale, text_position, text_thickness=1, text_color=(0, 255, 0),
                       background_color=(0, 0, 0), padding_x=3, padding_y=3):
    (text_width, text_hight), _ = cv2.getTextSize(text, font, font_scale, text_thickness)
    x, y = text_position
    cv2.rectangle(image, (x - padding_x, y + padding_y), (x + text_width + padding_x, y - text_hight - padding_y),
                  background_color, -1)
    cv2.putText(image, text, text_position, font, font_scale, text_color, text_thickness)
    return image
