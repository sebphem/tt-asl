import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
from crop_images import crop_and_zoom
import torchvision
import numpy as np


# Create a custom DrawingSpec for landmarks and connections.
landmarks_style = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=5)  # Red landmarks
connections_style = mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=5)  # Gray borders


def draw_hand_landmarks_and_connections(images : [str], write_back_location: str):
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
        for idx, file in enumerate(images):
            file = crop_and_zoom(file)
            # Read an image, flip it around y-axis for correct handedness output (see above).
            image_array = np.array(file)
            opencv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            image = cv2.flip(opencv_image, 1)
            
            # Convert the BGR image to RGB before processing.
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


            if not results.multi_hand_landmarks:
                continue
            annotated_image = image.copy()
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    landmarks_style,
                    connections_style)
            cv2.imwrite(
                write_back_location + '/' + str(idx) + '.png', cv2.flip(annotated_image, 1))


def draw_hand_landmarks_and_connections_return_image_path(image, write_back_location: str):
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
        for idx, file in enumerate(images):
            file = crop_and_zoom(file)
            # Read an image, flip it around y-axis for correct handedness output (see above).
            image_array = np.array(file)
            opencv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            image = cv2.flip(opencv_image, 1)
            
            # Convert the BGR image to RGB before processing.
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


            if not results.multi_hand_landmarks:
                continue
            annotated_image = image.copy()
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    landmarks_style,
                    connections_style)
            cv2.imwrite(
                write_back_location + '/' + str(idx) + '.png', cv2.flip(annotated_image, 1))
    return write_back_location + '/' + str(idx) + '.png'


import os

def get_all_file_abspaths(directory):
    return [os.path.abspath(os.path.join(directory, f)) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

# Example usage:
directory_path = '/home/user/tt-hackathon-2024/tenstorrent_images'
all_file_paths = get_all_file_abspaths(directory_path)
draw_hand_landmarks_and_connections(all_file_paths, '/home/user/tt-hackathon-2024/image_queue')