import cv2
import math
import mediapipe as mp
import paho.mqtt.client as mqtt

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
move_initial_y = None

call_stack = []
action = None


def check_middle_finger_below_half_screen(hand_landmarks):
    finger_orientations = [
        abs(
            math.degrees(
                math.atan2(
                    hand_landmarks.landmark[mp_hands.HandLandmark[f"{finger}_MCP"]].y
                    - hand_landmarks.landmark[mp_hands.HandLandmark[f"{finger}_TIP"]].y,
                    hand_landmarks.landmark[mp_hands.HandLandmark[f"{finger}_MCP"]].x
                    - hand_landmarks.landmark[mp_hands.HandLandmark[f"{finger}_TIP"]].x,
                )
            )
        )
        for finger in ["INDEX_FINGER", "MIDDLE_FINGER", "RING_FINGER", "PINKY"]
    ]
    finger_avg_orientation = sum(finger_orientations) / len(finger_orientations)
    if not all(
        [
            orientation - finger_avg_orientation < 10
            for orientation in finger_orientations
        ]
    ):
        return False
    return hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y > 0.5


def check_middle_finger_above_half_screen(hand_landmarks):
    finger_orientations = [
        abs(
            math.degrees(
                math.atan2(
                    hand_landmarks.landmark[mp_hands.HandLandmark[f"{finger}_MCP"]].y
                    - hand_landmarks.landmark[mp_hands.HandLandmark[f"{finger}_TIP"]].y,
                    hand_landmarks.landmark[mp_hands.HandLandmark[f"{finger}_MCP"]].x
                    - hand_landmarks.landmark[mp_hands.HandLandmark[f"{finger}_TIP"]].x,
                )
            )
        )
        for finger in ["INDEX_FINGER", "MIDDLE_FINGER", "RING_FINGER", "PINKY"]
    ]
    finger_avg_orientation = sum(finger_orientations) / len(finger_orientations)
    if not all(
        [
            orientation - finger_avg_orientation < 10
            for orientation in finger_orientations
        ]
    ):
        return False
    return hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y < 0.5


def check_fist(hand_landmarks):
    finger_mcp_above_pip = [
        hand_landmarks.landmark[mp_hands.HandLandmark[f"{finger}_MCP"]].y
        <= hand_landmarks.landmark[mp_hands.HandLandmark[f"{finger}_PIP"]].y
        for finger in ["INDEX_FINGER", "MIDDLE_FINGER", "RING_FINGER", "PINKY"]
    ]
    print("finger_mcp_above_pip", finger_mcp_above_pip)
    return all(finger_mcp_above_pip)


def check_thumbs_up(hand_landmarks):
    thumb_orientation = math.degrees(
        math.atan2(
            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y
            - hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x
            - hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x,
        )
    )
    finger_orientations = [
        abs(
            math.degrees(
                math.atan2(
                    hand_landmarks.landmark[mp_hands.HandLandmark[f"{finger}_MCP"]].y
                    - hand_landmarks.landmark[mp_hands.HandLandmark[f"{finger}_PIP"]].y,
                    hand_landmarks.landmark[mp_hands.HandLandmark[f"{finger}_MCP"]].x
                    - hand_landmarks.landmark[mp_hands.HandLandmark[f"{finger}_PIP"]].x,
                )
            )
        )
        for finger in ["INDEX_FINGER", "MIDDLE_FINGER", "RING_FINGER", "PINKY"]
    ]
    finger_avg_orientation = sum(finger_orientations) / len(finger_orientations)
    thumbs_normal_finger = (thumb_orientation + 90) - finger_avg_orientation < 30
    if not thumbs_normal_finger:
        return False
    finger_straight = all(
        [
            orientation - finger_avg_orientation < 10
            for orientation in finger_orientations
        ]
    )
    return finger_straight and thumbs_normal_finger and abs(thumb_orientation - 90) < 10


mqttc = mqtt.Client()
mqttc.connect("localhost", 1883, 60)

# For webcam input:
cap = cv2.VideoCapture(1)
with mp_hands.Hands(
    model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5
) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        curr_des = None
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            if check_thumbs_up(hand_landmarks):
                curr_des = "START"
            elif check_fist(hand_landmarks):
                curr_des = "STOP"
            elif check_middle_finger_above_half_screen(hand_landmarks):
                curr_des = "UP"
            elif check_middle_finger_below_half_screen(hand_landmarks):
                curr_des = "DOWN"

            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

        if len(call_stack) == 10 and all(des == call_stack[0] for des in call_stack):
            action = call_stack[0] if call_stack[0] is not None else action
        cv2.putText(
            image,
            str(action),
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (
                0,
                255 if action != "STOP" else 0,
                0 if action != "STOP" else 255,
            ),
            2,
        )
        mqttc.publish("action", action)
        call_stack.append(curr_des)
        if len(call_stack) > 10:
            call_stack.pop(0)
        print(call_stack)
        cv2.imshow("MediaPipe Hands", image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
