import cv2
import mediapipe as mp
import pyautogui

# Initialize the MediaPipe hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Webcam input
cap = cv2.VideoCapture(0)

screen_width, screen_height = pyautogui.size()

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the position of the index finger tip (landmark 8)
            index_finger_tip = hand_landmarks.landmark[8]
            x = int(index_finger_tip.x * screen_width)
            y = int(index_finger_tip.y * screen_height)

            # Move the mouse to the position of the index finger tip
            pyautogui.moveTo(x, y)

            # Get the position of the thumb tip (landmark 4)
            thumb_tip = hand_landmarks.landmark[4]
            thumb_x = int(thumb_tip.x * screen_width)
            thumb_y = int(thumb_tip.y * screen_height)

            # Calculate the distance between the index finger tip and thumb tip
            distance = ((x - thumb_x) ** 2 + (y - thumb_y) ** 2) ** 0.5

            # If the distance is less than a threshold, simulate a mouse click
            if distance < 50:  # Adjust the threshold as needed
                pyautogui.click()

    cv2.imshow("Virtual Mouse", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
