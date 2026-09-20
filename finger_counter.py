import cv2
import mediapipe as mp
import math

# Initialize mediapipe hands
mp_hands = mp.solutions.hands
# max_num_hands defines the maximum number of hands to detect
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=2, 
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Finger tip landmark IDs
# 4: Thumb, 8: Index, 12: Middle, 16: Ring, 20: Pinky
tip_ids = [4, 8, 12, 16, 20]

# Initialize video capture (0 is usually the default webcam)
cap = cv2.VideoCapture(0)

print("Starting webcam... Press 'ESC' to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a selfie-view display
    # Convert the BGR image to RGB for MediaPipe processing
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    
    # Improve performance by marking image as not writeable
    image.flags.writeable = False
    results = hands.process(image)

    # Convert back to BGR for OpenCV display
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    total_fingers = 0
    
    if results.multi_hand_landmarks:
        for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Draw landmarks and connections on the image
            mp_draw.draw_landmarks(
                image, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS
            )
            
            fingers = []
            
            # Thumb checking using Euclidean distance
            # A thumb is considered open if the distance from the Thumb Tip (4) to the Pinky Knuckle (17) 
            # is greater than the distance from the Thumb MCP (2) to the Pinky Knuckle (17).
            lm4 = hand_landmarks.landmark[4]
            lm2 = hand_landmarks.landmark[2]
            lm17 = hand_landmarks.landmark[17]
            
            dist_tip_to_pinky = math.hypot(lm4.x - lm17.x, lm4.y - lm17.y)
            dist_mcp_to_pinky = math.hypot(lm2.x - lm17.x, lm2.y - lm17.y)
            
            if dist_tip_to_pinky > dist_mcp_to_pinky:
                fingers.append(1)
            else:
                fingers.append(0)
            
            # 4 Fingers checking (y-coordinate comparison)
            for id in range(1, 5):
                # If tip y is less than the second lower joint y, finger is considered open
                if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id] - 2].y:
                    fingers.append(1)
                else:
                    fingers.append(0)
                    
            total_fingers += fingers.count(1)
            
    # Display the total finger count on the screen
    cv2.putText(image, f'Fingers: {total_fingers}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    # Show the image
    cv2.imshow('Finger Counter', image)
    
    # Press 'ESC' to exit
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
