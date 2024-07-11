import cv2
import mediapipe as mp

# Initialize mediapipe drawing utils and solutions
mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands

# Capture video from the webcam
cap = cv2.VideoCapture(0)

# Initialize face detection and hand landmark detection models
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to count the number of extended fingers
def count_fingers(hand_landmarks, hand_label):
    finger_tips = [4, 8, 12, 16, 20]
    count = 0

    # Thumb
    if hand_label == 'Left' and hand_landmarks.landmark[finger_tips[0]].x < hand_landmarks.landmark[finger_tips[0] - 1].x:
        count += 1
    if hand_label == 'Right' and hand_landmarks.landmark[finger_tips[0]].x > hand_landmarks.landmark[finger_tips[0] - 1].x:
        count += 1

    # Other fingers
    for tip in finger_tips[1:]:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            count += 1

    return count

# Function to determine the direction of the face relative to the frame center
def detect_face_direction(face_box, frame_width, frame_height):
    face_center_x = (face_box.xmin + face_box.width / 2) * frame_width
    face_center_y = (face_box.ymin + face_box.height / 2) * frame_height
    frame_center_x = frame_width / 2
    frame_center_y = frame_height / 2

    if face_center_x < frame_center_x - 50:
        direction = "Left"
    elif face_center_x > frame_center_x + 50:
        direction = "Right"
    elif face_center_y < frame_center_y - 50:
        direction = "Up"
    elif face_center_y > frame_center_y + 50:
        direction = "Down"
    else:
        direction = "Center"
    
    return direction

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape

        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image for face detection
        face_results = face_detection.process(image)

        # Process the image for hand detection
        hand_results = hands.process(image)

        # Draw face detection results and determine face direction
        if face_results.detections:
            for detection in face_results.detections:
                mp_drawing.draw_detection(frame, detection)
                face_box = detection.location_data.relative_bounding_box
                face_direction = detect_face_direction(face_box, frame_width, frame_height)
                cv2.putText(frame, f'Face Direction: {face_direction}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_CONSTANT,value=[255,0,0])
        # Draw hand landmarks and detect gestures
        if hand_results.multi_hand_landmarks:
            for hand_landmarks, hand_world_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_hand_world_landmarks, hand_results.multi_handedness):
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                hand_label = handedness.classification[0].label
                fingers_count = count_fingers(hand_landmarks, hand_label)
                cv2.putText(frame, f'Fingers: {fingers_count}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display the output
        cv2.imshow('Face and Hand Gesture Detection by Mahmood', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    face_detection.close()
    hands.close()
