import cv2
import mediapipe as mp
import numpy as np
import math
import time
import pygame
import threading

# Initialize Pygame and load a sound file ('sound.wav') for the posture alert
pygame.mixer.init()
sound = pygame.mixer.Sound('sound.wav')

played = 1

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Initialize MediaPipe DrawingSpec for drawing landmarks
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=2)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose

# Initialize webcam
cap = cv2.VideoCapture(0)

# Define the list of landmark indices to display
selected_landmarks = [157, 158, 159, 160, 161, 246, 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 190, 56, 28, 27, 29, 30, 247, 33, 130, 25, 110, 24, 23, 22, 26, 112, 243, 133]

# Define the indices of landmarks for right and left eyes
right_indices = [463, 362, 382, 381, 380, 374, 373, 390, 249, 263, 388, 387, 386, 385, 384, 398, 341, 256, 252, 253, 254, 339, 255, 359, 467, 260, 257, 258, 286, 414, 259]
left_indices = [157, 158, 159, 160, 161, 246, 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 190, 56, 28, 27, 29, 30, 247, 33, 130, 25, 110, 24, 23, 22, 26, 112, 243, 133]

# Initialize blink counter and blink flag
blink_counter = 0
blink_flag = False

# Set a cooldown period for blink detection (in seconds)
blink_cooldown = 1.0  # Adjust as needed

# Initialize the time of the last blink detection
last_blink_time = time.time()

# Initialize start time
start_time = time.time()

# Initialize last BPM update time as a global variable
last_bpm_update_time = time.time()

# Euclidean distance function
def euclideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

# Blink detection function
def detectBlink(img, landmarks, right_indices, left_indices):
    global blink_counter, blink_flag, last_blink_time

    # Calculate the blink ratio
    ratio = blinkRatio(img, landmarks, right_indices, left_indices)

    # Threshold to check for blinks
    blink_threshold = 3.35  # Adjust as needed

    # Check if the blink ratio is above the threshold
    if ratio > blink_threshold:
        current_time = time.time()
        # Check if enough time has passed since the last blink detection
        if current_time - last_blink_time >= blink_cooldown:
            blink_counter += 1
            last_blink_time = current_time
    else:
        blink_flag = False

    # Display the blink counter on the frame
    cv2.putText(img, f'Blinks: {blink_counter}', (350, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Blink ratio calculation function
def blinkRatio(img, landmarks, right_indices, left_indices):
    # Right eyes
    # horizontal line
    rh_right = landmarks[263]
    rh_left = landmarks[362]
    # vertical line
    rv_top = landmarks[386]
    rv_bottom = landmarks[374]

    # LEFT_EYE
    # horizontal line
    lh_right = landmarks[133]
    lh_left = landmarks[33]
    # vertical line
    lv_top = landmarks[159]
    lv_bottom = landmarks[145]

    # Finding Distance Right Eye
    rhDistance = euclideanDistance(rh_right, rh_left)
    rvDistance = euclideanDistance(rv_top, rv_bottom)
    # Finding Distance Left Eye
    lvDistance = euclideanDistance(lv_top, lv_bottom)
    lhDistance = euclideanDistance(lh_right, lh_left)

    # Handle division by zero error
    if rvDistance == 0 or lvDistance == 0:
        return 0  # Return 0 if the denominator is zero to avoid division by zero

    # Finding ratio of LEFT and Right Eyes
    reRatio = rhDistance / rvDistance
    leRatio = lhDistance / lvDistance
    ratio = (reRatio + leRatio) / 2
    return ratio

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = round(np.abs(radians * 180.0 / np.pi), 3)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Initialize variables for posture monitoring
perfect_posture = None
bad_posture_timer = None  # Timer for bad posture duration
bad_posture_alert = False  # Flag to indicate if bad posture alert is displayed
good_posture_start_time = None  # Timer for good posture duration
good_posture_duration = 0  # Duration of good posture

# Function to run the FaceMesh model
def run_face_mesh():
    global blink_counter, last_bpm_update_time
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate the elapsed time
        elapsed_time = time.time() - start_time
        elapsed_time_formatted = round(elapsed_time, 1)

        # Calculate the number of minutes that have passed
        minutes = elapsed_time / 60

        # Calculate the blink rate (blinks per minute)
        blink_rate = blink_counter / minutes

        # Convert BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and get the facial landmarks
        results = face_mesh.process(rgb_frame)

        # Calculate the pixel coordinates of landmarks
        if results.multi_face_landmarks:
            mesh_coord = [(int(point.x * frame.shape[1]), int(point.y * frame.shape[0])) for point in results.multi_face_landmarks[0].landmark]

            # Draw selected facial landmarks
            for idx, landmark in enumerate(results.multi_face_landmarks[0].landmark):
                if idx in selected_landmarks:
                    x, y = mesh_coord[idx]
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Call the blink detection function
            detectBlink(frame, mesh_coord, right_indices, left_indices)

        # Display the elapsed time and blink rate on the frame
        cv2.putText(frame, f'Time: {elapsed_time_formatted}s', (350, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Blink Rate: {blink_rate:.2f} ', (350, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame with the selected facial landmarks, blink counter, and elapsed time
        cv2.imshow('Blink Counter', frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Function to run the Pose model
def run_pose():
    global perfect_posture, bad_posture_timer, bad_posture_alert, good_posture_start_time, good_posture_duration
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
          
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                
                # Calculate angle
                angle = calculate_angle(shoulder, nose, wrist)
                
                # Visualize angle
                cv2.putText(image, str(angle), 
                            tuple(np.multiply(nose, [700, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Record angle when perfect posture is indicated
                if perfect_posture is None:
                    cv2.putText(image, 'Assume perfect posture', (15, 12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, 'Press "P" when ready', (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Check for key press to indicate readiness for perfect posture
                    key = cv2.waitKey(1)
                    if key == ord('p'):
                        perfect_posture = angle
                        bad_posture_timer = None  # Reset bad posture timer when perfect posture is set
                        good_posture_start_time = None  # Reset good posture timer when perfect posture is set
                        
                else:
                    if angle > (perfect_posture + 10):
                        counter = "Bad"
                        if bad_posture_timer is None:
                            bad_posture_timer = time.time()  # Start the timer for bad posture
                            good_posture_start_time = None  # Reset good posture timer
                        else:
                            bad_posture_duration = time.time() - bad_posture_timer
                            if bad_posture_duration >= 5.0:  # Check if bad posture is maintained for 5 seconds
                                bad_posture_alert = True
                                good_posture_start_time = None  # Reset good posture timer when bad posture is detected
                    else:
                        counter = "Good"
                        if good_posture_start_time is None:
                            good_posture_start_time = time.time()  # Start the timer for good posture
                            bad_posture_timer = None  # Reset bad posture timer when good posture is attained
                        else:
                            good_posture_duration = time.time() - good_posture_start_time
                            if good_posture_duration >= 2.0:  # Check if good posture is maintained for 3 seconds
                                bad_posture_alert = False  # Clear bad posture alert
                                played = 1
                    cv2.putText(image, 'Posture', (15, 12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(counter), 
                                (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    played #DO NOT DELETE, FIXES SOUND NOT PLAYING IN FIRST LOOP ISSUE
                    if bad_posture_alert:
                        cv2.putText(image, 'Bad Posture Alert!', (150, 240), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        if played == 1:
                            sound.play()
                            played = 0
                    
                    # Render detections
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
                                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2) 
                                             )               
            
            except:
                pass
            
            cv2.imshow('Posture Detector', image)

            key = cv2.waitKey(5)
            
            # Exit when 'q' is pressed
            if key == ord('q'):
                break

# Create threads for running FaceMesh and Pose models concurrently
face_mesh_thread = threading.Thread(target=run_face_mesh)
pose_thread = threading.Thread(target=run_pose)

# Start both threads
face_mesh_thread.start()
pose_thread.start()

# Wait for both threads to finish
face_mesh_thread.join()
pose_thread.join()

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Print the recorded angle after exiting the loop
if perfect_posture is not None:
    print(f"Perfect Posture Angle: {perfect_posture}")
else:
    print("No perfect posture angle recorded.")
