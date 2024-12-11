import cv2
import numpy as np
import time
import os
from datetime import datetime
import mediapipe as mp


# Function to calculate the dominant skin color in hex format
def estimate_skin_color(image):
    # Convert image to HSV (Hue, Saturation, Value) color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)  # Lower range for skin color
    upper_skin = np.array([25, 250, 255], dtype=np.uint8)  # Upper range for skin color

    # Mask the skin area
    skin_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)

    # Apply mask to the original image to isolate skin areas
    skin = cv2.bitwise_and(image, image, mask=skin_mask)

    # Get the mean color in BGR format
    mean_color = cv2.mean(skin, mask=skin_mask)[:3]

    # Convert BGR to RGB
    mean_color_rgb = tuple(int(x) for x in mean_color)

    # Convert RGB to hex format
    hex_color = '#{:02x}{:02x}{:02x}'.format(mean_color_rgb[2], mean_color_rgb[1], mean_color_rgb[0])
    return hex_color


# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize Mediapipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)  # Increase threshold for higher accuracy

# Cooldown timer
last_captured_time = 0
cooldown_seconds = 10
output_folder = "captured_faces"

# Create folder to save captured images
os.makedirs(output_folder, exist_ok=True)

# Open or create the skin-colors.txt file to write skin colors
skin_colors_file = os.path.join(output_folder, "skin-colors.txt")
if not os.path.exists(skin_colors_file):
    with open(skin_colors_file, 'w') as f:
        f.write("Skin Color Hex Codes:\n")  # Write header to the file

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Convert the BGR image to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    current_time = time.time()

    # Get the current date and time for filename and log
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # If faces are detected and cooldown has passed
    if results.detections and current_time - last_captured_time >= cooldown_seconds:
        for detection in results.detections:
            # Get bounding box for face detection
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            # Draw rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Crop the face tightly from the frame
            face_image = frame[y:y + h, x:x + w]

            # Resize the cropped face image to create a thumbnail (e.g., 100x100)
            thumbnail_size = (100, 100)
            face_thumbnail = cv2.resize(face_image, thumbnail_size)

            # Create the image file name with hex color
            hex_color = estimate_skin_color(face_image)
            image_filename = f"face_thumbnail_{timestamp}_{hex_color}.jpg"
            thumbnail_path = os.path.join(output_folder, image_filename)

            # Save the thumbnail
            cv2.imwrite(thumbnail_path, face_thumbnail)
            print(f"Face captured and saved as thumbnail at {thumbnail_path}")

            # Estimate the skin color from the face image
            skin_color_hex = estimate_skin_color(face_image)
            print(f"Estimated Skin Color: {skin_color_hex}")

            # Write the skin color hex code to skin-colors.txt with timestamp
            with open(skin_colors_file, 'a') as f:
                f.write(f"{timestamp} - {skin_color_hex}\n")

            # Update the cooldown timer after saving the image and writing the hex
            last_captured_time = current_time

    # Display the webcam feed
    cv2.imshow("Webcam", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()