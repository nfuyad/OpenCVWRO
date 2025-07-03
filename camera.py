import cv2

# Use the video device path directly (change if needed)
video_device = "/dev/video5"

# Create VideoCapture object
cap = cv2.VideoCapture(video_device)

# Set resolution to 1920x1080 (Full HD)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 864)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Verify if camera opened successfully
if not cap.isOpened():
    print(f"Error: Cannot open camera at {video_device}")
    exit()

print("Press 'q' to quit.")

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Show the frame
    cv2.imshow('External Camera Feed - 1080p', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
