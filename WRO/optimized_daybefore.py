import cv2
import numpy as np

# Use the video device path directly (change if needed)
video_device = "/dev/video5"

# Create VideoCapture object
cap = cv2.VideoCapture(video_device)

# Set resolution to 1920x1080 (Full HD)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 864)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

def get_color_mask(hsv):
    # Red color has two HSV ranges
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])

    # Green HSV range
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([85, 255, 255])

    # Red masks
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Green mask
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    return red_mask, green_mask

def find_boxes(mask, color_name, frame):
    boxes = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:  # Eliminate noise
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)

        # Accept roughly rectangular objects only
        if 0.0009 < aspect_ratio < 1.5:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0) if color_name == "green" else (0, 0, 255), 2)
            boxes.append((x, y, w, h, color_name))

    return boxes

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    red_mask, green_mask = get_color_mask(hsv)
    red_boxes = find_boxes(red_mask, "red", frame)
    green_boxes = find_boxes(green_mask, "green", frame)

    all_boxes = red_boxes + green_boxes

    # Find closest box by bottom line
    closest_box = None
    max_bottom = -1
    for (x, y, w, h, color) in all_boxes:
        bottom_line = y + h
        if bottom_line > max_bottom:
            max_bottom = bottom_line
            closest_box = (x, y, w, h, color)

    if closest_box:
        x, y, w, h, color = closest_box
        bottom = y + h
        center_x = x + w // 2
        frame_width = frame.shape[1]

        pseudo_distance = frame.shape[0] - bottom  # lower value = closer

        # Draw center point
        cv2.circle(frame, (center_x, y + h // 2), 5, (255, 255, 255), -1)
        cv2.putText(frame, f"Closest: {color}, pseudo-dist={pseudo_distance}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Check position
        if center_x > frame_width // 4:
            cv2.putText(frame, "Move camera RIGHT", (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)
            print(">> Move camera RIGHT")
        else:
            cv2.putText(frame, "Box in left fourth", (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    cv2.imshow("Red Mask", red_mask)
    cv2.imshow("Green Mask", green_mask)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
