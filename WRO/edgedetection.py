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

# Distance estimation equation from height
def estimate_distance(h):
    # Your earlier formula (can adjust if needed)
    return h * h * 0.001543 - 1.313 * h + 324.53

# HSV mask setup
def get_color_mask(hsv):
    # Red
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])
    red_mask = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1),
                              cv2.inRange(hsv, lower_red2, upper_red2))

    # Green
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    return red_mask, green_mask

# Find only rectangular objects using Canny + polygon filtering
def find_boxes(mask, color_name, frame):
    boxes = []

    # Blur then Canny
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Contours from Canny edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue

        # Approximate contour shape
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Only consider 4-sided shapes
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)

            # Filter only boxy rectangles (avoid long banners etc.)
            if 0.5 < aspect_ratio < 2.0:
                cv2.drawContours(frame, [approx], -1, (0, 255, 0) if color_name == "green" else (0, 0, 255), 2)
                boxes.append((x, y, w, h, color_name))

    return boxes

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get red/green masks
    red_mask, green_mask = get_color_mask(hsv)

    # Find rectangular boxes
    red_boxes = find_boxes(red_mask, "red", frame)
    green_boxes = find_boxes(green_mask, "green", frame)
    all_boxes = red_boxes + green_boxes

    # Find the closest box based on bottom line
    closest_box = None
    max_bottom = -1
    for (x, y, w, h, color) in all_boxes:
        bottom = y + h
        if bottom > max_bottom:
            max_bottom = bottom
            closest_box = (x, y, w, h, color)

    if closest_box:
        x, y, w, h, color = closest_box
        center_x = x + w // 2
        frame_width = frame.shape[1]
        bottom = y + h
        distance = estimate_distance(h)

        # Draw bounding box and text
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.putText(frame, f"{color.upper()} BOX", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Distance: {int(distance)} cm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Camera movement logic
        if center_x > frame_width // 4:
            cv2.putText(frame, "Move Camera RIGHT", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            print(">> Move camera RIGHT")
        else:
            cv2.putText(frame, "Box in LEFT area", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show results
    cv2.imshow("Detection", frame)
    cv2.imshow("Red Mask", red_mask)
    cv2.imshow("Green Mask", green_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
