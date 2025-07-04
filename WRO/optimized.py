import cv2
import numpy as np

# Use the video device path directly (change if needed)
video_device = "/dev/video5"

cap = cv2.VideoCapture(video_device)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 864)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

def estimate_distance(h):
    return h * h * 0.001543 - 1.313 * h + 324.53

def get_color_mask(hsv):
    # Red
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])
    red_mask = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1),
                              cv2.inRange(hsv, lower_red2, upper_red2))

    # Green
    lower_green = np.array([36, 100, 50])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    return red_mask, green_mask

# === Find good object candidates ===
def find_boxes(mask, color_name, frame):
    boxes = []

    # Step 1: Boost brightness for low light
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    edges = cv2.Canny(mask, 50, 150)

    # Step 2: Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue

        # Try to approximate shape
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        side_count = len(approx)

        if 4 <= side_count <= 8:  # Accept slightly curved/hexed shapes
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)

            if 0.4 < aspect_ratio < 2.5:
                cv2.drawContours(frame, [approx], -1,
                                 (0, 255, 0) if color_name == "green" else (0, 0, 255), 2)
                boxes.append((x, y, w, h, color_name))

    return boxes

# === Main Loop ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # === Brightness Boost for Low Light ===
    # frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=40)


    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Color masks
    red_mask, green_mask = get_color_mask(hsv)

    # Detect boxes
    red_boxes = find_boxes(red_mask, "red", frame)
    green_boxes = find_boxes(green_mask, "green", frame)
    all_boxes = red_boxes + green_boxes

    # Pick closest box (lowest bottom y)
    closest_box = None
    max_bottom = -1
    for (x, y, w, h, color) in all_boxes:
        bottom = y + h
        if bottom > max_bottom:
            max_bottom = bottom
            closest_box = (x, y, w, h, color)

    # === Draw and Calculate ===
    if closest_box:
        x, y, w, h, color = closest_box
        center_x = x + w // 2
        distance = estimate_distance(h)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.putText(frame, f"{color.upper()} BOX", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Distance: {int(distance)} cm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Camera turning suggestion
        # if center_x > frame.shape[1] // 4:
        #     cv2.putText(frame, "Move Camera RIGHT", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        #     print(">> Move camera RIGHT")
        # else:
        #     cv2.putText(frame, "Box in LEFT area", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        #Improved Camera Suggestion
        if color == "red":
            if center_x > frame.shape[1] // 4:
                cv2.putText(frame, "Move Camera RIGHT", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print(">> Move camera RIGHT")
            else:
                cv2.putText(frame, "Box in LEFT area", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        elif color == "green":
            print("Found Green")
            if center_x < frame.shape[1] * (3 // 4):
                cv2.putText(frame, "Move Camera LEFT", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print(">> Move camera LEFT")


    # Show output
    cv2.imshow("Detection", frame)
    cv2.imshow("Red Mask", red_mask)
    cv2.imshow("Green Mask", green_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
