import cv2
import numpy as np

# === Utilities ===

def order_points(pts):
    """Order points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left
    return rect

def four_point_transform(image, pts):
    """Apply perspective warp."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def process_image(image_path, photo_name, output, answer_key_given, *args):
    # === Step 1: Read and preprocess ===

    # Use your uploaded image
    # image_path = 'input/test10.jpeg'  # Your uploaded file path
    image = cv2.imread(image_path)
    if image.shape[0] > image.shape[1]:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    image = cv2.resize(image, (1200, 550))  # Resize for faster processing
    original = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_valus = [0, 1, 3, 5]

    for bv in blur_valus:
        if bv == 0:
            blurred = gray
        else:
            blurred = cv2.GaussianBlur(gray, (bv, bv), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 19, 3)

 
        kernel_size = 2
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # cv2.imshow("ji", thresh)
        # cv2.waitKey(0)
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        columns = []

        # === Step 2: Detect rectangles ===
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:  # Only quadrilaterals
                area = cv2.contourArea(cnt)
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = h / float(w) if w != 0 else 0

                # Filter based on size and aspect ratio
                if 20000 < area < 300000 and 1.3 < aspect_ratio < 5.0:
                    columns.append(approx.reshape(4, 2))
                    # print(area)

        # Sort left to right
        columns = sorted(columns, key=lambda c: np.min(c[:, 0]))
        detected_columns = None
        if len(columns) == 5:
            detected_columns = True
            break
        else:
            detected_columns = False


    # print(f"Detected {len(columns)} columns.")
    if not detected_columns:
    # Draw rectangles found so far for visualization
        visual = original.copy()
        for i, points in enumerate(columns):
            points = points.reshape((-1,1,2)).astype(np.int32)
            cv2.polylines(visual, [points], isClosed=True, color=(0,0,255), thickness=5)  # RED for error

        # Add error message text at bottom
        h, w = visual.shape[:2]
        new_image = np.ones((h + 180, w, 3), dtype=np.uint8) * 255  # white background
        new_image[:h, :w] = visual

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        y_start = h + 40

        cv2.putText(new_image, f"ERROR: Columns not detected", (50, y_start), font, font_scale, (0, 0, 255), thickness)
        cv2.putText(new_image, f"FILE: {photo_name}", (50, y_start + 40), font, font_scale, (0, 0, 0), thickness)

        # Save to error folder
        import os
        os.makedirs("error", exist_ok=True)
        error_filename = f"error/{photo_name}_error.jpg"
        cv2.imwrite(error_filename, new_image)

        print(f"Saved {error_filename} due to column detection failure.")

        # Skip further processing for this image
        return 0, 0, 0, 0, "00", "Column Detection Failed"

    # === Step 3: Draw rectangles for visualization ===

    visual = original.copy()

    for i, points in enumerate(columns):
        points = points.reshape((-1,1,2)).astype(np.int32)
        cv2.polylines(visual, [points], isClosed=True, color=(0,255,0), thickness=5)
        cv2.putText(visual, f"Col {i+1}", tuple(points[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)


    cv2.imwrite("with_rectangles.jpg", visual)
    # print("Saved visualization with rectangles -> with_rectangles.jpg")

    # === Step 4: Warp each column separately ===

    for i, corner_points in enumerate(columns):
        warped = four_point_transform(original, corner_points)
        warped = cv2.resize(warped, (186, 450))  # Resize for faster processing

        cv2.imwrite(f"column_{i+1}.jpg", warped)
        # print(f"Saved warped column -> column_{i+1}.jpg")

    # print("\nAll Done! Now you can see the results.")
    import json

    # === Step 5: Draw green unfilled circles based on locations.txt ===

    # Load locations.txt
    with open('locations.txt', 'r') as f:
        locations = json.load(f)

    # Draw circles on each column image
    for i in range(1, 6):  # For columns 1 to 5
        img = cv2.imread(f"column_{i}.jpg")
        points = locations[str(i)]  # Get points for this column

        for (x, y) in points:
            cv2.circle(img, (x, y), 10, (0, 255, 0), thickness=2)  # Green, unfilled circle

        cv2.imwrite(f"column_{i}_with_circles.jpg", img)
        # print(f"Saved -> column_{i}_with_circles.jpg with circles.")

    # print("\nAll circles drawn successfully!")


    # === Step 6: Canny and Measure White Pixels at Circle Locations ===

    final_data = {}

    question_counter = 1  # Start question number from 1

    for col_num in range(1, 6):
        img = cv2.imread(f"column_{col_num}_with_circles.jpg", cv2.IMREAD_GRAYSCALE)

        # Apply Canny edge detection
        # edges = cv2.Canny(img, threshold1=50, threshold2=150)
        _, edges = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)
        # cv2.imshow("hi", edges)
        # cv2.waitKey(0)

        points = locations[str(col_num)]  # Get corresponding points

        for i in range(0, len(points), 4):  # Every 4 points (A, B, C, D)
            question_data = []
            option_labels = ["A", "B", "C", "D"]
            
            for j in range(4):
                idx = i + j
                if idx >= len(points):
                    break
                x, y = points[idx]

                # Define a small ROI around the (x, y)
                roi_size = 10  # 10x10 square
                x1, y1 = max(0, x - roi_size), max(0, y - roi_size)
                x2, y2 = min(edges.shape[1], x + roi_size), min(edges.shape[0], y + roi_size)
                roi = edges[y1:y2, x1:x2]

                white_pixels = cv2.countNonZero(roi)
                # mask = np.zeros_like(roi)
                # white_pixels = cv2.countNonZero(cv2.bitwise_and(roi, roi, mask=mask))

                question_data.append([option_labels[j], x, y, white_pixels])

            final_data[str(question_counter)] = question_data
            question_counter += 1

    # === Step 7: Print Final JSON ===
    # print("\n\n=== Final Data ===\n")
    # print(json.dumps(final_data))

    # === Step 8: Filter options based on white pixel threshold ===

    filtered_data = {}

    for q_no, options in final_data.items():
        selected_options = []
        for opt in options:
            label, x, y, white_pixel_value = opt
            if white_pixel_value > 170:
                selected_options.append(label)
        filtered_data[q_no] = selected_options

    # print("\n\n=== Filtered Data (options with white pixels > 200) ===\n")
    # print(json.dumps(filtered_data))

    # === Step 9: Load correct answers ===

    with open(answer_key_given, 'r') as f:
        answer_key = json.load(f)

    # === Step 10: Compare answers ===

    correct = 0
    incorrect = 0
    unattempted = 0

    for q_no in range(1, 51):
        q_no_str = str(q_no)
        detected = filtered_data.get(q_no_str, [])
        actual = answer_key.get(q_no_str, [])

        if not detected:
            unattempted += 1
        elif len(detected) == 1 and detected[0] in actual:
            correct += 1
        else:
            incorrect += 1

    # === Step 11: Print result ===

    print("\n\n=== Result Summary ===")
    print(f"Correct answers: {correct}")
    print(f"Incorrect answers: {incorrect}")
    print(f"Unattempted questions: {unattempted}")
    # === Step 12: Draw final evaluation circles, pink dot, and tick/cross marks ===

    final_visual = original.copy()

    column_contours = {i+1: columns[i] for i in range(len(columns))}

    for col_num in range(1, 6):
        points = locations[str(col_num)]

        for q_idx in range(0, len(points), 4):
            question_number = (col_num - 1) * (len(points) // 4) + (q_idx // 4) + 1
            q_no_str = str(question_number)

            detected = filtered_data.get(q_no_str, [])
            actual = answer_key.get(q_no_str, [])

            # Transformation matrix for this column
            src_pts = np.array([
                [0, 0],
                [186 - 1, 0],
                [186 - 1, 450 - 1],
                [0, 450 - 1]
            ], dtype="float32")
            try:
                dst_pts = order_points(column_contours[col_num])
            except:
                return
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)

            # Draw circles
            for j in range(4):
                idx = q_idx + j
                if idx >= len(points):
                    break
                x, y = points[idx]
                pts = np.array([[[x, y]]], dtype="float32")
                transformed = cv2.perspectiveTransform(pts, M)
                x_orig, y_orig = transformed[0][0]
                x_orig, y_orig = int(x_orig), int(y_orig)

                option_label = ["A", "B", "C", "D"][j]

                if not detected:
                    circle_color = (255, 0, 0)  # Blue
                    cv2.circle(final_visual, (x_orig, y_orig), 11, circle_color, thickness=4)

                elif len(detected) > 1:
                    if option_label in detected:
                        circle_color = (0, 0, 255)  # Red
                        cv2.circle(final_visual, (x_orig, y_orig), 11, circle_color, thickness=4)

                else:
                    if option_label == detected[0]:
                        if detected[0] in actual:
                            circle_color = (0, 255, 0)  # Green
                        else:
                            circle_color = (0, 0, 255)  # Red
                        cv2.circle(final_visual, (x_orig, y_orig), 11, circle_color, thickness=4)

            # Draw pink dot on correct option
            if actual:
                for correct_option in actual:
                    correct_idx = ["A", "B", "C", "D"].index(correct_option)
                    correct_x, correct_y = points[q_idx + correct_idx]
                    pts = np.array([[[correct_x, correct_y]]], dtype="float32")
                    transformed = cv2.perspectiveTransform(pts, M)
                    cx, cy = transformed[0][0]
                    cx, cy = int(cx), int(cy)
                    cv2.circle(final_visual, (cx, cy), 5, (255, 0, 255), thickness=-1)  # Pink dot

            # Draw Tick or Cross at small rectangle (right of D)
            d_idx = q_idx + 3  # D option
            if d_idx < len(points):
                x_d, y_d = points[d_idx]
                pts = np.array([[[x_d, y_d]]], dtype="float32")
                transformed = cv2.perspectiveTransform(pts, M)
                x_d_orig, y_d_orig = transformed[0][0]
                x_d_orig, y_d_orig = int(x_d_orig), int(y_d_orig)

                # Shift to right to reach rectangle box
                x_box, y_box = x_d_orig + 23, y_d_orig + 10

                # Decide Tick or Cross
                if not detected:
                    # No attempt -> no tick or cross
                    cv2.putText(final_visual, "-", (x_box, y_box), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 3)
                elif len(detected) > 1:
                    # Multiple answers selected -> Cross
                    cv2.putText(final_visual, "X", (x_box, y_box), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                else:
                    if detected[0] in actual:
                        # Correct -> Tick
                        cv2.putText(final_visual, "OK", (x_box, y_box), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 3)
                    else:
                        # Wrong -> Cross
                        cv2.putText(final_visual, "X", (x_box, y_box), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

    # Save final visual
    cv2.imwrite("final_result.jpg", final_visual)
  #  print("\nSaved final evaluated image -> final_result.jpg")


    # Suppose final_visual is your final output
    h, w = final_visual.shape[:2]

    # Create a new image: add 150px extra height
    new_image = np.ones((h + 180, w, 3), dtype=np.uint8) * 255  # white background

    # Paste the original image on top
    new_image[:h, :w] = final_visual

    # Now, write the result summary at bottom
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    y_start = h + 40

    cv2.putText(new_image, f"CORRECT: {correct}", (50, y_start), font, font_scale, (0, 200, 0), thickness)
    cv2.putText(new_image, f"INCORRECT: {incorrect}", (50, y_start + 40), font, font_scale, (0, 0, 255), thickness)
    cv2.putText(new_image, f"NOT ATTEMPTED: {unattempted}", (50, y_start + 80), font, font_scale, (255, 0, 0), thickness)
    cv2.putText(new_image, f"TOTAL: {correct + incorrect + unattempted}", (50, y_start + 120), font, font_scale, (0, 0, 0), thickness)
    cv2.putText(new_image, f"NAME: {photo_name}", (450, y_start), font, font_scale, (0, 0, 0), thickness)

    # Save final image
    cv2.imwrite(f"{output}/{photo_name}_final_result_with_summary.jpg", new_image)
    print(f"Saved {output}{photo_name}_final_result_with_summary.jpg",)
  
    # return f"{output}/final_result_with_summary.jpg"
    return correct + incorrect + unattempted, correct, unattempted, incorrect, "00", "None"