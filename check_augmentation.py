import cv2

img = "000452673_jpg.rf.691b39608109cde124380edb13c2a485__aug_01"

image_path = "dataset/augmented/train/images/"+img+".jpg"
label_path = "dataset/augmented/train/labels/"+img+".txt"

img = cv2.imread(image_path)
h, w = img.shape[:2]

with open(label_path, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        box_w = float(parts[3])
        box_h = float(parts[4])

        # YOLO normalized -> pixel coords
        x_center *= w
        y_center *= h
        box_w *= w
        box_h *= h

        x1 = int(x_center - box_w / 2)
        y1 = int(y_center - box_h / 2)
        x2 = int(x_center + box_w / 2)
        y2 = int(y_center + box_h / 2)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            img,
            str(class_id),
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

cv2.imshow("YOLO labels", img)
cv2.waitKey(0)
cv2.destroyAllWindows()