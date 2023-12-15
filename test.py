from cv2 import imshow, waitKey
from module.init import init
from module.pose_landmarker import extract_pose_features, draw

# Khởi tạo webcam
cap, d = init()

while True:
    # Đọc frame từ webcam
    frame = cap.read()[1]

    # Trích xuất các điểm mốc
    landmarks = extract_pose_features(frame)

    # Nếu có điểm mốc thì vẽ các điểm mốc
    if landmarks is not None:
        frame = draw(frame, landmarks)

    # Hiển thị kết quả
    imshow("Pose Detection", frame)

    # Nếu nhấn phím Esc thì dừng chương trình
    if waitKey(1) == 27:
        break
