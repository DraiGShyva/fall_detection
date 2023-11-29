from module.pose_landmarker import extract_pose_features, draw
from module.init import capture, imshow, waitKey, destroyAllWindows, init_writer, init_csv, name_folder

# Khởi tạo Webcam
cap = capture()

# Khởi tạo folder_name
import os
folder_name = name_folder("fall_detection/video/vid_") # Tên thư mục
os.makedirs(folder_name) # Tạo thư mục

# Khởi tạo video_writer
video_writer = init_writer(cap, folder_name)

# Khởi tạo csv_writer
csv_writer = init_csv(folder_name+"/data.csv")

# Vòng lặp chính
while cap.isOpened():

    # Đọc frame từ webcam
    frame = cap.read()[1]

    # Trích xuất các điểm mốc
    landmarks = extract_pose_features(frame)

    # Nếu có điểm mốc thì vẽ các điểm mốc và các đường nối
    if landmarks is not None:
        frame = draw(frame.copy(), landmarks)

        #chuyển landmarks thành mảng 1 chiều
        list = [j for i in landmarks for j in i]

        # Ghi dữ liệu xuống file csv
        csv_writer.writerow(list)

    # Hiển thị kết quả
    imshow('Pose Detection', frame)

    # Ghi frame xuống video
    video_writer.write(frame)

    # Nhấn phím ESC để thoát
    if waitKey(1) & 0xFF == 27: 
        cap.release()
        video_writer.release()
        destroyAllWindows()