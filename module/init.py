from cv2 import VideoCapture, VideoWriter
import os
import datetime
import csv


def init():
    # Khởi tạo Webcam
    cap = VideoCapture(0)

    # Khởi tạo folder_name
    folder_name = name_folder("fall_detection/video/vid_")  # Tên thư mục
    os.makedirs(folder_name)  # Tạo thư mục

    # Khởi tạo video_writer
    video_writer = init_writer(cap, folder_name)

    return cap, video_writer


# Hàm lấy thời gian hiện tại
def name_folder(name):
    return name + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


# Hàm khởi tạo video_writer
def init_writer(cap, folder_name):
    video_name = f"{folder_name}/video.mp4"
    return VideoWriter(
        video_name,
        0x7634706D,
        cap.get(5),
        (int(cap.get(3)), int(cap.get(4))),
    )


# Hàm khởi tạo file csv
def init_csv(file_name):
    csv_file = open(file_name, "w", newline="")  # Tạo file csv
    csv_writer = csv.writer(csv_file)  # Khởi tạo csv_writer
    csv_writer.writerow(
        [
            "nose_x",
            "nose_y",
            "nose_z",
            "left_shoulder_x",
            "left_shoulder_y",
            "left_shoulder_z",
            "right_shoulder_x",
            "right_shoulder_y",
            "right_shoulder_z",
            "left_elbow_x",
            "left_elbow_y",
            "left_elbow_z",
            "right_elbow_x",
            "right_elbow_y",
            "right_elbow_z",
            "left_wrist_x",
            "left_wrist_y",
            "left_wrist_z",
            "right_wrist_x",
            "right_wrist_y",
            "right_wrist_z",
            "left_hip_x",
            "left_hip_y",
            "left_hip_z",
            "right_hip_x",
            "right_hip_y",
            "right_hip_z",
            "left_knee_x",
            "left_knee_y",
            "left_knee_z",
            "right_knee_x",
            "right_knee_y",
            "right_knee_z",
            "left_ankle_x",
            "left_ankle_y",
            "left_ankle_z",
            "right_ankle_x",
            "right_ankle_y",
            "right_ankle_z",
        ]
    )  # Viết tiêu đề cho file csv
    return csv_writer
