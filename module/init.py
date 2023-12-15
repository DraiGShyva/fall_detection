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
    return VideoWriter(video_name, 0x7634706D, 30, (640, 480))


# Hàm khởi tạo file csv
def init_csv(file_name):
    csv_file = open(file_name, "w", newline="")  # Tạo file csv
    csv_writer = csv.writer(csv_file)  # Khởi tạo csv_writer
    csv_writer.writerow(
        [
            "chest_x",
            "chest_y",
            "chest_z",
            "belt_x",
            "belt_y",
            "belt_z",
            "left_ankle_x",
            "left_ankle_y",
            "left_ankle_z",
            "right_ankle_x",
            "right_ankle_y",
            "right_ankle_z",
        ]
    )  # Viết tiêu đề cho file csv
    return csv_writer
