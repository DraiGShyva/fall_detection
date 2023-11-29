# Hàm lấy thời gian hiện tại
def name_folder(name):
    import datetime
    return name+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# Hàm khởi tạo video_writer
def init_writer(cap, folder_name):
    from cv2 import CAP_PROP_FOURCC, CAP_PROP_FPS, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, VideoWriter

    video_name = f"{folder_name}/video.mp4" # Tên video
    fourcc = int(cap.get(CAP_PROP_FOURCC)) # Lấy codec của video
    fps = cap.get(CAP_PROP_FPS) # Lấy số frame trên giây của video
    size = (int(cap.get(CAP_PROP_FRAME_WIDTH)),int(cap.get(CAP_PROP_FRAME_HEIGHT))) # Lấy kích thước của video

    return VideoWriter(video_name, fourcc, fps, size)

# Hàm khởi tạo file csv
def init_csv(file_name):
    import csv
    csv_file = open(file_name, 'w', newline='') # Tạo file csv
    csv_writer = csv.writer(csv_file) # Khởi tạo csv_writer
    csv_writer.writerow(["chest_x", "chest_y", "chest_z", "belt_x", "belt_y", "belt_z", "left_ankle_x", "left_ankle_y", "left_ankle_z", "right_ankle_x",  "right_ankle_y", "right_ankle_z"]) # Viết tiêu đề cho file csv
    return csv_writer

# Hàm khởi tạo webcam
def capture():
    from cv2 import VideoCapture
    return VideoCapture(0)

from cv2 import destroyAllWindows, imshow, waitKey