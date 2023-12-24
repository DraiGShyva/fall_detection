from cv2 import cvtColor, circle, line
from mediapipe.python.solutions.pose import Pose
import cv2

# tạo đối tượng Pose
mp_pose = Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1,
    enable_segmentation=True,
)

# tạo list các điểm mốc cần thiết
n_landmarks = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# tạo list các điểm cần nối
connections = [
    # thân
    [1, 2],
    [1, 7],
    [2, 8],
    [7, 8],
    # tay trái
    [1, 3],
    [3, 5],
    # tay phải
    [2, 4],
    [4, 6],
    # chân trái
    [7, 9],
    [9, 11],
    # chân phải
    [8, 10],
    [10, 12],
]


# Hàm trích xuất đặc trưng từ ảnh
def extract_pose_features(frame, segment=True):
    # chuyển ảnh sang RGB và resize về kích thước 640x480
    frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (640, 480))

    # lấy kết quả từ Pose
    results = mp_pose.process(frame).pose_landmarks  # type: ignore

    # nếu không có kết quả thì trả về landmarks với tất cả các tọa độ bằng 0
    if results is None:
        return [[0, 0, 0] for _ in range(len(n_landmarks))]

    # nếu có kết quả thì trích xuất tọa độ các điểm mốc theo tên và lưu vào list
    landmarks = []
    for i in n_landmarks:
        landmark = results.landmark[i]
        x = landmark.x
        y = landmark.y
        z = landmark.z
        landmarks.append([x, y, z])

    # nếu điểm nào có x, y <0 hoặc >1 thì gán bằng -1
    for i in range(len(landmarks)):
        if (
            landmarks[i][0] < 0
            or landmarks[i][0] > 1
            or landmarks[i][1] < 0
            or landmarks[i][1] > 1
        ):
            landmarks[i][0] = -1
            landmarks[i][1] = -1

    # nếu segment=True thì trả về (landmarks, segments)
    if segment:
        segments = mp_pose.process(frame).segmentation_mask  # type: ignore
        return landmarks, segments
    else:
        return landmarks


# Hàm vẽ các điểm mốc và các đường nối
def draw(frame, landmarks, segments):
    # lấy kích thước ảnh
    frame_height, frame_width, _ = frame.shape

    # vẽ segmentation
    frame[segments == 1] = [0, 0, 0]

    # vẽ các điểm mốc
    for landmark in landmarks:
        if landmark[0] != -1 and landmark[1] != -1:
            circle(
                frame,
                (int(landmark[0] * frame_width), int(landmark[1] * frame_height)),
                5,
                (0, 0, 255),
                -1,
            )

    # vẽ các đường nối theo connections
    for connection in connections:
        if (
            landmarks[connection[0]][0] != -1
            and landmarks[connection[0]][1] != -1
            and landmarks[connection[1]][0] != -1
            and landmarks[connection[1]][1] != -1
        ):
            line(
                frame,
                (
                    int(landmarks[connection[0]][0] * frame_width),
                    int(landmarks[connection[0]][1] * frame_height),
                ),
                (
                    int(landmarks[connection[1]][0] * frame_width),
                    int(landmarks[connection[1]][1] * frame_height),
                ),
                (0, 0, 255),
                3,
            )

    return frame
