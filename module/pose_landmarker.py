from cv2 import cvtColor, circle, line
from mediapipe.python.solutions.pose import Pose

# tạo đối tượng Pose
mp_pose = Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1,
    static_image_mode=False,
)
# tạo list các điểm mốc cần thiết
n_landmarks = [0, [11, 12], 13, 14, 15, 16, [23, 24], 25, 26, 27, 28]

# tạo list các điểm cần nối
connections = [
    [0, 1],
    [1, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [1, 6],
    [6, 7],
    [6, 8],
    [7, 9],
    [8, 10],
]


# Hàm trích xuất đặc trưng từ ảnh
def extract_pose_features(frame):
    # chuyển ảnh sang RGB
    frame_rgb = cvtColor(frame, 4)

    # xử lý ảnh
    results = mp_pose.process(frame_rgb).pose_landmarks  # type: ignore

    # nếu không có kết quả thì trả về landmarks với tất cả các tọa độ bằng 0
    if results is None:
        return [[0, 0, 0] for _ in range(len(n_landmarks))]

    # nếu có kết quả thì trích xuất tọa độ các điểm mốc theo tên và lưu vào list
    landmarks = []
    for i in n_landmarks:
        if type(i) == list:
            landmark1 = results.landmark[i[0]]
            landmark2 = results.landmark[i[1]]
            x = (landmark1.x + landmark2.x) / 2
            y = (landmark1.y + landmark2.y) / 2
            z = (landmark1.z + landmark2.z) / 2
            landmarks.append([x, y, z])
        else:
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

    return landmarks


# Hàm vẽ các điểm mốc và các đường nối
def draw(frame, landmarks):
    # lấy kích thước ảnh
    frame_height, frame_width, _ = frame.shape

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
