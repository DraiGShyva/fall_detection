from cv2 import cvtColor, circle
from mediapipe.python.solutions.pose import Pose

# tạo đối tượng Pose
mp_pose = Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1,
    static_image_mode=False,
)
# tạo list các điểm mốc cần thiết
n_landmarks = [[11, 12], [23, 24], 27, 28]


# Hàm trích xuất đặc trưng từ ảnh
def extract_pose_features(frame):
    # chuyển ảnh sang RGB
    frame_rgb = cvtColor(frame, 4)

    # xử lý ảnh
    results = mp_pose.process(frame_rgb).pose_landmarks  # type: ignore

    # nếu không có kết quả thì trả về landmarks với tất cả các tọa độ bằng 0
    if results is None:
        return [[0, 0, 0] for _ in range(len(n_landmarks))]

    # nếu điểm nào không nhận diện được thì gán tọa độ (0, 0, 0)
    for i in range(len(results.landmark)):
        if results.landmark[i].visibility < 0:
            results.landmark[i].x = 0
            results.landmark[i].y = 0
            results.landmark[i].z = 0

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

    return landmarks


# Hàm vẽ các điểm mốc và các đường nối
def draw(frame, landmarks):
    # lấy kích thước ảnh
    frame_height, frame_width, _ = frame.shape

    # vẽ các điểm mốc
    for landmark in landmarks:
        circle(
            frame,
            (int(landmark[0] * frame_width), int(landmark[1] * frame_height)),
            5,
            (0, 0, 255),
            -1,
        )

    return frame
