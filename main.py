import pandas as pd

flag = False
count = 0
data_list = []


def feature_extraction():
    from cv2 import imshow, waitKey, VideoCapture
    from module.pose_landmarker import extract_pose_features, draw

    cap = VideoCapture(0)
    df = []
    global flag, count, data_list

    while True:
        # Đọc frame từ webcam
        frame = cap.read()[1]

        # Trích xuất các điểm mốc
        landmarks = extract_pose_features(frame)

        list = [0] * 12

        # Nếu có điểm mốc thì vẽ các điểm mốc
        if landmarks is not None:
            frame = draw(frame, landmarks)

            # chuyển landmarks thành mảng 1 chiều
            list = [j for i in landmarks for j in i]

        # Thêm dữ liệu vào df
        df.append(list)

        try:
            print(count, len(df), len(data_list))
        except IndexError:
            pass

        if len(df) > 28:
            # Lấy mỗi 28 dòng dữ liệu đầu tiên và thêm vào data_list
            data_list.append([j for i in df[:28] for j in i])

            # Tăng biến đếm lên 1
            count += 1

            # Xóa 15 dòng dữ liệu đầu tiên
            df = df[15:]

        # Hiển thị kết quả
        imshow("Pose Detection", frame)

        # Nếu nhấn phím Esc thì dừng chương trình
        if waitKey(1) == 27:
            flag = True
            break


def predict():
    global flag, data_list, count
    # Khởi tạo model
    from pickle import load

    model = load(open("fall_detection/model/model.pkl", "rb"))
    count_2 = 0
    while flag is False:
        if count - count_2 * 5 == 5:
            count_2 += 1

            # Tạo bản sao của data_list chỉ lấy 5 dòng cuối cùng
            data_list_cp = pd.DataFrame(data_list[-5:])

            # Dự đoán
            print("Predicting...")
            result = model.predict(data_list_cp)

            # Nếu có 1 trong các dự đoán là ngã thì thông báo
            if 1 in result:
                print("Fall detected")


def main():
    import threading

    # Tạo thread để trích xuất đặc trưng
    feature = threading.Thread(target=feature_extraction)

    # Tạo thread để dự đoán
    pred = threading.Thread(target=predict)

    # Chạy 2 thread
    feature.start()
    pred.start()

    # Chờ thread kết thúc
    feature.join()

    print("Program stopped")


if __name__ == "__main__":
    main()
