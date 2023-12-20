from pickle import load
import pandas as pd
import threading

# Khởi tạo model
model = load(open("fall_detection/model/model.pkl", "rb"))

# Khởi tạo biến toàn cục
data_list = []


def feature_extraction():
    from cv2 import imshow, waitKey, VideoCapture, destroyAllWindows
    from module.pose_landmarker import extract_pose_features, draw

    cap, df, count = VideoCapture(0), [], 0
    global data_list

    while True:
        # Đọc frame từ webcam
        frame = cap.read()[1]

        # Trích xuất các điểm mốc
        landmarks = extract_pose_features(frame)

        # Nếu có điểm mốc thì vẽ các điểm mốc
        if landmarks is not None:
            frame = draw(frame, landmarks)

            # chuyển landmarks thành mảng 1 chiều và thêm vào df
            df.append([j for i in landmarks for j in i])

        # try:
        #     print(count, len(df), len(data_list))
        # except IndexError:
        #     pass

        if len(df) > 30:
            # Lấy mỗi 30 dòng dữ liệu đầu tiên chuyển thành mảng 1 chiều và thêm vào data_list
            data_list.append([j for i in df[:30] for j in i])

            # Tăng biến đếm lên 1
            count += 1

            # Xóa 5 dòng dữ liệu đầu tiên
            df = df[1:]

            # Mỗi 5s thì dự đoán 1 lần
            if count % 30 == 0 and count != 0:
                # Tạo thread để dự đoán
                pred = threading.Thread(target=predict)
                pred.start()

        # Hiển thị kết quả
        imshow("Pose Detection", frame)

        # Nếu nhấn phím Esc thì dừng chương trình
        if waitKey(1) == 27 or cap.read()[0] == False:
            print("Program stopped")
            cap.release()
            break


def predict():
    global data_list, model

    # Lấy 30 dòng dữ liệu cuối cùng
    data = pd.DataFrame(data_list[-30:])

    # Nếu dòng nào có trên 1/3 các giá trị bằng -1 hoặc bằng 0 thì xóa dòng đó
    data = data.drop(data[(data == -1).sum(axis=1) > 0.33 * data.shape[1]].index)
    data = data.drop(data[(data == 0).sum(axis=1) > 0.33 * data.shape[1]].index)

    # Nếu data không có dòng nào thì return
    if data.shape[0] == 0:
        print("No data")
        return

    # Dự đoán
    result = model.predict(data)

    # Nếu có 1 trong các dự đoán là ngã thì thông báo
    if 1 in result:
        print("Fall detected")
        # print(result)
    else:
        print("No fall detected")


def main():
    # Bắt đầu trích xuất đặc trưng
    feature_extraction()


if __name__ == "__main__":
    main()
