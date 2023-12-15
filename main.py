import threading
from pickle import load
import pandas as pd
from module.init import init
from cv2 import imshow, waitKey
from module.pose_landmarker import extract_pose_features, draw

flag = False
count = 0
df = []
data_list = []


def feature_extraction(cap, video):
    global flag, df, count, data_list

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

        # Bản sao của df (bỏ qua count * 15 dòng đầu tiên)
        df_cp = df[count * 15 :]

        # try:
        #     print(count, len(df_cp), len(data_list))
        # except IndexError:
        #     pass

        if len(df_cp) > 28:
            # Lấy mỗi 28 dòng dữ liệu đầu tiên và thêm vào data_list
            data_list.append([j for i in df_cp[:28] for j in i])  # type: ignore

            # Tăng biến đếm lên 1
            count += 1

        # Hiển thị kết quả
        imshow("Pose Detection", frame)

        # Ghi frame xuống video
        video.write(frame)

        # Nếu nhấn phím Esc thì dừng chương trình
        if waitKey(1) == 27:
            flag = True
            # Lưu dữ liệu vào file csv
            df = pd.DataFrame(df)
            df.to_csv("data.csv", index=False)
            break


def predict():
    global flag, data_list
    # Khởi tạo model
    model = load(open("fall_detection/model/model.pkl", "rb"))
    count_2 = 0
    while flag is False:
        if count - count_2 * 5 == 5:
            print("Predicting...")
            count_2 += 1

            # Tạo bản sao của data_list chỉ lấy 5 dòng cuối cùng
            data_list_cp = pd.DataFrame(data_list[-5:])

            # Dự đoán
            result = model.predict(data_list_cp)

            # Nếu có 1 trong các dự đoán là ngã thì thông báo
            if 1 in result:
                print("Fall detected")


def main():
    cap, video = init()

    # Tạo thread để trích xuất đặc trưng
    feature = threading.Thread(target=feature_extraction, args=(cap, video))

    # Tạo thread để dự đoán
    pred = threading.Thread(target=predict)

    # Chạy 2 thread
    feature.start()
    pred.start()

    # Chờ thread kết thúc
    feature.join()

    # Đóng webcam và video
    cap.release()
    video.release()

    print("Program stopped")


if __name__ == "__main__":
    main()
