def predict(folder_name, model):
    import pandas as pd

    # Đọc dữ liệu từ file csv
    try:
        df = pd.read_csv(folder_name + "/data.csv")
    except pd.errors.EmptyDataError:
        df = pd.DataFrame()

    # Nếu dữ liệu lớn hơn 28 dòng
    if len(df) > 28:
        # Lấy 28 dòng cuối cùng và chuyển thành mảng 1 chiều
        data_list = [[j for i in df.tail(28).values for j in i]]

        # Dự đoán kết quả mỗi 8 frame
        result = model.predict(data_list)

        # Nếu kết quả là 1 thì hiển thị cảnh báo
        if result == 1:
            print("Fall Detected")


def main():
    from module.pose_landmarker import extract_pose_features, draw
    from module.init import (
        imshow,
        waitKey,
        destroyAllWindows,
        init,
    )

    cap, video, csv, model, folder_name = init()

    # Vòng lặp chính
    counter = 0

    while True:
        # Đọc frame từ webcam
        frame = cap.read()[1]

        # Trích xuất các điểm mốc
        landmarks = extract_pose_features(frame)

        list = []

        # Nếu có điểm mốc thì vẽ các điểm mốc
        if landmarks is not None:
            frame = draw(frame.copy(), landmarks)

            # chuyển landmarks thành mảng 1 chiều
            list = [j for i in landmarks for j in i]

        # Ghi dữ liệu xuống file csv
        csv.writerow(list)

        if counter == 10:
            predict(folder_name, model)
            counter = 0

        # Hiển thị kết quả
        imshow("Pose Detection", frame)

        # Ghi frame xuống video
        video.write(frame)

        # Nhấn phím ESC để thoát
        if waitKey(1) & 0xFF == 27:
            cap.release()
            video.release()
            destroyAllWindows()

        counter += 1


if __name__ == "__main__":
    main()
