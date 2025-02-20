import numpy as np
import pandas as pd
from scipy.fftpack import fft

def extract_blink_features(data):
   
    # Kiểm tra xem các cột cần thiết có tồn tại không
    required_columns = ['eye_lmk_x_27', 'eye_lmk_y_27', 'AU45_r']
    if not set(required_columns).issubset(data.columns):
        raise ValueError("Dữ liệu không chứa đủ các cột cần thiết.")

    # 1. Tần suất nháy mắt (Blink Rate)
    blink_count = data['AU45_r'].sum()  # Giả sử AU45_r > 0 khi nháy mắt
    time_window = len(data) / 30  # Giả sử tốc độ khung hình là 30 fps
    blink_rate = blink_count / time_window

    # 2. Cường độ nháy mắt (Blink Intensity)
    blink_intensity = data['AU45_r'].mean()

    # 3. Thời gian nháy mắt (Blink Duration)
    closed_eye_time = data[data['AU45_r'] > 0]['AU45_r'].count() / 30  # Thời gian mắt đóng
    blink_duration = closed_eye_time / blink_count if blink_count > 0 else 0

    # 4. Tần số nháy mắt (FFT trên Blink Rate)
    blink_rate_signal = data['AU45_r'].values
    fft_blink_rate = np.abs(fft(blink_rate_signal))

    # Tạo DataFrame kết quả
    result = pd.DataFrame({
        'blink_rate': [blink_rate],
        'blink_intensity': [blink_intensity],
        'blink_duration': [blink_duration],
        'fft_blink_rate': [np.mean(fft_blink_rate)]
    })

    return result

# Ví dụ sử dụng
if __name__ == "__main__":
    data = pd.DataFrame({
        'eye_lmk_x_27': np.random.uniform(50, 100, 300),  # Tọa độ X của mắt
        'eye_lmk_y_27': np.random.uniform(50, 100, 300),  # Tọa độ Y của mắt
        'AU45_r': np.random.choice([0, 1], size=300)      # Nháy mắt (0: Không nháy, 1: Nháy)
    })

    # Trích xuất đặc trưng nháy mắt
    blink_features = extract_blink_features(data)
    print(blink_features)