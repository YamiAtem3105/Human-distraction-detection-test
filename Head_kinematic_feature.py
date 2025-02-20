import numpy as np
import pandas as pd
from scipy.fftpack import fft

def extract_head_kinematics_features(data, fps=30):
    """
    Trích xuất các đặc trưng động học của đầu từ dữ liệu góc quay đầu.
    - data: DataFrame chứa các cột pose_Rx, pose_Ry, pose_Rz (Yaw, Pitch, Roll).
    - fps: Tần số khung hình (frames per second) của video.
    """
    # Kiểm tra xem các cột cần thiết có tồn tại không
    required_columns = ['pose_Rx', 'pose_Ry', 'pose_Rz']
    if not set(required_columns).issubset(data.columns):
        raise ValueError("Dữ liệu không chứa đủ các cột cần thiết.")

    # 1. Tính tốc độ góc (Angular Velocity)
    angular_velocity_x = np.gradient(data['pose_Rx'], 1 / fps)
    angular_velocity_y = np.gradient(data['pose_Ry'], 1 / fps)
    angular_velocity_z = np.gradient(data['pose_Rz'], 1 / fps)

    # 2. Tính gia tốc góc (Angular Acceleration)
    angular_acceleration_x = np.gradient(angular_velocity_x, 1 / fps)
    angular_acceleration_y = np.gradient(angular_velocity_y, 1 / fps)
    angular_acceleration_z = np.gradient(angular_velocity_z, 1 / fps)

    # 3. Tính tần số dao động (Oscillation Frequency) bằng FFT
    def compute_fft(signal):
        N = len(signal)  # Số điểm trong tín hiệu
        frequencies = np.fft.fftfreq(N, 1 / fps)  # Tần số tương ứng
        fft_values = fft(signal)  # Biến đổi Fourier
        magnitude = np.abs(fft_values)  # Độ lớn của FFT
        dominant_frequency = frequencies[np.argmax(magnitude[:N // 2])]  # Tần số chính
        return dominant_frequency

    dominant_freq_x = compute_fft(data['pose_Rx'])
    dominant_freq_y = compute_fft(data['pose_Ry'])
    dominant_freq_z = compute_fft(data['pose_Rz'])

    # 4. Chuẩn hóa dữ liệu (Z-score Normalization)
    def z_score_normalization(values):
        mean_value = np.mean(values)
        std_value = np.std(values)
        normalized_values = (values - mean_value) / std_value
        return normalized_values

    normalized_velocity_x = z_score_normalization(angular_velocity_x)
    normalized_velocity_y = z_score_normalization(angular_velocity_y)
    normalized_velocity_z = z_score_normalization(angular_velocity_z)

    normalized_acceleration_x = z_score_normalization(angular_acceleration_x)
    normalized_acceleration_y = z_score_normalization(angular_acceleration_y)
    normalized_acceleration_z = z_score_normalization(angular_acceleration_z)

    # 5. Tạo DataFrame kết quả
    result = pd.DataFrame({
        'normalized_angular_velocity_x': normalized_velocity_x,
        'normalized_angular_velocity_y': normalized_velocity_y,
        'normalized_angular_velocity_z': normalized_velocity_z,
        'normalized_angular_acceleration_x': normalized_acceleration_x,
        'normalized_angular_acceleration_y': normalized_acceleration_y,
        'normalized_angular_acceleration_z': normalized_acceleration_z,
        'dominant_frequency_x': dominant_freq_x,
        'dominant_frequency_y': dominant_freq_y,
        'dominant_frequency_z': dominant_freq_z
    })

    return result

# Ví dụ sử dụng
if __name__ == "__main__":
   
    data = pd.DataFrame({
        'pose_Rx': np.random.uniform(-30, 30, 300),  # Góc quay ngang (Yaw)
        'pose_Ry': np.random.uniform(-20, 20, 300),  # Góc quay dọc (Pitch)
        'pose_Rz': np.random.uniform(-15, 15, 300)   # Góc nghiêng (Roll)
    })

    # Trích xuất đặc trưng động học của đầu
    head_kinematics_features = extract_head_kinematics_features(data)
    print(head_kinematics_features.head())