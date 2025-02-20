import numpy as np
import pandas as pd

def extract_gaze_head_angle(data):
    """
    Trích xuất và tính toán góc lệch giữa hướng nhìn và tư thế đầu từ dữ liệu đầu vào.
    """
    # Kiểm tra xem các cột cần thiết có tồn tại không
    required_columns = ['gaze_angle_x', 'gaze_angle_y', 'pose_Rx', 'pose_Ry', 'pose_Rz']
    if not set(required_columns).issubset(data.columns):
        raise ValueError("Dữ liệu không chứa đủ các cột cần thiết.")

    # 1. Tạo vector hướng nhìn (Gaze_Vector)
    gaze_vector = np.array([
        np.mean(np.cos(data['gaze_angle_x'])),
        np.mean(np.sin(data['gaze_angle_y'])),
        0  # Giả sử hướng nhìn không có thành phần Z
    ])

    # 2. Tạo vector tư thế đầu (Head_Vector)
    head_vector = np.array([
        np.mean(np.cos(data['pose_Rx'])),
        np.mean(np.sin(data['pose_Ry'])),
        np.mean(np.sin(data['pose_Rz']))
    ])

    # 3. Tính góc lệch giữa hai vector
    dot_product = np.dot(gaze_vector, head_vector)
    norm_gaze = np.linalg.norm(gaze_vector)
    norm_head = np.linalg.norm(head_vector)
    cosine_angle = dot_product / (norm_gaze * norm_head)
    angle_radians = np.arccos(cosine_angle)
    angle_degrees = np.degrees(angle_radians)

    # Tạo DataFrame kết quả
    result = pd.DataFrame({
        'angle_between_gaze_and_head': [angle_degrees]
    })

    return result

# Ví dụ sử dụng
if __name__ == "__main__":
 
    data = pd.DataFrame({
        'gaze_angle_x': np.random.uniform(-30, 30, 300),  # Góc nhìn ngang
        'gaze_angle_y': np.random.uniform(-20, 20, 300),  # Góc nhìn dọc
        'pose_Rx': np.random.uniform(-10, 10, 300),       # Tư thế đầu (Yaw)
        'pose_Ry': np.random.uniform(-10, 10, 300),       # Tư thế đầu (Pitch)
        'pose_Rz': np.random.uniform(-10, 10, 300)        # Tư thế đầu (Roll)
    })

    # Trích xuất đặc trưng góc lệch
    gaze_head_angle = extract_gaze_head_angle(data)
    print(gaze_head_angle)