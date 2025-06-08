import numpy as np

def classify_face_shape(landmarks):
    jaw_left = np.array(landmarks[234])
    jaw_right = np.array(landmarks[454])
    chin = np.array(landmarks[152])
    forehead = np.array(landmarks[10])

    jaw_width = np.linalg.norm(jaw_right - jaw_left)
    face_length = np.linalg.norm(forehead - chin)

    ratio = face_length / jaw_width

    if ratio > 1.6:
        return "Oval"
    elif abs(ratio - 1.0) < 0.2:
        return "Round"
    elif jaw_width > face_length:
        return "Square"
    else:
        return "Heart"
