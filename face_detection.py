import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
face_connections = mp_face_mesh.FACEMESH_TESSELATION

def get_landmarks(image):
    results = face_mesh.process(image)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        return [(lm.x, lm.y) for lm in landmarks.landmark]
    return None
