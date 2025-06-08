import numpy as np

def analyze_skin_tone(image, landmarks):
    h, w, _ = image.shape
    # Use cheek area for tone (landmark 234, 454)
    sample_points = [landmarks[234], landmarks[454]]
    colors = []
    for x, y in sample_points:
        px = int(x * w)
        py = int(y * h)
        color = image[py, px]
        colors.append(color)

    avg_color = np.mean(colors, axis=0)
    b, g, r = avg_color

    # Very basic tone logic
    if r > g and r - g > 15:
        return "Warm"
    elif b > r and b - r > 15:
        return "Cool"
    else:
        return "Neutral"
