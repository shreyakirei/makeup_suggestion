import streamlit as st
import cv2
import numpy as np
from utils.face_detection import get_landmarks
from utils.skin_tone import analyze_skin_tone
from utils.face_shape import classify_face_shape
from utils.makeup_overlay import apply_virtual_makeup

# Setup page
st.set_page_config(page_title="AI Makeup Suggestion", layout="wide")
st.title("💄 AI Makeup Suggestion App")
st.write("Live webcam-based skin tone and face shape analysis with smart makeup suggestions!")

run = st.checkbox('✅ Start Webcam')
FRAME_WINDOW = st.image([])

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize suggestion tracking
if 'last_suggestion' not in st.session_state:
    st.session_state.last_suggestion = ""

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("⚠️ Camera not working. Please check your webcam.")
        break

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get facial landmarks
    landmarks = get_landmarks(frame_rgb)

    if landmarks is not None:
        # Analyze features
        skin_tone = analyze_skin_tone(frame_rgb, landmarks)
        face_shape = classify_face_shape(landmarks)
        frame_rgb = apply_virtual_makeup(frame_rgb, landmarks, skin_tone, face_shape)

        # Unique key for current suggestion
        current_suggestion = skin_tone + face_shape

        # Update sidebar only if suggestion changed
        if st.session_state.last_suggestion != current_suggestion:
            st.sidebar.empty()
            st.sidebar.markdown("### 💡 Personalized Makeup Suggestions")

            # Skin tone suggestions
            if skin_tone == "Warm":
                st.sidebar.markdown("👩 **Skin Tone:** <span style='color:#e07a5f'>Warm</span>", unsafe_allow_html=True)
                st.sidebar.markdown("💄 Coral, bronze, peachy lipsticks.<br>✨ Gold or warm-toned eyeshadow.", unsafe_allow_html=True)

            elif skin_tone == "Cool":
                st.sidebar.markdown("👩 **Skin Tone:** <span style='color:#3b9ae1'>Cool</span>", unsafe_allow_html=True)
                st.sidebar.markdown("💄 Berry, rose, or plum lipsticks.<br>✨ Silver or icy tones for eyes.", unsafe_allow_html=True)

            else:
                st.sidebar.markdown("👩 **Skin Tone:** <span style='color:#9c27b0'>Neutral</span>", unsafe_allow_html=True)
                st.sidebar.markdown("💄 Mauve, nude, dusty pink lips.<br>✨ Both cool & warm eyeshadow work!", unsafe_allow_html=True)

            # Face shape suggestions
            st.sidebar.markdown(f"📐 **Face Shape:** <span style='color:#ff9800'>{face_shape}</span>", unsafe_allow_html=True)

            if face_shape == "Oval":
                st.sidebar.markdown("😊 Balanced face — try bold lips and winged eyeliner!", unsafe_allow_html=True)
            elif face_shape == "Round":
                st.sidebar.markdown("🎯 Use contour to elongate. Raised brows, angular lips help!", unsafe_allow_html=True)
            elif face_shape == "Square":
                st.sidebar.markdown("🌟 Soften edges with rounded brows. Try soft smokey eyes.", unsafe_allow_html=True)
            elif face_shape == "Heart":
                st.sidebar.markdown("💘 Add balance with contouring near temples. Gradient lips work great.", unsafe_allow_html=True)
            else:
                st.sidebar.markdown("✨ Explore freely! You're one of a kind.", unsafe_allow_html=True)

            # Save suggestion
            st.session_state.last_suggestion = current_suggestion

    # Show frame
    FRAME_WINDOW.image(frame_rgb)

else:
    cap.release()


   
