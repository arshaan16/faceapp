import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer
import av
from Home import face_rec


st.set_page_config(page_title='Registration Form')

st.subheader('Registration Form')

registration_form = face_rec.RegistrationForm()

# step1 - Collect person name and role
person_name = st.text_input(label="Name", placeholder='First & Last Name')
role = st.selectbox(label="Select your role", options=('Student', 'Teacher'))


# step2 - Collect facial embeddings of that person
def video_callback_func(frame):
    img = frame.to_ndarray(format='bgr24')

    reg_img, embedding = registration_form.get_embeddings(img)

    if embedding is not None:
        with open('face_embedding.txt', mode='ab') as f:
            np.savetxt(f, embedding)

    return av.VideoFrame.from_ndarray(reg_img, format='bgr24')


webrtc_streamer(key='registration', video_frame_callback=video_callback_func)

# step3 -  save the data in redis database

if st.button('Submit'):
    return_val = registration_form.save_data_in_redis_db(person_name, role)
    if return_val == True:
        st.success(f"{person_name} registered successfully")
    elif return_val == "name_false":
        st.error("Please enter name: Name cannot be empty or spaces")
    elif return_val == "file_false":
        st.error(
            "face_embedding.txt not found. Please refresh the page and start again")
