
import av
from streamlit_webrtc import webrtc_streamer
import streamlit as st
from Home import face_rec
import time

st.set_page_config(page_title='Predictions')


st.subheader('Real-Time Attendance System')

# Retrieve data

with st.spinner("Retreiving data from Redis DB ..."):
    redis_face_db = face_rec.retreive_data(name="academy:register")
    st.dataframe(redis_face_db)
st.success("Data successfully retreived from Redis DB")

waitTime = 10
setTime = int(time.time())
realtimepred = face_rec.RealTimePred()


def video_frame_callback(frame):
    global setTime

    img = frame.to_ndarray(format="bgr24")

    pred_img = realtimepred.face_recognition(img, redis_face_db, 'facial_features', [
        'Name', 'Role'], thresh=0.5)

    timenow = int(time.time())
    difftime = timenow-setTime

    if difftime >= waitTime:
        realtimepred.saveLogs_redis()
        setTime = int(time.time())
        print('Save Data to redis database')

    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")


webrtc_streamer(key="realtimePrediction",
                video_frame_callback=video_frame_callback)
