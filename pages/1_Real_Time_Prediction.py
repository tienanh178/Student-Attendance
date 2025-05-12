import re
from streamlit_webrtc import webrtc_streamer
import streamlit as st
from Home import face_rec
import av
import time

st.subheader('Real Time Prediction')
# Retrieve the data from Database
with st.spinner('Retrieving data from PostgreSQL DB...'):
  face_db = face_rec.retrieve_data()
  st.dataframe(face_db)

st.success('Data successfully retrieve from PostgreSQL')

# time
waitTime = 10 # time in sec
setTime = time.time()
realTimePred = face_rec.RealTimePred()

# Real Time Prediction
# Streamlit webrtc

# Callback function
def video_frame_callback(frame):
    global setTime

    img = frame.to_ndarray(format="bgr24") #3d numpy array
    pred_img = realTimePred.face_prediction(img, face_db,'facial_features', ['Name','student_code'], thresh=0.5)

    timeNow = time.time()
    diffTime = timeNow - setTime

    if diffTime >= waitTime:
       realTimePred.saveLogs_db()
       setTime = time.time()

       print('Save data to  database') 
 
    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")

webrtc_streamer(key="realTimePrediction", video_frame_callback=video_frame_callback)


