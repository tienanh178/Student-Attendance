import streamlit as st
from Home import face_rec 
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer
import av

st.subheader('Registration Form')

registration_form = face_rec.RegistrationForm() 

# Step 1: Collect person name and code
person_name = st.text_input(label='Name', placeholder='First and Last Name')
person_code = st.text_input(label='Code', placeholder='Student code')

# Step 2: Collect facial embedding of that person
def video_callback_func(frame):
  img = frame.to_ndarray(format='bgr24') # 3d array bgr
  reg_img, embedding = registration_form.get_embedding(img)
  
  # Save data into local computer txt
  if embedding is not None:
    with open('face_embedding.txt', mode='ab') as f:
      np.savetxt(f,embedding)

  return av.VideoFrame.from_ndarray(reg_img, format='bgr24')

webrtc_streamer(key='registration', video_frame_callback=video_callback_func)

# Step 3: Save the data in database

if st.button('Submit'):
  return_val = registration_form.save_data_in_postgres_db(person_name,person_code)
  if return_val == True:
    st.success(f'{person_name} registered successfully!')
  elif return_val == 'name_false':
    st.error('Please enter the name or the student_code!')
  elif return_val == 'file_false':
    st.error('face_embedding.txt is not found, please refresh the page or execute again!')
