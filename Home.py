#Python 3.10
import streamlit as st;
st.set_page_config('Attendance System', layout='centered')

st.header('Attendance System using Face Recognition')

with st.spinner('Loading models and connecting to database... '):
  import face_rec

st.success('Model loads successfully!')
st.success('PostgreSQL db successfully connected')