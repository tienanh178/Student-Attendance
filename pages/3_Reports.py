import streamlit as st
from Home import face_rec
import pandas as pd
import os
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
import os

# Load .env
load_dotenv()
 
# POSTGRESQL CONNECTION TO DATABASE
# Get info database from .env file
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT")

st.subheader('Reports')

def execute_query(query, values, columns):
    try:
        # Kết nối tới PostgreSQL
        connection = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cursor = connection.cursor()

        cursor.execute(query, values)
        result = cursor.fetchall()
        data = pd.DataFrame(result, columns=columns) 
        
        return data

    except psycopg2.Error as e:
        print(f"Lỗi khi lưu dữ liệu: {e}")
        connection.rollback()

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

def retrieve_data_class(class_code):
    query = """SELECT * FROM class WHERE class_code = %s"""
    columns = ['ID', 'Class Name', 'Study Limit', 'Class Code', 'Place of Study', 'Teacher ID']
    return execute_query(query,(class_code,),columns)


def retrieve_data_attendance(class_code):
    query = """SELECT 
            sa.id AS attendance_id,
            sa.check_in_time,
            s.student_name,
            c.class_name,
            c.class_code
            FROM 
                student_attendance sa
            JOIN 
                student s 
            ON 
                sa.student_id = s.id
            JOIN 
                class c 
            ON 
                sa.class_id = c.id
            WHERE
                c.class_code = %s
            """
    columns = ['ID', 'check_in_time', 'student_name', 'class_name','class_code']   
    return execute_query(query,(class_code,),columns)

class_code = st.text_input(label='Class Code', placeholder='Class code')
class_dataframe = retrieve_data_class(class_code)

if st.button('Submit'):
    st.dataframe(class_dataframe)

    st.dataframe(retrieve_data_attendance(class_code), hide_index=True)

