# eslint-disable no-alert, no-console 
from calendar import c
from anyio import current_time
from idna import encode
import numpy as np
import pandas as pd
import cv2
# Insight face
from insightface.app import FaceAnalysis 
from sklearn.metrics import pairwise
import time
from datetime import datetime
from streamlit import dataframe
import os
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

load_dotenv()
 
# POSTGRESQL CONNECTION TO DATABASE
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT")

def fetch_from_postgresql(query, params=None):
    try:
        connection = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cursor = connection.cursor()
        cursor.execute(query, params)

        results = cursor.fetchall()
        return results

    except psycopg2.Error as e:
        print(f"Lỗi xảy ra: {e}")
        return None

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

def registration_face_attendance(key, x_mean_bytes):
    try:
        connection = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cursor = connection.cursor()

        student_name = key.split('@')[0]  
        student_code = key.split('@')[1]
        
        # Chuyển đổi `x_mean_bytes` thành numpy array
        face_features = np.frombuffer(x_mean_bytes, dtype=np.float32).tolist()

        # Kiểm tra xem sinh viên đã tồn tại chưa
        cursor.execute("SELECT id FROM student WHERE student_code = %s", (student_code,))
        result = cursor.fetchone()

        if result:
            student_id = result[0] # Get student_id
        else:
            # Nếu sinh viên chưa tồn tại, thêm mới vào bảng `student`
            insert_student_query = """
            INSERT INTO student (student_name, student_code) 
            VALUES (%s, %s) RETURNING id
            """
            cursor.execute(insert_student_query, (student_name,student_code))
            student_id = cursor.fetchone()[0]

        # Lưu `face_features` vào bảng `face_features`
        insert_face_features_query = """
        INSERT INTO face_features (student_id, face_features)
        VALUES (%s, %s)
        """
        cursor.execute(insert_face_features_query, (student_id, face_features))
        connection.commit()

        print(f"Dữ liệu được lưu thành công cho sinh viên: {student_name} {student_code}")

    except psycopg2.Error as e:
        print(f"Lỗi xảy ra: {e}")
        connection.rollback()

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

# Retrieve Data from database
def retrieve_data():
    query = """
        SELECT 
            student.student_name AS name,
            student.student_code AS student_code,
            face_features.face_features AS facial_features
        FROM 
            student
        JOIN 
            face_features 
        ON 
            student.id = face_features.student_id;
        """
    # Thực thi truy vấn và lấy dữ liệu
    results = fetch_from_postgresql(query)
    
    if results:
        # Tạo DataFrame từ kết quả truy vấn
        retrieve_df = pd.DataFrame(results, columns=['Name','student_code', 'facial_features'])

        # Chuyển đổi cột `facial_features` thành mảng numpy
        retrieve_df['facial_features'] = retrieve_df['facial_features'].apply(
            lambda x: np.array(x, dtype=np.float32) if x else None
        )
        
        return retrieve_df[['Name', 'student_code', 'facial_features']]
    else:
        print("Không có dữ liệu được trả về.")
        return pd.DataFrame(columns=['Name','student_code', 'facial_features'])
    
def save_student_attendance(student_name, student_code, ctime):
    try:
        # Kết nối PostgreSQL
        connection = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cursor = connection.cursor()

        # Tìm `student_id` theo `student_code`
        cursor.execute("SELECT id FROM student WHERE student_code = %s", (student_code,))
        result = cursor.fetchone()

        if not result:
            print(f"Không tìm thấy sinh viên với mã {student_code}.")
            return

        student_id = result[0]
        
        class_id = 1  # ID lớp mặc định hoặc lấy từ đầu vào

        # Chèn dữ liệu vào bảng `student_attendance`
        insert_student_attendance_query = """
            INSERT INTO student_attendance (check_in_time, student_id, class_id) 
            VALUES (%s, %s, %s) RETURNING id
            """
        cursor.execute(insert_student_attendance_query, (ctime, student_id, class_id))

        # Commit dữ liệu
        connection.commit()

        print(f"Dữ liệu được lưu thành công cho sinh viên: {student_name} ({student_code}), lớp {class_id}, thời gian: {ctime}")

    except psycopg2.IntegrityError as e:
        print(f"Lỗi trùng lặp hoặc ràng buộc khóa ngoại: {e}")
        if connection:
            connection.rollback()

    except psycopg2.Error as e:
        print(f"Lỗi xảy ra khi truy vấn: {e}")
        if connection:
            connection.rollback()

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


# Config face analysis
faceapp = FaceAnalysis(name='buffalo_l', root='insight_face_models', providers=['CPUExecutionProvider'])
faceapp.prepare(ctx_id=0, det_size=(640,640), det_thresh = 0.5)

# ML search algorithm
def ml_search_algorithm(dataframe, feature_column, test_vector, name_code, thresh):
    # Step 1: Index face embedding from the data frame and convert to the array
    X_list = dataframe[feature_column].tolist()
    
    x = np.vstack(X_list)    
    
    # Step 2: Cal. cousine similarity
    similar = pairwise.cosine_similarity(x,test_vector.reshape(1,-1))
    similar_arr = np.array(similar).flatten()
    dataframe['cosine'] = similar_arr
    
    # Step 3: Filter the data
    datafilter = dataframe.query(f'cosine >= {thresh}')
    
    # Step 4: Get the person name
    if len(datafilter) > 0:
        datafilter.reset_index(drop=True, inplace=True)
        argmax = datafilter['cosine'].argmax()
        person_name, person_code = datafilter.loc[argmax][name_code]
    else:
        person_name = 'Unknown'
        person_code = 'Unknown'
        
    return person_name, person_code

########### Real Time Prediction

# Save logs for every 1 minute
class RealTimePred:
    def __init__(self):
        self.logs = dict(name=[], student_code=[], current_time=[])

    def reset_dict(self):
        self.logs = dict(name=[], student_code=[], current_time=[])

    def saveLogs_db(self):
        try:
            # Step 1: Create a log dataframe
            dataframe = pd.DataFrame(self.logs)

            if dataframe.empty:
                print("No logs to save.")
                return

            # Step 2: Drop duplicate entries based on 'name'
            dataframe.drop_duplicates(subset=['name'], inplace=True)

            # Step 3: Push data to database
            name_list = dataframe['name'].tolist()
            student_code_list = dataframe['student_code'].tolist()
            ctime_list = dataframe['current_time'].tolist()

            encoded_data = []

            for name, student_code, ctime in zip(name_list, student_code_list, ctime_list):
                if name != 'Unknown':
                    concat_string = f'{name}@{student_code}@{ctime}'
                    encoded_data.append(concat_string)
                    print(encoded_data)

            # Save logs to database
            for log in encoded_data:
                try:
                    name, student_code, ctime = log.split('@')
                    ctime_obj = datetime.strptime(ctime, '%Y-%m-%d %H:%M:%S.%f')
                    formatted_ctime = ctime_obj.strftime('%Y-%m-%d %H:%M:%S')
                    
                    save_student_attendance(name, student_code, formatted_ctime)
                except Exception as e:
                    print(f"Error processing log: {log} - {e}")
                  

        except Exception as e:
            print(f"Error in saveLogs_db: {e}")
        finally:
            # Clear logs after processing
            self.reset_dict()
    
    def face_prediction(self, test_image, dataframe,feature_column,name_code=['Name','student_code'], thresh=0.5):
        # Step 0: Find the time
        current_time = str(datetime.now())
        
        # Step 1: take the test image and apply to insight face
        results = faceapp.get(test_image)
        test_copy = test_image.copy()

        #Step 2: use for loop and extract each embedding and pass to ml_search_algorithm
        for res in results:
            x1,y1,x2,y2 = res['bbox'].astype(int)
            embeddings = res['embedding']
            person_name, student_code = ml_search_algorithm(dataframe, 
                                                        feature_column, 
                                                        test_vector=embeddings,
                                                        name_code=name_code, 
                                                        thresh=thresh)
        
            if person_name == 'Unknown':
                color = (0,0,255)
            else:
                color = (0,255,0)
        
            cv2.rectangle(test_copy, (x1,y1), (x2,y2), color)
        
            text_gen = person_name
            cv2.putText(test_copy,text_gen, (x1,y1), cv2.FONT_HERSHEY_DUPLEX,0.7, color, 2)
            #cv2.putText(test_copy,person_code, (x1,y2+10), cv2.FONT_HERSHEY_DUPLEX,0.7, color, 2)
            cv2.putText(test_copy, current_time, (x1,y2+10), cv2.FONT_HERSHEY_DUPLEX,0.7, color, 2)

            # save info in logs dict
            self.logs['name'].append(person_name)
            self.logs['student_code'].append(student_code)
            self.logs['current_time'].append(current_time)
            
        return test_copy

### Registration Form

class RegistrationForm:
    def __init__(self):
        self.sample = 0
    def reset(self):
        self.sample = 0
    def get_embedding(self, frame):
        # get result from insightface model
        results = faceapp.get(frame, max_num=1)
        embeddings = None

        for res in results:
            self.sample += 1
            x1,y1,x2,y2 = res['bbox'].astype(int)
            cv2.rectangle(frame,(x1,y1), (x2,y2), (0,255,0), 1)

            #put text sample info
            text=f'samples = {self.sample}'
            cv2.putText(frame, text, (x1,y1), cv2.FONT_HERSHEY_DUPLEX, 0.6, (25,255,0), 2)

            # facial features
            embeddings = res['embedding']

        return frame, embeddings
    
    def save_data_in_postgres_db(self, name, student_code):
        # Validation
        if name is not None and student_code is not None:
            if name.strip() != '':
                key=f'{name}@{student_code}'
            else:
                return 'name_false'
        else:
            return 'name_false'    
        
        #if face_embedding.txt exist
        if 'face_embedding.txt' not in os.listdir():
            return 'file_false'
        
        # Step 1: Load "face_embedding.txt"
        x_array = np.loadtxt('face_embedding.txt', dtype=np.float32) #flatten array

        # Step 2: convert into array
        received_samples = int(x_array.size/512)
        x_array = x_array.reshape(received_samples, 512)
        x_array = np.asarray(x_array)

        # Step 3: Call mean embeddings
        x_mean = x_array.mean(axis = 0)
        x_mean = x_mean.astype(np.float32)
        x_mean_bytes = x_mean.tobytes()

        # Step 4: Save into database
        registration_face_attendance(key, x_mean_bytes)

        os.remove('face_embedding.txt')
        self.reset()

        return True

