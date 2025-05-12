# Attendance System Setup Guide 

## Step 1: Install Dependencies 
Run the following commands to install the required packages:

pip install -r requirements.txt
pip install streamlit
pip install -U streamlit-webrtc

Create a folder call insight_face_models/models
Install buffalo_l model to that folder and unzip it
Link: "https://github.com/deepinsight/insightface/tree/master/model_zoo"

## Step 2: Create the Database 
Create a PostgreSQL database named Test_attendance.

## Step 3: Initialize Tables in PostgreSQL 
Run the following SQL commands to create the necessary tables:

CREATE TABLE teacher (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    code VARCHAR(50) UNIQUE NOT NULL,
    account VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password VARCHAR(100) NOT NULL,
    tel_number VARCHAR(20)
);

CREATE TABLE class (
    id SERIAL PRIMARY KEY,
    class_name VARCHAR(100) NOT NULL,
    number_of_study_limit INTEGER,
    class_code VARCHAR(50) UNIQUE NOT NULL,
    place_study VARCHAR(100),
    teacher_id INTEGER REFERENCES teacher(id)
);

CREATE TABLE student (
    id SERIAL PRIMARY KEY,
    student_name VARCHAR(100) NOT NULL,
    student_code VARCHAR(50) UNIQUE NOT NULL,
    phone_number VARCHAR(20)
);

CREATE TABLE face_features (
    id SERIAL PRIMARY KEY,
    student_id INTEGER REFERENCES student(id) ON DELETE CASCADE,
    face_features DOUBLE PRECISION[] NOT NULL  
);

CREATE TABLE student_attendance (
    id SERIAL PRIMARY KEY,
    check_in_time TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    student_id INTEGER REFERENCES student(id) ON DELETE CASCADE,
    class_id INTEGER REFERENCES class(id) ON DELETE CASCADE
);
Optional: Insert Sample Data

INSERT INTO teacher (name, code, account, email, password, tel_number)
VALUES ('Nguyễn Văn A', 'GV001', 'nguyenvana', 'nguyenvana@example.com', '123456', '0123456789');

INSERT INTO class (class_name, number_of_study_limit, class_code, place_study, teacher_id)
VALUES ('Lập trình Python cơ bản', 30, 'LPY101', 'Phòng 101', 1);

## Step 4: Configure Environment 
Update your .env file with your PostgreSQL database connection details (host, user, password, database name, etc.).

## Step 5: Run the Application 
Use the following command to start the application:

streamlit run Home.py

Usage Instructions:
Enter student_name and student_code.

Start the video to collect 50 or more face samples.

Enable real-time prediction to automatically mark attendance.

Go to the Report section and select a class to view student attendance reports.
