# Student Attendance Tracker & Classroom Summary

A real-time face and focus tracking system built with Python, OpenCV, and Mediapipe. This project uses facial landmarks to estimate a student's focus level and generates a summary report. A Streamlit dashboard is also included for easy visualization of the results.

---

## ⚙️ How It Works

- **Face Recognition:** Uses the `face_recognition` library to identify known students from a directory of known faces.
- **Focus Detection:** Employs Mediapipe's face mesh to get landmark points around the eyes. The system calculates the Eye Aspect Ratio (EAR) to determine if the eyes are open and looking forward, indicating focus.
- **Data Logging:** Tracks the number of total frames and focused frames for each identified student.
- **Summary Report:** After the tracking session, it generates a `attendance_summary.csv` file containing each student's name, the session timestamp, total frames, focused frames, and a calculated focus percentage.
- **Dashboard:** A Streamlit application reads the CSV file to display a summary report with metrics like the number of students logged, average focus percentage, and the most focused student.

---

## 📊 Features

- Real-time facial recognition and focus tracking.
- Calculates Eye Aspect Ratio (EAR) to measure focus.
- Generates a CSV summary file with focus statistics for each student.
- Streamlit dashboard for a quick visual overview of the session.
- Easily extensible to add more known student faces.

---

## 💻 Technologies Used

- Python
- OpenCV
- `face_recognition` library
- `mediapipe` for face mesh
- Pandas for data manipulation
- Streamlit for the web dashboard

---

## 🚀 Getting Started

### 1\. Place known faces

Add images of known students to the `known_faces` directory. The filename (without the extension) will be used as the student's name. For example, `sam.jpg`.

### 2\. Set up the Conda environment

This project uses a Conda environment to manage dependencies. You can create a new environment using the provided `environment.yml` file.

```bash
conda env create -f environment.yml
conda activate your_env_name
```

### 3\. Run the focus tracker

This script will open your webcam feed. Press `q` to quit the session and generate the `attendance_summary.csv` file.

```bash
python face_track.py
```

### 4\. Run the dashboard

Once the summary file is created, you can launch the Streamlit dashboard to view the results.

```bash
streamlit run dashboard.py
```

---

## 📁 Project Structure

```
attendance-face-detector
├── known_faces/
│   └── sam.jpg(example)
├── attendance_summary.csv(post program run)
├── dashboard.py
├── environment.yml
├── face_track.py
└── README.md
```
