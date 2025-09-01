# Emotion Detection App

A full-stack web application for real-time facial emotion recognition using deep learning and computer vision. Built with Flask, OpenCV, Deepface, and a custom CNN, it supports live webcam analysis and interactive dashboards for emotion analytics. The modular backend enables rapid model experimentation, while the frontend offers dynamic charting and session logs. Designed for extensibility, the system integrates therapist feedback, supports custom emotion categories, and provides RESTful APIs for emotion data. Ideal for research, mental health, and advanced analytics.
![chuck-norris-emotion](image.png)
# Real-Time Emotion Detection and Feedback System

This project is a Real-Time Emotion Detection and Feedback System built using Flask, OpenCV, and a Convolutional Neural Network (CNN) for emotion modeling. The application allows users to register, track their emotions, and visualize emotional trends over time.

## Features

- User registration and login
- Real-time emotion detection using webcam
- Emotion tracking over time
- Visualization of emotional trends using charts
- User-friendly dashboard

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd emotion-detection-app
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure the database connection in `config.py`:
   - Update the database credentials as needed.

4. Run the application:
   ```
   python app.py
   ```

5. Access the application in your web browser at `http://127.0.0.1:5000`.

## Usage

- **Register**: Create a new account to start tracking your emotions.
- **Login**: Access your dashboard with your registered credentials.
- **Emotion Detection**: Use your webcam to detect emotions in real-time.
- **Track Emotions**: View your emotional history and trends over time.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.