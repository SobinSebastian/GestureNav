# GestureNavApp

Hand Gesture Recognition for User Interface Control via Camera Feed
#
GestureNavApp is a Tkinter-based application that leverages hand gesture recognition to control various desktop functionalities. The application uses MediaPipe for hand landmark detection and a pre-trained scikit-learn model for gesture classification.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
-
## Overview

Machine Learning-Based Hand Gesture Recognition for User Interface Control via Camera Feed,Gesture-based interaction has emerged as a compelling paradigm for enhancing human-computer communication, offering a more intuitive alternative to traditional input methods. Leveraging Google Media Pipe Hands, a state-of-the-art machine learning solution, The model demonstrate real-time hand tracking and gesture recognition capabilities. The HaGRID Sample 120k 384p dataset, comprising over 127,331 hand gesture images, serves as the foundation for training our Gesture Recognition model using the Random Forest algorithm. This versatile algorithm, renowned for its robustness and effectiveness, enables accurate and stable predictions for activating corresponding UI control actions. Through a detailed exploration of these technologies, attendees will gain insights into the practical applications of gesture recognition, paving the way for immersive and intuitive human computer interaction experiences in diverse domains.
GestureNavApp provides a simple and intuitive way to control your computer using hand gestures. It can recognize several hand gestures and map them to specific actions like mouse clicks, mouse movements, and sound adjustments.

## Features

- **Hand Gesture Recognition**: Uses MediaPipe to detect hand landmarks.
- **Gesture Actions**: Maps gestures like "like", "dislike", "one", "fist", "four", and "three" to mouse and volume control actions.
- **GUI Interface**: Built using Tkinter for easy interaction and display.
- **Real-time Updates**: Continuously captures video feed for gesture recognition.

## Requirements

- Python 3.x
- OpenCV
- Tkinter
- MediaPipe
- NumPy
- scikit-learn
- pycaw
- Pillow
- pyautogui
- joblib
- comtypes

## Dataset

HaGRID Sample 120k 384p
https://www.kaggle.com/datasets/innominate817/hagrid-sample-120k-384p

