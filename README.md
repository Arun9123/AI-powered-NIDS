# AI-Powered Network Intrusion Detection System (NIDS)

This repository contains an AI-powered Network Intrusion Detection System (NIDS) designed to identify and classify network traffic patterns. Using a Random Forest Classifier, the system distinguishes between benign (safe) and malicious (attack) traffic. The project features a web-based dashboard for model training, performance evaluation, and manual packet analysis.


## Project Structure

The project is organized into a modular architecture to separate the user interface from the core analytical logic:

nids_main.py: The primary entry point for the Streamlit application. It manages the layout, user inputs, and visualizations.

logic.py: Contains the backend functions for synthetic data generation and the machine learning pipeline.

requirements.txt: Lists the necessary Python dependencies for the project.


## Features

Synthetic Data Generation: Generates 5,000 samples that mimic the structure of network traffic logs, specifically modeled after the CIC-IDS2017 dataset.

Pattern Recognition: The algorithm is designed to identify attacks based on specific network behaviors, such as high packet counts and short flow durations.

Interactive Control Panel: Users can adjust the training data split percentage and the number of trees in the Random Forest model directly from the sidebar.

Performance Evaluation: Provides real-time metrics including Accuracy and a Confusion Matrix heatmap to visualize true positives and false positives.

Live Traffic Simulator: Includes a manual testing module where users can input flow duration, packet count, and packet length mean to test the model's response.


## Installation and Usage

### Prerequisites
   
Ensure you have Python installed. You can install the required dependencies using the provided requirements file: 

pip install -r requirements.txt

### Running the Application

To launch the dashboard, run the following command in your terminal:

streamlit run nids_main.py


## Technical Implementation

The system utilizes the Scikit-Learn library for the Random Forest Classifier. Data manipulation is handled by Pandas and NumPy, while Seaborn and Matplotlib are used for data visualization. The model is stored in the Streamlit session state to allow for persistent use across the dashboard without the need for frequent retraining.
