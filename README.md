# AI-Powered Network Intrusion Detection System (NIDS)

This project is an AI-driven Network Intrusion Detection System (NIDS) that uses **Machine Learning** to classify network traffic and **Generative AI** to provide security insights.

## Overview

The system analyzes network flow data to distinguish between **Benign** traffic and potential **Attacks** (e.g., DDoS). It features a real-time dashboard for training models, simulating traffic, and generating AI-powered analysis for detected threats.

## Key Features

- **Machine Learning Analysis**: Uses a Random Forest Classifier to detect malicious patterns in network traffic.
- **Interactive Dashboard**: Built with Streamlit for a user-friendly experience.
- **AI Analyst**: Integrated with **Groq AI (Llama 3.3)** to provide plain-english explanations for detection results.
- **Traffic Simulator**: Capture and analyze random packets from the dataset to test the system's accuracy.

## Tech Stack

- **Languages**: Python
- **Frontend**: Streamlit
- **ML Framework**: Scikit-Learn (Random Forest)
- **Data Handling**: Pandas, NumPy
- **Generative AI**: Groq API

## Setup & Installation

1. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:

   ```bash
   streamlit run app.py
   ```

3. **Configure API Key**:
   To use the AI Analyst feature, obtain a free API key from [Groq Cloud](https://console.groq.com/keys) and enter it in the app's sidebar.

## Project Structure

- `app.py`: The main application file containing the UI and logic.
- `requirements.txt`: List of necessary Python libraries.
- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`: The dataset used for training and simulation.
