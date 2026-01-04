import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from logic import load_data, train_model

# PAGE CONFIGURATION
st.set_page_config(page_title="AI NIDS Dashboard", layout="wide")

st.title("AI-Powered Network Intrusion Detection System")
st.markdown("""
### Project Overview
This system uses Machine Learning (**Random Forest Algorithm**) to analyze network traffic.
""")

# Load Data
df = load_data()

# Sidebar Controls
st.sidebar.header("Control Panel")
split_size = st.sidebar.slider("Training Data Size (%)", 50, 90, 80)
n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100)

st.divider()
col_train, col_metrics = st.columns([1, 2])

with col_train:
    st.subheader("1. Model Training")
    if st.button("Train Model Now"):
        with st.spinner("Training Random Forest Classifier..."):
            model, acc, cm, X_test, y_test, y_pred = train_model(df, split_size, n_estimators)
            st.session_state['model'] = model
            st.session_state['metrics'] = {
                'acc': acc, 
                'cm': cm, 
                'y_pred': y_pred
            }
            st.success("Training Complete!")

    if 'model' in st.session_state:
        st.success("Model is Ready for Testing")

with col_metrics:
    st.subheader("2. Performance Metrics")
    if 'model' in st.session_state:
        metrics = st.session_state['metrics']
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{metrics['acc']*100:.2f}%")
        m2.metric("Total Samples", len(df))
        m3.metric("Detected Threats", np.sum(metrics['y_pred']))

        st.write("### Confusion Matrix")
        fig, ax = plt.subplots(figsize=(4, 2))
        sns.heatmap(metrics['cm'], annot=True, fmt='d', cmap='Reds', ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Please train the model first.")

# LIVE ATTACK SIMULATOR
st.divider()
st.subheader("3. Live Traffic Simulator")
st.write("Enter network packet details below to test the AI.")

c1, c2, c3, c4 = st.columns(4)
p_dur = c1.number_input("Flow Duration (ms)", 0, 100000, 500)
p_pkts = c2.number_input("Total Packets", 0, 500, 100)
p_len = c3.number_input("Packet Length Mean", 0, 1500, 500)
p_active = c4.number_input("Active Mean Time", 0, 1000, 50)

if st.button("Analyze Packet"):
    if 'model' in st.session_state:
        model = st.session_state['model']
        # Feature order: [Destination_Port, Flow_Duration, Total_Fwd_Packets, Packet_Length_Mean, Active_Mean]
        input_data = np.array([[80, p_dur, p_pkts, p_len, p_active]])
        pred = model.predict(input_data)
        
        if pred[0] == 1:
            st.error("ALERT: MALICIOUS TRAFFIC DETECTED!")
            st.write("**Reason:** Suspicious traffic pattern identified.")
        else:
            st.success("Traffic Status: BENIGN (Safe)")
    else:
        st.error("Please train the model first!")