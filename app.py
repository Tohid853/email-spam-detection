
import streamlit as st
import joblib

# Load Trained Model
model = joblib.load("spam_model.pkl")

# Page Configuration
st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="📧",
    layout="centered"
)

# Title
st.title("📧 Email Spam Detection System")

st.write("Enter your email or message to check whether it is Spam or Not Spam.")

# User Input
message = st.text_area(
    "Enter Your Message",
    placeholder="Type your email or message here..."
)

# Prediction Button
if st.button("Check Message", use_container_width=True):

    if message.strip() == "":
        st.warning("Please enter a message!")

    else:
        prediction = model.predict([message])[0]

        if prediction == "spam":
            st.error("🚨 This message is Spam!")
        else:
            st.success("✅ This message is Not Spam!")

