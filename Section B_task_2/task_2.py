import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="Delivery Time Prediction",
    layout="centered"
)

st.title("Food Delivery Time Prediction")

# Load the trained model only once
@st.cache_resource
def load_model():
    return joblib.load("delivery_model.pkl")


try:
    # Load saved model
    model = load_model()

    st.subheader("Enter Order Details")

    # Distance input
    distance = st.slider(
        "Distance (km)",
        min_value=1.0,
        max_value=50.0,
        value=5.0,
        step=0.5
    )

    # Order value input
    order_value = st.number_input(
        "Order Value (Rs)",
        min_value=0.0,
        value=500.0,
        step=50.0
    )

    # Time of day input
    time_of_day = st.selectbox(
        "Time of Day",
        ["Morning", "Afternoon", "Evening", "Night"]
    )

    # Encode time of day
    time_encoding = {
        "Morning": 0,
        "Afternoon": 1,
        "Evening": 2,
        "Night": 3
    }

    if st.button("Predict Delivery Time"):

        try:
            time_encoded = time_encoding[time_of_day]

            # Assemble feature array
            features = np.array([
                [distance, order_value, time_encoded]
            ])

            # Make prediction
            prediction = model.predict(features)

            predicted_time = float(prediction[0])

            st.success(
                f"Predicted Delivery Time: {predicted_time:.2f} minutes"
            )

        except ValueError as e:
            st.error(f"Invalid input value: {e}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

except (FileNotFoundError, OSError, ValueError) as e:
    st.error(
        f"Unable to load delivery_model.pkl. "
        f"Please make sure the model file is in the same folder as this app. "
        f"Error: {e}"
    )