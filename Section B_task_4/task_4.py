import streamlit as st
import tensorflow as tf
import numpy as np
import plotly.express as px


st.title("Food Delivery Demand Prediction")


# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("demand_model.keras")


try:
    model = load_model()

    # Inputs
    hour = st.slider("Hour of Day", 0, 23, 12)

    day = st.selectbox(
        "Day of Week",
        ["Monday", "Tuesday", "Wednesday",
         "Thursday", "Friday", "Saturday", "Sunday"]
    )

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=10.0,
        max_value=45.0,
        value=25.0
    )

    # Convert day to number
    day_number = [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ].index(day)

    # Prediction
    if st.button("Predict Demand"):

        # Input shape = (1, 3)
        data = np.array([
            [hour, day_number, temperature]
        ])

        prediction = model.predict(data, verbose=0)

        classes = ["Low", "Medium", "High"]

        index = np.argmax(prediction[0])
        predicted_class = classes[index]

        confidence = prediction[0][index] * 100

        # Display result
        if predicted_class == "High":
            st.warning(
                f"Demand: {predicted_class} "
                f"({confidence:.2f}% confidence)"
            )
        else:
            st.success(
                f"Demand: {predicted_class} "
                f"({confidence:.2f}% confidence)"
            )

        # Probability chart
        df = {
            "Class": classes,
            "Probability": prediction[0]
        }

        fig = px.bar(
            df,
            x="Class",
            y="Probability",
            title="Demand Probability"
        )

        st.plotly_chart(fig)


except Exception as e:
    st.error(f"Model loading or prediction error: {e}")