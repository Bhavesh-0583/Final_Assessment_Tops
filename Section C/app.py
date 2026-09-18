import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.express as px


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Food Delivery Intelligence",
    page_icon="🍔",
    layout="wide"
)


# --------------------------------------------------
# Load model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("demand_model.keras")


# --------------------------------------------------
# Sidebar navigation
# --------------------------------------------------

st.sidebar.title("Food Delivery Intelligence")

page = st.sidebar.selectbox(
    "Select Page",
    [
        "Home",
        "Data Explorer",
        "Demand Predictor",
        "Model Info"
    ]
)


# ==================================================
# HOME
# ==================================================

if page == "Home":

    st.title("🍔 Food Delivery Intelligence Dashboard")

    st.write(
        "This application provides food delivery data exploration, "
        "demand prediction, and model information."
    )

    st.subheader("Available Features")

    col1, col2 = st.columns(2)

    with col1:
        st.info(
            "**Data Explorer**\n\n"
            "Upload delivery data and explore orders, "
            "revenue, and delivery-time information."
        )

        st.info(
            "**Demand Predictor**\n\n"
            "Predict Low, Medium, or High delivery demand "
            "using the trained ANN model."
        )

    with col2:
        st.info(
            "**Model Info**\n\n"
            "View the neural-network architecture and "
            "training history."
        )


# ==================================================
# DATA EXPLORER
# ==================================================

elif page == "Data Explorer":

    st.title("📊 Data Explorer")

    uploaded_file = st.file_uploader(
        "Upload Food Delivery CSV",
        type=["csv"]
    )

    if uploaded_file is None:

        st.warning(
            "Please upload a CSV file to explore the delivery data."
        )

    else:

        try:
            df = pd.read_csv(uploaded_file)

            st.subheader("Dataset Preview")
            st.dataframe(df.head())

            # ------------------------------------------
            # Summary metrics
            # ------------------------------------------

            st.subheader("Summary Statistics")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Total Orders",
                    len(df)
                )

            with col2:

                if "delivery_time" in df.columns:
                    avg_time = df["delivery_time"].mean()

                    st.metric(
                        "Average Delivery Time",
                        f"{avg_time:.2f} minutes"
                    )
                else:
                    st.warning(
                        "delivery_time column not found."
                    )

            with col3:

                if "revenue" in df.columns:
                    revenue = df["revenue"].sum()

                    st.metric(
                        "Total Revenue",
                        f"₹{revenue:,.2f}"
                    )
                else:
                    st.warning(
                        "revenue column not found."
                    )

            # ------------------------------------------
            # Chart 1: Hourly order volume
            # ------------------------------------------

            if "hour" in df.columns:

                hourly_orders = (
                    df.groupby("hour")
                    .size()
                    .reset_index(name="orders")
                )

                fig1 = px.line(
                    hourly_orders,
                    x="hour",
                    y="orders",
                    title="Hourly Order Volume"
                )

                st.plotly_chart(
                    fig1,
                    use_container_width=True
                )

            else:

                st.info(
                    "The CSV does not contain an 'hour' column, "
                    "so the hourly order volume chart cannot be displayed."
                )

            # ------------------------------------------
            # Chart 2: Delivery time by restaurant
            # ------------------------------------------

            if (
                "restaurant" in df.columns
                and "delivery_time" in df.columns
            ):

                fig2 = px.box(
                    df,
                    x="restaurant",
                    y="delivery_time",
                    title="Delivery Time Distribution by Restaurant"
                )

                st.plotly_chart(
                    fig2,
                    use_container_width=True
                )

            else:

                st.info(
                    "The CSV needs 'restaurant' and 'delivery_time' "
                    "columns for this chart."
                )

        except Exception as e:

            st.error(
                f"Unable to read the CSV file: {e}"
            )


# ==================================================
# DEMAND PREDICTOR
# ==================================================

elif page == "Demand Predictor":

    st.title("🤖 Demand Predictor")

    try:

        model = load_model()

        st.write(
            "Enter the following information to predict delivery demand."
        )

        # ------------------------------------------
        # Inputs
        # ------------------------------------------

        hour = st.slider(
            "Hour of Day",
            min_value=0,
            max_value=23,
            value=12
        )

        day = st.selectbox(
            "Day of Week",
            [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday"
            ]
        )

        temperature = st.number_input(
            "Temperature (°C)",
            min_value=10.0,
            max_value=45.0,
            value=25.0
        )

        day_number = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ].index(day)

        # ------------------------------------------
        # Prediction
        # ------------------------------------------

        if st.button("Predict Demand"):

            input_data = np.array([
                [
                    hour,
                    day_number,
                    temperature
                ]
            ])

            prediction = model.predict(
                input_data,
                verbose=0
            )

            probabilities = prediction[0]

            classes = [
                "Low",
                "Medium",
                "High"
            ]

            predicted_index = np.argmax(
                probabilities
            )

            predicted_class = classes[
                predicted_index
            ]

            confidence = (
                probabilities[predicted_index] * 100
            )

            # --------------------------------------
            # Display prediction
            # --------------------------------------

            if predicted_class == "High":

                st.warning(
                    f"Predicted Demand: {predicted_class} "
                    f"({confidence:.2f}% confidence)"
                )

            else:

                st.success(
                    f"Predicted Demand: {predicted_class} "
                    f"({confidence:.2f}% confidence)"
                )

            # --------------------------------------
            # Probability chart
            # --------------------------------------

            probability_df = pd.DataFrame({
                "Class": classes,
                "Probability": probabilities
            })

            fig = px.bar(
                probability_df,
                x="Class",
                y="Probability",
                title="Demand Confidence Scores"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    except Exception as e:

        st.error(
            f"Model loading failed: {e}"
        )


# ==================================================
# MODEL INFO
# ==================================================

elif page == "Model Info":

    st.title("🧠 Model Information")

    try:

        model = load_model()

        st.subheader("Model Architecture")

        # ------------------------------------------
        # Architecture table
        # ------------------------------------------

        architecture = []

        for layer in model.layers:

            try:
                output_shape = str(
                    layer.output.shape
                )
            except Exception:
                output_shape = "N/A"

            architecture.append({
                "Layer Name": layer.name,
                "Output Shape": output_shape,
                "Parameters": layer.count_params()
            })

        architecture_df = pd.DataFrame(
            architecture
        )

        st.dataframe(
            architecture_df,
            use_container_width=True
        )

        st.write(
            f"**Total Parameters:** "
            f"{model.count_params():,}"
        )

        # ------------------------------------------
        # Training history
        # ------------------------------------------

        st.subheader("Training History")

        st.info(
            "Training history must be available from the "
            "model training process. The saved Keras model "
            "does not itself guarantee that the History object "
            "is available after restarting the application."
        )

    except Exception as e:

        st.error(
            f"Unable to load model information: {e}"
        )
