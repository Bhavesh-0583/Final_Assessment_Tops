import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Delivery Summary Dashboard",
    layout="wide"
)

st.title("Food Delivery Summary Dashboard")

# Requirement 1: CSV file uploader
uploaded_file = st.file_uploader(
    "Upload Food Delivery CSV File",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload a food delivery CSV file to view the dashboard.")

else:
    try:
        # Read uploaded CSV file
        df = pd.read_csv(uploaded_file)

        # Check required columns
        required_columns = [
            "city",
            "delivery_time",
            "revenue",
            "restaurant"
        ]

        missing_columns = [
            col for col in required_columns if col not in df.columns
        ]

        if missing_columns:
            st.error(
                f"Missing required columns: {', '.join(missing_columns)}"
            )
        else:
            # Requirement 3: City filter in sidebar
            cities = sorted(df["city"].dropna().unique())

            selected_city = st.sidebar.selectbox(
                "Select City",
                cities
            )

            # Filter data according to selected city
            filtered_df = df[df["city"] == selected_city]

            # Requirement 2: Three metric cards
            total_orders = len(filtered_df)
            average_delivery_time = filtered_df["delivery_time"].mean()
            total_revenue = filtered_df["revenue"].sum()

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Total Orders",
                    total_orders
                )

            with col2:
                st.metric(
                    "Average Delivery Time (Minutes)",
                    f"{average_delivery_time:.2f}"
                )

            with col3:
                st.metric(
                    "Total Revenue",
                    f"₹{total_revenue:,.2f}"
                )

            # Requirement 4: Order count per restaurant
            st.subheader(
                f"Order Count by Restaurant — {selected_city}"
            )

            restaurant_orders = (
                filtered_df["restaurant"]
                .value_counts()
                .rename_axis("restaurant")
                .to_frame("orders")
            )

            st.bar_chart(restaurant_orders)

    except Exception as e:
        st.error(f"Error while processing the uploaded file: {e}")
