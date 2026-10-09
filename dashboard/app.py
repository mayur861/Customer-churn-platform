import streamlit as st
import requests
import pandas as pd

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide"
)

API_URL = "https://customer-churn-platform-lbjk.onrender.com"


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main-title {
    font-size: 38px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    color: #888;
    font-size: 17px;
    margin-bottom: 25px;
}

.metric-card {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #333;
    text-align: center;
}

.risk-critical {
    font-size: 28px;
    font-weight: 700;
}

.section-title {
    font-size: 24px;
    font-weight: 600;
    margin-top: 20px;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.markdown(
    '<div class="main-title">📊 Customer Churn Intelligence Platform</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered churn prediction, SHAP explainability and retention recommendations</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Overview",
        "👤 Customer Prediction",
        "📁 Batch Prediction"
    ]
)


# ==================================================
# OVERVIEW
# ==================================================

if page == "🏠 Overview":

    st.header("Platform Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Model", "Random Forest")

    with col2:
        st.metric("Features", "10")

    with col3:
        st.metric("Explainability", "SHAP")

    with col4:
        st.metric("API Version", "2.0.0")

    st.markdown("---")

    st.subheader("What this platform does")

    st.write("""
    This platform predicts the probability that a customer will churn
    and provides actionable retention recommendations.
    """)

    st.markdown("### 🔍 Prediction")

    st.write("""
    The machine learning model analyzes customer information and
    predicts whether the customer is likely to churn.
    """)

    st.markdown("### 🧠 Explainability")

    st.write("""
    SHAP is used to identify the major factors increasing or decreasing
    the customer's churn risk.
    """)

    st.markdown("### 🎯 Retention Intelligence")

    st.write("""
    Based on the customer's profile and predicted risk,
    the system generates targeted retention recommendations.
    """)

    st.markdown("### ⚙️ Technology Stack")

    st.write("""
    **Python • Scikit-learn • Random Forest • SHAP • FastAPI • Streamlit • Pandas**
    """)


# ==================================================
# CUSTOMER PREDICTION
# ==================================================

elif page == "👤 Customer Prediction":

    st.header("👤 Customer Churn Prediction")

    st.write(
        "Enter customer information to generate a churn prediction."
    )

    st.markdown("### Customer Profile")

    col1, col2 = st.columns(2)

    with col1:

        senior_citizen = st.selectbox(
            "Senior Citizen",
            ["No", "Yes"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["No", "Yes"]
        )

        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=72,
            value=12
        )

        internet_service = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

        online_security = st.selectbox(
            "Online Security",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

    with col2:

        tech_support = st.selectbox(
            "Tech Support",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0,
            step=1.0
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=840.0,
            step=10.0
        )

    # Convert Yes/No to model format
    senior_citizen_value = 1 if senior_citizen == "Yes" else 0

    st.markdown("---")

    if st.button(
        "🔮 Predict Customer Churn",
        use_container_width=True
    ):

        payload = {
            "SeniorCitizen": senior_citizen_value,
            "Dependents": dependents,
            "tenure": tenure,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "TechSupport": tech_support,
            "Contract": contract,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }

        try:

            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=30
            )

            if response.status_code == 200:

                result = response.json()

                st.success("Prediction generated successfully!")

                st.markdown("## Prediction Result")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Churn Probability",
                        f"{result['churn_probability_percent']:.2f}%"
                    )

                with col2:
                    st.metric(
                        "Prediction",
                        result["prediction"]
                    )

                with col3:
                    st.metric(
                        "Risk Level",
                        result["risk_level"]
                    )

                st.markdown("---")

                # ------------------------------------------
                # RISK DRIVERS
                # ------------------------------------------

                st.markdown("### 🚨 Risk Drivers")

                drivers = result.get(
                    "risk_drivers",
                    []
                )

                if drivers:

                    driver_df = pd.DataFrame(drivers)

                    driver_df["shap_value"] = driver_df[
                        "shap_value"
                    ].round(4)

                    st.dataframe(
                        driver_df,
                        use_container_width=True,
                        hide_index=True
                    )

                else:
                    st.info("No significant risk drivers found.")

                # ------------------------------------------
                # PROTECTIVE FACTORS
                # ------------------------------------------

                st.markdown("### 🛡️ Protective Factors")

                protective = result.get(
                    "protective_factors",
                    []
                )

                if protective:

                    protective_df = pd.DataFrame(
                        protective
                    )

                    protective_df["shap_value"] = (
                        protective_df["shap_value"].round(4)
                    )

                    st.dataframe(
                        protective_df,
                        use_container_width=True,
                        hide_index=True
                    )

                else:
                    st.info("No significant protective factors found.")

                # ------------------------------------------
                # RECOMMENDATIONS
                # ------------------------------------------

                st.markdown("### 🎯 Retention Recommendations")

                recommendations = result.get(
                    "retention_recommendations",
                    []
                )

                for recommendation in recommendations:

                    st.write(
                        f"✅ {recommendation}"
                    )

            else:

                st.error(
                    f"API Error: {response.status_code}"
                )

                st.code(response.text)

        except requests.exceptions.ConnectionError:

            st.error(
                "❌ Cannot connect to FastAPI. "
                "Make sure the API server is running."
            )

        except Exception as e:

            st.error(
                f"Unexpected error: {str(e)}"
            )


# ==================================================
# BATCH PREDICTION
# ==================================================

elif page == "📁 Batch Prediction":

    st.header("📁 Batch Customer Prediction")

    st.write("""
    Upload a CSV file containing the 10 model features.
    The API will generate churn probability, risk level
    and retention recommendations for every customer.
    """)

    st.markdown("### Required CSV Columns")

    required_columns = [
        "SeniorCitizen",
        "Dependents",
        "tenure",
        "InternetService",
        "OnlineSecurity",
        "TechSupport",
        "Contract",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges"
    ]

    st.code(
        "\n".join(required_columns)
    )

    uploaded_file = st.file_uploader(
        "Upload Customer CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            df = pd.read_csv(uploaded_file)

            st.markdown("### Uploaded Data")

            st.dataframe(
                df.head(),
                use_container_width=True
            )

            missing_columns = [
                col for col in required_columns
                if col not in df.columns
            ]

            if missing_columns:

                st.error(
                    f"Missing columns: {missing_columns}"
                )

            else:

                st.success(
                    f"CSV validated successfully — {len(df)} customers found."
                )

                if st.button(
                    "🚀 Run Batch Prediction",
                    use_container_width=True
                ):

                    uploaded_file.seek(0)

                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "text/csv"
                        )
                    }

                    try:

                        response = requests.post(
                            f"{API_URL}/batch-predict",
                            files=files,
                            timeout=120
                        )

                        if response.status_code == 200:

                            output_file = response.content

                            st.success(
                                "Batch prediction completed successfully!"
                            )

                            st.download_button(
                                label="⬇️ Download Predictions",
                                data=output_file,
                                file_name="churn_predictions.csv",
                                mime="text/csv",
                                use_container_width=True
                            )

                        else:

                            st.error(
                                f"Batch API Error: {response.status_code}"
                            )

                            st.code(response.text)

                    except requests.exceptions.ConnectionError:

                        st.error(
                            "❌ Cannot connect to FastAPI."
                        )

        except Exception as e:

            st.error(
                f"Error reading CSV: {str(e)}"
            )
