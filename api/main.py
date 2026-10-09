from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import shap
import io
import os


# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI(
    title="Customer Churn Intelligence API",
    description="Customer churn prediction, SHAP explainability and retention recommendations",
    version="2.0.0"
)


# =====================================================
# LOAD MODEL
# =====================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "churn_model1.pkl"
)

model = joblib.load(MODEL_PATH)

preprocessor = model.named_steps["preprocessor"]
rf_model = model.named_steps["model"]


# =====================================================
# SHAP EXPLAINER
# =====================================================

explainer = shap.TreeExplainer(rf_model)

feature_names = preprocessor.get_feature_names_out()


# =====================================================
# INPUT SCHEMA — 10 SELECTED FEATURES
# =====================================================

class CustomerData(BaseModel):

    SeniorCitizen: int
    Dependents: str
    tenure: int

    InternetService: str
    OnlineSecurity: str
    TechSupport: str

    Contract: str
    PaymentMethod: str

    MonthlyCharges: float
    TotalCharges: float


# =====================================================
# CLEAN FEATURE NAMES
# =====================================================

def clean_feature_name(feature):

    feature = feature.replace("num__", "")
    feature = feature.replace("cat__", "")

    replacements = {

        "SeniorCitizen":
            "Senior Citizen",

        "tenure":
            "Customer Tenure",

        "MonthlyCharges":
            "Monthly Charges",

        "TotalCharges":
            "Total Charges",

        "Dependents_No":
            "No Dependents",

        "Dependents_Yes":
            "Has Dependents",

        "InternetService_DSL":
            "DSL Internet",

        "InternetService_Fiber optic":
            "Fiber Optic Internet",

        "InternetService_No":
            "No Internet Service",

        "OnlineSecurity_No":
            "No Online Security",

        "OnlineSecurity_Yes":
            "Has Online Security",

        "OnlineSecurity_No internet service":
            "No Internet Service",

        "TechSupport_No":
            "No Tech Support",

        "TechSupport_Yes":
            "Has Tech Support",

        "TechSupport_No internet service":
            "No Internet Service",

        "Contract_Month-to-month":
            "Month-to-Month Contract",

        "Contract_One year":
            "One-Year Contract",

        "Contract_Two year":
            "Two-Year Contract",

        "PaymentMethod_Bank transfer (automatic)":
            "Automatic Bank Transfer",

        "PaymentMethod_Credit card (automatic)":
            "Automatic Credit Card",

        "PaymentMethod_Electronic check":
            "Electronic Check",

        "PaymentMethod_Mailed check":
            "Mailed Check"
    }

    return replacements.get(
        feature,
        feature
    )


# =====================================================
# RISK LEVEL
# =====================================================

def get_risk_level(probability):

    if probability >= 0.75:
        return "Critical"

    elif probability >= 0.50:
        return "High"

    elif probability >= 0.25:
        return "Medium"

    else:
        return "Low"


# =====================================================
# RETENTION RECOMMENDATION ENGINE
# =====================================================

def generate_recommendations(
    customer,
    probability
):

    recommendations = []


    # Contract
    if customer["Contract"] == "Month-to-month":

        recommendations.append(
            "Offer an annual or two-year contract with a loyalty incentive."
        )


    # Early customer
    if customer["tenure"] <= 6:

        recommendations.append(
            "Prioritize early-life-cycle engagement and onboarding support."
        )


    # Technical support
    if customer["TechSupport"] == "No":

        recommendations.append(
            "Offer a technical support plan or proactive support assistance."
        )


    # Online security
    if customer["OnlineSecurity"] == "No":

        recommendations.append(
            "Recommend an online security add-on with a targeted offer."
        )


    # Payment method
    if customer["PaymentMethod"] == "Electronic check":

        recommendations.append(
            "Encourage automatic payment methods to simplify recurring billing."
        )


    # High monthly charges
    if (
        customer["MonthlyCharges"] >= 80
        and probability >= 0.25
    ):

        recommendations.append(
            "Review the customer's service bundle and consider a personalized value offer."
        )


    # Critical risk
    if probability >= 0.75:

        recommendations.append(
            "Trigger priority retention outreach from the customer success team."
        )


    # High risk
    elif probability >= 0.50:

        recommendations.append(
            "Schedule proactive retention communication."
        )


    # Low risk
    if probability < 0.25:

        recommendations.append(
            "No immediate retention intervention required; continue normal engagement."
        )


    return recommendations


# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model_loaded": True,
        "shap_enabled": True,
        "model": "Random Forest",
        "features": 10
    }


# =====================================================
# PREDICT CHURN
# =====================================================

@app.post("/predict")
def predict_churn(customer: CustomerData):

    # -------------------------------------------------
    # Convert request to DataFrame
    # -------------------------------------------------

    customer_dict = customer.model_dump()

    customer_df = pd.DataFrame([
        customer_dict
    ])


    # -------------------------------------------------
    # Prediction
    # -------------------------------------------------

    probability = model.predict_proba(
        customer_df
    )[0][1]

    prediction = model.predict(
        customer_df
    )[0]


    # -------------------------------------------------
    # Risk Level
    # -------------------------------------------------

    risk_level = get_risk_level(
        probability
    )


    # -------------------------------------------------
    # Transform Customer
    # -------------------------------------------------

    transformed_customer = (
        preprocessor.transform(
            customer_df
        )
    )


    if hasattr(
        transformed_customer,
        "toarray"
    ):

        transformed_customer = (
            transformed_customer.toarray()
        )


    # -------------------------------------------------
    # SHAP Values
    # -------------------------------------------------

    shap_result = explainer.shap_values(
        transformed_customer
    )


    # -------------------------------------------------
    # Handle SHAP Output
    # -------------------------------------------------

    if isinstance(
        shap_result,
        np.ndarray
    ):

        if shap_result.ndim == 3:

            customer_shap = (
                shap_result[0, :, 1]
            )

        else:

            customer_shap = (
                shap_result[0]
            )

    else:

        customer_shap = (
            shap_result[1][0]
        )


    # -------------------------------------------------
    # SHAP DataFrame
    # -------------------------------------------------

    explanation = pd.DataFrame({

        "Raw_Feature":
            feature_names,

        "SHAP_Value":
            customer_shap,

        "Encoded_Value":
            transformed_customer[0]

    })


    # -------------------------------------------------
    # Keep active features
    # -------------------------------------------------

    active_features = []


    for i, feature in enumerate(
        feature_names
    ):

        encoded_value = (
            explanation.iloc[i][
                "Encoded_Value"
            ]
        )


        # Numerical features
        if feature.startswith(
            "num__"
        ):

            active_features.append(i)


        # Active one-hot feature
        elif (
            feature.startswith("cat__")
            and encoded_value > 0.5
        ):

            active_features.append(i)


    explanation = explanation.iloc[
        active_features
    ].copy()


    # -------------------------------------------------
    # Clean Names
    # -------------------------------------------------

    explanation["Feature"] = (
        explanation[
            "Raw_Feature"
        ].apply(
            clean_feature_name
        )
    )


    # -------------------------------------------------
    # Risk Drivers
    # -------------------------------------------------

    risk_drivers = (

        explanation[
            explanation["SHAP_Value"] > 0
        ]

        .sort_values(
            "SHAP_Value",
            ascending=False
        )

        .head(5)
    )


    # -------------------------------------------------
    # Protective Factors
    # -------------------------------------------------

    protective_factors = (

        explanation[
            explanation["SHAP_Value"] < 0
        ]

        .sort_values(
            "SHAP_Value"
        )

        .head(5)
    )


    # -------------------------------------------------
    # Recommendations
    # -------------------------------------------------

    recommendations = (
        generate_recommendations(
            customer_dict,
            probability
        )
    )


    # -------------------------------------------------
    # Convert Risk Drivers
    # -------------------------------------------------

    risk_driver_list = [

        {
            "feature":
                row["Feature"],

            "shap_value":
                round(
                    float(
                        row["SHAP_Value"]
                    ),
                    4
                )
        }

        for _, row
        in risk_drivers.iterrows()

    ]


    # -------------------------------------------------
    # Convert Protective Factors
    # -------------------------------------------------

    protective_factor_list = [

        {
            "feature":
                row["Feature"],

            "shap_value":
                round(
                    float(
                        row["SHAP_Value"]
                    ),
                    4
                )
        }

        for _, row
        in protective_factors.iterrows()

    ]


    # -------------------------------------------------
    # FINAL RESPONSE
    # -------------------------------------------------

    return {

        "prediction":
            "Yes"
            if prediction == 1
            else "No",

        "churn_probability":
            round(
                float(probability),
                4
            ),

        "churn_probability_percent":
            round(
                float(
                    probability * 100
                ),
                2
            ),

        "risk_level":
            risk_level,

        "risk_drivers":
            risk_driver_list,

        "protective_factors":
            protective_factor_list,

        "retention_recommendations":
            recommendations
    }


# =====================================================
# BATCH PREDICTION
# =====================================================

@app.post("/batch-predict")
async def batch_predict(
    file: UploadFile = File(...)
):

    # -------------------------------------------------
    # Validate File
    # -------------------------------------------------

    if not file.filename.lower().endswith(
        ".csv"
    ):

        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file."
        )


    try:

        # -------------------------------------------------
        # Read CSV
        # -------------------------------------------------

        contents = await file.read()

        df = pd.read_csv(
            io.BytesIO(contents)
        )


        # -------------------------------------------------
        # Required 10 Features
        # -------------------------------------------------

        required_features = [

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


        # -------------------------------------------------
        # Check Missing Columns
        # -------------------------------------------------

        missing_columns = [

            col

            for col
            in required_features

            if col not in df.columns

        ]


        if missing_columns:

            raise HTTPException(

                status_code=400,

                detail={

                    "message":
                        "Missing required columns",

                    "missing_columns":
                        missing_columns

                }
            )


        # -------------------------------------------------
        # Model Features
        # -------------------------------------------------

        customer_data = df[
            required_features
        ].copy()


        # -------------------------------------------------
        # Prediction
        # -------------------------------------------------

        probabilities = (
            model.predict_proba(
                customer_data
            )[:, 1]
        )

        predictions = (
            model.predict(
                customer_data
            )
        )


        # -------------------------------------------------
        # Add Predictions
        # -------------------------------------------------

        result = df.copy()


        result[
            "Churn_Probability"
        ] = (
            probabilities * 100
        ).round(2)


        result["Prediction"] = [

            "Yes"
            if prediction == 1
            else "No"

            for prediction
            in predictions

        ]


        result["Risk_Level"] = [

            get_risk_level(
                probability
            )

            for probability
            in probabilities

        ]


        # -------------------------------------------------
        # Recommendations
        # -------------------------------------------------

        result[
            "Retention_Recommendation"
        ] = [

            "; ".join(

                generate_recommendations(

                    customer_data.iloc[
                        i
                    ].to_dict(),

                    probabilities[i]

                )

            )

            for i
            in range(
                len(customer_data)
            )

        ]


        # -------------------------------------------------
        # Convert to CSV
        # -------------------------------------------------

        output = io.StringIO()

        result.to_csv(
            output,
            index=False
        )

        output.seek(0)


        # -------------------------------------------------
        # Return CSV
        # -------------------------------------------------

        return StreamingResponse(

            iter([
                output.getvalue()
            ]),

            media_type="text/csv",

            headers={

                "Content-Disposition":
                    "attachment; "
                    "filename=churn_predictions.csv"

            }
        )


    except HTTPException:

        raise


    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=(
                f"Batch prediction failed: {str(e)}"
            )

        )