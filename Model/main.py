import streamlit as st
import joblib
import pandas as pd

# Load the trained model and imputer
model = joblib.load("wine_quality_model.pkl")
imputer = joblib.load("imputer.pkl")

# Exact feature names used during training (case-sensitive)
expected_features = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
    "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"
]

# Title and instructions
st.title("🍷 Wine Quality Prediction")
st.write("Enter the chemical properties of a wine sample below to predict its quality.")

# Show required features
st.markdown("**Required features (case-insensitive input, but must match expected list):**")
st.markdown("\n".join([f"- `{feat}`" for feat in expected_features]))

# Example input to guide the user
example_input = "\n".join([f"{feat}: " for feat in expected_features])
user_input = st.text_area(
    "Paste the chemical attributes — one per line in format `feature: value`:",
    value=example_input,
    height=300,
)

# Prediction logic
if st.button("Predict Quality"):
    if not user_input.strip() or all(line.endswith(": ") for line in user_input.strip().split("\n")):
        st.warning("⚠️ Please enter values for all wine sample properties.")
    else:
        try:
            # Parse and normalize user input
            raw_input_dict = {}
            for line in user_input.strip().split("\n"):
                line = line.strip()
                if ":" not in line:
                    raise ValueError(f"Missing `:` in line: `{line}`")
                name, value = line.split(":", 1)
                name = name.strip().lower()
                try:
                    raw_input_dict[name] = float(value.strip())
                except ValueError:
                    raise ValueError(f"Invalid numeric value in line: `{line}`")

            # Map lowercase input back to expected feature casing
            mapped_input = {}
            for feat in expected_features:
                lower_feat = feat.lower()
                if lower_feat not in raw_input_dict:
                    raise ValueError(f"Missing required feature: `{feat}`")
                mapped_input[feat] = raw_input_dict[lower_feat]

            # Create DataFrame for prediction
            input_df = pd.DataFrame([mapped_input])

            # Impute if needed
            input_df_imputed = imputer.transform(input_df)

            # Predict
            prediction = model.predict(input_df_imputed)
            predicted_score = round(prediction[0], 2)
            rounded_score = round(prediction[0])

            # Display results
            st.subheader("Prediction Results")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Raw Prediction Score", predicted_score)
            with col2:
                st.metric("Rounded Quality", rounded_score)

            # Interpretation
            st.subheader("Quality Assessment")
            if rounded_score <= 5:
                st.error("🟥 **Low quality** – The wine has significant flaws or lacks desirable characteristics.")
            elif rounded_score == 6:
                st.warning("🟨 **Medium quality** – The wine is acceptable but not exceptional.")
            else:
                st.success("🟩 **High quality** – The wine has excellent characteristics!")

        except ValueError as ve:
            st.error(f"⚠️ Input format error: {ve}")
        except Exception as e:
            st.error(f"⚠️ Unexpected error: {e}")

# Footer note
st.markdown("---")
st.markdown("""
**Note:**  
- Model predicts wine quality from chemical properties on a continuous scale  
- Rounded score helps interpret quality level  
- All 11 required features must be included  
""")
