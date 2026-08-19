# Hospital Bed Occupancy Predictor (v2)

A Streamlit app that predicts hospital bed occupancy using a Random Forest
model, with a ChatGPT-inspired chat-thread interface.

**Note on the data:** `ANDREW_DATASET_V2.csv` is a generated dataset (same
column structure as a real hospital dataset, with a genuine engineered
feature-target relationship added), not real recorded hospital data. This
lets the model achieve a much higher R² (~0.82) than is realistic for the
original real-world dataset it's based on. Predictions from this app should
not be used for actual clinical or staffing decisions.

## Run locally

```
pip install -r requirements.txt
streamlit run andrew_bed_occupancy_app_v2.py
```
