DATA_PATH = "../data/Rice2024.xlsx"
TARGET_COLUMN = "Class"
FEATURE_COLUMNS = [
    "Area",
    "Perimeter",
    "Major_Axis_Length",
    "Minor_Axis_Length",
    "Eccentricity",
    "Convex_Area",
    "Extent"
]
DATA_OUTPUT = "rice_processed.pkl"
FEATURES_PATH = "rice_feature_columns.pkl"
DESCRIPTION_PATH = "rice_data_description.pkl"
MODEL_PATH = "random_forest_rice_model.pkl"
SCALER_PATH = "scaler.pkl"