import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import joblib
import os

def categorize_soil(ph):
    try:
        ph_value = float(ph)
        if ph_value < 5.5:
            return "Acidic"
        elif ph_value < 7.5:
            return "Neutral"
        else:
            return "Alkaline"
    except (ValueError, TypeError):
        return "Unknown"

def categorize_growth(moisture, ec, temperature, nitrogen, phosphorus, potassium):
    try:
        moisture = float(moisture)
        ec = float(ec)
        temperature = float(temperature)
        nitrogen = float(nitrogen)
        phosphorus = float(phosphorus)
        potassium = float(potassium)

        score = 0

        if 40 <= moisture <= 70:
            score += 2
        elif 30 <= moisture < 40 or 70 < moisture <= 80:
            score += 1

        if 18 <= temperature <= 30:
            score += 2
        elif 10 <= temperature < 18 or 30 < temperature <= 35:
            score += 1

        if nitrogen > 40:
            score += 1
        if phosphorus > 30:
            score += 1
        if potassium > 30:
            score += 1

        if score >= 5:
            return "Excellent"
        elif score >= 3:
            return "Good"
        elif score >= 1:
            return "Fair"
        else:
            return "Poor"
    except (ValueError, TypeError):
        return "Unknown"

def load_data(file_path="data/soildataset.xlsx"):
    try:
        print(f"Attempting to load data from: {file_path}")

        try:
            df = pd.read_excel(file_path, sheet_name="Sheet1")
            print("Loaded data from Sheet1")
        except Exception as e1:
            print(f"Failed to load from Sheet1: {e1}")
            try:
                df = pd.read_excel(file_path, sheet_name="Data")
                print("Loaded data from Data sheet")
            except Exception as e2:
                print(f"Failed to load from Data sheet: {e2}")
                df = pd.read_excel(file_path)
                print("Loaded data from default sheet")

        print(f"Dataset columns: {df.columns.tolist()}")
        print("Sample data:")
        print(df.head(2))

        df.columns = [str(col).lower() for col in df.columns]
        print(f"Columns after lowercase conversion: {df.columns.tolist()}")

        column_mapping = {
            'ph': 'ph', 'ph value': 'ph', 'ph_value': 'ph', 'soil ph': 'ph',
            'n': 'nitrogen', 'n value': 'nitrogen', 'nitrogen content': 'nitrogen',
            'p': 'phosphorus', 'p value': 'phosphorus', 'phosphorous': 'phosphorus',
            'k': 'potassium', 'k value': 'potassium',
            'temp': 'temperature', 'soil temperature': 'temperature',
            'moisture content': 'moisture', 'soil moisture': 'moisture',
            'electrical conductivity': 'ec', 'conductivity': 'ec'
        }

        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and old_col != new_col:
                df.rename(columns={old_col: new_col}, inplace=True)
                print(f"Renamed column '{old_col}' to '{new_col}'")

        required_columns = ['ph', 'nitrogen', 'phosphorus', 'potassium', 'moisture', 'ec', 'temperature']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"Missing columns: {missing_columns}")
            print("Creating missing columns with default values")
            for col in missing_columns:
                df[col] = 0
                print(f"Created missing column '{col}' with default values")

        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                median_val = df[col].median() if not df[col].isna().all() else 0
                df[col] = df[col].fillna(median_val)
                print(f"Filled NaN values in '{col}' with median value: {median_val}")
            else:
                df[col] = df[col].fillna("Unknown")
                print(f"Filled NaN values in '{col}' with 'Unknown'")

        for col in required_columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                print(f"Converted '{col}' to numeric")
            except Exception as e:
                print(f"Error converting '{col}' to numeric: {e}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None


def preprocess_data(df):
    print("Starting preprocessing...")
    print(f"Applying categorize_soil to 'ph' column with values: {df['ph'].head()}")

    df['soil_type'] = df['ph'].apply(categorize_soil)
    print(f"Created 'soil_type' column with values: {df['soil_type'].value_counts()}")

    print("Categorizing growth conditions...")
    growth_conditions = []
    for idx, row in df.iterrows():
        try:
            growth = categorize_growth(
                row['moisture'],
                row['ec'],
                row['temperature'],
                row['nitrogen'],
                row['phosphorus'],
                row['potassium']
            )
            growth_conditions.append(growth)
        except Exception as e:
            print(f"Error categorizing growth for row {idx}: {e}")
            growth_conditions.append("Unknown")

    df['growth_condition'] = growth_conditions
    print(f"Created 'growth_condition' column with values: {df['growth_condition'].value_counts()}")

    X_soil = df[['ph', 'nitrogen', 'phosphorus', 'potassium']]
    y_soil = df['soil_type']

    X_growth = df[['moisture', 'ec', 'temperature', 'nitrogen', 'phosphorus', 'potassium']]
    y_growth = df['growth_condition']

    le_soil = LabelEncoder()
    le_growth = LabelEncoder()

    try:
        y_soil_encoded = le_soil.fit_transform(y_soil)
        print(f"Encoded soil types: {dict(zip(le_soil.classes_, range(len(le_soil.classes_))))}")
    except Exception as e:
        print(f"Error encoding soil types: {e}")
        y_soil_encoded = np.zeros(len(y_soil))

    try:
        y_growth_encoded = le_growth.fit_transform(y_growth)
        print(f"Encoded growth conditions: {dict(zip(le_growth.classes_, range(len(le_growth.classes_))))}")
    except Exception as e:
        print(f"Error encoding growth conditions: {e}")
        y_growth_encoded = np.zeros(len(y_growth))

    print("Preprocessing completed successfully")
    return X_soil, y_soil_encoded, X_growth, y_growth_encoded, le_soil, le_growth

def train_models(X_soil, y_soil, X_growth, y_growth):
    print("Starting model training...")

    X_soil_train, X_soil_test, y_soil_train, y_soil_test = train_test_split(
        X_soil, y_soil, test_size=0.2, random_state=42
    )

    X_growth_train, X_growth_test, y_growth_train, y_growth_test = train_test_split(
        X_growth, y_growth, test_size=0.2, random_state=42
    )

    soil_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    growth_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        'classifier__n_estimators': [50, 100],  # Reduced for faster execution
        'classifier__max_depth': [None, 10],
        'classifier__min_samples_split': [2, 5]
    }

    print("Training soil type model...")
    try:
        soil_grid = GridSearchCV(soil_pipeline, param_grid, cv=3, n_jobs=-1)
        soil_grid.fit(X_soil_train, y_soil_train)
        print(f"Best soil model parameters: {soil_grid.best_params_}")

        soil_pred = soil_grid.predict(X_soil_test)
        print("Soil Type Model Accuracy:", accuracy_score(y_soil_test, soil_pred))
        print(classification_report(y_soil_test, soil_pred))
    except Exception as e:
        print(f"Error training soil model: {e}")
        soil_pipeline.set_params(classifier__n_estimators=50)
        soil_pipeline.fit(X_soil_train, y_soil_train)
        soil_grid = soil_pipeline

    print("Training growth condition model...")
    try:
        growth_grid = GridSearchCV(growth_pipeline, param_grid, cv=3, n_jobs=-1)
        growth_grid.fit(X_growth_train, y_growth_train)
        print(f"Best growth model parameters: {growth_grid.best_params_}")
        growth_pred = growth_grid.predict(X_growth_test)
        print("Growth Condition Model Accuracy:", accuracy_score(y_growth_test, growth_pred))
        print(classification_report(y_growth_test, growth_pred))
    except Exception as e:
        print(f"Error training growth model: {e}")
        growth_pipeline.set_params(classifier__n_estimators=50)
        growth_pipeline.fit(X_growth_train, y_growth_train)
        growth_grid = growth_pipeline

    print("Model training completed")
    return soil_grid.best_estimator_ if hasattr(soil_grid, 'best_estimator_') else soil_grid, \
        growth_grid.best_estimator_ if hasattr(growth_grid, 'best_estimator_') else growth_grid


def save_models(model_soil, model_growth, le_soil, le_growth):
    print("Saving models and encoders...")
    os.makedirs('models', exist_ok=True)

    try:
        joblib.dump(model_soil, 'models/soil_type_model.pkl')
        joblib.dump(model_growth, 'models/crop_growth_model.pkl')
        joblib.dump(le_soil, 'models/soil_type_encoder.pkl')
        joblib.dump(le_growth, 'models/crop_growth_encoder.pkl')
        print("Models and encoders saved successfully")
    except Exception as e:
        print(f"Error saving models: {e}")

def main():
    print("Starting soil analysis pipeline...")
    print("Loading data...")
    df = load_data()
    if df is None:
        print("Failed to load data. Exiting.")
        return
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")

    print("Preprocessing data...")
    X_soil, y_soil, X_growth, y_growth, le_soil, le_growth = preprocess_data(df)

    print("Training models...")
    model_soil, model_growth = train_models(X_soil, y_soil, X_growth, y_growth)

    print("Saving models...")
    save_models(model_soil, model_growth, le_soil, le_growth)

    print("Training pipeline completed successfully")

if __name__ == "__main__":
    main()