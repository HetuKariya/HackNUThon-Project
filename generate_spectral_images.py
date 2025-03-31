import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import base64
import json
from io import BytesIO
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

print("Starting soil spectral data processing...")

os.makedirs('static/models', exist_ok=True)
os.makedirs('static/plots', exist_ok=True)
os.makedirs('static/results', exist_ok=True)

models = [
    ("0ml", 485, "Ph", 1),
    ("0ml", 560, "Nitro (mg/10 g)", 2),
    ("0ml", 560, "Posh Nitro (mg/10 g)", 3),
    ("0ml", 560, "Pota Nitro (mg/10 g)", 4),
    ("25ml", 510, "Ph", 5),
    ("25ml", 645, "Nitro (mg/10 g)", 6),
    ("25ml", 900, "Posh Nitro (mg/10 g)", 7),
    ("25ml", 900, "Pota Nitro (mg/10 g)", 8),
    ("50ml", 410, "Ph", 9),
    ("50ml", 560, "Nitro (mg/10 g)", 10),
    ("50ml", 560, "Posh Nitro (mg/10 g)", 11),
    ("50ml", 560, "Pota Nitro (mg/10 g)", 12),
]


def fig_to_base64(fig):
    img_buf = BytesIO()
    fig.savefig(img_buf, format='png', bbox_inches='tight')
    img_buf.seek(0)
    img_base64 = base64.b64encode(img_buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return img_base64


def save_plot(fig, plot_path):
    fig.savefig(plot_path, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)


def process_soil_data(df, volume, wavelength, target_var, model_number):
    print(f"\nProcessing Model {model_number}: {volume}, {wavelength}nm, {target_var}")

    df_filtered = df.copy()

    if volume != "All":
        df_filtered = df_filtered[df_filtered["Records"].astype(str).str.contains(volume, case=False, na=False)]

    if wavelength not in df_filtered.columns or target_var not in df_filtered.columns:
        print(f"⚠️ Model {model_number}: {wavelength} nm or {target_var} not found in dataset!")
        return None

    df_filtered = df_filtered[[wavelength, target_var]].dropna()

    correlation = df_filtered.corr().iloc[0, 1]
    print(f"Correlation: {correlation:.2f}")

    X = df_filtered[[wavelength]].values
    y = df_filtered[target_var].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    poly_degree = 2
    poly = PolynomialFeatures(degree=poly_degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    model_data = {
        'model': model,
        'poly': poly,
        'wavelength': wavelength,
        'target_var': target_var,
        'volume': volume
    }
    model_path = f'static/models/model_{model_number}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Model saved to {model_path}")

    y_pred = model.predict(X_test_poly)

    r2 = r2_score(y_test, y_pred) * 100
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    scale_range = max(y) - min(y)
    avg_dev = np.mean(np.abs(y_test - y_pred))
    plotting_accuracy = (1 - (avg_dev / scale_range)) * 100
    plotting_accuracy = np.clip(plotting_accuracy, 0, 100)

    plotting_precision = np.sqrt(np.mean((y_test - y_pred) ** 2))

    cv_scores = cross_val_score(model, poly.fit_transform(X), y, cv=5, scoring='r2')
    cv_r2 = np.mean(cv_scores) * 100

    print(f"R² Score: {r2:.2f}%")
    print(f"Cross-validated R² Score: {cv_r2:.2f}%")
    print(f"Mean Absolute Error: {mae:.3f}")
    print(f"Root Mean Squared Error: {rmse:.3f}")
    print(f"Plotting Accuracy: {plotting_accuracy:.2f}%")

    fig1 = plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', label=f'Actual {target_var}')
    plt.scatter(X_test, y_pred, color='red', label=f'Predicted {target_var}', alpha=0.7)
    plt.xlabel(f"{wavelength} nm Intensity")
    plt.ylabel(target_var)
    plt.title(f"Model {model_number}: Actual vs Predicted ({plotting_accuracy:.2f}% Accuracy)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot1_path = f'static/plots/model_{model_number}_actual_vs_predicted.png'
    save_plot(fig1, plot1_path)

    fig1 = plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', label=f'Actual {target_var}')
    plt.scatter(X_test, y_pred, color='red', label=f'Predicted {target_var}', alpha=0.7)
    plt.xlabel(f"{wavelength} nm Intensity")
    plt.ylabel(target_var)
    plt.title(f"Model {model_number}: Actual vs Predicted ({plotting_accuracy:.2f}% Accuracy)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plot1_img = fig_to_base64(fig1)

    fig2 = plt.figure(figsize=(10, 6))
    X_range = np.linspace(min(X), max(X), 100).reshape(-1, 1)
    X_range_poly = poly.transform(X_range)
    y_smooth = model.predict(X_range_poly)
    plt.scatter(X, y, color='blue', alpha=0.5, label='Data Points')
    plt.plot(X_range, y_smooth, color='red', linewidth=2, label='Model Prediction')
    plt.xlabel(f"{wavelength} nm Intensity")
    plt.ylabel(target_var)
    plt.title(f"Model {model_number}: {target_var} Prediction")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot2_path = f'static/plots/model_{model_number}_prediction.png'
    save_plot(fig2, plot2_path)

    fig2 = plt.figure(figsize=(10, 6))
    X_range = np.linspace(min(X), max(X), 100).reshape(-1, 1)
    X_range_poly = poly.transform(X_range)
    y_smooth = model.predict(X_range_poly)
    plt.scatter(X, y, color='blue', alpha=0.5, label='Data Points')
    plt.plot(X_range, y_smooth, color='red', linewidth=2, label='Model Prediction')
    plt.xlabel(f"{wavelength} nm Intensity")
    plt.ylabel(target_var)
    plt.title(f"Model {model_number}: {target_var} Prediction")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plot2_img = fig_to_base64(fig2)

    print(f"Plots saved to {plot1_path} and {plot2_path}")

    result = {
        'model_number': model_number,
        'volume': volume,
        'wavelength': wavelength,
        'target_var': target_var,
        'correlation': correlation,
        'r2': r2,
        'cv_r2': cv_r2,
        'mae': mae,
        'rmse': rmse,
        'plotting_accuracy': plotting_accuracy,
        'plotting_precision': plotting_precision,
        'plot1_img': plot1_img,
        'plot2_img': plot2_img
    }
    result_data = {k: v for k, v in result.items() if k not in ['plot1_img', 'plot2_img']}
    result_path = f'static/results/model_{model_number}_results.json'
    with open(result_path, 'w') as f:
        json.dump(result_data, f, indent=4)
    print(f"Results saved to {result_path}")

    return result


def process_all_models():
    print("Loading dataset...")
    try:
        file_path = "Copy of sorted_soildataset(1).xlsx"
        xls = pd.ExcelFile(file_path)
        df = pd.read_excel(xls, sheet_name="Sheet1")
        print(f"Dataset loaded successfully with {len(df)} rows")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return []

    results = []

    for volume, wavelength, target_var, model_number in models:
        result = process_soil_data(df, volume, wavelength, target_var, model_number)
        if result:
            results.append(result)

    with open('static/results/all_model_results.json', 'w') as f:
        json_results = []
        for result in results:
            json_result = {k: v for k, v in result.items() if k not in ['plot1_img', 'plot2_img']}
            json_results.append(json_result)
        json.dump(json_results, f, indent=4)

    for result in results:
        model_num = result['model_number']
        with open(f'static/results/model_{model_num}_plot1.txt', 'w') as f:
            f.write(result['plot1_img'])
        with open(f'static/results/model_{model_num}_plot2.txt', 'w') as f:
            f.write(result['plot2_img'])

    print(f"All {len(results)} model results saved to static/results/")
    return results


if __name__ == "__main__":
    print("Starting soil data processing and model generation...")
    results = process_all_models()
    print(f"Processed {len(results)} models successfully")
    print("Done! You can now run your Flask application to view the results.")