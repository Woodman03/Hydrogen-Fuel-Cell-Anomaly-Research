# NASA Hydrogen Fuel Cell Anomaly Research (Unsupervised Machine Learning)
# Dillon Wood - Arkansas Tech University - B.S. Electrical Engineering 2025

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.neighbors import LocalOutlierFactor

from sklearn.svm import OneClassSVM

import datetime

import seaborn as sns

# Load and prepare the battery dataset.
df_full = pd.read_csv('nasabattery_enhanced_input_20_percent.csv')  # Load data.
df_full['datetime'] = pd.to_datetime(df_full['datetime'])  # Convert datetime column.
df_full.dropna(inplace=True)  # Remove rows with missing values.

df = df_full.sample(frac=1, random_state=42)

# Define features used for anomaly detection.
features = [
    'cycle', 'ambient_temperature', 'capacity',
    'voltage_measured', 'current_measured',
    'temperature_measured', 'current_load', 'voltage_load', 'measured_power',
    'load_power', 'power_difference', 'average_power', 'temperature_difference',
    'measured_resistance', 'load_resistance', 'voltage_efficiency'
]

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(df[features])

# Initialize models
models = {
    'Isolation Forest': IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42
    ),
    'Local Outlier Factor': LocalOutlierFactor(
        n_neighbors=20,
        contamination=0.05,
        novelty=False
    ),
    'One-Class SVM': OneClassSVM(
        nu=0.05,
        kernel='rbf',
        gamma=0.1
    )
}

# Detect anomalies with all models
results = {}
for name, model in models.items():
    if name == 'Local Outlier Factor':
        anomalies = model.fit_predict(X)
    else:
        anomalies = model.fit(X).predict(X)
    results[name] = anomalies
    df[f'anomaly_{name.lower().replace(" ", "_")}'] = np.where(anomalies == -1, 1, 0)

# Create equally weighted anomaly column
df['anomaly_score'] = (
    df['anomaly_isolation_forest'] * (0.3885) +
    df['anomaly_local_outlier_factor'] * (0.1953) +
    df['anomaly_one-class_svm'] * (0.4162)
)

# Convert anomaly scores to binary anomaly labels (0 or 1)
df['anomaly'] = np.where(df['anomaly_score'] > 0.5, 1, 0)

# Extract anomalies (now 1 for anomaly, 0 for normal)
anomalies = df[df['anomaly'] == 1]


# -------------------- Capacity Anomaly Detection --------------------

# Visualize capacity degradation with anomalies (FULL DATA)
plt.figure(figsize=(12, 6))
# Creates a new figure for the plot with a specified size (12x6 inches).

plt.plot(df_full['cycle'], df_full['capacity'], label='Normal', color='blue', alpha=0.6)
# Plots the capacity degradation of the full dataset.
# 'df_full['cycle']' represents the x-axis (cycle number), and 'df_full['capacity']' represents the y-axis (capacity).
# 'label='Normal'' and 'color='blue'' assign a label and color to the plotted line.

plt.scatter(
    anomalies['cycle'], anomalies['capacity'],
    color='red', label='Anomaly', marker='x'
)
# Plots the detected anomalies on the same graph as scatter points.
# 'anomalies['cycle']' and 'anomalies['capacity']' provide the coordinates of the anomalies.
# 'color='red'', 'label='Anomaly'', and 'marker='x'' style the scatter points.

plt.xlabel('Cycle')
plt.ylabel('Capacity')
plt.title('Battery Capacity Degradation with Anomalies')
# Sets the labels for the x-axis, y-axis, and the title of the plot.

plt.legend()
# Displays a legend that explains the meaning of the plotted lines and points (Normal and Anomaly).

plt.grid(True)
# Adds a grid to the plot to improve readability.

plt.show(block=False)
# Displays the plot.
# block=False allows the script to continue execution without waiting for the plot window to close.

plt.pause(0.1)
# Pauses the script for 0.1 seconds, giving the plot window time to render.


# # -------------------- Measured Voltage Anomaly Detection --------------------

# plt.figure(figsize=(12, 6))
# # Creates a new figure for the plot with a specified size (12x6 inches).

# plt.plot(df_full['cycle'], df_full['voltage_measured'], label='Normal', color='blue', alpha=0.6)


# plt.scatter(
#     anomalies['cycle'], anomalies['voltage_measured'],
#     color='red', label='Anomaly', marker='x'
# )
# # Plots the detected anomalies on the same graph as scatter points.
# # 'anomalies['cycle']' and 'anomalies['voltage_measured']' provide the coordinates of the anomalies.
# # 'color='red'', 'label='Anomaly'', and 'marker='x'' style the scatter points.

# plt.xlabel('Cycle')
# plt.ylabel('Voltage Measured')
# plt.title('Voltage Measured with Anomalies')
# # Sets the labels for the x-axis, y-axis, and the title of the plot.

# plt.legend()
# # Displays a legend that explains the meaning of the plotted lines and points (Normal and Anomaly).

# plt.grid(True)
# # Adds a grid to the plot to improve readability.

# plt.show(block=False)
# # Displays the plot.
# # block=False allows the script to continue execution without waiting for the plot window to close.

# plt.pause(0.1)
# # Pauses the script for 0.1 seconds, giving the plot window time to render.


# # -------------------- Measured Current Anomaly Detection --------------------

# plt.figure(figsize=(12, 6))
# # Creates a new figure for the plot with a specified size (12x6 inches).

# plt.plot(df_full['cycle'], df_full['current_measured'], label='Normal', color='blue', alpha=0.6)


# plt.scatter(
#     anomalies['cycle'], anomalies['current_measured'],
#     color='red', label='Anomaly', marker='x'
# )
# # Plots the detected anomalies on the same graph as scatter points.
# # 'anomalies['cycle']' and 'anomalies['current_measured']' provide the coordinates of the anomalies.
# # 'color='red'', 'label='Anomaly'', and 'marker='x'' style the scatter points.

# plt.xlabel('Cycle')
# plt.ylabel('Current Measured')
# plt.title('Current Measured with Anomalies')
# # Sets the labels for the x-axis, y-axis, and the title of the plot.

# plt.legend()
# # Displays a legend that explains the meaning of the plotted lines and points (Normal and Anomaly).

# plt.grid(True)
# # Adds a grid to the plot to improve readability.

# plt.show(block=False)
# # Displays the plot.
# # block=False allows the script to continue execution without waiting for the plot window to close.

# plt.pause(0.1)
# # Pauses the script for 0.1 seconds, giving the plot window time to render.


# -------------------- Measured Power Anomaly Detection --------------------

# Visualize measured power with anomalies (FULL DATA)
plt.figure(figsize=(12, 6))
# Creates a new figure for the plot with a size of 12x6 inches.

plt.plot(df_full['cycle'], df_full['measured_power'], label='Normal', color='blue', alpha=0.6) #Make the blue line semi transparent
# Plots the measured power of the full dataset against the cycle number.
# 'df_full['cycle']' is the x-axis, and 'df_full['measured_power']' is the y-axis.
# 'label='Normal'' and 'color='blue'' style the normal data line.

plt.scatter(
    anomalies['cycle'], anomalies['measured_power'],
    color='red', label='Anomaly', marker='x'
)
# Overlays the detected anomalies (from the sampled data) as red 'x' markers.
# 'anomalies['cycle']' and 'anomalies['measured_power']' provide the anomaly coordinates.

plt.xlabel('Cycle')
plt.ylabel('Measured Power')
plt.title('Measured Power with Anomalies')
# Sets the x-axis label, y-axis label, and plot title.

plt.legend()
# Displays a legend to differentiate between normal data and anomalies.

plt.grid(True)
# Adds a grid to the plot for better readability.

plt.show(block=False)
# Displays the plot without blocking the script's execution.

plt.pause(0.1)
# Pauses the script for 0.1 seconds to allow the plot to render.


# -------------------- Measured Temperature Anomaly Detection --------------------

# Visualize measured power with anomalies (FULL DATA)
plt.figure(figsize=(12, 6))
# Creates a new figure for the plot with a size of 12x6 inches.

plt.plot(df_full['cycle'], df_full['temperature_measured'], label='Normal', color='blue', alpha=0.6) #Make the blue line semi transparent
# Plots the measured power of the full dataset against the cycle number.
# 'df_full['cycle']' is the x-axis, and 'df_full['measured_power']' is the y-axis.
# 'label='Normal'' and 'color='blue'' style the normal data line.

plt.scatter(
    anomalies['cycle'], anomalies['temperature_measured'],
    color='red', label='Anomaly', marker='x'
)
# Overlays the detected anomalies (from the sampled data) as red 'x' markers.
# 'anomalies['cycle']' and 'anomalies['measured_power']' provide the anomaly coordinates.

plt.xlabel('Cycle')
plt.ylabel('Measured Temperature')
plt.title('Measured Temperature with Anomalies')
# Sets the x-axis label, y-axis label, and plot title.

plt.legend()
# Displays a legend to differentiate between normal data and anomalies.

plt.grid(True)
# Adds a grid to the plot for better readability.

plt.show(block=False)
# Displays the plot without blocking the script's execution.

plt.pause(0.1)
# Pauses the script for 0.1 seconds to allow the plot to render.


# -------------------- Load Voltage Anomaly Detection --------------------

# Visualize measured power with anomalies (FULL DATA)
plt.figure(figsize=(12, 6))
# Creates a new figure for the plot with a size of 12x6 inches.

plt.plot(df_full['cycle'], df_full['voltage_load'], label='Normal', color='blue', alpha=0.6) #Make the blue line semi transparent
# Plots the measured power of the full dataset against the cycle number.
# 'df_full['cycle']' is the x-axis, and 'df_full['measured_power']' is the y-axis.
# 'label='Normal'' and 'color='blue'' style the normal data line.

plt.scatter(
    anomalies['cycle'], anomalies['voltage_load'],
    color='red', label='Anomaly', marker='x'
)
# Overlays the detected anomalies (from the sampled data) as red 'x' markers.
# 'anomalies['cycle']' and 'anomalies['measured_power']' provide the anomaly coordinates.

plt.xlabel('Cycle')
plt.ylabel('Load Voltage')
plt.title('Load Voltage with Anomalies')
# Sets the x-axis label, y-axis label, and plot title.

plt.legend()
# Displays a legend to differentiate between normal data and anomalies.

plt.grid(True)
# Adds a grid to the plot for better readability.

plt.show(block=False)
# Displays the plot without blocking the script's execution.

plt.pause(0.1)
# Pauses the script for 0.1 seconds to allow the plot to render.


# -------------------- Load Current Anomaly Detection --------------------

# Visualize measured power with anomalies (FULL DATA)
plt.figure(figsize=(12, 6))
# Creates a new figure for the plot with a size of 12x6 inches.

plt.plot(df_full['cycle'], df_full['current_load'], label='Normal', color='blue', alpha=0.6) #Make the blue line semi transparent
# Plots the measured power of the full dataset against the cycle number.
# 'df_full['cycle']' is the x-axis, and 'df_full['measured_power']' is the y-axis.
# 'label='Normal'' and 'color='blue'' style the normal data line.

plt.scatter(
    anomalies['cycle'], anomalies['current_load'],
    color='red', label='Anomaly', marker='x'
)
# Overlays the detected anomalies (from the sampled data) as red 'x' markers.
# 'anomalies['cycle']' and 'anomalies['measured_power']' provide the anomaly coordinates.

plt.xlabel('Cycle')
plt.ylabel('Load Current')
plt.title('Load Current with Anomalies')
# Sets the x-axis label, y-axis label, and plot title.

plt.legend()
# Displays a legend to differentiate between normal data and anomalies.

plt.grid(True)
# Adds a grid to the plot for better readability.

plt.show(block=False)
# Displays the plot without blocking the script's execution.

plt.pause(0.1)
# Pauses the script for 0.1 seconds to allow the plot to render.


# -------------------- Load Power Anomaly Detection --------------------

# Visualize measured power with anomalies (FULL DATA)
plt.figure(figsize=(12, 6))
# Creates a new figure for the plot with a size of 12x6 inches.

plt.plot(df_full['cycle'], df_full['load_power'], label='Normal', color='blue', alpha=0.6) #Make the blue line semi transparent
# Plots the measured power of the full dataset against the cycle number.
# 'df_full['cycle']' is the x-axis, and 'df_full['measured_power']' is the y-axis.
# 'label='Normal'' and 'color='blue'' style the normal data line.

plt.scatter(
    anomalies['cycle'], anomalies['load_power'],
    color='red', label='Anomaly', marker='x'
)
# Overlays the detected anomalies (from the sampled data) as red 'x' markers.
# 'anomalies['cycle']' and 'anomalies['measured_power']' provide the anomaly coordinates.

plt.xlabel('Cycle')
plt.ylabel('Load Power')
plt.title('Load Power with Anomalies')
# Sets the x-axis label, y-axis label, and plot title.

plt.legend()
# Displays a legend to differentiate between normal data and anomalies.

plt.grid(True)
# Adds a grid to the plot for better readability.

plt.show(block=False)
# Displays the plot without blocking the script's execution.

plt.pause(0.1)
# Pauses the script for 0.1 seconds to allow the plot to render.


# -------------------- Power Difference Anomaly Detection --------------------

# Visualize measured power with anomalies (FULL DATA)
plt.figure(figsize=(12, 6))
# Creates a new figure for the plot with a size of 12x6 inches.

plt.plot(df_full['cycle'], df_full['power_difference'], label='Normal', color='blue', alpha=0.6) #Make the blue line semi transparent
# Plots the measured power of the full dataset against the cycle number.
# 'df_full['cycle']' is the x-axis, and 'df_full['measured_power']' is the y-axis.
# 'label='Normal'' and 'color='blue'' style the normal data line.

plt.scatter(
    anomalies['cycle'], anomalies['power_difference'],
    color='red', label='Anomaly', marker='x'
)
# Overlays the detected anomalies (from the sampled data) as red 'x' markers.
# 'anomalies['cycle']' and 'anomalies['measured_power']' provide the anomaly coordinates.

plt.xlabel('Cycle')
plt.ylabel('Power Difference')
plt.title('Power Difference with Anomalies')
# Sets the x-axis label, y-axis label, and plot title.

plt.legend()
# Displays a legend to differentiate between normal data and anomalies.

plt.grid(True)
# Adds a grid to the plot for better readability.

plt.show(block=False)
# Displays the plot without blocking the script's execution.

plt.pause(0.1)
# Pauses the script for 0.1 seconds to allow the plot to render.


# -------------------- Average Power Anomaly Detection --------------------

# Visualize measured power with anomalies (FULL DATA)
plt.figure(figsize=(12, 6))
# Creates a new figure for the plot with a size of 12x6 inches.

plt.plot(df_full['cycle'], df_full['average_power'], label='Normal', color='blue', alpha=0.6) #Make the blue line semi transparent
# Plots the measured power of the full dataset against the cycle number.
# 'df_full['cycle']' is the x-axis, and 'df_full['measured_power']' is the y-axis.
# 'label='Normal'' and 'color='blue'' style the normal data line.

plt.scatter(
    anomalies['cycle'], anomalies['average_power'],
    color='red', label='Anomaly', marker='x'
)
# Overlays the detected anomalies (from the sampled data) as red 'x' markers.
# 'anomalies['cycle']' and 'anomalies['measured_power']' provide the anomaly coordinates.

plt.xlabel('Cycle')
plt.ylabel('Average Power')
plt.title('Average Power with Anomalies')
# Sets the x-axis label, y-axis label, and plot title.

plt.legend()
# Displays a legend to differentiate between normal data and anomalies.

plt.grid(True)
# Adds a grid to the plot for better readability.

plt.show(block=False)
# Displays the plot without blocking the script's execution.

plt.pause(0.1)
# Pauses the script for 0.1 seconds to allow the plot to render.


# -------------------- Temperature Difference Anomaly Detection --------------------

# Visualize measured power with anomalies (FULL DATA)
plt.figure(figsize=(12, 6))
# Creates a new figure for the plot with a size of 12x6 inches.

plt.plot(df_full['cycle'], df_full['temperature_difference'], label='Normal', color='blue', alpha=0.6) #Make the blue line semi transparent
# Plots the measured power of the full dataset against the cycle number.
# 'df_full['cycle']' is the x-axis, and 'df_full['measured_power']' is the y-axis.
# 'label='Normal'' and 'color='blue'' style the normal data line.

plt.scatter(
    anomalies['cycle'], anomalies['temperature_difference'],
    color='red', label='Anomaly', marker='x'
)
# Overlays the detected anomalies (from the sampled data) as red 'x' markers.
# 'anomalies['cycle']' and 'anomalies['measured_power']' provide the anomaly coordinates.

plt.xlabel('Cycle')
plt.ylabel('Temperature Difference')
plt.title('Temperature Difference with Anomalies')
# Sets the x-axis label, y-axis label, and plot title.

plt.legend()
# Displays a legend to differentiate between normal data and anomalies.

plt.grid(True)
# Adds a grid to the plot for better readability.

plt.show(block=False)
# Displays the plot without blocking the script's execution.

plt.pause(0.1)
# Pauses the script for 0.1 seconds to allow the plot to render.


# # -------------------- Measured Resistance Anomaly Detection --------------------

# # Visualize measured power with anomalies (FULL DATA)
# plt.figure(figsize=(12, 6))
# # Creates a new figure for the plot with a size of 12x6 inches.

# plt.plot(df_full['cycle'], df_full['measured_resistance'], label='Normal', color='blue', alpha=0.6) #Make the blue line semi transparent
# # Plots the measured power of the full dataset against the cycle number.
# # 'df_full['cycle']' is the x-axis, and 'df_full['measured_power']' is the y-axis.
# # 'label='Normal'' and 'color='blue'' style the normal data line.

# plt.scatter(
#     anomalies['cycle'], anomalies['measured_resistance'],
#     color='red', label='Anomaly', marker='x'
# )
# # Overlays the detected anomalies (from the sampled data) as red 'x' markers.
# # 'anomalies['cycle']' and 'anomalies['measured_power']' provide the anomaly coordinates.

# plt.xlabel('Cycle')
# plt.ylabel('Measured Resistance')
# plt.title('Measured Resistance with Anomalies')
# # Sets the x-axis label, y-axis label, and plot title.

# plt.legend()
# # Displays a legend to differentiate between normal data and anomalies.

# plt.grid(True)
# # Adds a grid to the plot for better readability.

# plt.show(block=False)
# # Displays the plot without blocking the script's execution.

# plt.pause(0.1)
# # Pauses the script for 0.1 seconds to allow the plot to render.


# -------------------- Load Resistance Anomaly Detection --------------------

# Visualize measured power with anomalies (FULL DATA)
plt.figure(figsize=(12, 6))
# Creates a new figure for the plot with a size of 12x6 inches.

plt.plot(df_full['cycle'], df_full['load_resistance'], label='Normal', color='blue', alpha=0.6) #Make the blue line semi transparent
# Plots the measured power of the full dataset against the cycle number.
# 'df_full['cycle']' is the x-axis, and 'df_full['measured_power']' is the y-axis.
# 'label='Normal'' and 'color='blue'' style the normal data line.

plt.scatter(
    anomalies['cycle'], anomalies['load_resistance'],
    color='red', label='Anomaly', marker='x'
)
# Overlays the detected anomalies (from the sampled data) as red 'x' markers.
# 'anomalies['cycle']' and 'anomalies['measured_power']' provide the anomaly coordinates.

plt.xlabel('Cycle')
plt.ylabel('Load Resistance')
plt.title('Load Resistance with Anomalies')
# Sets the x-axis label, y-axis label, and plot title.

plt.legend()
# Displays a legend to differentiate between normal data and anomalies.

plt.grid(True)
# Adds a grid to the plot for better readability.

plt.show(block=False)
# Displays the plot without blocking the script's execution.

plt.pause(0.1)
# Pauses the script for 0.1 seconds to allow the plot to render.


# # -------------------- Voltage Efficiency Anomaly Detection --------------------

# # Visualize measured power with anomalies (FULL DATA)
# plt.figure(figsize=(12, 6))
# # Creates a new figure for the plot with a size of 12x6 inches.

# plt.plot(df_full['cycle'], df_full['voltage_efficiency'], label='Normal', color='blue', alpha=0.6) #Make the blue line semi transparent
# # Plots the measured power of the full dataset against the cycle number.
# # 'df_full['cycle']' is the x-axis, and 'df_full['measured_power']' is the y-axis.
# # 'label='Normal'' and 'color='blue'' style the normal data line.

# plt.scatter(
#     anomalies['cycle'], anomalies['voltage_efficiency'],
#     color='red', label='Anomaly', marker='x'
# )
# # Overlays the detected anomalies (from the sampled data) as red 'x' markers.
# # 'anomalies['cycle']' and 'anomalies['measured_power']' provide the anomaly coordinates.

# plt.xlabel('Cycle')
# plt.ylabel('Voltage Efficiency')
# plt.title('Voltage Efficiency with Anomalies')
# # Sets the x-axis label, y-axis label, and plot title.

# plt.legend()
# # Displays a legend to differentiate between normal data and anomalies.

# plt.grid(True)
# # Adds a grid to the plot for better readability.

# plt.show(block=False)
# # Displays the plot without blocking the script's execution.

# plt.pause(0.1)
# # Pauses the script for 0.1 seconds to allow the plot to render.


# # -------------------- Measured Voltage vs Measured Current Anomaly Detection --------------------

# # Visualize measured power with anomalies (FULL DATA)
# plt.figure(figsize=(12, 6))
# # Creates a new figure for the plot with a size of 12x6 inches.

# plt.plot(df_full['current_measured'], df_full['voltage_measured'], label='Normal', color='blue', alpha=0.6) #Make the blue line semi transparent
# # Plots the measured power of the full dataset against the cycle number.
# # 'df_full['cycle']' is the x-axis, and 'df_full['measured_power']' is the y-axis.
# # 'label='Normal'' and 'color='blue'' style the normal data line.

# plt.scatter(
#     anomalies['current_measured'], anomalies['voltage_measured'],
#     color='red', label='Anomaly', marker='x'
# )
# # Overlays the detected anomalies (from the sampled data) as red 'x' markers.
# # 'anomalies['cycle']' and 'anomalies['measured_power']' provide the anomaly coordinates.

# plt.xlabel('Measured Current')
# plt.ylabel('Measured Voltage')
# plt.title('Measured Voltage vs Measured Current with Anomalies')
# # Sets the x-axis label, y-axis label, and plot title.

# plt.legend()
# # Displays a legend to differentiate between normal data and anomalies.

# plt.grid(True)
# # Adds a grid to the plot for better readability.

# plt.show(block=False)
# # Displays the plot without blocking the script's execution.

# plt.pause(0.1)
# # Pauses the script for 0.1 seconds to allow the plot to render.


# # -------------------- Measured Power vs Measured Current Anomaly Detection --------------------

# # Visualize measured power with anomalies (FULL DATA)
# plt.figure(figsize=(12, 6))
# # Creates a new figure for the plot with a size of 12x6 inches.

# plt.plot(df_full['current_measured'], df_full['measured_power'], label='Normal', color='blue', alpha=0.6) #Make the blue line semi transparent
# # Plots the measured power of the full dataset against the cycle number.
# # 'df_full['cycle']' is the x-axis, and 'df_full['measured_power']' is the y-axis.
# # 'label='Normal'' and 'color='blue'' style the normal data line.

# plt.scatter(
#     anomalies['current_measured'], anomalies['measured_power'],
#     color='red', label='Anomaly', marker='x'
# )
# # Overlays the detected anomalies (from the sampled data) as red 'x' markers.
# # 'anomalies['cycle']' and 'anomalies['measured_power']' provide the anomaly coordinates.

# plt.xlabel('Measured Current')
# plt.ylabel('Measured Power')
# plt.title('Measured Power vs Measured Current with Anomalies')
# # Sets the x-axis label, y-axis label, and plot title.

# plt.legend()
# # Displays a legend to differentiate between normal data and anomalies.

# plt.grid(True)
# # Adds a grid to the plot for better readability.

# plt.show(block=False)
# # Displays the plot without blocking the script's execution.

# plt.pause(0.1)
# # Pauses the script for 0.1 seconds to allow the plot to render.

# # -------------------- Measured Temperature vs Measured Current Anomaly Detection --------------------

# # Visualize measured power with anomalies (FULL DATA)
# plt.figure(figsize=(12, 6))
# # Creates a new figure for the plot with a size of 12x6 inches.

# plt.plot(df_full['current_measured'], df_full['temperature_measured'], label='Normal', color='blue', alpha=0.6) #Make the blue line semi transparent
# # Plots the measured power of the full dataset against the cycle number.
# # 'df_full['cycle']' is the x-axis, and 'df_full['measured_power']' is the y-axis.
# # 'label='Normal'' and 'color='blue'' style the normal data line.

# plt.scatter(
#     anomalies['current_measured'], anomalies['temperature_measured'],
#     color='red', label='Anomaly', marker='x'
# )
# # Overlays the detected anomalies (from the sampled data) as red 'x' markers.
# # 'anomalies['cycle']' and 'anomalies['measured_power']' provide the anomaly coordinates.

# plt.xlabel('Measured Current')
# plt.ylabel('Measured Temperature')
# plt.title('Measured Temperature vs Measured Current with Anomalies')
# # Sets the x-axis label, y-axis label, and plot title.

# plt.legend()
# # Displays a legend to differentiate between normal data and anomalies.

# plt.grid(True)
# # Adds a grid to the plot for better readability.

# plt.show(block=False)
# # Displays the plot without blocking the script's execution.

# plt.pause(0.1)
# # Pauses the script for 0.1 seconds to allow the plot to render.


# -------------------- Capacity vs Measured Voltage Anomaly Detection --------------------

# Visualize measured power with anomalies (FULL DATA)
plt.figure(figsize=(12, 6))
# Creates a new figure for the plot with a size of 12x6 inches.

plt.plot(df_full['voltage_measured'], df_full['capacity'], label='Normal', color='blue', alpha=0.6) #Make the blue line semi transparent
# Plots the measured power of the full dataset against the cycle number.
# 'df_full['cycle']' is the x-axis, and 'df_full['measured_power']' is the y-axis.
# 'label='Normal'' and 'color='blue'' style the normal data line.

plt.scatter(
    anomalies['voltage_measured'], anomalies['capacity'],
    color='red', label='Anomaly', marker='x'
)
# Overlays the detected anomalies (from the sampled data) as red 'x' markers.
# 'anomalies['cycle']' and 'anomalies['measured_power']' provide the anomaly coordinates.

plt.xlabel('Measured Voltage')
plt.ylabel('Capacity')
plt.title('Capacity vs Measured Voltage with Anomalies')
# Sets the x-axis label, y-axis label, and plot title.

plt.legend()
# Displays a legend to differentiate between normal data and anomalies.

plt.grid(True)
# Adds a grid to the plot for better readability.

plt.show(block=False)
# Displays the plot without blocking the script's execution.

plt.pause(0.1)
# Pauses the script for 0.1 seconds to allow the plot to render.


# # -------------------- Voltage Efficiency vs Measured Current Anomaly Detection --------------------

# # Visualize measured power with anomalies (FULL DATA)
# plt.figure(figsize=(12, 6))
# # Creates a new figure for the plot with a size of 12x6 inches.

# plt.plot(df_full['current_measured'], df_full['voltage_efficiency'], label='Normal', color='blue', alpha=0.6) #Make the blue line semi transparent
# # Plots the measured power of the full dataset against the cycle number.
# # 'df_full['cycle']' is the x-axis, and 'df_full['measured_power']' is the y-axis.
# # 'label='Normal'' and 'color='blue'' style the normal data line.

# plt.scatter(
#     anomalies['current_measured'], anomalies['voltage_efficiency'],
#     color='red', label='Anomaly', marker='x'
# )
# # Overlays the detected anomalies (from the sampled data) as red 'x' markers.
# # 'anomalies['cycle']' and 'anomalies['measured_power']' provide the anomaly coordinates.

# plt.xlabel('Measured Current')
# plt.ylabel('Voltage Efficiency')
# plt.title('Voltage Efficiency vs Measured Current with Anomalies')
# # Sets the x-axis label, y-axis label, and plot title.

# plt.legend()
# # Displays a legend to differentiate between normal data and anomalies.

# plt.grid(True)
# # Adds a grid to the plot for better readability.

# plt.show(block=False)
# # Displays the plot without blocking the script's execution.

# plt.pause(0.1)
# # Pauses the script for 0.1 seconds to allow the plot to render.


# -------------------- Capacity Histogram --------------------

plt.figure(figsize=(12, 6))
sns.histplot(df['capacity'], bins=50, kde=True, label='Normal Data', color='skyblue')
sns.histplot(anomalies['capacity'], bins=50, color='red', label='Anomalies')
plt.title('Histogram of Capacity with Anomalies Highlighted')
plt.xlabel('Capacity')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()


# -------------------- Load Power Histogram --------------------

plt.figure(figsize=(12, 6))
sns.histplot(df['load_power'], bins=50, kde=True, label='Normal Data', color='skyblue')
sns.histplot(anomalies['load_power'], bins=50, color='red', label='Anomalies')
plt.title('Histogram of Load Power with Anomalies Highlighted')
plt.xlabel('Load Power')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()




# -------------------- Measured Power Histogram --------------------

plt.figure(figsize=(12, 6))
sns.histplot(df['measured_power'], bins=50, kde=True, label='Normal Data', color='skyblue')
sns.histplot(anomalies['measured_power'], bins=50, color='red', label='Anomalies')
plt.title('Histogram of Measured Power with Anomalies Highlighted')
plt.xlabel('Measured Power')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()


# -------------------- Power Difference Histogram --------------------

plt.figure(figsize=(12, 6))
sns.histplot(df['power_difference'], bins=50, kde=True, label='Normal Data', color='skyblue')
sns.histplot(anomalies['power_difference'], bins=50, color='red', label='Anomalies')
plt.title('Histogram of Power Difference with Anomalies Highlighted')
plt.xlabel('Power Difference')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()


# -------------------- Correlation Matrix and Heatmap --------------------

# Define features to EXCLUDE from the correlation matrix.
exclude_features = ['time', 'cycle', 'datetime', 'ambient_temperature', 'batteryID', 'voltage_measured', 'current_measured', 'measured_resistance', 'voltage_efficiency', 'temperature_measured', 'current_load', 'voltage_load']

# Define features to INCLUDE in the correlation matrix.
include_features = [col for col in df.columns if col not in exclude_features and col in ['capacity','measured_power','load_power','power_difference','average_power','temperature_difference','load_resistance']]

# Create correlation matrix and heatmap
correlation_matrix = df[include_features].corr()

plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Important Features Correlation Matrix Heatmap')
plt.show()

# -------------------- PCA and Comparative Anomaly Detection --------------------

# Print anomaly statistics
anomaly_count = df['anomaly'].sum()
# Calculates the total number of anomalies detected in the sampled DataFrame 'df'.
print(f"\nDetected {anomaly_count} anomalies ({anomaly_count/len(df)*100:.2f}% of data)\n")
# Prints the number of detected anomalies and their percentage of the total sampled data.

# Visualize using PCA
scaled_data = X
# Assigns the normalized feature data 'X' to 'scaled_data' for PCA.
pca = PCA(n_components=3)
# Initializes a PCA (Principal Component Analysis) object with 2 components.
principal_components = pca.fit_transform(scaled_data)
# Fits the PCA model to the scaled data and transforms it into 2 principal components.

plt.figure(figsize=(10, 6))
# Creates a new figure for the PCA visualization with a size of 10x6 inches.
ax = plt.figure().add_subplot(projection='3d') # Create a 3D subplot

scatter = ax.scatter3D(
    principal_components[:, 0],
    principal_components[:, 1],
    principal_components[:, 2],
    c=df['anomaly'],
    cmap='plasma',
    alpha=0.6,  # Make data points semi-transparent
    s=50
)
# Creates a scatter plot of the two principal components, colored by anomaly labels.
# principal_components[:, 0] and principal_components[:, 1] represent the x and y coordinates.
# c=df['anomaly'] colors the points based on whether they are anomalies (1) or normal (0).
# cmap='viridis' sets the colormap.
# alpha=0.6 sets the transparency of the points.
# s=50 sets the size of the points.

ax.set_title('Anomaly Detection Visualization (PCA)')
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
plt.colorbar(scatter, label='Anomaly (1=Anomaly)', fraction=0.03, pad=0.04)  # Correct colorbar usage

plt.show(block=False)
# Displays the plot without blocking the script's execution.
plt.pause(0.1)
# Pauses the script for 0.1 seconds to allow the plot to render.

# Initialize models
models = {
    'Isolation Forest': IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42
    ),
    'Local Outlier Factor': LocalOutlierFactor(
        n_neighbors=20,
        contamination=0.05,
        novelty=False
    ),
    'One-Class SVM': OneClassSVM(
        nu=0.05,
        kernel='rbf',
        gamma=0.1
    )
}
# Creates a dictionary 'models' that stores instances of three anomaly detection models:
# Isolation Forest, Local Outlier Factor, and One-Class SVM.
# Each model is initialized with specific parameters, such as the number of estimators, contamination level, and kernel type.

# Detect anomalies with all models
results = {}
# Initializes an empty dictionary 'results' to store the anomaly predictions from each model.
for name, model in models.items():
    # Iterates through each model in the 'models' dictionary.
    if name == 'Local Outlier Factor':
        anomalies = model.fit_predict(scaled_data)
        # For Local Outlier Factor, uses fit_predict() to both train and predict anomalies directly.
    else:
        anomalies = model.fit(scaled_data).predict(scaled_data)
        # For Isolation Forest and One-Class SVM, fits the model to the scaled data and then predicts anomalies.

    results[name] = anomalies
    # Stores the anomaly predictions for each model in the 'results' dictionary.
    df[f'anomaly_{name.lower().replace(" ", "_")}'] = np.where(anomalies == -1, 1, 0)
    # Adds a new column to the sampled DataFrame 'df' for each model's anomaly predictions.
    # The anomaly labels (-1/1) are converted to 1/0 for consistency.
    # The column names are formatted to indicate the model used (e.g., 'anomaly_isolation_forest').
    

# Visualization 1: PCA projection (3D)
pca = PCA(n_components=3)
# Initializes a PCA object to reduce the data to three principal components (for 3D visualization).
principal_components = pca.fit_transform(scaled_data)
# Fits the PCA model to the scaled data and transforms it, obtaining the principal components.

fig = plt.figure(figsize=(15, 5))  # Create the figure OUTSIDE the loop
# Creates a new figure for the 3D plots, sized 15x5 inches.
for i, (name, _) in enumerate(models.items(), 1):
    # Iterates through the models, using enumerate to get both the index (i) and the model name.
    ax = fig.add_subplot(1, 3, i, projection='3d')  # Add subplot to the existing figure
    # Creates a 3D subplot in a 1x3 grid, with the current subplot's position determined by 'i'.

    scatter = ax.scatter3D(
        principal_components[:, 0],
        principal_components[:, 1],
        principal_components[:, 2],
        c=df[f'anomaly_{name.lower().replace(" ", "_")}'],
        cmap='plasma',
        alpha=0.6,
        s=40
    )
    # Creates a 3D scatter plot of the principal components, colored by the anomaly labels from the current model.
    # 'c' is set to the anomaly column corresponding to the current model.
    # 'cmap' sets the color map.
    # 'alpha' sets the transparency of the points.
    # 's' sets the size of the points.

    ax.set_title(f'{name} Anomaly Detection', fontsize=8)
    # Sets the title of the subplot, including the model name.
    ax.set_xlabel('PC 1', fontsize=6)
    ax.set_ylabel('PC 2', fontsize=6)
    ax.set_zlabel('PC 3', fontsize=6)
    # Sets the axis labels for the 3D subplot, including the z-axis label.
    plt.colorbar(scatter, label='Anomaly (1=Anomaly)', fraction=0.03, pad=0.04)
    # Adds a colorbar to the 3D subplot to indicate the meaning of the colors (anomaly vs. normal).

#plt.show(block=False) #removed block=false from the loop, to show all at once.
#plt.pause(0.1) #removed pause from the loop, to show all at once.
plt.show() #added show outside of the loop.
    
    
# Visualization 2: Capacity Comparison (FULL DATA)

plt.figure(figsize=(14, 7))
# Creates a new figure for the plot with a size of 14x7 inches.
plt.plot(df_full['cycle'], df_full['capacity'], 'gray', alpha=0.3, label='Normal')
# Plots the capacity degradation of the *full* dataset as a gray line with transparency.
# 'df_full['cycle']' and 'df_full['capacity']' provide the x and y coordinates.
# 'gray' and 'alpha=0.3' style the line.
# 'label='Normal'' adds a label for the legend.

colors = {'Isolation Forest': 'red', 'Local Outlier Factor': 'green', 'One-Class SVM': 'blue'}
# Defines a dictionary 'colors' to map model names to colors for the scatter plot.
for name in models.keys():
    # Iterates through the models.
    mask = df[f'anomaly_{name.lower().replace(" ", "_")}'] == 1
    # Creates a boolean mask to select rows where the current model detected anomalies in the *sampled* data.
    plt.scatter(
        df.loc[mask, 'cycle'],
        df.loc[mask, 'capacity'],
        color=colors[name],
        s=30,
        alpha=0.6, # Make data points semi-transparent
        label=f'{name} Anomaly'
    )
    # Creates a scatter plot of the anomalies detected by the current model, using the full dataset's cycle and capacity values.
    # 'df.loc[mask, 'cycle']' and 'df.loc[mask, 'capacity']' provide the x and y coordinates of the anomalies.
    # 'color=colors[name]' sets the color based on the model.
    # 's=30' sets the size of the points.
    # 'label=f'{name} Anomaly'' adds a label for the legend.

plt.title('Capacity with Comparative Anomaly Detection')
plt.xlabel('Cycle Number')
plt.ylabel('Capacity')
# Sets the plot title and axis labels.
plt.legend()
# Displays a legend to identify the normal data and the anomalies detected by each model.
plt.grid(True)
# Adds a grid to the plot for better readability.
plt.show(block=False)
# Displays the plot without blocking the script's execution.
plt.pause(0.1)
# Pauses the script for 0.1 seconds to allow the plot to render.


# Visualization 3: Measured power comparison (FULL DATA)

plt.figure(figsize=(14, 7))
# Creates a new figure for the plot with a size of 14x7 inches.
plt.plot(df_full['cycle'], df_full['measured_power'], 'gray', alpha=0.3, label='Normal')
# Plots the measured power of the *full* dataset against the cycle number as a gray line with transparency.
# 'df_full['cycle']' and 'df_full['measured_power']' provide the x and y coordinates.
# 'gray' and 'alpha=0.3' style the line.
# 'label='Normal'' adds a label for the legend.

colors = {'Isolation Forest': 'red', 'Local Outlier Factor': 'green', 'One-Class SVM': 'blue'}
# Defines a dictionary 'colors' to map model names to colors for the scatter plot.
for name in models.keys():
    # Iterates through the models.
    mask = df[f'anomaly_{name.lower().replace(" ", "_")}'] == 1
    # Creates a boolean mask to select rows where the current model detected anomalies in the *sampled* data.
    plt.scatter(
        df.loc[mask, 'cycle'],
        df.loc[mask, 'measured_power'],
        color=colors[name],
        s=30,
        alpha=0.6, # Make data points semi-transparent
        label=f'{name} Anomaly'
    )
    # Creates a scatter plot of the anomalies detected by the current model, using the full dataset's cycle and measured power values.
    # 'df.loc[mask, 'cycle']' and 'df.loc[mask, 'measured_power']' provide the x and y coordinates of the anomalies.
    # 'color=colors[name]' sets the color based on the model.
    # 's=30' sets the size of the points.
    # 'label=f'{name} Anomaly'' adds a label for the legend.

plt.title('Measured Power with Comparative Anomaly Detection')
plt.xlabel('Cycle Number')
plt.ylabel('Measured Power')
# Sets the plot title and axis labels.
plt.legend()
# Displays a legend to identify the normal data and the anomalies detected by each model.
plt.grid(True)
# Adds a grid to the plot for better readability.
plt.show(block=False)
# Displays the plot without blocking the script's execution.
plt.pause(0.1)
# Pauses the script for 0.1 seconds to allow the plot to render.


# Visualization 4: Power Difference comparison (FULL DATA)

plt.figure(figsize=(14, 7))
# Creates a new figure for the plot with a size of 14x7 inches.
plt.plot(df_full['cycle'], df_full['power_difference'], 'gray', alpha=0.3, label='Normal')
# Plots the measured power of the *full* dataset against the cycle number as a gray line with transparency.
# 'df_full['cycle']' and 'df_full['measured_power']' provide the x and y coordinates.
# 'gray' and 'alpha=0.3' style the line.
# 'label='Normal'' adds a label for the legend.

colors = {'Isolation Forest': 'red', 'Local Outlier Factor': 'green', 'One-Class SVM': 'blue'}
# Defines a dictionary 'colors' to map model names to colors for the scatter plot.
for name in models.keys():
    # Iterates through the models.
    mask = df[f'anomaly_{name.lower().replace(" ", "_")}'] == 1
    # Creates a boolean mask to select rows where the current model detected anomalies in the *sampled* data.
    plt.scatter(
        df.loc[mask, 'cycle'],
        df.loc[mask, 'power_difference'],
        color=colors[name],
        s=30,
        alpha=0.6, # Make data points semi-transparent
        label=f'{name} Anomaly'
    )
    # Creates a scatter plot of the anomalies detected by the current model, using the full dataset's cycle and measured power values.
    # 'df.loc[mask, 'cycle']' and 'df.loc[mask, 'measured_power']' provide the x and y coordinates of the anomalies.
    # 'color=colors[name]' sets the color based on the model.
    # 's=30' sets the size of the points.
    # 'label=f'{name} Anomaly'' adds a label for the legend.
    
plt.title('Power Difference with Comparative Anomaly Detection')
plt.xlabel('Cycle Number')
plt.ylabel('Power Difference')
    # Sets the plot title and axis labels.
plt.legend()
    # Displays a legend to identify the normal data and the anomalies detected by each model.
plt.grid(True)
    # Adds a grid to the plot for better readability.
plt.show(block=False)
    # Displays the plot without blocking the script's execution.
plt.pause(0.1)
    # Pauses the script for 0.1 seconds to allow the plot to render.
    
    
    # Visualization 5: Load Power comparison (FULL DATA)

plt.figure(figsize=(14, 7))
    # Creates a new figure for the plot with a size of 14x7 inches.
plt.plot(df_full['cycle'], df_full['load_power'], 'gray', alpha=0.3, label='Normal')
    # Plots the measured power of the *full* dataset against the cycle number as a gray line with transparency.
    # 'df_full['cycle']' and 'df_full['measured_power']' provide the x and y coordinates.
    # 'gray' and 'alpha=0.3' style the line.
    # 'label='Normal'' adds a label for the legend.

colors = {'Isolation Forest': 'red', 'Local Outlier Factor': 'green', 'One-Class SVM': 'blue'}
    # Defines a dictionary 'colors' to map model names to colors for the scatter plot.
for name in models.keys():
        # Iterates through the models.
    mask = df[f'anomaly_{name.lower().replace(" ", "_")}'] == 1
        # Creates a boolean mask to select rows where the current model detected anomalies in the *sampled* data.
    plt.scatter(
        df.loc[mask, 'cycle'],
        df.loc[mask, 'load_power'],
        color=colors[name],
        s=30,
        alpha=0.6, # Make data points semi-transparent
        label=f'{name} Anomaly'
        )
        # Creates a scatter plot of the anomalies detected by the current model, using the full dataset's cycle and measured power values.
        # 'df.loc[mask, 'cycle']' and 'df.loc[mask, 'measured_power']' provide the x and y coordinates of the anomalies.
        # 'color=colors[name]' sets the color based on the model.
        # 's=30' sets the size of the points.
        # 'label=f'{name} Anomaly'' adds a label for the legend.
        
plt.title('Load Power with Comparative Anomaly Detection')
plt.xlabel('Cycle Number')
plt.ylabel('Load Power')
        # Sets the plot title and axis labels.
plt.legend()
        # Displays a legend to identify the normal data and the anomalies detected by each model.
plt.grid(True)
        # Adds a grid to the plot for better readability.
plt.show(block=False)
        # Displays the plot without blocking the script's execution.
plt.pause(0.1)
        # Pauses the script for 0.1 seconds to allow the plot to render.
        
        
    # Visualization 6: Temperature Difference comparison (FULL DATA)

plt.figure(figsize=(14, 7))
    # Creates a new figure for the plot with a size of 14x7 inches.
plt.plot(df_full['cycle'], df_full['temperature_difference'], 'gray', alpha=0.3, label='Normal')
    # Plots the measured power of the *full* dataset against the cycle number as a gray line with transparency.
    # 'df_full['cycle']' and 'df_full['measured_power']' provide the x and y coordinates.
    # 'gray' and 'alpha=0.3' style the line.
    # 'label='Normal'' adds a label for the legend.

colors = {'Isolation Forest': 'red', 'Local Outlier Factor': 'green', 'One-Class SVM': 'blue'}
    # Defines a dictionary 'colors' to map model names to colors for the scatter plot.
for name in models.keys():
        # Iterates through the models.
    mask = df[f'anomaly_{name.lower().replace(" ", "_")}'] == 1
        # Creates a boolean mask to select rows where the current model detected anomalies in the *sampled* data.
    plt.scatter(
        df.loc[mask, 'cycle'],
        df.loc[mask, 'temperature_difference'],
        color=colors[name],
        s=30,
        alpha=0.6, # Make data points semi-transparent
        label=f'{name} Anomaly'
        )
        # Creates a scatter plot of the anomalies detected by the current model, using the full dataset's cycle and measured power values.
        # 'df.loc[mask, 'cycle']' and 'df.loc[mask, 'measured_power']' provide the x and y coordinates of the anomalies.
        # 'color=colors[name]' sets the color based on the model.
        # 's=30' sets the size of the points.
        # 'label=f'{name} Anomaly'' adds a label for the legend.
        

plt.title('Temperature Difference with Comparative Anomaly Detection')
plt.xlabel('Cycle Number')
plt.ylabel('Temperature Difference')
# Sets the plot title and axis labels.
plt.legend()
# Displays a legend to identify the normal data and the anomalies detected by each model.
plt.grid(True)
# Adds a grid to the plot for better readability.
plt.show(block=False)
# Displays the plot without blocking the script's execution.
plt.pause(0.1)
# Pauses the script for 0.1 seconds to allow the plot to render.
        



# Print comparison statistics
print("Anomaly Detection Results Comparison:")
# Prints a header for the comparison statistics.
stats = []
# Initializes an empty list 'stats' to store the anomaly counts and percentages for each model.
for name in models.keys():
    # Iterates through the models.
    col_name = f'anomaly_{name.lower().replace(" ", "_")}'
    # Creates the column name corresponding to the current model's anomaly predictions.
    count = df[col_name].sum()
    # Calculates the total number of anomalies detected by the current model.
    percentage = count / len(df) * 100
    # Calculates the percentage of anomalies detected by the current model.
    stats.append([name, count, percentage])
    # Appends the model name, anomaly count, and percentage to the 'stats' list.

stats_df = pd.DataFrame(stats, columns=['Algorithm', 'Anomaly Count', 'Percentage'])
# Creates a pandas DataFrame 'stats_df' from the 'stats' list, with column names.
print(stats_df)
# Prints the DataFrame, displaying the anomaly statistics for each model.

# Overlap Analysis (without ground truth)
print("\n\nAnomaly Overlap Analysis:")
# Prints a header for the overlap analysis.
anomaly_sets = {}
anomaly_counts = {} # Store total anomaly count for each model
# Initializes empty dictionaries 'anomaly_sets' and 'anomaly_counts'.
for name in models.keys():
    # Iterates through the models.
    col_name = f'anomaly_{name.lower().replace(" ", "_")}'
    # Creates the column name corresponding to the current model's anomaly predictions.
    anomaly_sets[name] = set(df[df[col_name] == 1].index)
    # Creates a set of indices where the current model detected anomalies.
    anomaly_counts[name] = df[col_name].sum() #store anomaly counts
    #store the total anomaly counts for each model.

for name1 in models.keys():
    # Outer loop to iterate through models.
    for name2 in models.keys():
        # Inner loop to iterate through models.
        if name1 != name2:
            # Checks if the models are different.
            overlap_indices = anomaly_sets[name1].intersection(anomaly_sets[name2])
            # Calculates the intersection of anomaly indices between two models.
            overlap_count = len(overlap_indices)
            # Calculates the number of overlapping anomalies.
            overlap_percent_name1 = (overlap_count / anomaly_counts[name1]) * 100
            overlap_percent_name2 = (overlap_count / anomaly_counts[name2]) * 100
            #calculate the percentage of overlapping anomalies for each algorithm.

            print(f"\nOverlap between {name1} and {name2}: {overlap_count}\n")
            # Prints the overlap count between two models.
            print(f"  Overlap Percentage ({name1}): {overlap_percent_name1:.2f}%")
            print(f"  Overlap Percentage ({name2}): {overlap_percent_name2:.2f}%\n")
            #prints the overlap percentage for each algorithm.
            if overlap_count > 0:
                print(f"  Overlap Indices: {sorted(list(overlap_indices))}")
                #prints the overlap indices if any overlap exists.
                
# Overlap between all three algorithms
all_three_overlap = anomaly_sets['Isolation Forest'].intersection(
    anomaly_sets['Local Outlier Factor'], anomaly_sets['One-Class SVM']
)
# Calculates the intersection of anomaly indices among all three models, finding the anomalies detected by all of them.
all_three_overlap_count = len(all_three_overlap)
# Calculates the number of anomalies detected by all three models.
print(f"\n\nOverlap between all three algorithms: {all_three_overlap_count}\n")
# Prints the number of anomalies detected by all three models.

all_three_overlap_percent = {}
# Initializes an empty dictionary to store the overlap percentage for each algorithm.
for name in models.keys():
    # Iterates through the models.
    all_three_overlap_percent[name] = (all_three_overlap_count / anomaly_counts[name]) * 100
    # Calculates the percentage of anomalies detected by all three models relative to the total anomalies detected by each individual model.
    print(f"  Overlap Percentage ({name}): {all_three_overlap_percent[name]:.2f}%")
    # Prints the overlap percentage for each model.

if all_three_overlap_count > 0:
    print(f"  \nOverlap Indices: {sorted(list(all_three_overlap))}")
    # Prints the overlap indices if anomalies are detected by all three algorithms.

# Display sample anomalies from each method
print("\n\nSample Anomalies from Each Algorithm:")
# Prints a header for the sample anomalies.
for name in models.keys():
    # Iterates through the models.
    col_name = f'anomaly_{name.lower().replace(" ", "_")}'
    # Creates the column name corresponding to the current model's anomaly predictions.
    print(f"\n{name}:")
    # Prints the model name.
    print(df[df[col_name] == 1].head(3))
    # Prints the first 3 rows of the DataFrame where the current model detected anomalies.

# -------------------- Measured Power vs Time for All Batteries on One Plot --------------------
# This section generates a single plot that displays the measured power over time for all four batteries
# on the same graph. It also marks the detected anomalies on this combined plot.

if 'batteryID' in df_full.columns:
    # Check if the 'batteryID' column exists in the DataFrame.
    unique_batteries = df_full['batteryID'].unique()
    # Get a list of unique battery identifiers.
    colors = plt.cm.get_cmap('viridis', len(unique_batteries))
    # Create a colormap to assign different colors to each battery.
    plt.figure(figsize=(14, 8))
    # Create a new figure for the combined plot with a specified size.

    for i, batteryID in enumerate(unique_batteries):
        # Loop through each unique battery ID.
        battery_data_full = df_full[df_full['batteryID'] == batteryID].sort_values(by='time').copy()
        # Filter the full DataFrame to get data for the current battery and sort by time.
        if not battery_data_full.empty:
            # Proceed only if there is data for the current battery.
            color = colors(i)
            # Get the color for the current battery from the colormap.
            plt.plot(battery_data_full['time'], battery_data_full['measured_power'],
                     label=f'Battery {batteryID}', color=color, linewidth=1.2, alpha=0.4)
            # Plot the measured power vs. time for the current battery with a label, color, line width, and transparency.

            # Mark anomalies for this battery
            if 'anomaly' in df.columns:
                # Check if the 'anomaly' column exists in the sampled DataFrame.
                df_battery_sample = df[df['batteryID'] == batteryID].copy()
                # Filter the sampled DataFrame for the current battery.
                if not df_battery_sample.empty:
                    # Proceed if there is sample data for the current battery.
                    anomalies_battery_sample = df_battery_sample[df_battery_sample['anomaly'] == 1].copy()
                    # Filter the sample data to get only the anomaly rows for the current battery.
                    if not anomalies_battery_sample.empty:
                        # Proceed if anomalies were detected for this battery in the sample.
                        if 'time' in anomalies_battery_sample.columns:
                            # Check if the 'time' column exists in the anomaly data.
                            anomaly_times_sample = anomalies_battery_sample['time'].tolist()
                            # Get the time values of the anomalies in the sample.
                            anomaly_powers_full = df_full.loc[(df_full['batteryID'] == batteryID) & (df_full['time'].isin(anomaly_times_sample)), 'measured_power']
                            # Find the corresponding measured power values in the full dataset for the anomaly times.
                            anomaly_times_full = df_full.loc[(df_full['batteryID'] == batteryID) & (df_full['time'].isin(anomaly_times_sample)), 'time']
                            # Find the corresponding time values in the full dataset for the anomaly times.

                            if not anomaly_times_full.empty and not anomaly_powers_full.empty and len(anomaly_times_full) == len(anomaly_powers_full):
                                # If the time and power arrays have the same size, plot the anomalies.
                                plt.scatter(anomaly_times_full, anomaly_powers_full,
                                            color='red', marker='x', s=120, linewidths=2)
                                # Plot the anomalies as red 'x' markers with increased size and line width.
                            else:
                                print(f"Warning: Could not reliably plot anomalies for Battery {batteryID} on the combined plot.")
                        else:
                            print(f"Warning: 'time' column not found in the sampled anomalies for Battery {batteryID} (combined plot).")
                    else:
                        print(f"No anomalies detected for Battery {batteryID} in the sampled data (combined plot).")
                else:
                    print(f"No data found for Battery {batteryID} in the sampled DataFrame (combined plot).")
        else:
            print(f"No data found for Battery {batteryID} in the full DataFrame (combined plot).")

    plt.xlabel('Time', fontsize=12)
    # Set the label for the x-axis.
    plt.ylabel('Measured Power', fontsize=12)
    # Set the label for the y-axis.
    plt.title('Measured Power vs Time for All Batteries with Anomalies', fontsize=14)
    # Set the title of the plot.
    plt.legend(fontsize=10)
    # Display the legend to identify the batteries.
    plt.grid(True)
    # Add a grid to the plot for better readability.
    plt.tight_layout()
    # Adjust the plot layout to prevent labels from overlapping.
    plt.show(block=False)
    # Display the plot without blocking the script.
    plt.pause(0.1)
    # Pause briefly to allow the plot to render.
else:
    print("\nWarning: 'batteryID' column not found in the DataFrame. Cannot generate the combined plot.")
    # If the 'batteryID' column is missing, print a warning.

# -------------------- Measured Power vs Time for Each Battery (Separate Plots) --------------------
# This section generates four separate plots, one for each individual battery, showing the
# measured power over time and marking the detected anomalies on each plot.

if 'batteryID' in df_full.columns:
    # Check if the 'batteryID' column exists.
    unique_batteries = df_full['batteryID'].unique()
    # Get unique battery identifiers.
    colors = plt.cm.get_cmap('viridis', len(unique_batteries))
    # Get the colormap.

    for i, batteryID in enumerate(unique_batteries):
        # Loop through each battery.
        plt.figure(figsize=(10, 6))  # Create a new figure for each plot.
        battery_data_full = df_full[df_full['batteryID'] == batteryID].sort_values(by='time').copy()
        # Get the data for the current battery.
        if not battery_data_full.empty:
            # If there is data for the battery.
            color = colors(i)
            # Get the color for the battery.
            plt.plot(battery_data_full['time'], battery_data_full['measured_power'],
                     label=f'Battery {batteryID}', color=color, linewidth=1.2, alpha=0.7)
            # Plot the power vs. time for the battery.

            # Mark anomalies for this battery
            if 'anomaly' in df.columns:
                # If the 'anomaly' column exists in the sample data.
                df_battery_sample = df[df['batteryID'] == batteryID].copy()
                # Get the sample data for the current battery.
                if not df_battery_sample.empty:
                    # If there is sample data.
                    anomalies_battery_sample = df_battery_sample[df_battery_sample['anomaly'] == 1].copy()
                    # Get anomaly data from the sample for this battery.
                    if not anomalies_battery_sample.empty:
                        # If anomalies were found in the sample.
                        if 'time' in anomalies_battery_sample.columns:
                            # If the 'time' column exists in the anomalies.
                            anomaly_times_sample = anomalies_battery_sample['time'].tolist()
                            # Get the anomaly times from the sample.
                            anomaly_powers_full = df_full.loc[(df_full['batteryID'] == batteryID) & (df_full['time'].isin(anomaly_times_sample)), 'measured_power']
                            # Find the corresponding powers in the full dataset.
                            anomaly_times_full = df_full.loc[(df_full['batteryID'] == batteryID) & (df_full['time'].isin(anomaly_times_sample)), 'time']
                            # Find the corresponding times in the full dataset.

                            if not anomaly_times_full.empty and not anomaly_powers_full.empty and len(anomaly_times_full) == len(anomaly_powers_full):
                                # If the anomaly data sizes match.
                                plt.scatter(anomaly_times_full, anomaly_powers_full,
                                            color='red', marker='x', s=80, linewidths=2)
                                # Plot the anomalies as red 'x' markers.
                            else:
                                print(f"Warning: Could not reliably plot anomalies for Battery {batteryID} on the separate plot.")
                        else:
                            print(f"Warning: 'time' column not found in the sampled anomalies for Battery {batteryID} (separate plot).")
                    else:
                        print(f"No anomalies detected for Battery {batteryID} in the sampled data (separate plot).")
                else:
                    print(f"No data found for Battery {batteryID} in the sampled DataFrame (separate plot).")
        else:
            print(f"No data found for Battery {batteryID} in the full DataFrame (separate plot).")

        plt.xlabel('Time', fontsize=12)
        # Set x-axis label.
        plt.ylabel('Measured Power', fontsize=12)
        # Set y-axis label.
        plt.title(f'Measured Power vs Time for Battery {batteryID} with Anomalies', fontsize=14)
        # Set the plot title.
        plt.legend(fontsize=10)
        # Display the legend.
        plt.grid(True)
        # Add a grid.
        plt.tight_layout()
        # Adjust layout.
        plt.show(block=False)
        # Show the plot.
        plt.pause(0.1)
        # Pause.
else:
    print("\nWarning: 'batteryID' column not found in the DataFrame. Cannot generate the separate plots.")
    # If 'batteryID' is missing, print a warning.


# Anomaly Overlap Analysis
anomaly_sets = {}
for name in models.keys():
    col_name = f'anomaly_{name.lower().replace(" ", "_")}'
    anomaly_sets[name] = set(df[df[col_name] == 1].index)
    # Creates a set of indices where the current model detected anomalies.

# Overlap between all three algorithms
all_three_overlap = anomaly_sets['Isolation Forest'].intersection(
    anomaly_sets['Local Outlier Factor'], anomaly_sets['One-Class SVM']
)
# Calculates the intersection of anomaly indices among all three models.

# Add "overlap anomalies" column
df['overlap_anomalies'] = 0
# Initializes a new column named "true_anomalies" with all values set to 0.
df.loc[df.index.isin(all_three_overlap), 'overlap_anomalies'] = 1
# Updates the "overlap_anomalies" column to 1 for rows where all three models detected anomalies.


# 3D PCA plot for true_anomalies
pca = PCA(n_components=3)
principal_components = pca.fit_transform(X)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter3D(
    principal_components[:, 0],
    principal_components[:, 1],
    principal_components[:, 2],
    c=df['overlap_anomalies'],
    cmap='plasma',
    alpha=0.6,
    s=50
)

ax.set_title('3D PCA Plot of Overlap Anomalies')
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
plt.colorbar(scatter, label='Overlap Anomaly (1=Anomaly)')

plt.show()

from astropy.timeseries import LombScargle
from scipy.stats import skew, kurtosis


# -------------------- F1 Scores --------------------


from sklearn.metrics import f1_score

# Add "overlap anomalies" column
df['overlap_anomalies'] = 0
df.loc[df.index.isin(all_three_overlap), 'overlap_anomalies'] = 1

# Calculate F1 scores for each model
f1_isolation_forest = f1_score(df['anomaly'], df['anomaly_isolation_forest'])
f1_lof = f1_score(df['anomaly'], df['anomaly_local_outlier_factor'])
f1_svm = f1_score(df['anomaly'], df['anomaly_one-class_svm'])

print ("\n\nF1 Scores by Model:\n")
print(f"F1 Score (Isolation Forest): {f1_isolation_forest}")
print(f"F1 Score (Local Outlier Factor): {f1_lof}")
print(f"F1 Score (One-Class SVM): {f1_svm}")


# -------------------- Anomalies by Battery --------------------

# Count anomalies per battery ID for each category
anomalies_per_battery = df.groupby('batteryID').agg({
    'anomaly': 'sum',
    'anomaly_isolation_forest': 'sum',
    'anomaly_local_outlier_factor': 'sum',
    'anomaly_one-class_svm': 'sum',
    'overlap_anomalies': 'sum'
}).sort_index()

# Generate bar graph
anomalies_per_battery.plot(kind='bar', figsize=(15, 7))
plt.title('Anomalies per Battery ID')
plt.xlabel('Battery ID')
plt.ylabel('Number of Anomalies')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# # -------------------- Statistical Analysis of Anomalies --------------------
# # This section calculates and prints descriptive statistics of the features for each model and the true anomalies.
# print("\n\nStatistical Analysis of Anomalies:")
# model_statistics = {} # Initialize dictionary to hold model statistic dataframes
# for name in models.keys():
#     col_name = f'anomaly_{name.lower().replace(" ", "_")}'
#     anomaly_data = df[df[col_name] == 1][features]
#     model_statistics[name] = anomaly_data.describe() # store statistics for each model
#     print(f"\n{name} Anomaly Statistics:")
#     print(model_statistics[name])

# true_anomaly_data = df[df['overlap_anomalies'] == 1][features]
# model_statistics["Overlap Anomalies"] = true_anomaly_data.describe() # store statistics for overlap anomalies
# print("\nOverlap Anomalies Statistics:")
# print(model_statistics["Overlap Anomalies"])

# # Output the statistics to a separate Excel file
# with pd.ExcelWriter("nasabattery_model_anomalies_statistics_new2.xlsx") as writer:
#     for model_name, stats_df in model_statistics.items():
#         stats_df.to_excel(writer, sheet_name=model_name)
# # Saves the statistics for each model and true anomalies to a separate Excel file, each on a different sheet.
# print("\nModel and Overlap anomaly statistics saved to nasabattery_model_anomalies_statistics_new2.xlsx")


#-------------------- Statistical Analysis of Anomalies --------------------

#This section calculates and prints descriptive statistics of the features for each model and the true anomalies.
print("\n\nStatistical Analysis of Anomalies:")
model_statistics = {}  # Initialize dictionary to hold model statistic dataframes
for name in models.keys():
    col_name = f'anomaly_{name.lower().replace(" ", "_")}'
    anomaly_data = df[df[col_name] == 1][features]
    model_statistics[name] = anomaly_data.describe()  # store statistics for each model
    print(f"\n{name} Anomaly Statistics:")
    print(model_statistics[name])

# Include the 'anomaly' and 'true_anomalies' columns in the statistics
anomaly_data_anomaly = df[df['anomaly'] == 1][features]
model_statistics["Anomaly"] = anomaly_data_anomaly.describe()
print("\nAnomaly Statistics:")
print(model_statistics["Anomaly"])

true_anomaly_data = df[df['overlap_anomalies'] == 1][features]
model_statistics["Overlap Anomalies"] = true_anomaly_data.describe()
print("\nOverlap Anomalies Statistics:")
print(model_statistics["Overlap Anomalies"])

# Output the statistics to a separate Excel file
with pd.ExcelWriter("nasabattery_model_anomalies_statistics_new2.xlsx") as writer:
    for model_name, stats_df in model_statistics.items():
        stats_df.to_excel(writer, sheet_name=model_name)
# Saves the statistics for each model and true anomalies to a separate Excel file, each on a different sheet.
print("\nModel and anomaly statistics saved to nasabattery_model_anomalies_statistics_new2.xlsx")

# Output anomaly detection to Excel
df.to_excel("nasabattery_with_anomaliesTest_new2.xlsx", index=False)
# Saves the DataFrame, including the "overlap_anomalies" column, to an Excel file.
print("\n\nAnomaly detection results saved to nasabattery_with_anomaliesTest_new2.xlsx")
# Prints a message indicating that the Excel file has been saved.