import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os

# Load your CSV file
PHD_ROOT = os.getenv("PHD_ROOT")
sys.path.append(PHD_ROOT)
csv_file_path = "~/PhD/results/logs/ppo.csv"
data = pd.read_csv(csv_file_path)


key_word_exclude = ["MIN", "MAX", "cai", "baseline", "sparse","dense", "SAC"]
key_word_must = []
possible_lables = ["Sparse", "Dense", "Tool_dist"]
# filtered_columns = [col for col in data.columns if "SAC" in col]
filtered_columns = []
print(data["Step"])
for col in data.columns:
    musts = [key in col for key in key_word_must]
    excludes = [key in col for key in key_word_exclude]
    if all(musts) and not any(excludes): filtered_columns.append(col)

# Set the smoothing factor
smoothing_factor = 10 # Adjust this value according to your preference

# Apply average smoothing to each column
smoothed_data = data.rolling(window=smoothing_factor).mean()
filtered_data = smoothed_data[filtered_columns]
print(filtered_data)
print(filtered_columns)

# Create a line plot for each column
sns.set(style="whitegrid")  # Set the style of the plot
plt.figure(figsize=(10, 6))  # Set the figure size

# Iterate through each column and plot the line
counter = 0
for column in filtered_data.columns:
    print(column)
    label = "Sparse"
    y_scale = 1

    if "Step" in column: 
        continue
    
    for possible_lable in possible_lables:
        if possible_lable.lower() in column: 
            label = possible_lable
            y_scale = 1
        
        if "dense" in column: y_scale = 1

    if "PPO" in column:
        step = 20 * data["Step"] // 2
    else:
        step = data["Step"]
    if "SAC" in column: y_scale = 100

    counter += 1
    sns.lineplot(x= step, y=data[column] / y_scale, label=f"Run {counter}")

plt.axhline(y=-5, c="r") 

plt.ylim(-110, 0)  # Adjust the y-axis limits as needed
plt.title('3 Runs - PPO with Dense + Tool Dist function')
plt.xlabel('Num Episodes')
plt.ylabel('Mean Episode Reward')
plt.legend()  # Show legend with column names
plt.show()
