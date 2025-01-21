import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tkinter.filedialog import askopenfilename
from sklearn.linear_model import LinearRegression

# Draw linear regression function
def plot_linear_regression(x, y1, y2):
    plt.rcParams['figure.figsize'] = (6, 5)
    plt.rcParams['savefig.dpi'] = 200
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.scatter(x, y1, marker='o', s=15, c='white', edgecolors='red')
    plt.scatter(x, y2, marker='o', s=15, c='white', edgecolors='blue')
    font1 = {'family': 'Arial', 'weight': 'normal', 'size': 18}
    plt.xlabel('concentration', font1)
    plt.ylabel('area', font1)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Transform to array
    T_x = np.array(x).reshape((len(x), 1))
    T_y1 = np.array(y1).reshape((len(y1), 1))
    T_y2 = np.array(y2).reshape((len(y2), 1))

    # LR
    lineModel1 = LinearRegression()
    lineModel1.fit(T_x, T_y1)
    lineModel2 = LinearRegression()
    lineModel2.fit(T_x, T_y2)

    # Draw the regression line
    plt.plot(T_x, lineModel1.predict(T_x), linestyle='dotted', color='red')
    plt.plot(T_x, lineModel2.predict(T_x), linestyle='dotted', color='blue')
    ax = plt.gca()
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    plt.tick_params(axis='both', width=1, length=5)
    plt.draw()
    parameters = [lineModel1.coef_[0][0], lineModel1.intercept_[0], lineModel2.coef_[0][0], lineModel2.intercept_[0]]
    return parameters

# Calculate nitrogen concentration by nitrite
def Nitrite_N(data, parameters, lower_limit, higher_limit):
    Nitrite_N_result = []
    for row in data:
        if row[1] > lower_limit:
            nitrogen = (row[1] + parameters[1]) / parameters[0] * float(row[0][-3:-1]) / 46.005 * 14.007
            Nitrite_N_result.append(nitrogen)
        else:
            nitrogen = 0
            Nitrite_N_result.append(nitrogen)
    NO2N = np.array(Nitrite_N_result).reshape((len(Nitrite_N_result), 1))
    return NO2N

# Calculate nitrogen concentration by nitrate
def Nitrate_N(data, parameters, lower_limit, higher_limit):
    Nitrate_N_result = []
    for row in data:
        if row[2] > lower_limit:
            nitrogen = (row[2] + parameters[3]) / parameters[2] * float(row[0][-3:-1]) / 62.004 * 14.007
            Nitrate_N_result.append(nitrogen)
        else:
            nitrogen = 0
            Nitrate_N_result.append(nitrogen)
    NO3N = np.array(Nitrate_N_result).reshape((len(Nitrate_N_result), 1))
    return NO3N

# Plot the result curves for MA, OA, MB, OB
def plot_result_curves(output_data):
    # Filter data to only include MA, MB, OA, OB
    filtered_data = output_data[output_data['Sample'].isin(['MA', 'MB', 'OA', 'OB'])].copy()
    
    # Convert columns to numeric values and fill NaN with 0 for other anions
    anions = ['Fluoride', 'Chloride', 'Sulfate', 'Phosphate']
    for anion in anions:
        filtered_data[f'{anion}-Concentration'] = pd.to_numeric(filtered_data[f'{anion}-Concentration'], errors='coerce').fillna(0)
    
    # Sort by Time in ascending order
    filtered_data = filtered_data.sort_values(by='Time', ascending=True)
    
    # Define colors for each sample
    colors = {'MA': 'red', 'OA': 'blue', 'MB': 'green', 'OB': 'orange'}
    
    # Plot concentration for each anion
    for anion in anions:
        plt.figure(figsize=(10, 6))
        for sample, color in colors.items():
            sample_data = filtered_data[filtered_data['Sample'] == sample]
            if not sample_data.empty:
                plt.plot(sample_data['Time'], sample_data[f'{anion}-Concentration'], label=f'{sample} {anion}', color=color, linestyle='-')
        
        plt.xlabel('Time', fontsize=14)
        plt.ylabel(f'{anion} Concentration', fontsize=14)
        plt.title(f'{anion} Concentration Over Time', fontsize=16)
        plt.legend()
        plt.grid(True)
        
        # Set y-axis to integer values with adaptive scaling
        concentration_values = filtered_data[f'{anion}-Concentration']
        max_value = int(np.nanmax(concentration_values)) + 1  # Add 1 to avoid cutting off the top value
        min_value = int(np.nanmin(concentration_values))
        
        # Calculate step size based on the range of values
        value_range = max_value - min_value
        if value_range <= 10:
            step = 1  # Small range, use step 1
        elif value_range <= 20:
            step = 2  # Medium range, use step 2
        else:
            step = max(1, value_range // 10)  # Large range, use adaptive step
        
        plt.yticks(np.arange(min_value, max_value + 1, step))
        
        plt.savefig(f'output_{anion}_concentration_curves.png')
        plt.show()

def main():
    # Delete remaining data
    try:
        os.remove('output data.xlsx')
        os.remove('output LR graph.png')
        os.remove('output_Fluoride_concentration_curves.png')
        os.remove('output_Chloride_concentration_curves.png')
        os.remove('output_Sulfate_concentration_curves.png')
        os.remove('output_Phosphate_concentration_curves.png')
    except:
        pass

    # Window setting
    root = tk.Tk()
    root.title("MagIC Data Processing")
    root.geometry("800x400")

    # Username part
    username_frame = ttk.Frame(root, padding="10")
    username_frame.grid(row=0, column=0, sticky="ew")

    ttk.Label(username_frame, text="Please enter your name, which is same as the data label:").grid(row=0, column=0, sticky="w")
    username_value = ttk.Entry(username_frame)
    username_value.grid(row=0, column=1, sticky="ew")

    def printget():
        global username
        username = username_value.get()
        ttk.Label(username_frame, text=f"Confirm your name: {username}").grid(row=0, column=2, sticky="w")

    ttk.Button(username_frame, text="Confirm", command=printget).grid(row=0, column=3, sticky="e")

    # Data extraction
    data_frame = ttk.Frame(root, padding="10")
    data_frame.grid(row=1, column=0, sticky="ew")

    def openfile(username):
        global data_path
        data_path = askopenfilename(title="Please choose the export file", filetypes=[("MagIC output file", "*.csv")], defaultextension=".csv")
        ttk.Label(data_frame, text=f"Confirm your data path: {data_path}").grid(row=0, column=1, sticky="w")

        def calculation_and_finish(username, data_path):
            calculation(username, data_path)
            ttk.Label(root, text="All calculations are done, close the window to finish the program.").grid(row=3, column=0, sticky="w")

        ttk.Button(root, text="Press to start calculation", command=lambda: calculation_and_finish(username, data_path)).grid(row=2, column=0, sticky="ew")

    ttk.Button(data_frame, text="Please choose your data file", command=lambda: openfile(username)).grid(row=0, column=0, sticky="w")

    root.mainloop()

def calculation(username, data_path):
    # Read data from data path
    original_data = pd.read_csv(data_path)
    data_array = np.nan_to_num(np.array(original_data))

    # Extract standard line data
    std_data = original_data[original_data['Ident'].str.startswith('STD')].copy()
    
    # Convert standard data to numeric values
    std_data.loc[:, 'Anions.Fluoride.Area'] = pd.to_numeric(std_data['Anions.Fluoride.Area'], errors='coerce')
    std_data.loc[:, 'Anions.Chloride.Area'] = pd.to_numeric(std_data['Anions.Chloride.Area'], errors='coerce')
    std_data.loc[:, 'Anions.Sulfate.Area'] = pd.to_numeric(std_data['Anions.Sulfate.Area'], errors='coerce')
    std_data.loc[:, 'Anions.Phosphate.Area'] = pd.to_numeric(std_data['Anions.Phosphate.Area'], errors='coerce')

    # Manually set standard concentrations (assuming STD 1 = 1, STD 2 = 2, etc.)
    std_data['Concentration'] = std_data['Ident'].str.extract(r'STD (\d+)').astype(float)

    # Perform linear regression for each anion
    anions = ['Fluoride', 'Chloride', 'Sulfate', 'Phosphate']
    regression_parameters = {}

    for anion in anions:
        area_col = f'Anions.{anion}.Area'
        
        # Filter out rows with NaN values
        valid_data = std_data[[area_col, 'Concentration']].dropna()
        
        if len(valid_data) > 1:
            X = valid_data[area_col].values.reshape(-1, 1)
            y = valid_data['Concentration'].values
            
            model = LinearRegression()
            model.fit(X, y)
            regression_parameters[anion] = {
                'slope': model.coef_[0],
                'intercept': model.intercept_
            }
        else:
            regression_parameters[anion] = None

    # Extract user's data
    userdata = []
    labels = []
    username_length = len(username)
    for row in data_array:
        if isinstance(row[1], float):
            pass
        else:
            if row[1][:username_length] == username:
                userdata.append([row[1], row[7], row[9], row[3], row[5], row[11], row[13]])  # Add other anions data
                labels.append(row[1])

    # Calculate concentration for all samples
    results = []
    for label, data in zip(labels, userdata):
        time = label.split('-')[1]
        sample = label.split('-')[2]
        result_row = [label, time, sample]
        
        for anion in anions:
            area_col = f'Anions.{anion}.Area'
            area_value = data[anions.index(anion) + 3]  # Adjust index based on userdata structure
            
            if regression_parameters[anion] is not None and area_value != '':
                slope = regression_parameters[anion]['slope']
                intercept = regression_parameters[anion]['intercept']
                concentration = slope * float(area_value) + intercept
                result_row.append(concentration)
            else:
                result_row.append(np.nan)
        
        results.append(result_row)

    # Create DataFrame
    output_data = pd.DataFrame(results, columns=['Label', 'Time', 'Sample'] + [f'{anion}-Concentration' for anion in anions])

    # Sort by time in ascending order
    output_data = output_data.sort_values(by='Time', ascending=True)

    # Define the desired order for samples
    sample_order = ['MA', 'OA', 'MB', 'OB']
    output_data['Sample'] = pd.Categorical(output_data['Sample'], categories=sample_order, ordered=True)
    output_data = output_data.sort_values(by=['Sample', 'Time'], ascending=[True, True])

    # Save to Excel
    with pd.ExcelWriter('output data.xlsx') as writer:
        output_data.to_excel(writer, sheet_name='All Samples', index=False)

    # Plot the result curves
    plot_result_curves(output_data)

main()