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
    print("debug:plot_linear_regression start")
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
    print("debug:plot_linear_regression end")
    return parameters

# Calculate nitrogen concentration by nitrite
def Nitrite_N(data, parameters, lower_limit, higher_limit):
    print("debug:Nitrite_N start")
    Nitrite_N_result = []
    for row in data:
        if row[1] > lower_limit:
            nitrogen = (row[1] + parameters[1]) / parameters[0] * float(row[0][-3:-1]) / 46.005 * 14.007
            Nitrite_N_result.append(nitrogen)
        else:
            nitrogen = 0
            Nitrite_N_result.append(nitrogen)
    NO2N = np.array(Nitrite_N_result).reshape((len(Nitrite_N_result), 1))
    print("debug:Nitrite_N end")
    return NO2N

# Calculate nitrogen concentration by nitrate
def Nitrate_N(data, parameters, lower_limit, higher_limit):
    print("debug:Nitrate_N start")
    Nitrate_N_result = []
    for row in data:
        if row[2] > lower_limit:
            nitrogen = (row[2] + parameters[3]) / parameters[2] * float(row[0][-3:-1]) / 62.004 * 14.007
            Nitrate_N_result.append(nitrogen)
        else:
            nitrogen = 0
            Nitrate_N_result.append(nitrogen)
    NO3N = np.array(Nitrate_N_result).reshape((len(Nitrate_N_result), 1))
    print("debug:Nitrate_N end")
    return NO3N

# Plot the result curves for MA, OA, MB, OB
def plot_result_curves(output_data):
    print("debug:plot_result_curves start")
    # Filter data to only include MA, MB, OA, OB
    filtered_data = output_data[output_data['Sample'].isin(['MA', 'MB', 'OA', 'OB'])].copy()
    
    # Convert 'Nitrite-N' and 'Nitrate-N' columns to numeric values
    filtered_data.loc[:, 'Nitrite-N'] = pd.to_numeric(filtered_data['Nitrite-N'], errors='coerce')
    filtered_data.loc[:, 'Nitrate-N'] = pd.to_numeric(filtered_data['Nitrate-N'], errors='coerce')
    
    # Sort by Time in ascending order
    filtered_data = filtered_data.sort_values(by='Time', ascending=True)
    
    # Define colors for each sample
    colors = {'MA': 'red', 'OA': 'blue', 'MB': 'green', 'OB': 'orange'}
    
    # Plot Nitrite-N
    plt.figure(figsize=(10, 6))
    for sample, color in colors.items():
        sample_data = filtered_data[filtered_data['Sample'] == sample]
        if not sample_data.empty:
            plt.plot(sample_data['Time'], sample_data['Nitrite-N'], label=f'{sample} Nitrite-N', color=color, linestyle='-')
    
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Nitrite-N Concentration', fontsize=14)
    plt.title('Nitrite-N Concentration Over Time', fontsize=16)
    plt.legend()
    plt.grid(True)
    
    # Set y-axis to integer values with step 10
    nitrite_max = filtered_data['Nitrite-N'].max()
    if pd.isna(nitrite_max):
        plt.close()
        print('Warning: No valid Nitrite-N data to plot.')
    else:
        max_value = int(nitrite_max) + 10
        plt.yticks(np.arange(0, max_value, step=10))
        plt.savefig('output_Nitrite-N_curves.png')
        plt.show()
    
    # Plot Nitrate-N
    plt.figure(figsize=(10, 6))
    for sample, color in colors.items():
        sample_data = filtered_data[filtered_data['Sample'] == sample]
        if not sample_data.empty:
            plt.plot(sample_data['Time'], sample_data['Nitrate-N'], label=f'{sample} Nitrate-N', color=color, linestyle='--')
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Nitrate-N Concentration', fontsize=14)
    plt.title('Nitrate-N Concentration Over Time', fontsize=16)
    plt.legend()
    plt.grid(True)
    # Set y-axis to integer values with step 10
    nitrate_max = filtered_data['Nitrate-N'].max()
    if pd.isna(nitrate_max):
        plt.close()
        print('Warning: No valid Nitrate-N data to plot.')
    else:
        max_value = int(nitrate_max) + 10
        plt.yticks(np.arange(0, max_value, step=10))
        plt.savefig('output_Nitrate-N_curves.png')
        plt.show()
    print("debug:plot_result_curves end")

def main():
    # Delete remaining data
    try:
        os.remove('output data.xlsx')
        os.remove('output LR graph.png')
        os.remove('output_result_curves.png')
    except:
        pass

    # Window setting

    root = tk.Tk()
    root.title("MagIC Data Processing")
    root.geometry("700x320")
    style = ttk.Style()
    try:
        style.theme_use('clam')
    except:
        pass
    default_font = ("Segoe UI", 12)
    root.option_add("*Font", default_font)
    root.configure(bg="#f7f7fa")

    # Title
    title_label = ttk.Label(root, text="MagIC Data Processing", font=("Segoe UI", 18, "bold"), anchor="center")
    title_label.grid(row=0, column=0, columnspan=3, pady=(18, 8), sticky="ew")

    # Username input
    username_frame = ttk.Frame(root, padding="10 5 10 5")
    username_frame.grid(row=1, column=0, columnspan=3, sticky="ew")
    username_frame.columnconfigure(1, weight=1)
    ttk.Label(username_frame, text="User Name (same as data label):").grid(row=0, column=0, sticky="w", padx=(0,8))
    username_value = ttk.Entry(username_frame, width=24)
    username_value.grid(row=0, column=1, sticky="ew")
    confirm_label = ttk.Label(username_frame, text="", foreground="#0078d4")
    confirm_label.grid(row=0, column=2, sticky="w", padx=(8,0))

    def printget():
        global username
        username = username_value.get()
        confirm_label.config(text=f"Confirmed: {username}")

    confirm_btn = ttk.Button(username_frame, text="Confirm", command=printget)
    confirm_btn.grid(row=0, column=3, padx=(8,0))

    # Separator
    sep1 = ttk.Separator(root, orient="horizontal")
    sep1.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(5,5))

    # Data file selection
    data_frame = ttk.Frame(root, padding="10 5 10 5")
    data_frame.grid(row=3, column=0, columnspan=3, sticky="ew")
    data_frame.columnconfigure(1, weight=1)
    data_path_label = ttk.Label(data_frame, text="No file selected", foreground="#888")
    data_path_label.grid(row=0, column=1, sticky="w", padx=(8,0))

    def openfile():
        global data_path
        data_path = askopenfilename(title="Select MagIC output file", filetypes=[("CSV Files", "*.csv")], defaultextension=".csv")
        if data_path:
            data_path_label.config(text=f"Selected: {os.path.basename(data_path)}", foreground="#0078d4")
            start_btn.state(["!disabled"])
        else:
            data_path_label.config(text="No file selected", foreground="#888")
            start_btn.state(["disabled"])

    select_btn = ttk.Button(data_frame, text="Select Data File", command=openfile)
    select_btn.grid(row=0, column=0, sticky="w")

    # Start calculation
    def calculation_and_finish():
        calculation(username_value.get(), data_path)
        finish_label.config(text="All calculations are done. The window will close in 3 seconds.", foreground="#228B22")
        root.after(3000, root.destroy)

    start_btn = ttk.Button(root, text="Start Calculation", command=calculation_and_finish, state="disabled")
    start_btn.grid(row=4, column=0, columnspan=3, pady=(18,0), sticky="ew")

    # Finish label
    finish_label = ttk.Label(root, text="", font=("Segoe UI", 11), anchor="center")
    finish_label.grid(row=5, column=0, columnspan=3, pady=(10,0), sticky="ew")

    # Responsive resizing
    for i in range(3):
        root.grid_columnconfigure(i, weight=1)

    root.mainloop()

def calculation(username, data_path):
    print("debug:calculation start")
    # Read data from data path
    original_data = pd.read_csv(data_path)
    filled_original_data = original_data.fillna(0)
    data_array = np.nan_to_num(np.array(filled_original_data))

    # Extract standard line data
    standard_line = []
    nitrite = []
    nitrate = []
    filtered_data_array = data_array
    row_counter = -1
    for row in data_array:
        if isinstance(row[0], float):
            pass
        else:
            row_counter += 1
            if row[0][:2] == "20":
                if row[1][:3] == "STD":
                    standard_line.append(float(row[1][4:]))
                    nitrite.append(row[7])
                    nitrate.append(row[9])
            else:
                row_counter -= 1
                filtered_data_array = np.delete(filtered_data_array, row_counter, axis=0)

    standard_line.sort()
    nitrite.sort()
    nitrate.sort()

    # Extract user's data
    userdata = []
    labels = []
    username_length = len(username)
    for row in filtered_data_array:
        if isinstance(row[1], float):
            pass
        else:
            if row[1][:username_length] == username:
                userdata.append([row[1], row[7], row[9]])
                labels.append(row[1])

    # Detect standard line data
    try:
        standard_line = [float(stdnum) for stdnum in standard_line]
    except:
        print("No standard line data detected, please check the naming format of input file")

    # Calculate linear regression result
    parameters = plot_linear_regression(standard_line, nitrite, nitrate)

    # Calculate concentration of all samples
    NO2_N = np.nan_to_num(Nitrite_N(userdata, parameters, nitrite[0], nitrite[-1]))
    NO3_N = np.nan_to_num(Nitrate_N(userdata, parameters, nitrate[0], nitrate[-1]))

    # Extract time and sample name from labels
    times = []
    samples = []
    for label in labels:
        parts = label.split('-')
        times.append(parts[1])
        samples.append(parts[2])

    # Combine all data into a DataFrame
    Nitrogen_result = np.hstack((np.array(labels).reshape(len(labels), 1),
                                 np.array(times).reshape(len(times), 1),
                                 np.array(samples).reshape(len(samples), 1),
                                 NO2_N, NO3_N))

    # Create DataFrame
    output_data = pd.DataFrame(Nitrogen_result, columns=['Label', 'Time', 'Sample', 'Nitrite-N', 'Nitrate-N'])

    # Sort by time in reverse order
    output_data = output_data.sort_values(by='Time', ascending=False)

    # Define the desired order for samples
    sample_order = ['MA', 'OA', 'MB', 'OB']
    output_data['Sample'] = pd.Categorical(output_data['Sample'], categories=sample_order, ordered=True)
    output_data = output_data.sort_values(by=['Sample', 'Time'], ascending=[True, False])
    
    # Debug print
    print(output_data)
    
    # Save to Excel
    with pd.ExcelWriter('output data.xlsx') as writer:
        output_data.to_excel(writer, sheet_name='All Samples', index=False)

    # Save the regression line graph
    plot_title = "[nitrite]:F(x)=" + str(parameters[0])[:8] + "x + " + str(parameters[1])[:8] + ", " + "[nitrate]:F(x)=" + str(parameters[2])[:8] + "x + " + str(parameters[3])[:8]
    plt.title(plot_title)
    plt.draw()
    plt.savefig('output LR graph.png')
    plt.show()

    # Plot the result curves
    plot_result_curves(output_data)
    print("debug:calculation end")
    return

main()