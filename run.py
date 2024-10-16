import tkinter as tk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tkinter.filedialog import askopenfilename
from sklearn.linear_model import LinearRegression

# Draw linear regression function
def plot_linear_regression(x,y1,y2):
    # Window config
    plt.rcParams['figure.figsize'] = (6,5)
    plt.rcParams['savefig.dpi'] = 200 
    plt.rcParams['figure.dpi'] = 200 
    plt.rcParams['font.sans-serif']=['Arial']
    plt.scatter(x,y1,marker='o',s=15,c='white',edgecolors='red')
    plt.scatter(x,y2,marker='o',s=15,c='white',edgecolors='blue')
    font1 = {'family':'Arial', 'weight':'normal', 'size':18,}
    plt.xlabel('concentration',font1)
    plt.ylabel('area',font1)
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
    plt.plot(T_x,lineModel1.predict(T_x),linestyle='dotted',color='red')
    plt.plot(T_x,lineModel2.predict(T_x),linestyle='dotted',color='blue')
    ax=plt.gca()
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    plt.tick_params(axis='both',width=1,length=5)
    plt.draw()
    parameters = [lineModel1.coef_[0][0],lineModel1.intercept_[0],lineModel2.coef_[0][0],lineModel2.intercept_[0]]
    return parameters

# Calculate nitrogen concentration by nitrite
def Nitrite_N(data,parameters,lower_limit,higher_limit):
    Nitrite_N_result = []
    for row in data:
        if row[1] > lower_limit:
            # if row[1] > higher_limit:
            #     nitrogen = "overlimit"
            #     Nitrite_N_result.append(nitrogen)
            # else:
                nitrogen = (row[1]+parameters[1])/parameters[0]*float(row[0][-3:-1])/46.005*14.007
                Nitrite_N_result.append(nitrogen)
        else:
            nitrogen = 0
            Nitrite_N_result.append(nitrogen)
    NO2N = np.array(Nitrite_N_result).reshape((len(Nitrite_N_result), 1))
    return NO2N

# Calculate nitrogen concentration by nitrate
def Nitrate_N(data,parameters,lower_limit,higher_limit):
    Nitrate_N_result = []
    for row in data:
        if row[2] > lower_limit:
            # if row[2] > higher_limit:
            #     nitrogen = "overlimit"
            #     Nitrate_N_result.append(nitrogen)
            # else:
                nitrogen = (row[2]+parameters[3])/parameters[2]*float(row[0][-3:-1])/62.004*14.007
                Nitrate_N_result.append(nitrogen)
        else:
            nitrogen = 0
            Nitrate_N_result.append(nitrogen)
    NO3N = np.array(Nitrate_N_result).reshape((len(Nitrate_N_result), 1))
    return NO3N

def main():
    
    # Delete remaining data
    try:
        os.remove('output data.xlsx')
        os.remove('output LR graph.png')
    except:
        None
    
    '''
    GUI packing start
    '''
    
    # Window setting
    root = tk.Tk()
    root.title("MagIC data processing")
    root.geometry("800x300")
    
    # Username part
    username_frame = tk.Frame(root)
    username_frame.grid(row=0)
    
    ## Username label
    username_requiring_text = tk.Label(username_frame, text="Please enter your name, which is same as the data label:")
    username_requiring_text.grid(row=0, column=0)
    
    ## Username input box and button
    username_value = tk.Entry(username_frame)
    username_value.grid(row=0, column=1)
    def printget(): # Get username data from input box to global variable "username" and return confirm message
        global username
        username = username_value.get()
        username_confirm = tk.Label(username_frame, text="Confirm your name: " + username)
        username_confirm.grid(row=0, column=4)
    tk.Button(username_frame, text = "Confirm", command = lambda: printget()).grid(row=0, column=2)
    
    # Data extraction
    data_frame = tk.Frame(root)
    data_frame.grid(row=1)
    
    ## Data file choosing button
    def openfile(username): # Get exported data from filemanager to global variable "data_path" and return confirm message
        global data_path
        data_path = askopenfilename(title = "Please choose the export file", filetypes=[("MagIC output file", "*.csv")], defaultextension=".csv")
        data_confirm = tk.Label(data_frame, text="Confirm your data path: " + data_path)
        data_confirm.grid(row=0,column=1)
        # Generate confirm button, which is to continue the calculation part
        def calculation_and_finish(username,data_path):
            calculation(username,data_path)
            tk.Label(root, text="All calculations are done, close the window to finish the program.").grid(row=3)
        tk.Button(root, text = "Press to start calculation", command = lambda: calculation_and_finish(username,data_path)).grid(row=2)
     
    file_path_button = tk.Button(data_frame, text = "Please choose your data file", command = lambda: openfile(username))
    file_path_button.grid(row=0, column=0)
    
    tk.mainloop()
    
def calculation(username,data_path):
    # Read data from data path
    original_data = pd.read_csv(data_path)
    data_array = np.nan_to_num(np.array(original_data))

    # Extract standard line data from it for linear regression
    standard_line = []
    nitrite = []
    nitrate = []
    row_counter = -1
    for row in data_array:
        row_counter+=1
        if row[0][:2] == "20":
            if row[1][:3] == "STD":
                standard_line.append(float(row[1][4:]))
                nitrite.append(row[7])
                nitrate.append(row[9])
        else:
            filtered_data_array = np.delete(data_array, row_counter, axis = 0)
        
    standard_line.sort()
    nitrite.sort()
    nitrate.sort()
    
    # Extract user's data
    userdata = []
    labels = []
    username_length = len(username)
    for row in filtered_data_array:
        if row[1][:username_length] == username:
            userdata.append([row[1],row[7],row[9]])
            labels.append(row[1])
            
    # Detect standard line data and transform into proper format
    try:
        standard_line = [float(stdnum) for stdnum in standard_line]
    except:
        print("No standard line data detected, please check the naming format of input file")
        
    # Calculate linear regression result
    parameters = plot_linear_regression(standard_line,nitrite,nitrate)
    
    # Calculate concentration of all samples
    NO2_N = np.nan_to_num(Nitrite_N(userdata, parameters , nitrite[0] , nitrite[-1]))
    NO3_N = np.nan_to_num(Nitrate_N(userdata, parameters , nitrate[0] , nitrate[-1]))
    
    # Output excel sheet
    labels = np.array(labels).reshape(len(labels), 1)
    Nitrogen_result = np.hstack((labels, NO2_N, NO3_N))
    output_data = pd.DataFrame(Nitrogen_result)   
    output_data.columns = ['Label','Nitrite-N','Nitrate-N'] 
    output_data.to_excel('output data.xlsx', index = False)
    
    # Save the regression line pragh
    plot_title = "[nitrite]:F(x)="+str(parameters[0])[:8]+"x + "+str(parameters[1])[:8]+", "+"[nitrate]:F(x)="+str(parameters[2])[:8]+"x + "+str(parameters[3])[:8]
    plt.title(plot_title)
    plt.draw()
    plt.savefig('output LR graph.png')
    plt.show()
    
main()
