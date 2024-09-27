from tkinter.filedialog import askopenfilename
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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
    print("[nitrite]:F(x)=",lineModel1.coef_[0][0],"x + ",lineModel1.intercept_[0])
    print("[nitrate]:F(x)=",lineModel2.coef_[0][0],"x + ",lineModel2.intercept_[0])
    parameters = [lineModel1.coef_[0][0],lineModel1.intercept_[0],lineModel2.coef_[0][0],lineModel2.intercept_[0]]
    return parameters

# Calculate nitrogen concentration by nitrite
def Nitrite_N(data,parameters,lower_limit,higher_limit):
    Nitrite_N_result = []
    for row in data:
        if row[1] > lower_limit:
            if row[1] > higher_limit:
                nitrogen = "overlimit"
                Nitrite_N_result.append(nitrogen)
            else:
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
            if row[2] > higher_limit:
                nitrogen = "overlimit"
                Nitrate_N_result.append(nitrogen)
            else:
                nitrogen = (row[2]+parameters[3])/parameters[2]*float(row[0][-3:-1])/62.004*14.007
                Nitrate_N_result.append(nitrogen)
        else:
            nitrogen = 0
            Nitrate_N_result.append(nitrogen)
    NO3N = np.array(Nitrate_N_result).reshape((len(Nitrate_N_result), 1))
    return NO3N

def main():
    # Enter user's name which is marked on data label
    username = input("Please enter your name, which is same as the data label: ")
    
    # Choose .csv file and read data from it
    print("Please choose the data file with ending of .csv")
    data_path = askopenfilename()
    original_data = pd.read_csv(data_path)
    data_array = np.array(original_data)

    # Extract standard line data from it for linear regression
    standard_line = []
    nitrite = []
    nitrate = []
    for row in data_array:
        if row[1][:3]=="STD":
            standard_line.append(float(row[1][4:]))
            nitrite.append(row[7])
            nitrate.append(row[9])
    standard_line.sort()
    nitrite.sort()
    nitrate.sort()
    
    # Extract user's data
    userdata = []
    labels = []
    username_length = len(username)
    for row in data_array:
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
    output_data.to_excel('./output/output data.xlsx', index = False)
    
    # Save the regression line pragh
    print("All calculations are done, close the graph to finish the program.")
    plot_title = "[nitrite]:F(x)="+str(parameters[0])[:8]+"x + "+str(parameters[1])[:8]+", "+"[nitrate]:F(x)="+str(parameters[2])[:8]+"x + "+str(parameters[3])[:8]
    plt.title(plot_title)
    plt.draw()
    plt.savefig('./output/output LR graph.png')
    plt.show()

main()
