import pandas as pd
import numpy as np
from functions import preprocess_min_max_group
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import json
import matplotlib as mpl
import random
from dtaidistance import dtw
from scipy.stats import ttest_rel
from matplotlib.gridspec import GridSpec
random.seed(2)
np.random.seed(42)
import os 
os.environ['PATH'] = "/Library/TeX/texbin:" + os.environ.get('PATH', '')
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['text.latex.preamble'] = r'\usepackage{lmodern}\usepackage[T1]{fontenc}'

def boot(data, num_samples=1000, statistic=np.mean):
    n = len(data)
    bootstrap_estimates = np.empty(num_samples)
    for i in range(num_samples):
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_estimates[i] = statistic(bootstrap_sample)
    standard_error = np.std(bootstrap_estimates)
    return standard_error

micro_states={"Dominica":54,
              "Grenada":55,
              "Saint Lucia":56,
              "Saint Vincent and the Grenadines":57,
              "Antigua & Barbuda":58,
              "Saint Kitts and Nevis":60,
              "Monaco":221,
              "Liechtenstein":223,
              "San Marino":331,
              "Andorra":232,
              "Abkhazia":396,
              "South Ossetia":397,
              "São Tomé and Principe":403,
              "Seychelles":591,
              "Vanuatu":935,
              "Kiribati":970,
              "Nauru":971,
              "Tonga":972,
              "Tuvalu":973,
              "Marshall Islands":983,
              "Palau":986,
              "Micronesia":987,
              "Samoa":990}

df=pd.read_csv("data/df.csv",index_col=0)
df = df[~df['gw_codes'].isin(list(micro_states.values()))]
df = df.reset_index(drop=True)

# Main results
df_linear = pd.read_csv("data/df_linear.csv",index_col=0)
df_nonlinear =  pd.read_csv("data/df_nonlinear.csv",index_col=0)
print("Linear models")
print(mean_squared_error(df_linear.fatalities, df_linear.preds_ols))
print(mean_squared_error(df_linear.fatalities, df_linear.preds_olsx))
print(mean_squared_error(df_linear.fatalities, df_linear.preds_dols))
print(mean_squared_error(df_linear.fatalities, df_linear.preds_dolsx))
print("Non-linear models")
print(mean_squared_error(df_nonlinear.fatalities, df_nonlinear.preds_rf))
print(mean_squared_error(df_nonlinear.fatalities, df_nonlinear.preds_rfx))
print(mean_squared_error(df_nonlinear.fatalities, df_nonlinear.preds_drf))
print(mean_squared_error(df_nonlinear.fatalities, df_nonlinear.preds_drfx))

df_nonlinear["mse_rf"]=((df_nonlinear["fatalities"] - df_nonlinear["preds_rf"]) ** 2) 
df_nonlinear["mse_rfx"]=((df_nonlinear["fatalities"] - df_nonlinear["preds_rfx"]) ** 2)
df_nonlinear["mse_drf"]=((df_nonlinear["fatalities"] - df_nonlinear["preds_drf"]) ** 2) 
df_nonlinear["mse_drfx"]=((df_nonlinear["fatalities"] - df_nonlinear["preds_drfx"]) ** 2) 
means = [df_nonlinear["mse_rf"].mean(),df_nonlinear["mse_rfx"].mean(),df_nonlinear["mse_drf"].mean(),df_nonlinear["mse_drfx"].mean()]
std_error = [boot(df_nonlinear["mse_rf"]),boot(df_nonlinear["mse_rfx"]),boot(df_nonlinear["mse_drf"]),boot(df_nonlinear["mse_drfx"])]
mean_mse_rf = pd.DataFrame({'mean': means,'std': std_error})
 
df_linear["mse_ols"]=((df_linear["fatalities"] - df_linear["preds_ols"]) ** 2) 
df_linear["mse_olsx"]=((df_linear["fatalities"] - df_linear["preds_olsx"]) ** 2)
df_linear["mse_dols"]=((df_linear["fatalities"] - df_linear["preds_dols"]) ** 2) 
df_linear["mse_dolsx"]=((df_linear["fatalities"] - df_linear["preds_dolsx"]) ** 2) 
means = [df_linear["mse_ols"].mean(),df_linear["mse_olsx"].mean(),df_linear["mse_dols"].mean(),df_linear["mse_dolsx"].mean()]
std_error = [boot(df_linear["mse_ols"]),boot(df_linear["mse_olsx"]),boot(df_linear["mse_dols"]),boot(df_linear["mse_dolsx"])]
mean_mse_linear = pd.DataFrame({'mean': means,'std': std_error})

print(mean_mse_linear)
print(round(ttest_rel(df_linear["mse_ols"], df_linear["mse_olsx"])[1],5))
print(round(ttest_rel(df_linear["mse_ols"], df_linear["mse_dols"])[1],5))
print(round(ttest_rel(df_linear["mse_olsx"], df_linear["mse_dolsx"])[1],5))

print(mean_mse_rf)
print(round(ttest_rel(df_nonlinear["mse_rf"], df_nonlinear["mse_rfx"])[1],5))
print(round(ttest_rel(df_nonlinear["mse_rf"], df_nonlinear["mse_drf"])[1],5))
print(round(ttest_rel(df_nonlinear["mse_rfx"], df_nonlinear["mse_drfx"])[1],5))

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 7))
marker_size = 150
linewidth = 3
fonts=25
ax1.scatter(mean_mse_linear.index, mean_mse_linear['mean'], color="black", marker='o', s=marker_size)
ax1.errorbar(mean_mse_linear.index, mean_mse_linear['mean'], yerr=mean_mse_linear['std'], fmt='none', color="black", linewidth=linewidth)
ax2.scatter(mean_mse_rf.index, mean_mse_rf['mean'], color="black", marker='o', s=marker_size)
ax2.errorbar(mean_mse_rf.index, mean_mse_rf['mean'], yerr=mean_mse_rf['std'], fmt='none', color="black", linewidth=linewidth)
ax1.grid(False)
ax2.grid(False)
ax1.set_ylim(0.0118, 0.0165)
ax2.set_ylim(0.0091, 0.013)
ax1.set_yticks([0.011,0.012,0.013,0.014,0.015,0.016,0.017],[0.011,0.012,0.013,0.014,0.015,0.016,0.017],fontsize=18)
ax2.set_yticks([0.0091,0.0096,0.0101,0.0106,0.0111,0.0116,0.0121,0.0126,0.0131],[0.0091,0.0096,0.0101,0.0106,0.0111,0.0116,0.0121,0.0126,0.0131],size=18)
ax2.yaxis.set_ticks_position('right')
ax1.set_xticks([*range(4)],['RR','RRX','DRR','DRRX'],fontsize=18)
ax2.set_xticks([*range(4)],['RF','RFX','DRF','DRFX'],fontsize=18)
plt.subplots_adjust(wspace=0.05)
ax1.set_ylabel("Mean squared error (MSE)",size=22)

ax1.plot([0,1],[0.0135,0.0135],linewidth=0.5,color="black")
ax1.plot([0,0],[0.0135,0.0136],linewidth=0.5,color="black")
ax1.plot([1,1],[0.0135,0.0136],linewidth=0.5,color="black")
ax1.text(0.47, 0.01338, "x", fontsize=12)

ax1.plot([0,2],[0.0116,0.0116],linewidth=0.5,color="black")
ax1.plot([0,0],[0.0116,0.0117],linewidth=0.5,color="black")
ax1.plot([2,2],[0.0116,0.0117],linewidth=0.5,color="black")
ax1.text(0.92, 0.01143, "***", fontsize=12)

ax1.plot([1,3],[0.0112,0.0112],linewidth=0.5,color="black")
ax1.plot([1,1],[0.0112,0.0113],linewidth=0.5,color="black")
ax1.plot([3,3],[0.0112,0.0113],linewidth=0.5,color="black")
ax1.text(1.92, 0.011021, "***", fontsize=12)

ax2.plot([0,1],[0.01007,0.01007],linewidth=0.5,color="black")
ax2.plot([0,0],[0.01007,0.01014],linewidth=0.5,color="black")
ax2.plot([1,1],[0.01007,0.01014],linewidth=0.5,color="black")
ax2.text(0.42,0.00996, "***", fontsize=12)

ax2.plot([0,2],[0.0098,0.0098],linewidth=0.5,color="black")
ax2.plot([0,0],[0.0098,0.00987],linewidth=0.5,color="black")
ax2.plot([2,2],[0.0098,0.00987],linewidth=0.5,color="black")
ax2.text(0.91,0.00968, "***", fontsize=12)

ax2.plot([1,3],[0.012,0.012],linewidth=0.5,color="black")
ax2.plot([1,1],[0.012,0.0119],linewidth=0.5,color="black")
ax2.plot([3,3],[0.012,0.0119],linewidth=0.5,color="black")
ax2.text(1.92,0.01199, "***", fontsize=12)

plt.savefig("out/results_main_plot.eps",dpi=300,bbox_inches="tight")
plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_dissertation/out/results_main_plot.eps",dpi=300,bbox_inches='tight')

# All shapes 
df_linear = pd.read_csv("data/df_linear.csv",index_col=0)
df_nonlinear =  pd.read_csv("data/df_nonlinear.csv",index_col=0)

with open("data/rf_shapes.json", 'r') as json_file:
    shapes_rf = json.load(json_file)
    
with open("data/ols_shapes.json", 'r') as json_file:
    shapes_ols = json.load(json_file)
    
countries=df_linear.country.unique()
fig, axs = plt.subplots(17, 10, figsize=(26, 37))
for c,i,j in zip(countries,[0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,9,9,10,10,10,10,10,10,10,10,10,10,11,11,11,11,11,11,11,11,11,11,12,12,12,12,12,12,12,12,12,12,13,13,13,13,13,13,13,13,13,13,14,14,14,14,14,14,14,14,14,14,15,15,15,15,15,15,15,15,15,15,16,16,16,16,16,16,16,16,16,16],[0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9]):
    cmap = plt.get_cmap('gray')
    colors = cmap(np.linspace(0, 1, len(shapes_rf[f"drfx_{c}"][1])+1))
    for x,col in zip(range(len(shapes_rf[f"drfx_{c}"][1])),colors):  
        seq=shapes_rf[f"drfx_{c}"][1][x]
        axs[i, j].plot(seq,color=col)
        axs[i, j].set_yticks([],[])
        axs[i, j].set_xticks([],[])
        axs[i, j].set_axis_off()
        if c =='Democratic Republic of Congo':
            axs[i, j].set_title(f'DRC',size=25)
        elif c =='Central African Republic':
            axs[i, j].set_title(f'CAR',size=25)
        elif c =='Bosnia and Herzegovina':
            axs[i, j].set_title(f'BIH',size=25)      
        elif c =='Domican Republic':
            axs[i, j].set_title(f'Domican Rep.',size=25)  
        elif c =='Equatorial Guinea':
            axs[i, j].set_title(f'Eq. Guinea',size=25)     
        elif c =='United Arab Emirates':
            axs[i, j].set_title(f'UAE',size=25)                
        else: 
            axs[i, j].set_title(f'{c}',size=25)
         
plt.subplots_adjust(wspace=0.05)            
for ax in axs[16, 8:]:
    ax.remove()            
            
plt.savefig("out/results_cluster_grid_drfx.eps",dpi=300,bbox_inches="tight")
plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_dissertation/out/results_cluster_grid_drfx.eps",dpi=300,bbox_inches='tight')
 
fig, axs = plt.subplots(17, 10, figsize=(26, 37))
for c,i,j in zip(countries,[0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,9,9,10,10,10,10,10,10,10,10,10,10,11,11,11,11,11,11,11,11,11,11,12,12,12,12,12,12,12,12,12,12,13,13,13,13,13,13,13,13,13,13,14,14,14,14,14,14,14,14,14,14,15,15,15,15,15,15,15,15,15,15,16,16,16,16,16,16,16,16,16,16],[0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9]):
    cmap = plt.get_cmap('gray')
    colors = cmap(np.linspace(0, 1, len(shapes_ols[f"dolsx_{c}"][1])+1))
    for x,col in zip(range(len(shapes_ols[f"dolsx_{c}"][1])),colors):  
        seq=shapes_ols[f"dolsx_{c}"][1][x]
        axs[i, j].plot(seq,color=col)
        axs[i, j].set_yticks([],[])
        axs[i, j].set_xticks([],[])
        axs[i, j].set_axis_off()
        if c =='Democratic Republic of Congo':
            axs[i, j].set_title(f'DRC',size=25)
        elif c =='Central African Republic':
            axs[i, j].set_title(f'CAR',size=25)
        elif c =='Bosnia and Herzegovina':
            axs[i, j].set_title(f'BIH',size=25)      
        elif c =='Domican Republic':
            axs[i, j].set_title(f'Domican Rep.',size=25)  
        elif c =='Equatorial Guinea':
            axs[i, j].set_title(f'Eq. Guinea',size=25)     
        elif c =='United Arab Emirates':
            axs[i, j].set_title(f'UAE',size=25)                
        else: 
            axs[i, j].set_title(f'{c}',size=25)
         
plt.subplots_adjust(wspace=0.05)            
for ax in axs[16, 8:]:
    ax.remove()           
            
plt.savefig("out/results_cluster_grid_dolsx.eps",dpi=300,bbox_inches="tight") 
plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_dissertation/out/results_cluster_grid_dolsx.eps",dpi=300,bbox_inches='tight')

# Prediction plots 
df_linear = pd.read_csv("data/df_linear.csv",index_col=0)
df_nonlinear =  pd.read_csv("data/df_nonlinear.csv",index_col=0)

fig = plt.figure(figsize=(10, 13))
grid = GridSpec(4, 2, figure=fig, wspace=0.05, hspace=0.4)
for n,y,i,j in zip(["India","India", "Philippines","Philippines", "Somalia","Somalia","Somalia","Somalia"],[2022,2023,2022,2023,2018,2019,2020,2021],[0,0,1,1,2,2,3,3],[0,1,0,1,0,1,0,1]):
    print(n,y,i,j )
    ax = fig.add_subplot(grid[i, j])
    plt.plot(df_linear["dd"].loc[(df_linear["country"]==n)&(df_linear["dd"]>=f"{y}-01")&(df_linear["dd"]<=f"{y}-12")],df_linear["fatalities"].loc[(df_linear["country"]==n)&(df_linear["dd"]>=f"{y}-01")&(df_linear["dd"]<=f"{y}-12")],linestyle="solid",color="black",linewidth=1)
    plt.plot(df_linear["dd"].loc[(df_linear["country"]==n)&(df_linear["dd"]>=f"{y}-01")&(df_linear["dd"]<=f"{y}-12")],df_linear["preds_ols"].loc[(df_linear["country"]==n)&(df_linear["dd"]>=f"{y}-01")&(df_linear["dd"]<=f"{y}-12")],linestyle="dotted",color="black",linewidth=1)
    plt.plot(df_linear["dd"].loc[(df_linear["country"]==n)&(df_linear["dd"]>=f"{y}-01")&(df_linear["dd"]<=f"{y}-12")],df_linear["preds_dolsx"].loc[(df_linear["country"]==n)&(df_linear["dd"]>=f"{y}-01")&(df_linear["dd"]<=f"{y}-12")],linestyle="dashed",color="black",linewidth=1)
   
    if y==2023:
        plt.xticks(["2023-01","2023-03","2023-05","2023-07","2023-09","2023-11"],["01-23","03-23","05-23","07-23","09-23","11-23"],fontsize=15)
    elif y==2022:
        plt.xticks(["2022-01","2022-03","2022-05","2022-07","2022-09","2022-11"],["01-22","03-22","05-22","07-22","09-22","11-22"],fontsize=15)
    elif y==2021:
        plt.xticks(["2021-01","2021-03","2021-05","2021-07","2021-09","2021-11"],["01-21","03-21","05-21","07-21","09-21","11-21"],fontsize=15)
    elif y==2020:
        plt.xticks(["2020-01","2020-03","2020-05","2020-07","2020-09","2020-11"],["01-20","03-20","05-20","07-20","09-20","11-20"],fontsize=15)
    elif y==2019:
        plt.xticks(["2019-01","2019-03","2019-05","2019-07","2019-09","2019-11"],["01-19","03-19","05-19","07-19","09-19","11-19"],fontsize=15)
    elif y==2018:
        plt.xticks(["2018-01","2018-03","2018-05","2018-07","2018-09","2018-11"],["01-18","03-18","05-18","07-18","09-18","11-18"],fontsize=15)
    elif y==2017:
        plt.xticks(["2017-01","2017-03","2017-05","2017-07","2017-09","2017-11"],["01-17","03-17","05-17","07-17","09-17","11-17"],fontsize=15)
    elif y==2016:
        plt.xticks(["2016-01","2016-03","2016-05","2016-07","2016-09","2016-11"],["01-16","03-16","05-16","07-16","09-16","11-16"],fontsize=15) 
 
    if i==0:
        plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6],[0,0.1,0.2,0.3,0.4,0.5,0.6],fontsize=15) 
    if i==1:
        plt.yticks([0,0.1,0.2,0.3,0.4,0.5],[0,0.1,0.2,0.3,0.4,0.5],fontsize=15) 
    if i==2:
        plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7],[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7],fontsize=15)       
    if i==3:
        plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7],[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7],fontsize=15)   

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if j==1:
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])
        if n=="India":
            plt.text(-2, 0.6, n, fontsize=20)
        elif n=="Philippines":
            plt.text(-3, 0.5, n, fontsize=20)
        elif n=="Somalia":
            plt.text(-2.9, 0.7, n, fontsize=20)

plt.savefig("out/preds_best_select_ols.eps",dpi=300,bbox_inches="tight")
plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_dissertation/out/preds_best_select_ols.eps",dpi=300,bbox_inches='tight')

fig = plt.figure(figsize=(10, 13))
grid = GridSpec(4, 2, figure=fig, wspace=0.05, hspace=0.4)
for n,y,i,j in zip(["Thailand","Thailand", "Niger","Niger", "Nigeria","Nigeria","Nigeria","Nigeria"],[2022,2023,2017,2018,2016,2017,2018,2019],[0,0,1,1,2,2,3,3],[0,1,0,1,0,1,0,1]):
    print(n,y,i,j )
    ax = fig.add_subplot(grid[i, j])
    plt.plot(df_linear["dd"].loc[(df_linear["country"]==n)&(df_linear["dd"]>=f"{y}-01")&(df_linear["dd"]<=f"{y}-12")],df_linear["fatalities"].loc[(df_linear["country"]==n)&(df_linear["dd"]>=f"{y}-01")&(df_linear["dd"]<=f"{y}-12")],linestyle="solid",color="black",linewidth=1)
    plt.plot(df_linear["dd"].loc[(df_linear["country"]==n)&(df_linear["dd"]>=f"{y}-01")&(df_linear["dd"]<=f"{y}-12")],df_linear["preds_ols"].loc[(df_linear["country"]==n)&(df_linear["dd"]>=f"{y}-01")&(df_linear["dd"]<=f"{y}-12")],linestyle="dotted",color="black",linewidth=1)
    plt.plot(df_linear["dd"].loc[(df_linear["country"]==n)&(df_linear["dd"]>=f"{y}-01")&(df_linear["dd"]<=f"{y}-12")],df_linear["preds_dolsx"].loc[(df_linear["country"]==n)&(df_linear["dd"]>=f"{y}-01")&(df_linear["dd"]<=f"{y}-12")],linestyle="dashed",color="black",linewidth=1)
   
    if y==2023:
        plt.xticks(["2023-01","2023-03","2023-05","2023-07","2023-09","2023-11"],["01-23","03-23","05-23","07-23","09-23","11-23"],fontsize=15)
    elif y==2022:
        plt.xticks(["2022-01","2022-03","2022-05","2022-07","2022-09","2022-11"],["01-22","03-22","05-22","07-22","09-22","11-22"],fontsize=15)
    elif y==2021:
        plt.xticks(["2021-01","2021-03","2021-05","2021-07","2021-09","2021-11"],["01-21","03-21","05-21","07-21","09-21","11-21"],fontsize=15)
    elif y==2020:
        plt.xticks(["2020-01","2020-03","2020-05","2020-07","2020-09","2020-11"],["01-20","03-20","05-20","07-20","09-20","11-20"],fontsize=15)
    elif y==2017:
        plt.xticks(["2017-01","2017-03","2017-05","2017-07","2017-09","2017-11"],["01-17","03-17","05-17","07-17","09-17","11-17"],fontsize=15)
    elif y==2018:
        plt.xticks(["2018-01","2018-03","2018-05","2018-07","2018-09","2018-11"],["01-18","03-18","05-18","07-18","09-18","11-18"],fontsize=15)
    elif y==2016:
        plt.xticks(["2016-01","2016-03","2016-05","2016-07","2016-09","2016-11"],["01-16","03-16","05-16","07-16","09-16","11-16"],fontsize=15)
    elif y==2019:
        plt.xticks(["2019-01","2019-03","2019-05","2019-07","2019-09","2019-11"],["01-19","03-19","05-19","07-19","09-19","11-19"],fontsize=15)

    if i==0:
        plt.yticks([0,0.1,0.2,0.3,0.4,0.5],[0,0.1,0.2,0.3,0.4,0.5],fontsize=15) 
    if i==1:
        plt.yticks([0,0.1,0.2,0.3,0.4],[0,0.1,0.2,0.3,0.4],fontsize=15) 
    if i==2:
        plt.yticks([0,0.1,0.2,0.3,0.4,0.5],[0,0.1,0.2,0.3,0.4,0.5],fontsize=15)       
    if i==3:
        plt.yticks([0,0.1,0.2,0.3,0.4],[0,0.1,0.2,0.3,0.4],fontsize=15)       

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if j==1:
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])
        if n=="Thailand":
            plt.text(-2.9, 0.5, n, fontsize=20)
        elif n=="Niger":
            plt.text(-2.5, 0.4, n, fontsize=20)
        elif (n=="Nigeria")&(i==2):
            plt.text(-2.7, 0.5, n, fontsize=20)
        elif (n=="Nigeria")&(i==3):
            plt.text(-2.7, 0.4, n, fontsize=20)

plt.savefig("out/preds_worst_select_ols.eps",dpi=300,bbox_inches="tight")
plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_dissertation/out/preds_worst_select_ols.eps",dpi=300,bbox_inches='tight')
 
fig = plt.figure(figsize=(10, 13))
grid = GridSpec(4, 2, figure=fig, wspace=0.05, hspace=0.4)
for n,y,i,j in zip(["India","India", "Philippines","Philippines", "Somalia","Somalia","Somalia","Somalia"],[2022,2023,2022,2023,2018,2019,2020,2021],[0,0,1,1,2,2,3,3],[0,1,0,1,0,1,0,1]):
    print(n,y,i,j )
    ax = fig.add_subplot(grid[i, j])
    plt.plot(df_nonlinear["dd"].loc[(df_nonlinear["country"]==n)&(df_nonlinear["dd"]>=f"{y}-01")&(df_nonlinear["dd"]<=f"{y}-12")],df_nonlinear["fatalities"].loc[(df_nonlinear["country"]==n)&(df_nonlinear["dd"]>=f"{y}-01")&(df_nonlinear["dd"]<=f"{y}-12")],linestyle="solid",color="black",linewidth=1)
    plt.plot(df_nonlinear["dd"].loc[(df_nonlinear["country"]==n)&(df_nonlinear["dd"]>=f"{y}-01")&(df_nonlinear["dd"]<=f"{y}-12")],df_nonlinear["preds_rf"].loc[(df_nonlinear["country"]==n)&(df_nonlinear["dd"]>=f"{y}-01")&(df_nonlinear["dd"]<=f"{y}-12")],linestyle="dotted",color="black",linewidth=1)
    plt.plot(df_nonlinear["dd"].loc[(df_nonlinear["country"]==n)&(df_nonlinear["dd"]>=f"{y}-01")&(df_nonlinear["dd"]<=f"{y}-12")],df_nonlinear["preds_drfx"].loc[(df_nonlinear["country"]==n)&(df_nonlinear["dd"]>=f"{y}-01")&(df_nonlinear["dd"]<=f"{y}-12")],linestyle="dashed",color="black",linewidth=1)
   
    if y==2023:
        plt.xticks(["2023-01","2023-03","2023-05","2023-07","2023-09","2023-11"],["01-23","03-23","05-23","07-23","09-23","11-23"],fontsize=15)
    elif y==2022:
        plt.xticks(["2022-01","2022-03","2022-05","2022-07","2022-09","2022-11"],["01-22","03-22","05-22","07-22","09-22","11-22"],fontsize=15)
    elif y==2021:
        plt.xticks(["2021-01","2021-03","2021-05","2021-07","2021-09","2021-11"],["01-21","03-21","05-21","07-21","09-21","11-21"],fontsize=15)
    elif y==2020:
        plt.xticks(["2020-01","2020-03","2020-05","2020-07","2020-09","2020-11"],["01-20","03-20","05-20","07-20","09-20","11-20"],fontsize=15)
    elif y==2019:
        plt.xticks(["2019-01","2019-03","2019-05","2019-07","2019-09","2019-11"],["01-19","03-19","05-19","07-19","09-19","11-19"],fontsize=15)
    elif y==2018:
        plt.xticks(["2018-01","2018-03","2018-05","2018-07","2018-09","2018-11"],["01-18","03-18","05-18","07-18","09-18","11-18"],fontsize=15)
    elif y==2017:
        plt.xticks(["2017-01","2017-03","2017-05","2017-07","2017-09","2017-11"],["01-17","03-17","05-17","07-17","09-17","11-17"],fontsize=15)
    elif y==2016:
        plt.xticks(["2016-01","2016-03","2016-05","2016-07","2016-09","2016-11"],["01-16","03-16","05-16","07-16","09-16","11-16"],fontsize=15) 
 
    if i==0:
        plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6],[0,0.1,0.2,0.3,0.4,0.5,0.6],fontsize=15) 
    if i==1:
        plt.yticks([0,0.1,0.2,0.3,0.4,0.5],[0,0.1,0.2,0.3,0.4,0.5],fontsize=15) 
    if i==2:
        plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7],[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7],fontsize=15)       
    if i==3:
        plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7],[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7],fontsize=15)   

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if j==1:
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])
        if n=="India":
            plt.text(-2, 0.6, n, fontsize=20)
        elif n=="Philippines":
            plt.text(-3, 0.5, n, fontsize=20)
        elif n=="Somalia":
            plt.text(-2.9, 0.7, n, fontsize=20)

plt.savefig("out/preds_best_select_rf.eps",dpi=300,bbox_inches="tight")
plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_dissertation/out/preds_best_select_rf.eps",dpi=300,bbox_inches='tight')
       
fig = plt.figure(figsize=(10, 13))
grid = GridSpec(4, 2, figure=fig, wspace=0.05, hspace=0.4)
for n,y,i,j in zip(["Thailand","Thailand", "Niger","Niger", "Nigeria","Nigeria","Nigeria","Nigeria"],[2022,2023,2017,2018,2016,2017,2018,2019],[0,0,1,1,2,2,3,3],[0,1,0,1,0,1,0,1]):
    print(n,y,i,j )
    ax = fig.add_subplot(grid[i, j])
    plt.plot(df_nonlinear["dd"].loc[(df_nonlinear["country"]==n)&(df_nonlinear["dd"]>=f"{y}-01")&(df_nonlinear["dd"]<=f"{y}-12")],df_nonlinear["fatalities"].loc[(df_nonlinear["country"]==n)&(df_nonlinear["dd"]>=f"{y}-01")&(df_nonlinear["dd"]<=f"{y}-12")],linestyle="solid",color="black",linewidth=1)
    plt.plot(df_nonlinear["dd"].loc[(df_nonlinear["country"]==n)&(df_nonlinear["dd"]>=f"{y}-01")&(df_nonlinear["dd"]<=f"{y}-12")],df_nonlinear["preds_rf"].loc[(df_nonlinear["country"]==n)&(df_nonlinear["dd"]>=f"{y}-01")&(df_nonlinear["dd"]<=f"{y}-12")],linestyle="dotted",color="black",linewidth=1)
    plt.plot(df_nonlinear["dd"].loc[(df_nonlinear["country"]==n)&(df_nonlinear["dd"]>=f"{y}-01")&(df_nonlinear["dd"]<=f"{y}-12")],df_nonlinear["preds_drfx"].loc[(df_nonlinear["country"]==n)&(df_nonlinear["dd"]>=f"{y}-01")&(df_nonlinear["dd"]<=f"{y}-12")],linestyle="dashed",color="black",linewidth=1)
   
    if y==2023:
        plt.xticks(["2023-01","2023-03","2023-05","2023-07","2023-09","2023-11"],["01-23","03-23","05-23","07-23","09-23","11-23"],fontsize=15)
    elif y==2022:
        plt.xticks(["2022-01","2022-03","2022-05","2022-07","2022-09","2022-11"],["01-22","03-22","05-22","07-22","09-22","11-22"],fontsize=15)
    elif y==2021:
        plt.xticks(["2021-01","2021-03","2021-05","2021-07","2021-09","2021-11"],["01-21","03-21","05-21","07-21","09-21","11-21"],fontsize=15)
    elif y==2020:
        plt.xticks(["2020-01","2020-03","2020-05","2020-07","2020-09","2020-11"],["01-20","03-20","05-20","07-20","09-20","11-20"],fontsize=15)
    elif y==2017:
        plt.xticks(["2017-01","2017-03","2017-05","2017-07","2017-09","2017-11"],["01-17","03-17","05-17","07-17","09-17","11-17"],fontsize=15)
    elif y==2018:
        plt.xticks(["2018-01","2018-03","2018-05","2018-07","2018-09","2018-11"],["01-18","03-18","05-18","07-18","09-18","11-18"],fontsize=15)
    elif y==2016:
        plt.xticks(["2016-01","2016-03","2016-05","2016-07","2016-09","2016-11"],["01-16","03-16","05-16","07-16","09-16","11-16"],fontsize=15)
    elif y==2019:
        plt.xticks(["2019-01","2019-03","2019-05","2019-07","2019-09","2019-11"],["01-19","03-19","05-19","07-19","09-19","11-19"],fontsize=15)

    if i==0:
        plt.yticks([0,0.1,0.2,0.3,0.4,0.5],[0,0.1,0.2,0.3,0.4,0.5],fontsize=15) 
    if i==1:
        plt.yticks([0,0.1,0.2,0.3,0.4],[0,0.1,0.2,0.3,0.4],fontsize=15) 
    if i==2:
        plt.yticks([0,0.1,0.2,0.3,0.4,0.5],[0,0.1,0.2,0.3,0.4,0.5],fontsize=15)       
    if i==3:
        plt.yticks([0,0.1,0.2,0.3,0.4],[0,0.1,0.2,0.3,0.4],fontsize=15)       

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if j==1:
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])
        if n=="Thailand":
            plt.text(-2.9, 0.5, n, fontsize=20)
        elif n=="Niger":
            plt.text(-2.5, 0.4, n, fontsize=20)
        elif (n=="Nigeria")&(i==2):
            plt.text(-2.7, 0.5, n, fontsize=20)
        elif (n=="Nigeria")&(i==3):
            plt.text(-2.7, 0.4, n, fontsize=20)

plt.savefig("out/preds_worst_select_rf.eps",dpi=300,bbox_inches="tight")
plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_dissertation/out/preds_worst_select_rf.eps",dpi=300,bbox_inches='tight')
         
for n in countries: 
    s=df_linear.loc[(df_linear["country"]==n)]
    s["year"]=df_linear['dd'].loc[(df_linear["country"]==n)].str.extract('(\d{4})')
    grouped_counts = s.groupby('year').size().reset_index()
    years=list(grouped_counts["year"].loc[grouped_counts[0]==12])
    cols = 2 
    rows = len(years) // cols + (len(years) % cols > 0)  
    fig=plt.figure(figsize=(10, 3 * rows))
    for y,counts in zip(years,range(len(years))):
        plt.subplot(rows, cols, counts+1)
        plt.plot(df_linear["dd"].loc[(df_linear["country"]==n)&(df_linear["dd"]>=f"{y}-01")&(df_linear["dd"]<=f"{y}-12")],df_linear["fatalities"].loc[(df_linear["country"]==n)&(df_linear["dd"]>=f"{y}-01")&(df_linear["dd"]<=f"{y}-12")],linestyle="solid",color="black",linewidth=1)
        plt.plot(df_linear["dd"].loc[(df_linear["country"]==n)&(df_linear["dd"]>=f"{y}-01")&(df_linear["dd"]<=f"{y}-12")],df_linear["preds_ols"].loc[(df_linear["country"]==n)&(df_linear["dd"]>=f"{y}-01")&(df_linear["dd"]<=f"{y}-12")],linestyle="dotted",color="black",linewidth=1)
        plt.plot(df_linear["dd"].loc[(df_linear["country"]==n)&(df_linear["dd"]>=f"{y}-01")&(df_linear["dd"]<=f"{y}-12")],df_linear["preds_dolsx"].loc[(df_linear["country"]==n)&(df_linear["dd"]>=f"{y}-01")&(df_linear["dd"]<=f"{y}-12")],linestyle="dashed",color="black",linewidth=1)
        plt.yticks(fontsize=18)
        if y=="2023":
            plt.xticks(["2023-01","2023-03","2023-05","2023-07","2023-09","2023-11"],["01-23","03-23","05-23","07-23","09-23","11-23"],fontsize=15)
        elif y=="2022":
            plt.xticks(["2022-01","2022-03","2022-05","2022-07","2022-09","2022-11"],["01-22","03-22","05-22","07-22","09-22","11-22"],fontsize=15)
        elif y=="2021":
            plt.xticks(["2021-01","2021-03","2021-05","2021-07","2021-09","2021-11"],["01-21","03-21","05-21","07-21","09-21","11-21"],fontsize=15)
        elif y=="2020":
            plt.xticks(["2020-01","2020-03","2020-05","2020-07","2020-09","2020-11"],["01-20","03-20","05-20","07-20","09-20","11-20"],fontsize=15)
        elif y=="2019":
            plt.xticks(["2019-01","2019-03","2019-05","2019-07","2019-09","2019-11"],["01-19","03-19","05-19","07-19","09-19","11-19"],fontsize=15)
        elif y=="2018":
            plt.xticks(["2018-01","2018-03","2018-05","2018-07","2018-09","2018-11"],["01-18","03-18","05-18","07-18","09-18","11-18"],fontsize=15)
        elif y=="2017":
            plt.xticks(["2017-01","2017-03","2017-05","2017-07","2017-09","2017-11"],["01-17","03-17","05-17","07-17","09-17","11-17"],fontsize=15)
        elif y=="2016":
            plt.xticks(["2016-01","2016-03","2016-05","2016-07","2016-09","2016-11"],["01-16","03-16","05-16","07-16","09-16","11-16"],fontsize=15)
    plt.savefig(f"out/results_preds_ols_{n}.eps",dpi=300,bbox_inches="tight")
                                                                                                                                                       
for n in countries: 
    s=df_nonlinear.loc[(df_nonlinear["country"]==n)]
    s["year"]=df_nonlinear['dd'].loc[(df_nonlinear["country"]==n)].str.extract('(\d{4})')
    grouped_counts = s.groupby('year').size().reset_index()
    years=list(grouped_counts["year"].loc[grouped_counts[0]==12])
    cols = 2 
    rows = len(years) // cols + (len(years) % cols > 0)  
    fig=plt.figure(figsize=(10, 3 * rows))
    for y,counts in zip(years,range(len(years))):
        plt.subplot(rows, cols, counts+1)
        plt.plot(df_nonlinear["dd"].loc[(df_nonlinear["country"]==n)&(df_nonlinear["dd"]>=f"{y}-01")&(df_nonlinear["dd"]<=f"{y}-12")],df_nonlinear["fatalities"].loc[(df_nonlinear["country"]==n)&(df_nonlinear["dd"]>=f"{y}-01")&(df_nonlinear["dd"]<=f"{y}-12")],linestyle="solid",color="black",linewidth=1)
        plt.plot(df_nonlinear["dd"].loc[(df_nonlinear["country"]==n)&(df_nonlinear["dd"]>=f"{y}-01")&(df_nonlinear["dd"]<=f"{y}-12")],df_nonlinear["preds_rf"].loc[(df_nonlinear["country"]==n)&(df_nonlinear["dd"]>=f"{y}-01")&(df_nonlinear["dd"]<=f"{y}-12")],linestyle="dotted",color="black",linewidth=1)
        plt.plot(df_nonlinear["dd"].loc[(df_nonlinear["country"]==n)&(df_nonlinear["dd"]>=f"{y}-01")&(df_nonlinear["dd"]<=f"{y}-12")],df_nonlinear["preds_drfx"].loc[(df_nonlinear["country"]==n)&(df_nonlinear["dd"]>=f"{y}-01")&(df_nonlinear["dd"]<=f"{y}-12")],linestyle="dashed",color="black",linewidth=1)
        plt.yticks(fontsize=18)
        if y=="2023":
            plt.xticks(["2023-01","2023-03","2023-05","2023-07","2023-09","2023-11"],["01-23","03-23","05-23","07-23","09-23","11-23"],fontsize=15)
        elif y=="2022":
            plt.xticks(["2022-01","2022-03","2022-05","2022-07","2022-09","2022-11"],["01-22","03-22","05-22","07-22","09-22","11-22"],fontsize=15)
        elif y=="2021":
            plt.xticks(["2021-01","2021-03","2021-05","2021-07","2021-09","2021-11"],["01-21","03-21","05-21","07-21","09-21","11-21"],fontsize=15)
        elif y=="2020":
            plt.xticks(["2020-01","2020-03","2020-05","2020-07","2020-09","2020-11"],["01-20","03-20","05-20","07-20","09-20","11-20"],fontsize=15)
        elif y=="2019":
            plt.xticks(["2019-01","2019-03","2019-05","2019-07","2019-09","2019-11"],["01-19","03-19","05-19","07-19","09-19","11-19"],fontsize=15)
        elif y=="2018":
            plt.xticks(["2018-01","2018-03","2018-05","2018-07","2018-09","2018-11"],["01-18","03-18","05-18","07-18","09-18","11-18"],fontsize=15)
        elif y=="2017":
            plt.xticks(["2017-01","2017-03","2017-05","2017-07","2017-09","2017-11"],["01-17","03-17","05-17","07-17","09-17","11-17"],fontsize=15)
        elif y=="2016":
            plt.xticks(["2016-01","2016-03","2016-05","2016-07","2016-09","2016-11"],["01-16","03-16","05-16","07-16","09-16","11-16"],fontsize=15)
    plt.savefig(f"out/results_preds_rf_{n}.eps",dpi=300,bbox_inches="tight")
                                                                                                                            
# Dangerous shapes 
df_linear = pd.read_csv("data/df_linear.csv",index_col=0)
df_nonlinear =  pd.read_csv("data/df_nonlinear.csv",index_col=0)

with open("data/rf_shapes.json", 'r') as json_file:
    shapes_rf = json.load(json_file)
    
with open("data/ols_shapes.json", 'r') as json_file:
    shapes_ols = json.load(json_file)
  
countries=df_linear.country.unique()
results={"country":[],"change":[]}
dangerous={"country":[],"change":[],"shape":[],"n":[]}

for n in countries: 
    preds=df_nonlinear.loc[df_nonlinear["country"]==n]
    preds["clusters"]=shapes_rf[f"drfx_{n}"][2]
    means=preds.groupby('clusters')["fatalities"].mean()
    means=means.sort_values()
    results["country"].append(n)
    results["change"].append(means.max())
    dist=preds.groupby("clusters").size()
    cols = 4  
    rows = len(list(means.index)) // cols + (len(list(means.index)) % cols > 0)  
    fig=plt.figure(figsize=(10, 3 * rows))
    for i,counts in zip(list(means.index),range(len(list(means.index)))):
        seq=shapes_rf[f"drfx_{n}"][1][i]
        plt.subplot(rows, cols, counts+1)
        plt.plot(seq,color="black")
        plt.yticks([],[])
        plt.xticks([],[])
        plt.title(f'{round(means[i],5)}')
        dangerous["country"].append(n)
        dangerous["change"].append(means[i])
        dangerous["shape"].append(sum(seq, []))
        dangerous["n"].append(dist[i])
results=pd.DataFrame(results)
results=results.sort_values(by=["change"])

# Top dangerous shapes.
dangerous=pd.DataFrame(dangerous)
dangerous=dangerous.sort_values(by=["change"])

fig, axs = plt.subplots(5, 4, figsize=(16, 11))
for c,i,j in zip(range(1,31),[0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4],[0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]):
    axs[i, j].plot(dangerous["shape"].iloc[-c],color="black")
    axs[i, j].set_yticks([],[])
    axs[i, j].set_xticks([],[])
    axs[i, j].set_axis_off()
    if dangerous['country'].iloc[-c] =='Democratic Republic of Congo':
        axs[i, j].set_title("DRC",size=29)
    elif dangerous['country'].iloc[-c] =='Central African Republic':
        axs[i, j].set_title("CAR",size=29)
    else:
        axs[i, j].set_title(f"{dangerous['country'].iloc[-c]}",size=29)
plt.subplots_adjust(wspace=0.01)
plt.savefig(f"out/results_dang_shapes_rf_top.eps",dpi=300,bbox_inches="tight")
plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_dissertation/out/results_dang_shapes_rf_top.eps",dpi=300,bbox_inches='tight')

# Top harmless shapes.
dangerous=pd.DataFrame(dangerous)
dangerous=dangerous.sort_values(by=["change"])
dangerous=dangerous.loc[dangerous["change"]==0]

for i in range(len(dangerous)):
    plt.figure(figsize=(12, 8))
    plt.plot(dangerous["shape"].iloc[i],color="black")
    plt.title(f"{dangerous['country'].iloc[i]}, {round(dangerous['change'].iloc[i],4)}",size=23)
    plt.show()
    
# Random sample
dangerous = dangerous.sample(n=31, random_state=30) 
fig, axs = plt.subplots(5, 4, figsize=(16, 11))
for c,i,j in zip(range(1,31),[0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4],[0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]):
    axs[i, j].plot(dangerous["shape"].iloc[c],color="black")
    axs[i, j].set_yticks([],[])
    axs[i, j].set_xticks([],[])
    axs[i, j].set_axis_off()
    if dangerous['country'].iloc[c] =='Democratic Republic of Congo':
        axs[i, j].set_title("DRC",size=29)
    elif dangerous['country'].iloc[c] =='Central African Republic':
        axs[i, j].set_title("CAR",size=29)
    else:
        axs[i, j].set_title(f"{dangerous['country'].iloc[c]}",size=29)
plt.subplots_adjust(wspace=0.01)
plt.savefig(f"out/results_dang_shapes_rf_bottom.eps",dpi=300,bbox_inches="tight")
plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_dissertation/out/results_dang_shapes_rf_bottom.eps",dpi=300,bbox_inches='tight')

results={"country":[],"change":[]}
dangerous={"country":[],"change":[],"shape":[],"n":[]}

for n in countries: 
    preds=df_linear.loc[df_linear["country"]==n]
    preds["clusters"]=shapes_ols[f"dolsx_{n}"][2]
    means=preds.groupby('clusters')["fatalities"].mean()
    means=means.sort_values()
    results["country"].append(n)
    results["change"].append(means.max())
    dist=preds.groupby("clusters").size()
    cols = 4  
    rows = len(list(means.index)) // cols + (len(list(means.index)) % cols > 0)  
    fig=plt.figure(figsize=(10, 3 * rows))
    for i,counts in zip(list(means.index),range(len(list(means.index)))):
        seq=shapes_ols[f"dolsx_{n}"][1][i]
        plt.subplot(rows, cols, counts+1)
        plt.plot(seq,color="black")
        plt.yticks([],[])
        plt.xticks([],[])
        plt.title(f'{round(means[i],5)}')
        dangerous["country"].append(n)
        dangerous["change"].append(means[i])
        dangerous["shape"].append(sum(seq, []))
        dangerous["n"].append(dist[i])

results=pd.DataFrame(results)
results=results.sort_values(by=["change"])

# Top dangerous shapes.
dangerous=pd.DataFrame(dangerous)
dangerous=dangerous.sort_values(by=["change"])

fig, axs = plt.subplots(5, 4, figsize=(16, 11))
for c,i,j in zip(range(1,31),[0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4],[0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]):
    axs[i, j].plot(dangerous["shape"].iloc[-c],color="black")
    axs[i, j].set_yticks([],[])
    axs[i, j].set_xticks([],[])
    axs[i, j].set_axis_off()
    if dangerous['country'].iloc[-c] =='Democratic Republic of Congo':
        axs[i, j].set_title("DRC",size=29)
    elif dangerous['country'].iloc[-c] =='Central African Republic':
        axs[i, j].set_title("CAR",size=29)
    else:
        axs[i, j].set_title(f"{dangerous['country'].iloc[-c]}",size=29)
plt.subplots_adjust(wspace=0.01)
plt.savefig(f"out/results_dang_shapes_ols_top.eps",dpi=300,bbox_inches="tight")
plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_dissertation/out/results_dang_shapes_ols_top.eps",dpi=300,bbox_inches='tight')

# Top harmless shapes.
dangerous=pd.DataFrame(dangerous)
dangerous=dangerous.sort_values(by=["change"])
dangerous=dangerous.loc[dangerous["change"]==0]

for i in range(len(dangerous)):
    plt.figure(figsize=(12, 8))
    plt.plot(dangerous["shape"].iloc[i],color="black")
    plt.title(f"{dangerous['country'].iloc[i]}, {round(dangerous['change'].iloc[i],4)}",size=23)
    plt.show()
    
# Random sample
dangerous = dangerous.sample(n=31, random_state=30) 
fig, axs = plt.subplots(5, 4, figsize=(16, 11))
for c,i,j in zip(range(1,31),[0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4],[0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]):
    axs[i, j].plot(dangerous["shape"].iloc[c],color="black")
    axs[i, j].set_yticks([],[])
    axs[i, j].set_xticks([],[])
    axs[i, j].set_axis_off()
    if dangerous['country'].iloc[c] =='Democratic Republic of Congo':
        axs[i, j].set_title("DRC",size=29)
    elif dangerous['country'].iloc[c] =='Central African Republic':
        axs[i, j].set_title("CAR",size=29)
    else:
        axs[i, j].set_title(f"{dangerous['country'].iloc[c]}",size=29)
plt.subplots_adjust(wspace=0.01)
plt.savefig(f"out/results_dang_shapes_ols_bottom.eps",dpi=300,bbox_inches="tight")
plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_dissertation/out/results_dang_shapes_ols_bottom.eps",dpi=300,bbox_inches='tight')


# Co-variation 
preprocess_min_max_group(df,"n_protest_events","country")
preprocess_min_max_group(df,"fatalities","country")
      
df_s = df.loc[(df["country"]=="Egypt")]
fig=plt.figure(figsize=(12,3))
years=[2013,2014,2015]
for y,i in zip(years,range(len(years))):
    ax1=plt.subplot(rows, cols, i+1)
    plt.plot(df_s["dd"].loc[df_s["year"]==y],df_s["n_protest_events_norm"].loc[df_s["year"]==y],linestyle="solid",color="black",linewidth=2)
    ax1.set_ylim(-0.02, 1.02)
    ax2 = ax1.twinx()
    ax2.plot(df_s["dd"].loc[df_s["year"]==y],df_s["fatalities_norm"].loc[df_s["year"]==y],linestyle="solid",color="gray",linewidth=2)
    ax2.set_ylim(-0.02, 1.02)
    plt.title(f'{y}',size=25)
    plt.xticks([],[])
    ax2.set_yticks([])
    ax1.set_yticks([])
plt.tight_layout()
fig.savefig(f"out/covar_Egypt.eps",dpi=300,bbox_inches="tight")
plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_dissertation/out/covar_Egypt.eps",dpi=300,bbox_inches='tight')
plt.show()

df_s = df.loc[(df["country"]=="Myanmar")]
fig=plt.figure(figsize=(12,3))
years=[2021,2022,2023]
for y,i in zip(years,range(len(years))):
    ax1=plt.subplot(rows, cols, i+1)
    plt.plot(df_s["dd"].loc[df_s["year"]==y],df_s["n_protest_events_norm"].loc[df_s["year"]==y],linestyle="solid",color="black",linewidth=2)
    ax1.set_ylim(-0.02, 1.02)
    ax2 = ax1.twinx()
    ax2.plot(df_s["dd"].loc[df_s["year"]==y],df_s["fatalities_norm"].loc[df_s["year"]==y],linestyle="solid",color="gray",linewidth=2)
    ax2.set_ylim(-0.02, 1.02)
    plt.title(f'{y}',size=25)
    plt.xticks([],[])
    ax2.set_yticks([])
    ax1.set_yticks([])
plt.tight_layout()
fig.savefig(f"out/covar_Myanmar.eps",dpi=300,bbox_inches="tight")
plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_dissertation/out/covar_Myanmar.eps",dpi=300,bbox_inches='tight')
plt.show()


# DTW 
t1 = [0.6,0.75,0.8,0.91,0.82,0.6,0.52,0.55,0.62,0.67,0.69,0.8,]
t2 = [0.5,0.52,0.6,0.55,0.4,0.34,0.29,0.2,0.35,0.5,0.52,0.67]
y=df.loc[(df["country"]=="Egypt")].n_protest_events
Egypt=df[["dd","n_protest_events"]].loc[(df["country"]=="Egypt")].reset_index(drop=True)
matrix=[]
for i in range(12,len(y)):
    matrix.append(y.iloc[i-12:i])  
matrix=np.array(matrix)
matrix_l= pd.DataFrame(matrix).T
matrix_l=(matrix_l-matrix_l.min())/(matrix_l.max()-matrix_l.min())
matrix_l=matrix_l.fillna(0) 
matrix_l=np.array(matrix_l.T)

t1=matrix_l[100] #  05-2005 - 04-2006
t1=t1+1.2
t2=matrix_l[210] # 07-2014 - 06-2015

distance, paths = dtw.warping_paths(t1, t2)
best_path = dtw.best_path(paths) 
path_x, path_y = zip(*best_path)  
fig, ax = plt.subplots(figsize=(7, 6))
plt.plot(t1, label="X", color='black',linewidth=2)
plt.plot(t2, label="Y", color='black',linewidth=2)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylim(-0.2,2.5)
ax.text(0.4, 1.92, "Egypt (05-2005---04-2006)", fontsize=23, color="black")
ax.text(2.1, -0.12, "Egypt (07-2014---06-2015)", fontsize=23, color="black")

for i, j in zip(path_x, path_y):
    plt.plot([i, j], [t1[i], t2[j]], color="gray", alpha=0.5,linewidth=1)
ax.set_yticks([])  
plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11],[1,2,3,4,5,6,7,8,9,10,11,12],size=22)
plt.tight_layout()
plt.savefig("out/dtw1.eps",dpi=300,bbox_inches="tight")
plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_dissertation/out/dtw1.eps",dpi=300,bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(10, 10))
paths=paths[1:,1:]
im = ax.imshow(paths, cmap='Greys',origin="lower")
plt.plot(path_y, path_x, 'black', linewidth=2, label="Warping Path")  
for i in range(paths.shape[0]):     
    for j in range(paths.shape[1]):  
        ax.text(j, i,                
                str(round(paths[i, j],2)),
                ha='center', va='center', color='black', fontsize=20)
plt.ylabel("Egypt (05-2005---04-2006)",size=35)
plt.xlabel("Egypt (07-2014---06-2015)",size=35)
plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11],[1,2,3,4,5,6,7,8,9,10,11,12],size=35)
plt.yticks([0,1,2,3,4,5,6,7,8,9,10,11],[1,2,3,4,5,6,7,8,9,10,11,12],size=35)
plt.tight_layout()
plt.savefig("out/dtw2.eps",dpi=300,bbox_inches="tight")
plt.show()




