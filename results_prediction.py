import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import json
import matplotlib as mpl
import random
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
from scipy.stats import ttest_rel
random.seed(2)
np.random.seed(42)
import os 
os.environ['PATH'] = "/Library/TeX/texbin:" + os.environ.get('PATH', '')
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['text.latex.preamble'] = r'\usepackage{lmodern}\usepackage[T1]{fontenc}'

# Mircostates
# http://ksgleditsch.com/data-4.html
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

# Load data 
df = pd.read_csv("data/df.csv",index_col=0)
df = df[~df['gw_codes'].isin(list(micro_states.values()))]
df = df.reset_index(drop=True)

####################
### Main results ###
####################

# Load predictions
df_linear = pd.read_csv("data/df_linear.csv",index_col=0)
df_nonlinear =  pd.read_csv("data/df_nonlinear.csv",index_col=0)

print("Linear models")
print(mean_squared_error(df_linear.fatalities, df_linear.preds_ols))
print(mean_squared_error(df_linear.fatalities, df_linear.preds_olsx))
print(mean_squared_error(df_linear.fatalities, df_linear.preds_dols))
print(mean_squared_error(df_linear.fatalities, df_linear.preds_dolsx))

# Get the error for each observation, ols
df_linear["mse_ols"]=((df_linear["fatalities"] - df_linear["preds_ols"]) ** 2) 
df_linear["mse_olsx"]=((df_linear["fatalities"] - df_linear["preds_olsx"]) ** 2)
df_linear["mse_dols"]=((df_linear["fatalities"] - df_linear["preds_dols"]) ** 2) 
df_linear["mse_dolsx"]=((df_linear["fatalities"] - df_linear["preds_dolsx"]) ** 2) 

print(round(df_linear["mse_ols"].mean(),5))
print(round(df_linear["mse_olsx"].mean(),5))
print(round(df_linear["mse_dols"].mean(),5))
print(round(df_linear["mse_dolsx"].mean(),5))

print(round(df_linear["mse_ols"].std(),3))
print(round(df_linear["mse_olsx"].std(),3))
print(round(df_linear["mse_dols"].std(),3))
print(round(df_linear["mse_dolsx"].std(),3))

# t-test for paired samples on whether the means are equal 
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html
print(round(ttest_rel(df_linear["mse_ols"], df_linear["mse_olsx"])[1],5))
print(round(ttest_rel(df_linear["mse_ols"], df_linear["mse_dols"])[1],5))
print(round(ttest_rel(df_linear["mse_olsx"], df_linear["mse_dolsx"])[1],5))

print("Non-linear models")
print(mean_squared_error(df_nonlinear.fatalities, df_nonlinear.preds_rf))
print(mean_squared_error(df_nonlinear.fatalities, df_nonlinear.preds_rfx))
print(mean_squared_error(df_nonlinear.fatalities, df_nonlinear.preds_drf))
print(mean_squared_error(df_nonlinear.fatalities, df_nonlinear.preds_drfx))

# Get the error for each observation, rf
df_nonlinear["mse_rf"]=((df_nonlinear["fatalities"] - df_nonlinear["preds_rf"]) ** 2) 
df_nonlinear["mse_rfx"]=((df_nonlinear["fatalities"] - df_nonlinear["preds_rfx"]) ** 2)
df_nonlinear["mse_drf"]=((df_nonlinear["fatalities"] - df_nonlinear["preds_drf"]) ** 2) 
df_nonlinear["mse_drfx"]=((df_nonlinear["fatalities"] - df_nonlinear["preds_drfx"]) ** 2) 

print(round(df_nonlinear["mse_rf"].mean(),5))
print(round(df_nonlinear["mse_rfx"].mean(),5))
print(round(df_nonlinear["mse_drf"].mean(),5))
print(round(df_nonlinear["mse_drfx"].mean(),5))

print(round(df_nonlinear["mse_rf"].std(),3))
print(round(df_nonlinear["mse_rfx"].std(),3))
print(round(df_nonlinear["mse_drf"].std(),3))
print(round(df_nonlinear["mse_drfx"].std(),3))

# t-test for paired samples on whether the means are equal 
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html
print(round(ttest_rel(df_nonlinear["mse_rf"], df_nonlinear["mse_rfx"])[1],5))
print(round(ttest_rel(df_nonlinear["mse_rf"], df_nonlinear["mse_drf"])[1],5))
print(round(ttest_rel(df_nonlinear["mse_rfx"], df_nonlinear["mse_drfx"])[1],5))

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 7))

# OLS in plot 1
yerrs=[1.65 *(df_linear["mse_ols"].std()/np.sqrt(len(df_linear))),1.65 *(df_linear["mse_dols"].std()/np.sqrt(len(df_linear))),1.65 *(df_linear["mse_olsx"].std()/np.sqrt(len(df_linear))),1.65 *(df_linear["mse_dolsx"].std()/np.sqrt(len(df_linear)))]
ax1.scatter([0,1,2,3], [df_linear["mse_ols"].mean(),df_linear["mse_dols"].mean(),df_linear["mse_olsx"].mean(),df_linear["mse_dolsx"].mean()],color="black",marker='o',s=50)
ax1.errorbar([0,1,2,3], [df_linear["mse_ols"].mean(),df_linear["mse_dols"].mean(),df_linear["mse_olsx"].mean(),df_linear["mse_dolsx"].mean()], yerr=yerrs, color="black", linewidth=1, fmt='none')
ax1.grid(False)
ax1.set_ylim(0.0155,0.0255)
ax1.set_yticks([0.016,0.017,0.018,0.019,0.02,0.021,0.022,0.023,0.024,0.025],[0.016,0.017,0.018,0.019,0.02,0.021,0.022,0.023,0.024,0.025],fontsize=18)
ax1.set_ylabel("Mean squared error (MSE)",size=22)
ax1.set_xticks([0,1,2,3],['RR','DRR','RRX','DRRX'],fontsize=18)

# RF in plot 2
yerrs=[1.65 *(df_nonlinear["mse_rf"].std()/np.sqrt(len(df_nonlinear))),1.65 *(df_nonlinear["mse_drf"].std()/np.sqrt(len(df_nonlinear))),1.65 *(df_nonlinear["mse_rfx"].std()/np.sqrt(len(df_nonlinear))),1.65 *(df_nonlinear["mse_drfx"].std()/np.sqrt(len(df_nonlinear)))]
ax2.scatter([0,1,2,3], [df_nonlinear["mse_rf"].mean(),df_nonlinear["mse_drf"].mean(),df_nonlinear["mse_rfx"].mean(),df_nonlinear["mse_drfx"].mean()], color="black", marker='o',s=50)
ax2.errorbar([0,1,2,3], [df_nonlinear["mse_rf"].mean(),df_nonlinear["mse_drf"].mean(),df_nonlinear["mse_rfx"].mean(),df_nonlinear["mse_drfx"].mean()], yerr=yerrs, color="black", linewidth=1, fmt='none')
ax2.grid(False)
ax2.set_ylim(0.0125,0.0225)
ax2.set_yticks([0.013,0.014,0.015,0.016,0.017,0.018,0.019,0.02,0.021,0.022],[0.013,0.014,0.015,0.016,0.017,0.018,0.019,0.02,0.021,0.022],size=18)
ax2.yaxis.set_ticks_position('right')
ax2.set_xticks([0,1,2,3],['RF','DRF','RFX','DRFX'],fontsize=18)
plt.subplots_adjust(wspace=0.05)

# Manually add results for the t-test
ax1.plot([0,2],[0.0246,0.0246],linewidth=0.5,color="black")
ax1.plot([0,0],[0.0246,0.0244],linewidth=0.5,color="black")
ax1.plot([2,2],[0.0246,0.0244],linewidth=0.5,color="black")
ax1.text(0.92, 0.0247, "o", fontsize=12)

ax1.plot([0,1],[0.0169,0.0169],linewidth=0.5,color="black")
ax1.plot([0,0],[0.0169,0.0171],linewidth=0.5,color="black")
ax1.plot([1,1],[0.0169,0.0171],linewidth=0.5,color="black")
ax1.text(0.42, 0.0166, "***", fontsize=12)

ax1.plot([2,3],[0.0166,0.0166],linewidth=0.5,color="black")
ax1.plot([2,2],[0.0166,0.0168],linewidth=0.5,color="black")
ax1.plot([3,3],[0.0166,0.0168],linewidth=0.5,color="black")
ax1.text(2.42, 0.0163, "***", fontsize=12)

ax2.plot([0,2],[0.0218,0.0218],linewidth=0.5,color="black")
ax2.plot([0,0],[0.0218,0.0216],linewidth=0.5,color="black")
ax2.plot([2,2],[0.0218,0.0216],linewidth=0.5,color="black")
ax2.text(0.92,0.0218, "***", fontsize=12)

ax2.plot([0,1],[0.0158,0.0158],linewidth=0.5,color="black")
ax2.plot([0,0],[0.0158,0.016],linewidth=0.5,color="black")
ax2.plot([1,1],[0.0158,0.016],linewidth=0.5,color="black")
ax2.text(0.42,0.0155, "***", fontsize=12)

ax2.plot([2,3],[0.014,0.014],linewidth=0.5,color="black")
ax2.plot([2,2],[0.014,0.0142],linewidth=0.5,color="black")
ax2.plot([3,3],[0.014,0.0142],linewidth=0.5,color="black")
ax2.text(2.42,0.0137, "***", fontsize=12)

# Save
plt.savefig("out/results_main_plot.eps",dpi=300,bbox_inches="tight")
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_dissertation/out/results_main_plot.eps",dpi=300,bbox_inches='tight')
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/protest_armed_conflict_diss/out/results_main_plot.eps",dpi=300,bbox_inches='tight')
plt.show()

##########################################
### Plot dangerous and harmless shapes ###                                                                                                                           
##########################################

# Load predictions
df_linear = pd.read_csv("data/df_linear.csv",index_col=0)
df_nonlinear =  pd.read_csv("data/df_nonlinear.csv",index_col=0)

# Load shapes
with open("data/rf_shapes.json", 'r') as json_file:
    shapes_rf = json.load(json_file)
    
with open("data/ols_shapes.json", 'r') as json_file:
    shapes_ols = json.load(json_file)
  
# Get unique countries
countries=df_linear.country.unique()

### OLS ###
results={"country":[],"fatalities":[],"shape":[],"n":[]}

# For each country
for n in countries: 
    # Get subset
    preds=df_linear.loc[df_linear["country"]==n]
    # Add cluster assignments
    preds["clusters"]=shapes_ols[f"dolsx_{n}"][2]
    # For each cluster calculate the mean fatalities and the number of observations assigned
    means=preds.groupby('clusters')["fatalities"].mean()
    dist=preds.groupby("clusters").size()
    
    # Loop over clusters and append country, fatalities, the centroid, and
    # the number of cases assigned to the pattern
    for i in list(means.index):
        seq=shapes_ols[f"dolsx_{n}"][1][i]
        results["country"].append(n)
        results["fatalities"].append(means[i])
        # Convert centroid into list
        results["shape"].append(sum(seq, []))
        results["n"].append(dist[i])

# Get df
results=pd.DataFrame(results)

# Plot top dangerous shapes
dangerous=results.sort_values(by=["fatalities"])

# Initiate plot
fig, axs = plt.subplots(5, 4, figsize=(16, 11))

# Fill each subplot with a shape, starting from the last country in dangerous
for c,i,j in zip(range(1,21),[0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4],[0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]):
    axs[i, j].plot(dangerous["shape"].iloc[-c],color="black")
    axs[i, j].set_yticks([],[])
    axs[i, j].set_xticks([],[])
    axs[i, j].set_axis_off()
    
    # Manually fix country names if too long and set country as title
    if dangerous['country'].iloc[-c] =='Democratic Republic of Congo':
        axs[i, j].set_title("DRC",size=29)
    elif dangerous['country'].iloc[-c] =='Central African Republic':
        axs[i, j].set_title("CAR",size=29)
    else:
        axs[i, j].set_title(f"{dangerous['country'].iloc[-c]}",size=29)
plt.subplots_adjust(wspace=0.01)

# Save
plt.savefig("out/results_dang_shapes_ols_top.eps",dpi=300,bbox_inches="tight")
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_dissertation/out/results_dang_shapes_ols_top.eps",dpi=300,bbox_inches='tight')
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/protest_armed_conflict_diss/out/results_dang_shapes_ols_top.eps",dpi=300,bbox_inches='tight')
plt.show()

# Plot a random sample of harmless shapes, which have zero fatalities
harmless=results.loc[results["fatalities"]==0]
harmless = harmless.sample(n=20, random_state=30) 

# Fill in each subplot with a shape in harmless
fig, axs = plt.subplots(5, 4, figsize=(16, 11))
for c,i,j in zip(range(0,20),[0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4],[0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]):
    axs[i, j].plot(harmless["shape"].iloc[c],color="black")
    axs[i, j].set_yticks([],[])
    axs[i, j].set_xticks([],[])
    axs[i, j].set_axis_off()
    
    # Manually fix country names if too long and set country as title
    if harmless['country'].iloc[c] =='Democratic Republic of Congo':
        axs[i, j].set_title("DRC",size=29)
    elif harmless['country'].iloc[c] =='Central African Republic':
        axs[i, j].set_title("CAR",size=29)
    else:
        axs[i, j].set_title(f"{harmless['country'].iloc[c]}",size=29)
plt.subplots_adjust(wspace=0.01)

# Save
plt.savefig("out/results_dang_shapes_ols_bottom.eps",dpi=300,bbox_inches="tight")
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_dissertation/out/results_dang_shapes_ols_bottom.eps",dpi=300,bbox_inches='tight')
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/protest_armed_conflict_diss/out/results_dang_shapes_ols_bottom.eps",dpi=300,bbox_inches='tight')
plt.show()

### RF ###
results={"country":[],"fatalities":[],"shape":[],"n":[]}

# For each country
for n in countries: 
    # Get subset
    preds=df_nonlinear.loc[df_nonlinear["country"]==n]
    # Add cluster assignments
    preds["clusters"]=shapes_rf[f"drfx_{n}"][2]
    # For each cluster calculate the mean fatalities and the number of observations assigned
    means=preds.groupby('clusters')["fatalities"].mean()
    dist=preds.groupby("clusters").size()
    
    # Loop over clusters and append country, fatalities, the centroid, and
    # the number of cases assigned to the pattern
    for i in list(means.index):
        seq=shapes_rf[f"drfx_{n}"][1][i]
        results["country"].append(n)
        results["fatalities"].append(means[i])
        # Convert centroid into list
        results["shape"].append(sum(seq, []))
        results["n"].append(dist[i])

# Get df
results=pd.DataFrame(results)

# Plot top dangerous shapes
dangerous=results.sort_values(by=["fatalities"])

# Initiate plot
fig, axs = plt.subplots(5, 4, figsize=(16, 11))

# Fill each subplot with a shape, starting from the last country in dangerous
for c,i,j in zip(range(1,21),[0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4],[0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]):
    axs[i, j].plot(dangerous["shape"].iloc[-c],color="black")
    axs[i, j].set_yticks([],[])
    axs[i, j].set_xticks([],[])
    axs[i, j].set_axis_off()
    
    # Manually fix country names if too long and set country as title
    if dangerous['country'].iloc[-c] =='Democratic Republic of Congo':
        axs[i, j].set_title("DRC",size=29)
    elif dangerous['country'].iloc[-c] =='Central African Republic':
        axs[i, j].set_title("CAR",size=29)
    else:
        axs[i, j].set_title(f"{dangerous['country'].iloc[-c]}",size=29)
plt.subplots_adjust(wspace=0.01)

# Save
plt.savefig("out/results_dang_shapes_rf_top.eps",dpi=300,bbox_inches="tight")
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_dissertation/out/results_dang_shapes_rf_top.eps",dpi=300,bbox_inches='tight')
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/protest_armed_conflict_diss/out/results_dang_shapes_rf_top.eps",dpi=300,bbox_inches='tight')
plt.show()

# Plot a random sample of harmless shapes, which have zero fatalities
harmless=results.loc[results["fatalities"]==0]
harmless = harmless.sample(n=20, random_state=30) 

# Fill in each subplot with a shape in harmless
fig, axs = plt.subplots(5, 4, figsize=(16, 11))
for c,i,j in zip(range(0,20),[0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4],[0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]):
    axs[i, j].plot(harmless["shape"].iloc[c],color="black")
    axs[i, j].set_yticks([],[])
    axs[i, j].set_xticks([],[])
    axs[i, j].set_axis_off()
    
    # Manually fix country names if too long and set country as title
    if harmless['country'].iloc[c] =='Democratic Republic of Congo':
        axs[i, j].set_title("DRC",size=29)
    elif harmless['country'].iloc[c] =='Central African Republic':
        axs[i, j].set_title("CAR",size=29)
    else:
        axs[i, j].set_title(f"{harmless['country'].iloc[c]}",size=29)
plt.subplots_adjust(wspace=0.01)

# Save
plt.savefig("out/results_dang_shapes_rf_bottom.eps",dpi=300,bbox_inches="tight")
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_dissertation/out/results_dang_shapes_rf_bottom.eps",dpi=300,bbox_inches='tight')
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/protest_armed_conflict_diss/out/results_dang_shapes_rf_bottom.eps",dpi=300,bbox_inches='tight')
plt.show()

#########################################################
###  Plot co-variation between protest and fatalities ###
#########################################################

# Min-max normalize protest and fatalitiy time series
def preprocess_min_max_group(df, x, group):
    out = pd.DataFrame()
    for i in df[group].unique():
        Y = df[x].loc[df[group] == i]
        mini = np.min(Y)
        maxi = np.max(Y)
        Y = (Y - mini) / (maxi - mini)
        Y=Y.fillna(0) 
        out = pd.concat([out, pd.DataFrame(Y)], ignore_index=True)
    df[f"{x}_norm"] = out

preprocess_min_max_group(df,"n_protest_events","country")
preprocess_min_max_group(df,"fatalities","country")
      
# Make plot for Egypt
df_s = df.loc[(df["country"]=="Egypt")]

# Initiate figure
fig=plt.figure(figsize=(12,3))
years=[2013,2014,2015]

# For each year
for y,i in zip(years,range(len(years))):
    
    # Plot protest events
    ax1=plt.subplot(1, 3, i+1)
    plt.plot(df_s["dd"].loc[df_s["year"]==y],df_s["n_protest_events_norm"].loc[df_s["year"]==y],linestyle="solid",color="black",linewidth=2)
    ax1.set_ylim(-0.02, 1.02)
    
    # Plot fatalities on second axis
    ax2 = ax1.twinx()
    ax2.plot(df_s["dd"].loc[df_s["year"]==y],df_s["fatalities_norm"].loc[df_s["year"]==y],linestyle="solid",color="gray",linewidth=2)
    ax2.set_ylim(-0.02, 1.02)
    
    # Add country as titel and remove ticks
    plt.title(f'{y}',size=25)
    plt.xticks([],[])
    ax2.set_yticks([])
    ax1.set_yticks([])
plt.tight_layout()

# Save
fig.savefig("out/covar_Egypt.eps",dpi=300,bbox_inches="tight")
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_dissertation/out/covar_Egypt.eps",dpi=300,bbox_inches='tight')
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/protest_armed_conflict_diss/out/covar_Egypt.eps",dpi=300,bbox_inches='tight')
plt.show()

# Make plot for Myanmar
df_s = df.loc[(df["country"]=="Myanmar")]

# Initiate figure
fig=plt.figure(figsize=(12,3))
years=[2021,2022,2023]

# For each year
for y,i in zip(years,range(len(years))):
    
    # Plot protest events
    ax1=plt.subplot(1, 3, i+1)
    plt.plot(df_s["dd"].loc[df_s["year"]==y],df_s["n_protest_events_norm"].loc[df_s["year"]==y],linestyle="solid",color="black",linewidth=2)
    ax1.set_ylim(-0.02, 1.02)
    
    # Plot fatalities on second axis
    ax2 = ax1.twinx()
    ax2.plot(df_s["dd"].loc[df_s["year"]==y],df_s["fatalities_norm"].loc[df_s["year"]==y],linestyle="solid",color="gray",linewidth=2)
    ax2.set_ylim(-0.02, 1.02)
    
    # Add country as titel and remove ticks
    plt.title(f'{y}',size=25)
    plt.xticks([],[])
    ax2.set_yticks([])
    ax1.set_yticks([])
plt.tight_layout()

# Save
fig.savefig("out/covar_Myanmar.eps",dpi=300,bbox_inches="tight")
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_dissertation/out/covar_Myanmar.eps",dpi=300,bbox_inches='tight')
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/protest_armed_conflict_diss/out/covar_Myanmar.eps",dpi=300,bbox_inches='tight')
plt.show()

############################
### Dynamic Time Warping ###
############################

# Use Egypt as example
y=df.loc[(df["country"]=="Egypt")].n_protest_events
Egypt=df.loc[(df["country"]=="Egypt")].reset_index(drop=True)

# Get min-max normalized subsequences with length 12
# Adapted from: https://github.com/ThomasSchinca/ShapeF/blob/Thomas_draft/functions.py

matrix=[]
for i in range(12,len(y)):
    matrix.append(y.iloc[i-12:i])  
matrix=np.array(matrix)
matrix_l= pd.DataFrame(matrix).T
matrix_l=(matrix_l-matrix_l.min())/(matrix_l.max()-matrix_l.min())
matrix_l=matrix_l.fillna(0) 
matrix_l=np.array(matrix_l.T)

# Select two subseqeucnes, and find actual months by checking "Egypt"
t1=matrix_l[100] # 05-2005 - 04-2006
t1=t1+1.2
t2=matrix_l[210] # 07-2014 - 06-2015

# Plot the two time series
fig, ax = plt.subplots(figsize=(7, 6))
plt.plot(t1, label="X", color='black',linewidth=2)
plt.plot(t2, label="Y", color='black',linewidth=2)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylim(-0.2,2.5)
ax.text(0.4, 1.92, "Egypt (05-2005---04-2006)", fontsize=23, color="black")
ax.text(2.1, -0.12, "Egypt (07-2014---06-2015)", fontsize=23, color="black")

# Run dtw on the two time series, and obtain warping path
dist, cost_matrix = dtw.warping_paths(t1, t2)
dtw_path = dtw.best_path(cost_matrix) 
print(dist)

# Save x and y indices of the warping path
index_x, index_y = zip(*dtw_path)  

# Plot the assignments of observations in dtw
for x, y in zip(index_x, index_y):
    plt.plot([x, y], [t1[x], t2[y]], color="gray", alpha=0.5, linewidth=1)

# Add ticks
ax.set_yticks([])  
plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11],[1,2,3,4,5,6,7,8,9,10,11,12],size=22)
plt.tight_layout()

# Save
plt.savefig("out/dtw1.eps",dpi=300,bbox_inches="tight")
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_dissertation/out/dtw1.eps",dpi=300,bbox_inches='tight')
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/protest_armed_conflict_diss/out/dtw1.eps",dpi=300,bbox_inches='tight')
plt.show()

# Plot cost matrix
fig, ax = plt.subplots(figsize=(10, 10))

# Get cost matrix and remove first row and first column which are inf 
dist, cost_matrix = dtw.warping_paths(t1, t2)
cost_matrix=cost_matrix[1:,1:]

# Cost matrix
im = ax.imshow(cost_matrix,cmap='Greys',origin="lower")

# And add warping path
plt.plot(index_x, index_y, 'black', linewidth=2)  

# Fill each cell in cost matrix with distance
for x in range(12):     
    for y in range(12):  
        ax.text(y, x, str(round(cost_matrix[x, y],2)),ha='center',va='center',color='black',fontsize=20)

# Add labels and ticks
plt.ylabel("Egypt (05-2005---04-2006)",size=35)
plt.xlabel("Egypt (07-2014---06-2015)",size=35)
plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11],[1,2,3,4,5,6,7,8,9,10,11,12],size=35)
plt.yticks([0,1,2,3,4,5,6,7,8,9,10,11],[1,2,3,4,5,6,7,8,9,10,11,12],size=35)
plt.tight_layout()

# Save
plt.savefig("out/dtw2.eps",dpi=300,bbox_inches="tight")
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_dissertation/out/dtw2.eps",dpi=300,bbox_inches='tight')
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/protest_armed_conflict_diss/out/dtw2.eps",dpi=300,bbox_inches='tight')
plt.show()

# Verify with the library build in visualization tools
fig, ax = dtwvis.plot_warping(t1, t2, dtw_path)
plt.savefig("out/dtw_check.png",dpi=300,bbox_inches="tight")

dist, cost_matrix = dtw.warping_paths(t1, t2)
dtw_path = dtw.best_path(cost_matrix) 
fig, ax = dtwvis.plot_warpingpaths(t1, t2, cost_matrix, dtw_path, shownumbers=True)
plt.savefig("out/dtw_check2.png",dpi=300,bbox_inches="tight")





