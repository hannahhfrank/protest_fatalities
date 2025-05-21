import pandas as pd
import numpy as np
from functions import general_model,general_dynamic_model,preprocess_min_max_group
import matplotlib.pyplot as plt
import json
import matplotlib as mpl
from sklearn.linear_model import Ridge
import os 
os.environ['PATH'] = "/Library/TeX/texbin:" + os.environ.get('PATH', '')
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['text.latex.preamble'] = r'\usepackage{lmodern}\usepackage[T1]{fontenc}'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 10, stop = 2000, num = 7)],
               'max_depth': [int(x) for x in np.linspace(10, 50, num = 5)]}

param_grid_lasso = {'alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10]}

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
preprocess_min_max_group(df,"fatalities","country")
df['fatalities_norm_lag1'] = df.groupby('gw_codes')['fatalities_norm'].shift(1).fillna(0)
df["SP.POP.TOTL_log"]=np.log(df["SP.POP.TOTL"])
df["NY.GDP.PCAP.CD_log"]=np.log(df["NY.GDP.PCAP.CD"])
df["fatalities_log"]=np.log(df["fatalities"]+1)
df.isnull().any()

######################
### Dynamic models ###
######################

countries=df.country.unique()
final_dynamic=pd.DataFrame()
shapes_rf={}
final_dynamic_linear=pd.DataFrame()
shapes_ols={}
#flat_countries=[]
#non_flat_countries=[]

for c in countries:
    print(c)
    ts=df["n_protest_events"].loc[df["country"]==c]
    Y=df["fatalities"].loc[df["country"]==c]
    X=df[["fatalities_norm_lag1",'NY.GDP.PCAP.CD_log','SP.POP.TOTL_log',"v2x_libdem","v2x_clphy","v2x_corr","v2x_rule","v2x_civlib","v2x_neopat"]].loc[df["country"]==c]
   
    # Return 0 if training data is flat
    #if Y[int(0.7*len(Y))-12:int(0.7*len(Y))].max()==0:
    #    flat_countries.append(c)
    #    print("flat")
    #    data_lin_rf = {'dd': list(df["dd"].loc[df["country"]==c][int(0.7*len(ts)):]),
    #        'country': [c] * len(list(df["dd"].loc[df["country"]==c][int(0.7*len(ts)):])),
    #        'fatalities_revert': [0] * len(list(df["dd"].loc[df["country"]==c][int(0.7*len(ts)):])),
    #        'fatalities': [0] * len(list(df["dd"].loc[df["country"]==c][int(0.7*len(ts)):])),
    #        'preds_drf': [0] * len(list(df["dd"].loc[df["country"]==c][int(0.7*len(ts)):])),
    #        'preds_drf_reverted': [0] * len(list(df["dd"].loc[df["country"]==c][int(0.7*len(ts)):])),
    #        'preds_drfx': [0] * len(list(df["dd"].loc[df["country"]==c][int(0.7*len(ts)):])),
    #        'preds_drfx_reverted': [0] * len(list(df["dd"].loc[df["country"]==c][int(0.7*len(ts)):])),}
    #    preds = pd.DataFrame(data_lin_rf)
    #    final_dynamic = pd.concat([final_dynamic, preds])
    #    final_dynamic=final_dynamic.reset_index(drop=True)
    #    final_dynamic.to_csv(f"data/preds_dynamic_nonlinear.csv")
    #    
    #    data_lin_ols = {'dd': list(df["dd"].loc[df["country"]==c][int(0.7*len(ts)):]),
    #        'country': [c] * len(list(df["dd"].loc[df["country"]==c][int(0.7*len(ts)):])),
    #        'fatalities_revert': [0] * len(list(df["dd"].loc[df["country"]==c][int(0.7*len(ts)):])),
    #        'fatalities': [0] * len(list(df["dd"].loc[df["country"]==c][int(0.7*len(ts)):])),
    #        'preds_dols': [0] * len(list(df["dd"].loc[df["country"]==c][int(0.7*len(ts)):])),
    #        'preds_dols_reverted': [0] * len(list(df["dd"].loc[df["country"]==c][int(0.7*len(ts)):])),
    #        'preds_dolsx': [0] * len(list(df["dd"].loc[df["country"]==c][int(0.7*len(ts)):])),
    #        'preds_dolsx_reverted': [0] * len(list(df["dd"].loc[df["country"]==c][int(0.7*len(ts)):])),}
    #    preds = pd.DataFrame(data_lin_ols) 
    #    final_dynamic_linear = pd.concat([final_dynamic_linear, preds])
    #    final_dynamic_linear=final_dynamic_linear.reset_index(drop=True)
    #    final_dynamic_linear.to_csv(f"data/preds_dynamic_linear.csv")        
        
    #else: 
     #   non_flat_countries.append(c)

     # DRF
    drf = general_dynamic_model(ts,Y,opti_grid=random_grid,norm=True,metric="mse") 
    preds = pd.DataFrame(df["dd"].loc[df["country"]==c][-len(drf["actuals"]):])
    preds.columns = ["dd"]  
    preds["country"] = c
    preds["fatalities_revert"] = df["fatalities"].loc[df["country"]==c][-len(drf["actuals"]):]
    preds["fatalities"] = list(drf["actuals"])
    preds["preds_drf"] = list(drf["drf_pred"])
    preds["preds_drf_reverted"] = list(drf["drf_pred_revert"])
    shapes_rf.update({f"drf_{c}":[drf["s"],drf["shapes"].tolist(),drf["clusters"].tolist()]})
           
    # DRFX
    drfx = general_dynamic_model(ts,Y,X=X,norm=True,opti_grid=random_grid,metric="mse")
    preds["preds_drfx"] = list(drfx["drf_pred"])
    preds["preds_drfx_reverted"] = list(drfx["drf_pred_revert"])  
    shapes_rf.update({f"drfx_{c}":[drfx["s"],drfx["shapes"].tolist(),drfx["clusters"].tolist()]})
    final_dynamic = pd.concat([final_dynamic, preds])
    final_dynamic.to_csv("data/preds_dynamic_nonlinear2.csv")  
     
    # Linear
    dOLS = general_dynamic_model(ts,Y,model_pred=Ridge(max_iter=5000),opti_grid=param_grid_lasso,norm=True,metric="mse") 
    preds = pd.DataFrame(df["dd"].loc[df["country"]==c][-len(dOLS["actuals"]):])
    preds.columns = ["dd"]  
    preds["country"] = c
    preds["fatalities_revert"] = df["fatalities"].loc[df["country"]==c][-len(dOLS["actuals"]):]
    preds["fatalities"] = list(dOLS["actuals"])
    preds["preds_dols"] = list(dOLS["drf_pred"])
    preds["preds_dols_reverted"] = list(dOLS["drf_pred_revert"])
    shapes_ols.update({f"dols_{c}":[dOLS["s"],dOLS["shapes"].tolist(),dOLS["clusters"].tolist()]})
           
    # Linear X
    dOLSx = general_dynamic_model(ts,Y,X=X,model_pred=Ridge(max_iter=5000),opti_grid=param_grid_lasso,norm=True,metric="mse")
    preds["preds_dolsx"] = list(dOLSx["drf_pred"])
    preds["preds_dolsx_reverted"] = list(dOLSx["drf_pred_revert"])  
    shapes_ols.update({f"dolsx_{c}":[dOLSx["s"],dOLSx["shapes"].tolist(),dOLSx["clusters"].tolist()]})
    final_dynamic_linear = pd.concat([final_dynamic_linear, preds])
    final_dynamic_linear.to_csv("data/preds_dynamic_linear2.csv")  
    
#with open("data/rf_shapes2.json", 'w') as json_file:
#    json.dump(shapes_rf, json_file)

#with open("data/ols_shapes2.json", 'w') as json_file:
#    json.dump(shapes_ols, json_file)
    
#with open("data/flat_countries.txt", "w") as file:
#    for item in flat_countries:
#        file.write(f"{item}\n")

#with open("data/non_flat_countries.txt", "w") as file:
#    for item in non_flat_countries:
#        file.write(f"{item}\n")

#####################
### Static models ###
#####################

countries=df.country.unique()
final_preds=pd.DataFrame()
final_preds_linear=pd.DataFrame()

for c in countries:
    print(c)
    ts=df["n_protest_events"].loc[df["country"]==c]
    Y=df["fatalities"].loc[df["country"]==c]
    X=df[["fatalities_norm_lag1",'NY.GDP.PCAP.CD_log','SP.POP.TOTL_log',"v2x_libdem","v2x_clphy","v2x_corr","v2x_rule","v2x_civlib","v2x_neopat"]].loc[df["country"]==c]

    # Return 0 if training data is flat
    #if Y[int(0.7*len(Y))-12:int(0.7*len(Y))].max()==0:
    #    print("flat")
    #    data_lin_rf = {'dd': list(df["dd"].loc[df["country"]==c][int(0.7*len(ts)):]),
    #        'country': [c] * len(list(df["dd"].loc[df["country"]==c][int(0.7*len(ts)):])),
    #        'fatalities_revert': [0] * len(list(df["dd"].loc[df["country"]==c][int(0.7*len(ts)):])),
    #        'fatalities': [0] * len(list(df["dd"].loc[df["country"]==c][int(0.7*len(ts)):])),
    #        'preds_rf': [0] * len(list(df["dd"].loc[df["country"]==c][int(0.7*len(ts)):])),
    #        'preds_rf_reverted': [0] * len(list(df["dd"].loc[df["country"]==c][int(0.7*len(ts)):])),
    #        'preds_rfx': [0] * len(list(df["dd"].loc[df["country"]==c][int(0.7*len(ts)):])),
    #        'preds_rfx_reverted': [0] * len(list(df["dd"].loc[df["country"]==c][int(0.7*len(ts)):])),}
    #    preds = pd.DataFrame(data_lin_rf)
    #    final_preds = pd.concat([final_preds, preds])
    #    final_preds=final_preds.reset_index(drop=True)
    #    final_preds.to_csv(f"data/preds_static_nonlinear.csv")
    #    
    #    data_lin_ols = {'dd': list(df["dd"].loc[df["country"]==c][int(0.7*len(ts)):]),
    #        'country': [c] * len(list(df["dd"].loc[df["country"]==c][int(0.7*len(ts)):])),
    #        'fatalities_revert': [0] * len(list(df["dd"].loc[df["country"]==c][int(0.7*len(ts)):])),
    #        'fatalities': [0] * len(list(df["dd"].loc[df["country"]==c][int(0.7*len(ts)):])),
    #        'preds_ols': [0] * len(list(df["dd"].loc[df["country"]==c][int(0.7*len(ts)):])),
    #        'preds_ols_reverted': [0] * len(list(df["dd"].loc[df["country"]==c][int(0.7*len(ts)):])),
    #        'preds_olsx': [0] * len(list(df["dd"].loc[df["country"]==c][int(0.7*len(ts)):])),
    #        'preds_olsx_reverted': [0] * len(list(df["dd"].loc[df["country"]==c][int(0.7*len(ts)):])),}
    #    preds = pd.DataFrame(data_lin_ols) 
    #    final_preds_linear = pd.concat([final_preds_linear, preds])
    #    final_preds_linear=final_preds_linear.reset_index(drop=True)
    #    final_preds_linear.to_csv(f"data/preds_static_linear.csv")
    #        
    # Fit static models
    #else:
        
    # RF
    rf = general_model(ts,Y,opti_grid=random_grid,norm=True,metric="mse") 
    preds = pd.DataFrame(df["dd"].loc[df["country"]==c][-len(rf["actuals"]):])
    preds.columns = ["dd"]  
    preds["country"] = c
    preds["fatalities_revert"] = df["fatalities"].loc[df["country"]==c][-len(rf["actuals"]):]
    preds["fatalities"] = list(rf["actuals"])
    preds["preds_rf"] = list(rf["rf_pred"])
    preds["preds_rf_reverted"] = list(rf["rf_pred_revert"])
        
    # RFX
    rfx = general_model(ts,Y,X=X,opti_grid=random_grid,norm=True,metric="mse") 
    preds["preds_rfx"] = list(rfx["rf_pred"])
    preds["preds_rfx_reverted"] = list(rfx["rf_pred_revert"])    
    final_preds = pd.concat([final_preds, preds])
    final_preds=final_preds.reset_index(drop=True)
    final_preds.to_csv("data/preds_static_nonlinear.csv")  
        
    # Linear
    OLS = general_model(ts,Y,model_pred=Ridge(max_iter=5000),opti_grid=param_grid_lasso,norm=True,metric="mse") 
    preds = pd.DataFrame(df["dd"].loc[df["country"]==c][-len(OLS["actuals"]):])
    preds.columns = ["dd"]  
    preds["country"] = c
    preds["fatalities_revert"] = df["fatalities"].loc[df["country"]==c][-len(OLS["actuals"]):]
    preds["fatalities"] = list(OLS["actuals"])
    preds["preds_ols"] = list(OLS["rf_pred"])
    preds["preds_ols_reverted"] = list(OLS["rf_pred_revert"])
        
    # Linea X
    OLSx = general_model(ts,Y,X=X,model_pred=Ridge(max_iter=5000),opti_grid=param_grid_lasso,norm=True,metric="mse") 
    preds["preds_olsx"] = list(OLSx["rf_pred"])
    preds["preds_olsx_reverted"] = list(OLSx["rf_pred_revert"])    
    final_preds_linear = pd.concat([final_preds_linear, preds])
    final_preds_linear=final_preds_linear.reset_index(drop=True)
    final_preds_linear.to_csv("data/preds_static_linear.csv")  
            
# Merge cases 
def boot(data, num_samples=1000, statistic=np.mean):
    n = len(data)
    bootstrap_estimates = np.empty(num_samples)
    for i in range(num_samples):
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_estimates[i] = statistic(bootstrap_sample)
    standard_error = np.std(bootstrap_estimates)
    return standard_error

# Nonlinear     
final_rf_static = pd.read_csv("data/preds_static_nonlinear.csv",index_col=0)
duplicate_rows = final_rf_static[final_rf_static.duplicated(subset=['dd', 'country'], keep=False)]
for thres in [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
    df_n_country_month = {}
    countries=df.country.unique()
    for i in countries:
        ts = df["v2x_polyarchy"].loc[df["country"]==i][:int(0.7*len(df["v2x_polyarchy"].loc[df["country"]==i]))]
        df_n_country_month[i] = ts.mean()
    df_n_country_month = pd.DataFrame.from_dict(df_n_country_month, orient="index").reset_index()
    df_n_country_month.rename(columns = {'index':'country', 0:'avg'}, inplace = True) 
    country_keep = df_n_country_month.loc[df_n_country_month["avg"]<thres].country.unique()
    
    # Dynamic models
    final_rf_dynamic = pd.read_csv("data/preds_dynamic_nonlinear.csv",index_col=0)
    final_rf_dynamic = final_rf_dynamic[final_rf_dynamic['country'].isin(country_keep)]
    duplicate_rows = final_rf_dynamic[final_rf_dynamic.duplicated(subset=['dd', 'country'], keep=False)]
    
    # Merge
    df_nonlinear=pd.merge(final_rf_static,final_rf_dynamic[["dd","country",'preds_drf','preds_drf_reverted','preds_drfx','preds_drfx_reverted']],on=["dd","country"],how="right")
    df_nonlinear=df_nonlinear.sort_values(by=["country","dd"])
    df_nonlinear=df_nonlinear.reset_index(drop=True)
    df_nonlinear.to_csv("data/df_nonlinear.csv")  
 
    df_nonlinear["mse_rf"]=((df_nonlinear["fatalities"] - df_nonlinear["preds_rf"]) ** 2) 
    df_nonlinear["mse_rfx"]=((df_nonlinear["fatalities"] - df_nonlinear["preds_rfx"]) ** 2)
    df_nonlinear["mse_drf"]=((df_nonlinear["fatalities"] - df_nonlinear["preds_drf"]) ** 2) 
    df_nonlinear["mse_drfx"]=((df_nonlinear["fatalities"] - df_nonlinear["preds_drfx"]) ** 2) 
    means = [df_nonlinear["mse_rf"].mean(),df_nonlinear["mse_rfx"].mean(),df_nonlinear["mse_drf"].mean(),df_nonlinear["mse_drfx"].mean()]
    std_error = [boot(df_nonlinear["mse_rf"]),boot(df_nonlinear["mse_rfx"]),boot(df_nonlinear["mse_drf"]),boot(df_nonlinear["mse_drfx"])]
    mean_mse = pd.DataFrame({'mean': means,'std': std_error})
    means_imporve=[(df_nonlinear["mse_rf"]-df_nonlinear["mse_rfx"]).mean(),(df_nonlinear["mse_rf"]-df_nonlinear["mse_drf"]).mean(),(df_nonlinear["mse_rf"]-df_nonlinear["mse_drfx"]).mean()]
    std_improve=[boot(df_nonlinear["mse_rf"]-df_nonlinear["mse_rfx"]),boot(df_nonlinear["mse_rf"]-df_nonlinear["mse_drf"]),boot(df_nonlinear["mse_rf"]-df_nonlinear["mse_drfx"])]
    mean_mse_imporv = pd.DataFrame({'mean': means_imporve,'std': std_improve})
    print(thres, means_imporve)
    
    # Plot
    fig, ax1 = plt.subplots(figsize=(12,8))
    marker_size = 150
    linewidth = 3
    fonts=25
    ax1.scatter(mean_mse.index, mean_mse['mean'], color="black", marker='o', s=marker_size)
    ax1.errorbar(mean_mse.index, mean_mse['mean'], yerr=mean_mse['std'], fmt='none', color="black", linewidth=linewidth)
    ax2 = ax1.twinx()
    ax2.grid(False)
    ax2.scatter(mean_mse_imporv.index+1.2, mean_mse_imporv['mean'], color="gray", marker='o', s=marker_size)
    ax2.errorbar(mean_mse_imporv.index+1.2, mean_mse_imporv['mean'], yerr=mean_mse_imporv['std'], fmt='none', color="gray", linewidth=linewidth)
    ax2.hlines(0, 0, 3.5, linestyles='--', color="gray", linewidth=linewidth)
    ax1.grid(False)
    ax2.set_xticks([*range(4)],['RF','RFX','DRF','DRFX'],fontsize=18)
    ax1.set_ylabel("Mean squared error (WMSE)",size=25)
    ax2.set_ylabel("Improvement in MSE",size=25)
    plt.title(f"thres {thres}")
    plt.savefig(f"out/results_main_plot_{thres}_nonlin.jpeg",dpi=300,bbox_inches="tight")

# Linear     
final_rf_static = pd.read_csv("data/preds_static_linear.csv",index_col=0)
duplicate_rows = final_rf_static[final_rf_static.duplicated(subset=['dd', 'country'], keep=False)]
for thres in [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
    df_n_country_month = {}
    countries=df.country.unique()
    for i in countries:
        ts = df["v2x_polyarchy"].loc[df["country"]==i][:int(0.7*len(df["v2x_polyarchy"].loc[df["country"]==i]))]
        df_n_country_month[i] = ts.mean()
    df_n_country_month = pd.DataFrame.from_dict(df_n_country_month, orient="index").reset_index()
    df_n_country_month.rename(columns = {'index':'country', 0:'avg'}, inplace = True) 
    country_keep = df_n_country_month.loc[df_n_country_month["avg"]<thres].country.unique()
    
    # Dynamic models
    final_rf_dynamic = pd.read_csv("data/preds_dynamic_linear.csv",index_col=0)
    final_rf_dynamic = final_rf_dynamic[final_rf_dynamic['country'].isin(country_keep)]
    duplicate_rows = final_rf_dynamic[final_rf_dynamic.duplicated(subset=['dd', 'country'], keep=False)]
    
    # Merge
    df_nonlinear=pd.merge(final_rf_static,final_rf_dynamic[["dd","country",'preds_dols','preds_dols_reverted','preds_dolsx','preds_dolsx_reverted']],on=["dd","country"],how="right")
    df_nonlinear=df_nonlinear.sort_values(by=["country","dd"])
    df_nonlinear=df_nonlinear.reset_index(drop=True)
    df_nonlinear.to_csv("data/df_linear.csv")  
 
    df_nonlinear["mse_ols"]=((df_nonlinear["fatalities"] - df_nonlinear["preds_ols"]) ** 2) 
    df_nonlinear["mse_olsx"]=((df_nonlinear["fatalities"] - df_nonlinear["preds_olsx"]) ** 2)
    df_nonlinear["mse_dols"]=((df_nonlinear["fatalities"] - df_nonlinear["preds_dols"]) ** 2) 
    df_nonlinear["mse_dolsx"]=((df_nonlinear["fatalities"] - df_nonlinear["preds_dolsx"]) ** 2) 
    means = [df_nonlinear["mse_ols"].mean(),df_nonlinear["mse_olsx"].mean(),df_nonlinear["mse_dols"].mean(),df_nonlinear["mse_dolsx"].mean()]
    std_error = [boot(df_nonlinear["mse_ols"]),boot(df_nonlinear["mse_olsx"]),boot(df_nonlinear["mse_dols"]),boot(df_nonlinear["mse_dolsx"])]
    mean_mse = pd.DataFrame({'mean': means,'std': std_error})
    means_imporve=[(df_nonlinear["mse_ols"]-df_nonlinear["mse_olsx"]).mean(),(df_nonlinear["mse_ols"]-df_nonlinear["mse_dols"]).mean(),(df_nonlinear["mse_ols"]-df_nonlinear["mse_dolsx"]).mean()]
    std_improve=[boot(df_nonlinear["mse_ols"]-df_nonlinear["mse_olsx"]),boot(df_nonlinear["mse_ols"]-df_nonlinear["mse_dols"]),boot(df_nonlinear["mse_ols"]-df_nonlinear["mse_dolsx"])]
    mean_mse_imporv = pd.DataFrame({'mean': means_imporve,'std': std_improve})
    
    # Plot
    fig, ax1 = plt.subplots(figsize=(12,8))
    marker_size = 150
    linewidth = 3
    fonts=25
    ax1.scatter(mean_mse.index, mean_mse['mean'], color="black", marker='o', s=marker_size)
    ax1.errorbar(mean_mse.index, mean_mse['mean'], yerr=mean_mse['std'], fmt='none', color="black", linewidth=linewidth)
    ax2 = ax1.twinx()
    ax2.grid(False)
    ax2.scatter(mean_mse_imporv.index+1.2, mean_mse_imporv['mean'], color="gray", marker='o', s=marker_size)
    ax2.errorbar(mean_mse_imporv.index+1.2, mean_mse_imporv['mean'], yerr=mean_mse_imporv['std'], fmt='none', color="gray", linewidth=linewidth)
    ax2.hlines(0, 0, 3.5, linestyles='--', color="gray", linewidth=linewidth)
    ax1.grid(False)
    ax2.set_xticks([*range(4)],['OLS','OLSX','DOLS','DOLSX'],fontsize=18)
    ax1.set_ylabel("Mean squared error (WMSE)",size=25)
    ax2.set_ylabel("Improvement in MSE",size=25)
    plt.title(f"thres {thres}")
    plt.savefig(f"out/results_main_plot_{thres}_lin.jpeg",dpi=300,bbox_inches="tight")


