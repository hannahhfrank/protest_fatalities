import pandas as pd
import numpy as np
from functions import general_model,general_dynamic_model,preprocess_min_max_group
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

grid = {'n_estimators': [10, 341, 673, 1005, 1336, 1668, 2000],
        'max_depth': [10, 20, 30, 40, 50]}

grid_lasso = {'alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10]}

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

for c in countries:
    print(c)
    ts=df["n_protest_events"].loc[df["country"]==c]
    Y=df["fatalities"].loc[df["country"]==c]
    X=df[["fatalities_norm_lag1",'NY.GDP.PCAP.CD_log','SP.POP.TOTL_log',"v2x_libdem","v2x_clphy","v2x_corr","v2x_rule","v2x_civlib","v2x_neopat"]].loc[df["country"]==c]
   
    # DRF
    drf = general_dynamic_model(ts,Y,grid=None,norm=True,metric="mse") 
    preds = pd.DataFrame(df["dd"].loc[df["country"]==c][-len(drf["actuals"]):])
    preds.columns = ["dd"]  
    preds["country"] = c
    preds["fatalities_revert"] = df["fatalities"].loc[df["country"]==c][-len(drf["actuals"]):]
    preds["fatalities"] = list(drf["actuals"])
    preds["preds_drf"] = list(drf["drf_pred"])
    preds["preds_drf_reverted"] = list(drf["drf_pred_revert"])
    shapes_rf.update({f"drf_{c}":[drf["s"],drf["shapes"].tolist(),drf["clusters"].tolist()]})
           
    # DRFX
    drfx = general_dynamic_model(ts,Y,X=X,norm=True,grid=None,metric="mse")
    preds["preds_drfx"] = list(drfx["drf_pred"])
    preds["preds_drfx_reverted"] = list(drfx["drf_pred_revert"])  
    shapes_rf.update({f"drfx_{c}":[drfx["s"],drfx["shapes"].tolist(),drfx["clusters"].tolist()]})
    final_dynamic = pd.concat([final_dynamic, preds])
    final_dynamic.to_csv("data/preds_dynamic_nonlinear.csv")  
     
    # Linear
    dOLS = general_dynamic_model(ts,Y,model_pred=Ridge(max_iter=5000),grid=grid_lasso,norm=True,metric="mse") 
    preds = pd.DataFrame(df["dd"].loc[df["country"]==c][-len(dOLS["actuals"]):])
    preds.columns = ["dd"]  
    preds["country"] = c
    preds["fatalities_revert"] = df["fatalities"].loc[df["country"]==c][-len(dOLS["actuals"]):]
    preds["fatalities"] = list(dOLS["actuals"])
    preds["preds_dols"] = list(dOLS["drf_pred"])
    preds["preds_dols_reverted"] = list(dOLS["drf_pred_revert"])
    shapes_ols.update({f"dols_{c}":[dOLS["s"],dOLS["shapes"].tolist(),dOLS["clusters"].tolist()]})
           
    # Linear X
    dOLSx = general_dynamic_model(ts,Y,X=X,model_pred=Ridge(max_iter=5000),grid=grid_lasso,norm=True,metric="mse")
    preds["preds_dolsx"] = list(dOLSx["drf_pred"])
    preds["preds_dolsx_reverted"] = list(dOLSx["drf_pred_revert"])  
    shapes_ols.update({f"dolsx_{c}":[dOLSx["s"],dOLSx["shapes"].tolist(),dOLSx["clusters"].tolist()]})
    final_dynamic_linear = pd.concat([final_dynamic_linear, preds])
    final_dynamic_linear.to_csv("data/preds_dynamic_linear.csv")  
    
with open("data/rf_shapes.json", 'w') as json_file:
    json.dump(shapes_rf, json_file)

with open("data/ols_shapes.json", 'w') as json_file:
    json.dump(shapes_ols, json_file)
    

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
        
    # RF
    rf = general_model(ts,Y,grid=None,norm=True,metric="mse") 
    preds = pd.DataFrame(df["dd"].loc[df["country"]==c][-len(rf["actuals"]):])
    preds.columns = ["dd"]  
    preds["country"] = c
    preds["fatalities_revert"] = df["fatalities"].loc[df["country"]==c][-len(rf["actuals"]):]
    preds["fatalities"] = list(rf["actuals"])
    preds["preds_rf"] = list(rf["rf_pred"])
    preds["preds_rf_reverted"] = list(rf["rf_pred_revert"])
        
    # RFX
    rfx = general_model(ts,Y,X=X,grid=None,norm=True,metric="mse") 
    preds["preds_rfx"] = list(rfx["rf_pred"])
    preds["preds_rfx_reverted"] = list(rfx["rf_pred_revert"])    
    final_preds = pd.concat([final_preds, preds])
    final_preds=final_preds.reset_index(drop=True)
    final_preds.to_csv("data/preds_static_nonlinear.csv")  
        
    # Linear
    OLS = general_model(ts,Y,model_pred=Ridge(max_iter=5000),grid=grid_lasso,norm=True,metric="mse") 
    preds = pd.DataFrame(df["dd"].loc[df["country"]==c][-len(OLS["actuals"]):])
    preds.columns = ["dd"]  
    preds["country"] = c
    preds["fatalities_revert"] = df["fatalities"].loc[df["country"]==c][-len(OLS["actuals"]):]
    preds["fatalities"] = list(OLS["actuals"])
    preds["preds_ols"] = list(OLS["rf_pred"])
    preds["preds_ols_reverted"] = list(OLS["rf_pred_revert"])
        
    # Linea X
    OLSx = general_model(ts,Y,X=X,model_pred=Ridge(max_iter=5000),grid=grid_lasso,norm=True,metric="mse") 
    preds["preds_olsx"] = list(OLSx["rf_pred"])
    preds["preds_olsx_reverted"] = list(OLSx["rf_pred_revert"])    
    final_preds_linear = pd.concat([final_preds_linear, preds])
    final_preds_linear=final_preds_linear.reset_index(drop=True)
    final_preds_linear.to_csv("data/preds_static_linear.csv")  
            

# Merge
df_linear=pd.merge(final_preds_linear,final_dynamic_linear[["dd","country",'preds_dols','preds_dols_reverted','preds_dolsx','preds_dolsx_reverted']],on=["dd","country"])
df_linear=df_linear.sort_values(by=["country","dd"])
df_linear=df_linear.reset_index(drop=True)
df_linear.to_csv("data/df_linear.csv")  

df_nonlinear=pd.merge(final_preds,final_dynamic[["dd","country",'preds_drf','preds_drf_reverted','preds_drfx','preds_drfx_reverted']],on=["dd","country"])
df_nonlinear=df_nonlinear.sort_values(by=["country","dd"])
df_nonlinear=df_nonlinear.reset_index(drop=True)
df_nonlinear.to_csv("data/df_nonlinear.csv")  


