import pandas as pd
import numpy as np
from functions import clustering,preprocess_min_max_group
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import json
import itertools
from matplotlib.lines import Line2D
from dtaidistance import dtw
from scipy.cluster.hierarchy import linkage, fcluster,dendrogram
import geopandas as gpd
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm 
from sklearn.linear_model import LinearRegression,Lasso,Ridge
import statsmodels.api as sm
import numpy as np
from tslearn.metrics import dtw
import statistics
from scipy.spatial.distance import euclidean
from statsmodels.iolib.summary2 import summary_col
from tslearn.clustering import silhouette_score
from dtaidistance import dtw
from scipy.cluster.hierarchy import linkage, fcluster,dendrogram
from scipy.spatial.distance import squareform
from statsmodels.stats.anova import anova_lm
from statsmodels.iolib.summary2 import summary_col
import os 
os.environ['PATH'] = "/Library/TeX/texbin:" + os.environ.get('PATH', '')
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['text.latex.preamble'] = r'\usepackage{lmodern}\usepackage[T1]{fontenc}'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# Simple random forest
random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 10, stop = 2000, num = 7)],
               'max_depth': [int(x) for x in np.linspace(10, 50, num = 5)]}

param_grid_lasso = {'alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10]}

# Country definitions: http://ksgleditsch.com/statelist.html

# List of microstates: 
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

# Exclude micro states
df = df[~df['gw_codes'].isin(list(micro_states.values()))]
df = df.reset_index(drop=True)

# Lagged dependent variable
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
final_dynamic_linear=pd.DataFrame()
shapes_ols={}

for c in countries:
    print(c)
    df_s=df.loc[df["country"]==c].copy()
    ts=df["n_protest_events"].loc[df["country"]==c]
    dOLSx = clustering(ts)
    min_val = np.min(ts)
    max_val = np.max(ts)
    ts_norm = (ts - min_val) / (max_val - min_val)
    ts_norm=ts_norm.fillna(0) 
    df_s["n_protest_events_norm"]=ts_norm
    
    df_s=df_s[-len(dOLSx["clusters"]):]
    df_s["clusters"]=dOLSx["clusters"]
    df_s['n_protest_events_lag_1']=(df_s['n_protest_events'].shift(1))
    df_s['n_protest_events_lag_2']=(df_s['n_protest_events'].shift(2))
    df_s['n_protest_events_lag_3']=(df_s['n_protest_events'].shift(3))
    df_s['n_protest_events_lag_4']=(df_s['n_protest_events'].shift(4))
    df_s['n_protest_events_lag_5']=(df_s['n_protest_events'].shift(5))
    
    df_s['n_protest_events_norm_lag_1']=(df_s['n_protest_events_norm'].shift(1))
    df_s['n_protest_events_norm_lag_2']=(df_s['n_protest_events_norm'].shift(2))
    df_s['n_protest_events_norm_lag_3']=(df_s['n_protest_events_norm'].shift(3))
    df_s['n_protest_events_norm_lag_4']=(df_s['n_protest_events_norm'].shift(4))
    df_s['n_protest_events_norm_lag_5']=(df_s['n_protest_events_norm'].shift(5))
        
    final_dynamic_linear = pd.concat([final_dynamic_linear, df_s])
    final_dynamic_linear.to_csv("data/preds_dynamic_linear_reg.csv")  
    shapes_ols.update({f"dols_{c}":[dOLSx["s"],dOLSx["shapes"].tolist(),dOLSx["clusters"].tolist()]})
    
with open("data/ols_shapes_reg.json", 'w') as json_file:
    json.dump(shapes_ols, json_file)
 
###################################
### Clustering of the centroids ###
###################################

final_dynamic_linear=pd.read_csv("data/preds_dynamic_linear_reg.csv",index_col=0)  
 
with open("data/ols_shapes_reg.json", 'r') as json_file:
    shapes_rf = json.load(json_file)
    
shapes_rf = dict(filter(lambda item: item[0].startswith("dols_"), shapes_rf.items()))


score_test=-1
for k in [3,5,7]:
    df_cen=pd.DataFrame()
    for d in shapes_rf.keys():
        for i in range(len(shapes_rf[d][1])):
            lst = pd.DataFrame([[d[5:], i]], columns=['country', 'clusters'])
            add=pd.DataFrame([item for sublist in shapes_rf[d][1][i] for item in sublist]).T
            lst=pd.concat([lst,add],axis=1)
            df_cen=pd.concat([df_cen,lst])
    
    arr=df_cen[[0,1,2,3,4,5,6,7,8,9,10,11]].values
    rows_without_nan = []
    # Iterate over each row in the dataset
    for row in arr:
        row=row.astype(float)
        # Remove missing values from the row and append to the list
        rows_without_nan.append(row[~np.isnan(row)])
    
    distance_matrix = dtw.distance_matrix_fast(rows_without_nan)    
    condensed_dist_matrix = squareform(distance_matrix)
    linkage_matrix = linkage(condensed_dist_matrix, method='ward')
    clusters = fcluster(linkage_matrix, t=k, criterion='maxclust')
    df_cen["clusters_cen"]=clusters

    score = silhouette_score(rows_without_nan, clusters,metric="dtw")
    print(score)
    
    if score>score_test: 
        score_test=score
        df_cen_final=df_cen

        unique_clusters = np.unique(clusters)
        representatives = []
            
        for cluster_id in unique_clusters:
            cluster_sequences = [rows_without_nan[i] for i, cluster in enumerate(clusters) if cluster == cluster_id]
            distance_matrix = dtw.distance_matrix_fast(cluster_sequences)
            representative_idx = np.argmin(distance_matrix.sum(axis=0))
            representatives.append(cluster_sequences[representative_idx])
                
        n_clusters = len(representatives)
        cols = 3
        rows = n_clusters // cols + (n_clusters % cols > 0)
            
        plt.figure(figsize=(10, 3 * rows))
        for i, seq in enumerate(representatives):
            plt.subplot(rows, cols, i+1)
            plt.plot(seq,linewidth=2,c="black")
            plt.title(f'Cluster {i+1}',size=25)
            plt.yticks([],[])
            plt.xticks([],[])
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.01)
        plt.savefig("out/clusters_clusters.jpeg",dpi=300,bbox_inches="tight")
        plt.show()


df_final=pd.merge(final_dynamic_linear, df_cen_final[["country","clusters","clusters_cen"]],on=["clusters","country"])
dummies = pd.get_dummies(df_final['clusters_cen'], prefix='cluster').astype(int)
final_shapes_s = pd.concat([df_final, dummies], axis=1)
final_shapes_s['fatalities_log_lag1'] = final_shapes_s.groupby('gw_codes')['fatalities_log'].shift(1)
final_shapes_s['fatalities_log_lag2'] = final_shapes_s.groupby('gw_codes')['fatalities_log'].shift(2)
final_shapes_s['fatalities_log_lag3'] = final_shapes_s.groupby('gw_codes')['fatalities_log'].shift(3)
final_shapes_s['fatalities_log_lag4'] = final_shapes_s.groupby('gw_codes')['fatalities_log'].shift(4)
final_shapes_s['fatalities_log_lag5'] = final_shapes_s.groupby('gw_codes')['fatalities_log'].shift(5)
final_shapes_s['fatalities_log_lag6'] = final_shapes_s.groupby('gw_codes')['fatalities_log'].shift(6)
final_shapes_s['fatalities_log_lag7'] = final_shapes_s.groupby('gw_codes')['fatalities_log'].shift(7)
final_shapes_s['fatalities_log_lag8'] = final_shapes_s.groupby('gw_codes')['fatalities_log'].shift(8)
final_shapes_s['fatalities_log_lag9'] = final_shapes_s.groupby('gw_codes')['fatalities_log'].shift(9)
final_shapes_s['fatalities_log_lag10'] = final_shapes_s.groupby('gw_codes')['fatalities_log'].shift(10)

final_shapes_s.to_csv("data/final_shapes_s.csv")  

