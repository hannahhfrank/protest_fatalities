import pandas as pd
import numpy as np
from functions import clustering,preprocess_min_max_group
import matplotlib.pyplot as plt
import json
from dtaidistance import dtw
from scipy.cluster.hierarchy import linkage,fcluster
import matplotlib as mpl
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.clustering import silhouette_score
from scipy.spatial.distance import squareform
import os 
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['PATH'] = "/Library/TeX/texbin:" + os.environ.get('PATH', '')
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['text.latex.preamble'] = r'\usepackage{lmodern}\usepackage[T1]{fontenc}'

# Remove microstates
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
df=pd.read_csv("data/df.csv",index_col=0)
df = df[~df['gw_codes'].isin(list(micro_states.values()))]
df = df.reset_index(drop=True)

# Transforms
preprocess_min_max_group(df,"fatalities","country")
df['fatalities_norm_lag1'] = df.groupby('gw_codes')['fatalities_norm'].shift(1).fillna(0)
df["SP.POP.TOTL_log"]=np.log(df["SP.POP.TOTL"])
df["NY.GDP.PCAP.CD_log"]=np.log(df["NY.GDP.PCAP.CD"])
df["fatalities_log"]=np.log(df["fatalities"]+1)

##############################################
### Step 1:  Get clusters for each country ###
##############################################

# Get within country clusters
countries=df.country.unique()
final_out=pd.DataFrame()
shapes={}

for c in countries:
    print(c)
    df_s=df.loc[df["country"]==c].copy()
    ts=df["n_protest_events"].loc[df["country"]==c]
    
    # Get clusters for each country
    cluster_out = clustering(ts)
    
    # Min-max normalize input sequences
    min_val = np.min(ts)
    max_val = np.max(ts)
    ts_norm = (ts - min_val) / (max_val - min_val)
    ts_norm=ts_norm.fillna(0) 
    df_s["n_protest_events_norm"]=ts_norm
    
    # Add cluster assignments
    df_s=df_s[-len(cluster_out["clusters"]):]
    df_s["clusters"]=cluster_out["clusters"]
    
    # Add static protest variables
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
        
    # Save ts with cluster assignments
    final_out = pd.concat([final_out, df_s])
    final_out.to_csv("data/cluster_reg.csv")  
    
    # Save corresponding centroids
    shapes.update({f"s_{c}":[cluster_out["s"],cluster_out["shapes"].tolist(),cluster_out["clusters"].tolist()]})
    
with open("data/ols_shapes_reg.json", 'w') as json_file:
    json.dump(shapes, json_file)
    
###########################################
### Step 2: Clustering of the centroids ###
###########################################

# Load within country clusters
with open("data/ols_shapes_reg.json", 'r') as json_file:
    shapes = json.load(json_file)
    
# Cluster the within country centroids
score_test=-1
for k in [3,5,7]:
        
    # Get centroids
    df_cen=pd.DataFrame()
    # For each country
    for d in shapes.keys():
        # Access the centroids
        for i in range(len(shapes[d][1])):
            # For each centroid, save country, cluster number, and centroid in df
            lst = pd.DataFrame([[d[2:], i]], columns=['country', 'clusters'])
            add=pd.DataFrame([item for sublist in shapes[d][1][i] for item in sublist]).T
            lst=pd.concat([lst,add],axis=1)
            df_cen=pd.concat([df_cen,lst])
    
    # Remove missing values which occur because the centroids can have
    # different lengths
    arr=df_cen[[0,1,2,3,4,5,6,7,8,9,10,11]].values
    matrix_in = []
    for row in arr:
        row=row.astype(float)
        matrix_in.append(row[~np.isnan(row)])
        
    # Hierachical clustering  
    matrix_d = dtw.distance_matrix_fast(matrix_in)    
    dist_matrix = squareform(matrix_d)
    link_matrix = linkage(dist_matrix, method='complete')
    clusters = fcluster(link_matrix, t=k, criterion='maxclust')
    df_cen["clusters_cen"]=clusters
    
    # Silhouette score
    score = silhouette_score(matrix_in, clusters, metric="dtw")
    print(score)
    
    # If s score is larger than test score update results
    if score>score_test: 
        score_test=score
        df_cen_final=df_cen
        clusters_s = np.unique(clusters)
            
        # Get centroids
        centroids = []
        # Loop over clusters
        for ids in clusters_s:
            # Get all within centroids assigned to the specific cluster
            cluster_seq = [matrix_in[i] for i, cluster in enumerate(clusters) if cluster == ids]
            # Then calculate the centroid using DTW Barycenter Averaging (DBA)
            # takes the mean for time series
            cen = dtw_barycenter_averaging(cluster_seq, barycenter_size=7)
            centroids.append(cen.ravel())
        
        # Plot
        plt.figure(figsize=(10, 6))
        for i, seq in enumerate(centroids):
            plt.subplot(2, 3, i+1)
            plt.plot(seq,linewidth=2,c="black")
            plt.title(f'Cluster {i+1}',size=25)
            plt.yticks([],[])
            plt.xticks([],[])
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.01)
        plt.savefig("out/clusters_clusters.eps",dpi=300,bbox_inches="tight")
        plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_dissertation/out/clusters_clusters.eps",dpi=300,bbox_inches='tight')
        plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/protest_armed_conflict_diss/out/clusters_clusters.eps",dpi=300,bbox_inches='tight')
        plt.show()


# Merge centroids of the centroids with the original data
final_out=pd.read_csv("data/cluster_reg.csv",index_col=0)  
df_final=pd.merge(final_out, df_cen_final[["country","clusters","clusters_cen"]],on=["clusters","country"])

# Create a dummy set for the cluster assignments
dummies = pd.get_dummies(df_final['clusters_cen'], prefix='cluster').astype(int)
final_shapes_s = pd.concat([df_final, dummies], axis=1)

# Calculate lagged dependent variable
final_shapes_s['fatalities_log_lag1'] = final_shapes_s.groupby('gw_codes')['fatalities_log'].shift(1)

# Save df
final_shapes_s.to_csv("data/final_shapes_s.csv")  

# ---> move to R for regression analysis



