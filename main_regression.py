import pandas as pd
import numpy as np
from functions import clustering
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

# Transforms
df["SP.POP.TOTL_log"]=np.log(df["SP.POP.TOTL"])
df["NY.GDP.PCAP.CD_log"]=np.log(df["NY.GDP.PCAP.CD"])
df["fatalities_log"]=np.log(df["fatalities"]+1)

##############################################
### Step 1:  Get clusters for each country ###
##############################################

# Define unique countries
countries=df.country.unique()

# Define out dfs
final_out=pd.DataFrame()
shapes={}

# Loop through every country
for c in countries:
    print(c)
    
    # Get subset data for country
    df_s=df.loc[df["country"]==c].copy()
    ts=df["n_protest_events"].loc[df["country"]==c]

    # Get clusters for each country using the protest time series
    cluster_out = clustering(ts)
    
    # Min-max normalize input sequences (needed for regression)
    min_val = np.min(ts)
    max_val = np.max(ts)
    ts_norm = (ts - min_val) / (max_val - min_val)
    ts_norm = ts_norm.fillna(0) 
    df_s["n_protest_events_norm"]=ts_norm
    
    # Add cluster assignments
    df_s=df_s[-len(cluster_out["clusters"]):]
    df_s["clusters"]=cluster_out["clusters"]
    
    # Add static protest variables
    df_s['n_protest_events_lag_1']=df_s['n_protest_events'].shift(1)
    df_s['n_protest_events_lag_2']=df_s['n_protest_events'].shift(2)
    df_s['n_protest_events_lag_3']=df_s['n_protest_events'].shift(3)
    df_s['n_protest_events_lag_4']=df_s['n_protest_events'].shift(4)
    df_s['n_protest_events_lag_5']=df_s['n_protest_events'].shift(5)
    
    # Add static protest variables, min-max normalized    
    df_s['n_protest_events_norm_lag_1']=df_s['n_protest_events_norm'].shift(1)
    df_s['n_protest_events_norm_lag_2']=df_s['n_protest_events_norm'].shift(2)
    df_s['n_protest_events_norm_lag_3']=df_s['n_protest_events_norm'].shift(3)
    df_s['n_protest_events_norm_lag_4']=df_s['n_protest_events_norm'].shift(4)
    df_s['n_protest_events_norm_lag_5']=df_s['n_protest_events_norm'].shift(5)
        
    # Merge to out df and save
    final_out = pd.concat([final_out, df_s])
    final_out.to_csv("data/cluster_reg.csv")  
    
    # Add centroids
    shapes.update({f"s_{c}":[cluster_out["s"],cluster_out["shapes"].tolist(),cluster_out["clusters"].tolist()]})

# Save centroids        
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
        
    # (1) Get centroids (within)
    
    df_cen=pd.DataFrame()
    # For each country
    for d in shapes.keys():
        # For each centroid
        for i in range(len(shapes[d][1])):
            # save country (d[2:]) and cluster number (i) --> needed for merging later
            row = pd.DataFrame([[d[2:], i]], columns=['country', 'clusters'])
            # Obtain corresponding centroid, by converting list of lists into list
            cen=[]
            for x in range(len(shapes[d][1][i])):
                cen.append(shapes[d][1][i][x][0])
            # Convert list with centroid into df, so that each column is one point                    
            centro=pd.DataFrame(cen).T
            # Add centroid to country and cluster number
            row=pd.concat([row,centro],axis=1)
            # Add row to out df
            df_cen=pd.concat([df_cen,row])
    
    # Remove missing values which occur because the centroids can have
    # different lengths
    arr=df_cen[[0,1,2,3,4,5,6,7,8,9,10,11]].values
    matrix_in = []
    # Loop through every row and append centroid values after removing na
    for row in arr:
        matrix_in.append(row[~np.isnan(row)])
        
    # (2) Hierachical clustering with DTW as distance metric --> cross country protest patterns
    
    # Get condensed distance matrix using dtw for distance
    matrix_d = dtw.distance_matrix_fast(matrix_in)    
    dist_matrix = squareform(matrix_d)
    # and run hierachical clustering algorithm on distance matrix using method complete
    link_matrix = linkage(dist_matrix,method='complete')
    
    # Obtain k clusters
    clusters = fcluster(link_matrix,t=k,criterion='maxclust')
    df_cen["clusters_cen"]=clusters
    
    # Silhouette score
    score = silhouette_score(matrix_in, clusters, metric="dtw")
    print(score)
    
    # If s score is larger than test score update results
    if score>score_test: 
        score_test=score
        df_cen_final=df_cen
        clusters_s = np.unique(clusters)
            
        # (3) Get centroids --> cross country protest patterns
        
        # Define out list
        centroids = []
        # Loop over clusters
        for k_num in clusters_s:
            # Get all centroids assigned to the specific cluster
            cluster_seq=[]
            # Loop over each case
            for i, cluster in enumerate(clusters): 
                # and save centroid if the case belongs to cluster k
                if cluster == k_num:
                    cluster_seq.append(matrix_in[i])
                    
            # Then calculate the centroid using DTW Barycenter Averaging (DBA)
            # takes the mean for time series
            cen = dtw_barycenter_averaging(cluster_seq, barycenter_size=7)
            # Save centroid
            centroids.append(cen.ravel())
        
        # Plot centroids
        
        # Specify plot 
        plt.figure(figsize=(10, 6))
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.01)
        
        # Loop over each centroid
        for i, seq in enumerate(centroids):
            # Add subplot at index i+1 (starts at 1)
            plt.subplot(2, 3, i+1)
            # and plot centroid
            plt.plot(seq,linewidth=2,c="black")
            plt.title(f'Cluster {i+1}',size=25)
            plt.ylim(-0.05,1)
            plt.yticks([],[])
            plt.xticks([],[])
            
        # Save
        plt.savefig("out/clusters_clusters.eps",dpi=300,bbox_inches="tight")
        plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_dissertation/out/clusters_clusters.eps",dpi=300,bbox_inches='tight')
        plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/protest_armed_conflict_diss/out/clusters_clusters.eps",dpi=300,bbox_inches='tight')
        plt.show()

#################################################
### Step 3: Prepare data for regression model ###
#################################################

# Merge centroids of the centroids with the original data

# Load data from step 1, including within country clusters
final_out=pd.read_csv("data/cluster_reg.csv",index_col=0)  

# Merge across country clusters on "clusters" and country --> "clusters" denote the within country clusters (see above)
df_final=pd.merge(final_out, df_cen_final[["country","clusters","clusters_cen"]],on=["clusters","country"])

# Create a dummy set for the cluster assignments
dummies = pd.get_dummies(df_final['clusters_cen'],prefix='cluster').astype(int)
final_shapes_s = pd.concat([df_final,dummies],axis=1)

# Calculate lagged dependent variable
final_shapes_s['fatalities_log_lag1'] = final_shapes_s.groupby('gw_codes')['fatalities_log'].shift(1)

# Save df
final_shapes_s.to_csv("data/final_shapes_s.csv")  

# ---> move to R for regression analysis



