import pandas as pd
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import PredefinedSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from tslearn.clustering import TimeSeriesKMeans,silhouette_score

def preprocess_min_max_group(df, x, group):
    out = pd.DataFrame()
    for i in df[group].unique():
        Y = df[x].loc[df[group] == i]
        
        # Train
        y_train = Y[:int(0.7*len(Y))]
        mini = np.min(y_train)
        maxi = np.max(y_train)
        y_train = (y_train - mini) / (maxi - mini)
        y_train=y_train.fillna(0) 
        
        # Test
        y_test =  Y[int(0.7*len(Y)):]           
        mini = np.min(y_test)
        maxi = np.max(y_test)
        y_test = (y_test - mini) / (maxi - mini)
        y_test=y_test.fillna(0) 
        
        # Merge
        Y=pd.concat([y_train,y_test])  
        out = pd.concat([out, pd.DataFrame(Y)], ignore_index=True)
    df[f"{x}_norm"] = out

def simple_imp_grouped(df, group, vars_input):
    
    # Split 
    train = pd.DataFrame()
    test = pd.DataFrame()
    for c in df.country.unique():
        df_s = df.loc[df["country"] == c]
        train_s = df_s[:int(0.7*len(df_s))]
        test_s = df_s[int(0.7*len(df_s)):]
        train = pd.concat([train, train_s])
        test = pd.concat([test, test_s])
            
    # Fill
    df_filled = pd.DataFrame()
    for c in df[group].unique():
        
        # Training
        df_s = train.loc[train[group] == c]
        imp = df_s[vars_input]
        df_imp = imp.copy(deep=True)
        imputer = SimpleImputer(strategy='mean')
        imputer.fit(df_imp)
        df_imp_train = imputer.transform(df_imp)
        df_imp_train_df = pd.DataFrame(df_imp_train)
        
        # Test
        df_s = test.loc[test[group] == c]
        imp = df_s[vars_input]
        df_imp = imp.copy(deep=True)        
        df_imp_test = imputer.transform(df_imp)
        df_imp_test_df = pd.DataFrame(df_imp_test)        

        # Merge
        df_imp_final = pd.concat([df_imp_train_df, df_imp_test_df])
        df_filled = pd.concat([df_filled, df_imp_final])

    # Merge
    df_filled.columns = vars_input
    df_filled = df_filled.reset_index(drop=True)
    feat_complete = df.drop(columns=vars_input)
    out = pd.concat([feat_complete, df_filled], axis=1)
    out=out.reset_index(drop=True)
    
    return out


def linear_imp_grouped(df, group, vars_input):
    
    # Split data 
    train = pd.DataFrame()
    test = pd.DataFrame()
    for c in df.country.unique():
        df_s = df.loc[df["country"] == c]
        train_s = df_s[:int(0.7*len(df_s))]
        test_s = df_s[int(0.7*len(df_s)):]
        train = pd.concat([train, train_s])
        test = pd.concat([test, test_s])

    # Fill    
    df_filled = pd.DataFrame()
    for c in df[group].unique():
        
        # Train
        df_s = train.loc[train[group] == c]
        imp = df_s[vars_input]
        df_imp = imp.copy(deep=True)
        df_imp_train_df = df_imp.interpolate(limit_direction="both")
        
        # Test
        df_s = test.loc[test[group] == c]
        imp = df_s[vars_input]
        df_imp = imp.copy(deep=True)        
        df_imp_test_df = df_imp.interpolate(limit_direction="forward")
        
        # Merge
        df_imp_trans_df = pd.concat([df_imp_train_df, df_imp_test_df])
        df_filled = pd.concat([df_filled, df_imp_trans_df])
        
    # Merge   
    df_filled.columns = vars_input
    df_filled = df_filled.reset_index(drop=True)
    feat_complete = df.drop(columns=vars_input)
    out = pd.concat([feat_complete, df_filled], axis=1)
    out=out.reset_index(drop=True)
    
    return out


def general_model(ts, Y, model_pred=RandomForestRegressor(random_state=0),  X=None, norm=False, grid=None, ar_test=[1,2,4,6,12]):

    # Normalize output    
    if norm==True:
        y_train = Y[:int(0.7*len(Y))]
        mini = np.min(y_train)
        maxi = np.max(y_train)
        y_train = (y_train - mini) / (maxi - mini)
        y_train=y_train.fillna(0) 
        
        y_test = Y[int(0.7*len(Y)):]      
        mini = np.min(y_test)
        maxi = np.max(y_test)
        y_test = (y_test - mini) / (maxi - mini)
        y_test=y_test.fillna(0) 
    
        Y=pd.concat([y_train,y_test])  
               
    min_test=np.inf
    for ar in ar_test:

        # Function to get matrix with lags
        def lags(series):
            last = series.iloc[-ar:].fillna(0)
            return last.tolist() + [0] * (ar - len(last))
           
        # Get matrix with temporal lags for ts
        data_matrix = []
        for i in range(ar, len(ts) + 1):
            data_matrix.append(lags(ts.iloc[:i]))
        in_put=pd.DataFrame(data_matrix)
        
        # Set index to align with ts 
        index=list(range(len(ts)-len(in_put), len(ts)))
        in_put.set_index(pd.Index(index),inplace=True)
        
        # Add X variables if available 
        if X is not None:
            # Reset index to avoid misalignment
            # Concat with X as base and delete missing observations
            X=X.reset_index(drop=True)
            in_put=pd.concat([X,in_put],axis=1)
            in_put = in_put.dropna()
            
        # Make sure column names are character
        in_put.columns = in_put.columns.map(str)
                    
        # Obtain output
        # Reset index to avoid misalignment and 
        # crop output to the size of the input   
        output=Y.reset_index(drop=True)
        output=output.iloc[-len(in_put):]
                    
        # Data split
        y_train = output[:-(len(ts)-int(0.7*len(ts)))]
        x_train = in_put[:-(len(ts)-int(0.7*len(ts)))]
        
        y_test = output[-(len(ts)-int(0.7*len(ts))):]        
        x_test = in_put[-(len(ts)-int(0.7*len(ts))):]     

        # If optimization
        if grid is not None:
            # Optimize in trainig data, based on a 50/50 split
            
            # Get splits
            val_train_ids = list(y_train[:int(0.5*len(y_train))].index)
            val_test_ids = list(y_train[int(0.5*len(y_train)):].index)
            splits = np.array([-1] * len(val_train_ids) + [0] * len(val_test_ids))
            splits = PredefinedSplit(test_fold=splits)
            grid_search = GridSearchCV(estimator=model_pred, param_grid=grid, cv=splits, verbose=0, n_jobs=-1)
            grid_search.fit(x_train, y_train)
            pred = grid_search.predict(x_test)
        # If no optimization
        else: 
            model_pred.fit(x_train, y_train)
            pred = model_pred.predict(x_test)        
       
        # MSE
        error=((y_test.reset_index(drop=True)-pd.Series(pred))**2).mean()
        
        # If MSE is lower than test mse, update results
        if error<min_test:
            min_test=error
            para=[ar]
            preds_final=pred

    print(f"Final RF {para}, with MSE: {min_test}")

    return({'pred':pd.Series(preds_final),'actuals':y_test.reset_index(drop=True)})
    
            
        
def general_dynamic_model(ts, Y, model_pred=RandomForestRegressor(random_state=0), X=None, norm=False, model=TimeSeriesKMeans(n_clusters=5,metric="dtw",max_iter_barycenter=100,verbose=0,random_state=0),grid=None,ar_test=[1,2,4,6,12],cluster_n=[3,5,7],w_length=[3,5,7,9]):
    
    # Normalize output
    if norm==True:
        y_train = Y[:int(0.7*len(Y))]
        mini = np.min(y_train)
        maxi = np.max(y_train)
        y_train = (y_train - mini) / (maxi - mini)
        y_train=y_train.fillna(0) 
        
        y_test = Y[int(0.7*len(Y)):]       
        mini = np.min(y_test)
        maxi = np.max(y_test)
        y_test = (y_test - mini) / (maxi - mini)
        y_test=y_test.fillna(0) 
    
        Y=pd.concat([y_train,y_test])  
    
    # (1) Clustering  
    min_test=np.inf
    for k in cluster_n:
        for w in w_length:
                
            # Adapted from: https://github.com/ThomasSchinca/ShapeF/blob/Thomas_draft/functions.py
            # Get input matrix for training data 
            y_s=ts.iloc[:int(0.7*len(ts))]
            seq_matrix=[]
            for i in range(w,len(y_s)):
                seq_matrix.append(ts.iloc[i-w:i])   
            seq_matrix=np.array(seq_matrix)
                
            # Min-max normalize the time subsequences
            seq_matrix_l= pd.DataFrame(seq_matrix).T
            seq_matrix_l=(seq_matrix_l-seq_matrix_l.min())/(seq_matrix_l.max()-seq_matrix_l.min())
            seq_matrix_l=seq_matrix_l.fillna(0) 
            seq_matrix_l=np.array(seq_matrix_l.T)
            seq_matrix_l=seq_matrix_l.reshape(len(seq_matrix_l),w,1)
                
            # Run the clustering algorithm on the training data 
            model.n_clusters=k
            model_clust = model.fit(seq_matrix_l)
            cl_train= model_clust.labels_
            
            # Get dummy set of cluster assignments
            cl_train=pd.Series(cl_train)
            cl_train=pd.get_dummies(cl_train).astype(int)
            cl_b=pd.DataFrame(columns=range(k))
            cl_final=pd.concat([cl_b,cl_train],axis=0)   
            cl_final=cl_final.fillna(0)
            
            # Get input matrix for test data 
            seq_matrix=[]
            for i in range(len(y_s),len(ts)):
                seq_matrix.append(ts.iloc[i-w:i])
            seq_matrix=np.array(seq_matrix)
                
            # Min-max normalize the time subsequences
            seq_matrix_l= pd.DataFrame(seq_matrix).T
            seq_matrix_l=(seq_matrix_l-seq_matrix_l.min())/(seq_matrix_l.max()-seq_matrix_l.min())
            seq_matrix_l=seq_matrix_l.fillna(0) 
            seq_matrix_l=np.array(seq_matrix_l.T)
            seq_matrix_l=seq_matrix_l.reshape(len(seq_matrix_l),w,1)
                
            # Use the clustering model from the training data
            # to assign test subsequences to a cluster
            cl_test = model_clust.predict(seq_matrix_l)
            y_test_seq = model_clust.predict(seq_matrix_l) # Copy cluster series
            
            # Get dummy set of cluster assignments
            cl_test=pd.Series(cl_test)
            cl_test=pd.get_dummies(cl_test).astype(int)
            y_t=pd.DataFrame(columns=range(k))
            cl_test=pd.concat([y_t,cl_test],axis=0)   
            cl_test=cl_test.fillna(0)  
                
            # Merge training and testing cluster assignments
            # and reset index to avoid misalignment
            clusters=pd.concat([cl_final,cl_test],axis=0,ignore_index=True)
            index=list(range(len(ts)-len(clusters), len(ts)))
            clusters.set_index(pd.Index(index),inplace=True)
                    
            # (2) Predictions 
            for ar in ar_test:
                
                # Function to get matrix with lags
                def lags(series):
                    last = series.iloc[-ar:].fillna(0)
                    return last.tolist() + [0] * (ar - len(last))
                
                # Get matrix with temporal lags for ts
                data_matrix = []
                for i in range(ar, len(ts) + 1):
                    data_matrix.append(lags(ts.iloc[:i]))
                in_put=pd.DataFrame(data_matrix)
                        
                # Set index to align with ts 
                index=list(range(len(ts)-len(in_put), len(ts)))
                in_put.set_index(pd.Index(index),inplace=True)
                
                # Merge lags with clusters
                # If clusters are longer, take as base to avoid misalignment
                if len(clusters)>=len(in_put):
                    in_put=pd.concat([clusters,in_put],axis=1)
                # If lags are longer, take as base to avoid misalignment
                else: 
                    in_put=pd.concat([in_put,clusters],axis=1)
                    
                # Fill missing values with zero
                # These oocur due to the potentially different start
                # indices in clusters and the input matrix
                in_put=in_put.fillna(0)
                            
                # Add X variables if available 
                if X is not None:
                    # Reset index to avoid misalignment
                    # Concat with X as base and delete missing observations
                    X=X.reset_index(drop=True)
                    in_put=pd.concat([X,in_put],axis=1)
                    in_put = in_put.dropna()
                
                # Make sure column names are character
                in_put.columns = in_put.columns.map(str)
                        
                # Obtain output
                # Reset index to avoid misalignment and 
                # crop output to the size of the input   
                output=Y.reset_index(drop=True)
                output=output[-len(in_put):]
        
                # Data split
                y_train = output[:-(len(ts)-int(0.7*len(ts)))]
                x_train = in_put[:-(len(ts)-int(0.7*len(ts)))]
        
                y_test = output[-(len(ts)-int(0.7*len(ts))):]        
                x_test = in_put[-(len(ts)-int(0.7*len(ts))):] 
                    
                # If optimization
                if grid is not None: 
                    # Optimize in trainig data, based on a 50/50 split
                    
                    # Get splits
                    val_train_ids = list(y_train[:int(0.5*len(y_train))].index)
                    val_test_ids = list(y_train[int(0.5*len(y_train)):].index)
                    splits = np.array([-1] * len(val_train_ids) + [0] * len(val_test_ids))
                    splits = PredefinedSplit(test_fold=splits)
                    grid_search = GridSearchCV(estimator=model_pred, param_grid=grid, cv=splits, verbose=0, n_jobs=-1)                       
                    grid_search.fit(x_train, y_train.values.ravel())
                    pred = grid_search.predict(x_test)
                
                # If no optimization
                else: 
                    model_pred.fit(x_train, y_train.values.ravel())
                    pred = model_pred.predict(x_test)                           

                # MSE
                error=((y_test.reset_index(drop=True)-pd.Series(pred))**2).mean()
         
                # If MSE is lower than test mse, update results
                if error<min_test:
                    min_test=error
                    para=[ar,k,w]
                    preds_final=pred
                    shapes=model_clust.cluster_centers_
                    seq=y_test_seq
                    # If all test sequences are assigned to one cluster,
                    # s score cannot be computed --> set to na
                    if y_test_seq.max()==0:
                        s=np.nan
                    else: 
                        s=silhouette_score(seq_matrix_l, y_test_seq, metric="dtw") 

    print(f"Final DRF {para}, with MSE: {min_test}")
    
    return({'pred':pd.Series(preds_final),'actuals':y_test.reset_index(drop=True),"shapes":shapes,"s":s,"clusters":seq})
    
                    

def clustering(ts, model=TimeSeriesKMeans(n_clusters=5,metric="dtw",max_iter_barycenter=100,verbose=0,random_state=0),cluster_n=[3,5,7],w_length=[5,6,7,8,9,10,11,12]):

    score_test=-1
    for k in cluster_n:
        for w in w_length:
            
            # Adapted from: https://github.com/ThomasSchinca/ShapeF/blob/Thomas_draft/functions.py
            
            # Create input matrix for clustering
            seq_matrix=[]
            for i in range(w,len(ts)):
                seq_matrix.append(ts.iloc[i-w:i])  
            seq_matrix=np.array(seq_matrix)
            seq_matrix_l= pd.DataFrame(seq_matrix).T
            seq_matrix_l=(seq_matrix_l-seq_matrix_l.min())/(seq_matrix_l.max()-seq_matrix_l.min())
            seq_matrix_l=seq_matrix_l.fillna(0) 
            seq_matrix_l=np.array(seq_matrix_l.T)
            seq_matrix_l=seq_matrix_l.reshape(len(seq_matrix_l),w,1)
            
            # Cluster subsequences            
            model.n_clusters=k
            model_clust = model.fit(seq_matrix_l)
            clusters=model_clust.labels_
            
            # Get s score
            s=silhouette_score(seq_matrix_l,clusters, metric="dtw")
            
            # If s score is larger than than test value, update results
            if s>score_test:
                score_test=s
                s_final=s
                shapes_final=model_clust.cluster_centers_
                seq_final=model_clust.labels_
                                            
    return({"shapes":shapes_final,"s":s_final,"clusters":seq_final})
    
            
        