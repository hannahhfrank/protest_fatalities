import pandas as pd
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import PredefinedSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from tslearn.clustering import TimeSeriesKMeans,silhouette_score

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
            
    df_filled = pd.DataFrame()
    for c in df[group].unique():
        df_s = train.loc[train[group] == c]
        df_imp = df_s[vars_input]

        # If completely missing
        if df_imp.isnull().all().all():
            df_imp_train_df=df_imp
            df_imp_trans_df=df[vars_input].loc[df["country"] == c]
            
        else: 
            # Train
            imputer = SimpleImputer(strategy='mean')
            imputer.fit(df_imp)
            df_imp_train = imputer.transform(df_imp)
            df_imp_train_df = pd.DataFrame(df_imp_train)
            df_imp_train_df.columns=vars_input
            
            # Test
            df_s = test.loc[test[group] == c]
            df_imp = df_s[vars_input]
            
            if df_imp.isnull().all().all():
                df_imp_test_df=df_imp.fillna(df_imp_train_df.mean().mean())
                
            else:
                imputer.fit(df_imp)
                df_imp_test = imputer.transform(df_imp)
                df_imp_test_df = pd.DataFrame(df_imp_test)    
                df_imp_test_df.columns=vars_input
        
            # Merge
            df_imp_trans_df = pd.concat([df_imp_train_df, df_imp_test_df])
        df_s = df.loc[df["country"] == c]
        df_filled = pd.concat([df_filled, df_imp_trans_df])
    
    df_filled.columns = vars_input
    feat_complete = df.drop(columns=vars_input)
    df_filled = df_filled.set_index(feat_complete.index)
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
    
    df_filled = pd.DataFrame()
    for c in df[group].unique():
        
        # Train
        df_s = train.loc[train[group] == c]
        df_imp = df_s[vars_input]
        df_imp = df_imp.copy(deep=True)
        df_imp_train_df = df_imp.interpolate(limit_direction="both")
        
        # Test
        df_s = test.loc[test[group] == c]
        df_imp = df_s[vars_input]
        df_imp = df_imp.copy(deep=True)        
        df_imp_test_df = df_imp.interpolate(limit_direction="forward")
        
        # Merge
        df_imp_trans_df = pd.concat([df_imp_train_df, df_imp_test_df])
        df_filled = pd.concat([df_filled, df_imp_trans_df])
    df_filled.columns = vars_input
    df_filled = df_filled.reset_index(drop=True)
    feat_complete = df.drop(columns=vars_input)
    out = pd.concat([feat_complete.reset_index(drop=True), df_filled.reset_index(drop=True)], axis=1)
    
    return out


def general_model(ts,
                  Y,
                  model_pred=RandomForestRegressor(random_state=0),
                  X: bool = None,
                  norm: bool=False,
                  metric: str='mse',
                  train_test_split=0.7,
                  opti_grid=None,
                  ar_test: list=[1,2,4,6,12]):
    
    if norm==True:
        mini = np.min(Y)
        maxi = np.max(Y)
        Y = (Y - mini) / (maxi - mini)
        Y=Y.fillna(0) 
               
    min_test=np.inf
    for ar in ar_test:
        try:        
            def lags(series):
                last = series.iloc[-ar:].fillna(0)
                return last.tolist() + [0] * (ar - len(last))
                    
            data_matrix = []
            for i in range(ar, len(ts) + 1):
                data_matrix.append(lags(ts.iloc[:i]))
                        
            cols_name=[]
            for i in range(ar):
                cols_name.append(f"t-{i}")  
            cols_name=cols_name[::-1]
            in_put=pd.DataFrame(data_matrix,columns=cols_name)
            output=Y.iloc[-len(in_put):]
                    
            if X is not None:
                X=X.iloc[-len(in_put):].reset_index(drop=True)
                in_put=pd.concat([X, in_put],axis=1)
                        
            y_train = output[:-(len(ts)-int(train_test_split*len(ts)))]
            x_train = in_put[:-(len(ts)-int(train_test_split*len(ts)))]
            
            y_test = output[-(len(ts)-int(train_test_split*len(ts))):]        
            x_test = in_put[-(len(ts)-int(train_test_split*len(ts))):]     
            
            if opti_grid is not None:
                val_train_index = list(y_train[:int(0.5*len(y_train))].index)
                val_test_index = list(y_train[int(0.5*len(y_train)):].index)
                splits = np.array([-1] * len(val_train_index) + [0] * len(val_test_index))
                ps = PredefinedSplit(test_fold=splits)
                grid_search = GridSearchCV(estimator=model_pred, param_grid=opti_grid, cv=ps, verbose=0, n_jobs=-1)
                grid_search.fit(x_train, y_train)
                pred = grid_search.predict(x_test)
                
            else: 
                model_pred.fit(x_train, y_train)
                pred = model_pred.predict(x_test)        
           
            if y_test.max()==0:
                error=((y_test.reset_index(drop=True)-pd.Series(pred))**2).mean()
    
            else:   
                if metric=="mae":
                    error=abs(y_test.reset_index(drop=True)-pd.Series(pred)).mean()
                elif metric=="mse":
                    error=((y_test.reset_index(drop=True)-pd.Series(pred))**2).mean()
                elif metric=="wmse":
                    error=mean_squared_error(y_test,pred,sample_weight=y_test)
                                
            if error<min_test:
                min_test=error
                para=[ar]
                preds_final=pred
        except:
            continue

    print(f"Final RF {para}, with {metric}: {min_test}")
    if norm==True: 
        y_revert = y_test * (maxi - mini) + mini
        y_pred_revert = pd.Series(preds_final) * (maxi - mini) + mini
        
        return({'rf_pred':pd.Series(preds_final),'actuals':y_test.reset_index(drop=True),
               'rf_pred_revert':y_pred_revert, 'actuals_revert':y_revert.reset_index(drop=True)})
    
    return({'rf_pred':pd.Series(preds_final),'actuals':y_test.reset_index(drop=True)})
            
        
def general_dynamic_model(y,
                    Y,
                    model_pred=RandomForestRegressor(random_state=0),
                    X: bool = None,
                    norm: bool=False,
                    model=TimeSeriesKMeans(n_clusters=5,metric="dtw",max_iter_barycenter=100,verbose=0,random_state=0),
                    metric: str='mse',
                    train_test_split=0.7,
                    opti_grid=None,
                    ar_test: list=[1,2,4,6,12],
                    cluster_n:list=[3,5,7],
                    w_length:list=[3,5,7,9]):
    
    y_cluster=y
    if norm==True:
        mini = np.min(Y)
        maxi = np.max(Y)
        Y = (Y - mini) / (maxi - mini)
        Y=Y.fillna(0) 
    
    # Clustering  
    min_test=np.inf
    for k in cluster_n:
        for w in w_length:
            model.n_clusters=k
            try: 
                # Training 
                y_s=y_cluster.iloc[:int(train_test_split*len(y_cluster))]
                seq_matrix=[]
                for i in range(w,len(y_s)):
                    seq_matrix.append(y_cluster.iloc[i-w:i])   
                seq_matrix=np.array(seq_matrix)
                    
                # Scaling
                seq_matrix_l= pd.DataFrame(seq_matrix).T
                seq_matrix_l=(seq_matrix_l-seq_matrix_l.min())/(seq_matrix_l.max()-seq_matrix_l.min())
                seq_matrix_l=seq_matrix_l.fillna(0) 
                seq_matrix_l=np.array(seq_matrix_l.T)
                seq_matrix_l=seq_matrix_l.reshape(len(seq_matrix_l),w,1)
                    
                # Clustering 
                model.n_clusters=k
                model_clust = model.fit(seq_matrix_l)
                cl_train= model_clust.labels_
                cl_train=pd.Series(cl_train)
                cl_train=pd.get_dummies(cl_train).astype(int)
                    
                cl_b=pd.DataFrame(columns=range(k))
                cl_final=pd.concat([cl_b,cl_train],axis=0)   
                cl_final=cl_final.fillna(0)
                
                # Test data 
                seq_matrix=[]
                for i in range(len(y_s),len(y_cluster)):
                    seq_matrix.append(y_cluster.iloc[i-w:i])
                seq_matrix=np.array(seq_matrix)
                    
                # Sacling
                seq_matrix_l= pd.DataFrame(seq_matrix).T
                seq_matrix_l=(seq_matrix_l-seq_matrix_l.min())/(seq_matrix_l.max()-seq_matrix_l.min())
                seq_matrix_l=seq_matrix_l.fillna(0) 
                seq_matrix_l=np.array(seq_matrix_l.T)
                seq_matrix_l=seq_matrix_l.reshape(len(seq_matrix_l),w,1)
                    
                cl_test = model_clust.predict(seq_matrix_l)
                y_test_seq = model_clust.predict(seq_matrix_l)
                cl_test=pd.Series(cl_test)
                cl_test=pd.get_dummies(cl_test).astype(int)
                    
                y_t=pd.DataFrame(columns=range(k))
                cl_test=pd.concat([y_t,cl_test],axis=0)   
                cl_test=cl_test.fillna(0)  
                    
                clusters=pd.concat([cl_final,cl_test],axis=0,ignore_index=True)
                index=list(range(len(y)-len(clusters), len(y)))
                clusters.set_index(pd.Index(index),inplace=True)
                    
                # Predictions 
                for ar in ar_test:
                    try:                            
                        def lags(series):
                            last = series.iloc[-ar:].fillna(0)
                            return last.tolist() + [0] * (ar - len(last))
                                
                        data_matrix = []
                        for i in range(ar, len(y) + 1):
                            data_matrix.append(lags(y.iloc[:i]))
                                    
                        cols_name=[]
                        for i in range(ar):
                            cols_name.append(f"t-{i}")  
                        cols_name=cols_name[::-1]
                        in_put=pd.DataFrame(data_matrix,columns=cols_name)
                                
                        index=list(range(len(y)-len(in_put), len(y)))
                        in_put.set_index(pd.Index(index),inplace=True)
                                    
                        if len(clusters)>=len(in_put):
                            in_put=pd.concat([clusters,in_put],axis=1)
                        else: 
                            in_put=pd.concat([in_put,clusters],axis=1)
                                    
                            in_put=in_put.fillna(0)
                                    
                        if X is not None:
                            X=X.reset_index(drop=True)
                            in_put=pd.concat([X,in_put],axis=1)
                            in_put = in_put.dropna()
                                
                        in_put.columns = in_put.columns.map(str)
                                
                        output=Y.reset_index(drop=True)
                        output=output[-len(in_put):]
                
                        y_train = output[:-(len(y)-int(train_test_split*len(y)))]
                        x_train = in_put[:-(len(y)-int(train_test_split*len(y)))]
                
                        y_test = output[-(len(y)-int(train_test_split*len(y))):]        
                        x_test = in_put[-(len(y)-int(train_test_split*len(y))):] 
                                
                        if opti_grid is not None: 
                            val_train_index = list(y_train[:int(0.5*len(y_train))].index)
                            val_test_index = list(y_train[int(0.5*len(y_train)):].index)
                            splits = np.array([-1] * len(val_train_index) + [0] * len(val_test_index))
                            ps = PredefinedSplit(test_fold=splits)
                            grid_search = GridSearchCV(estimator=model_pred, param_grid=opti_grid, cv=ps, verbose=0, n_jobs=-1)
                            grid_search.fit(x_train, y_train.values.ravel())
                            pred = grid_search.predict(x_test)
                            
                        else: 
                            model_pred.fit(x_train, y_train.values.ravel())
                            pred = model_pred.predict(x_test)                           
        
                        if y_test.max()==0:
                            error=((y_test.reset_index(drop=True)-pd.Series(pred))**2).mean()
                                        
                        else:
                            if metric=="mae":
                                error=abs(y_test.reset_index(drop=True)-pd.Series(pred)).mean()
                            elif metric=="mse":
                                error=((y_test.reset_index(drop=True)-pd.Series(pred))**2).mean()
                            elif metric=="wmse":
                                error=mean_squared_error(y_test,pred,sample_weight=y_test)
                                                    
                        if error<min_test:
                            min_test=error
                            para=[ar,k,w]
                            preds_final=pred
                            shapes=model_clust.cluster_centers_
                            seq=y_test_seq
                            if y_test_seq.max()==0:
                                s=np.nan
                            else: 
                                s=silhouette_score(seq_matrix_l, y_test_seq, metric="dtw") 

                    except:
                        continue
            except: 
                continue

    print(f"Final DRF {para}, with {metric}: {min_test}")
    
    if norm==True:
        y_revert = y_test * (maxi - mini) + mini
        y_pred_revert = pd.Series(preds_final) * (maxi - mini) + mini
        
        return({'drf_pred':pd.Series(preds_final),'actuals':y_test.reset_index(drop=True),
               'drf_pred_revert':y_pred_revert, 'actuals_revert':y_revert.reset_index(drop=True),
               "shapes":shapes,"s":s,"clusters":seq})
    
    return({'drf_pred':pd.Series(preds_final),'actuals':y_test.reset_index(drop=True),"shapes":shapes,"s":s})
                    

def clustering(y,
               model=TimeSeriesKMeans(n_clusters=5,metric="dtw",max_iter_barycenter=100,verbose=0,random_state=0),
               cluster_n:list=[3,5,7],
               w_length:list=[5,6,7,8,9,10,11,12]):

    score_test=-1
    for k in cluster_n:
        for w in w_length:
            model.n_clusters=k
            seq_matrix=[]
            for i in range(w,len(y)):
                seq_matrix.append(y.iloc[i-w:i])  
            seq_matrix=np.array(seq_matrix)
            seq_matrix_l= pd.DataFrame(seq_matrix).T
            seq_matrix_l=(seq_matrix_l-seq_matrix_l.min())/(seq_matrix_l.max()-seq_matrix_l.min())
            seq_matrix_l=seq_matrix_l.fillna(0) 
            seq_matrix_l=np.array(seq_matrix_l.T)
            model.n_clusters=k
            model_clust = model.fit(seq_matrix_l)
            clusters=model_clust.labels_
            s=silhouette_score(seq_matrix_l,clusters, metric="dtw")
            
            if s>score_test:
                score_test=s
                s_final=s
                shapes_final=model_clust.cluster_centers_
                seq_final=model_clust.labels_
                                            
    return({"shapes":shapes_final,"s":s_final,"clusters":seq_final})
    
            
        