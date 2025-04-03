import pandas as pd
from sklearn.model_selection import GridSearchCV
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.model_selection import PredefinedSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from tslearn.clustering import TimeSeriesKMeans,silhouette_score

def preprocess_min_max_group(df, x, group):
    out = pd.DataFrame()
    for i in df[group].unique():
        Y = df[x].loc[df[group] == i]
        min_val = np.min(Y)
        max_val = np.max(Y)
        Y = (Y - min_val) / (max_val - min_val)
        Y=Y.fillna(0) 
        out = pd.concat([out, pd.DataFrame(Y)], ignore_index=True)
    df[f"{x}_norm"] = out
    
###################
### Imputation ####
###################

def simple_imp_grouped(df, group, vars_input):
    
    ### Split data ###
    train = pd.DataFrame()
    test = pd.DataFrame()
    for c in df.country.unique():
        df_s = df.loc[df["country"] == c]
    
        # Train, test
        train_s = df_s[:int(0.7*len(df_s))]
        test_s = df_s[int(0.7*len(df_s)):]
    
        # Merge
        train = pd.concat([train, train_s])
        test = pd.concat([test, test_s])
            
    ### Training ###
    df_filled = pd.DataFrame()
    
    for c in df[group].unique():
                    
        df_s = train.loc[train[group] == c]
        feat_imp = df_s[vars_input]
        
        # If completely missing
        if feat_imp.isnull().all().all():
            df_MICE_train_df=feat_imp
            df_MICE_trans_df=df[vars_input].loc[df["country"] == c]
            
        else: 
            # Train
            imputer = SimpleImputer(strategy='mean')
            imputer.fit(feat_imp)
            df_MICE_train = imputer.transform(feat_imp)
            df_MICE_train_df = pd.DataFrame(df_MICE_train)
            df_MICE_train_df.columns=vars_input
            
            # Test
            df_s = test.loc[test[group] == c]
            feat_imp = df_s[vars_input]
            
            if feat_imp.isnull().all().all():
                df_MICE_test_df=feat_imp.fillna(df_MICE_train_df.mean().mean())
                
            else:
                imputer.fit(feat_imp)
                df_MICE_test = imputer.transform(feat_imp)
                df_MICE_test_df = pd.DataFrame(df_MICE_test)    
                df_MICE_test_df.columns=vars_input
        
            # Merge
            df_MICE_trans_df = pd.concat([df_MICE_train_df, df_MICE_test_df])
        df_s = df.loc[df["country"] == c]
       
        df_filled = pd.concat([df_filled, df_MICE_trans_df])
    
    # Merge
    df_filled.columns = vars_input
    feat_complete = df.drop(columns=vars_input)
    df_filled = df_filled.set_index(feat_complete.index)
    _ = pd.concat([feat_complete, df_filled], axis=1)
    _=_.reset_index(drop=True)
    
    return _

def linear_imp_grouped(df, group, vars_input):
    
    ### Split data ###
    train = pd.DataFrame()
    test = pd.DataFrame()
    for c in df.country.unique():
        df_s = df.loc[df["country"] == c]

        # Train, test
        train_s = df_s[:int(0.7*len(df_s))]
        test_s = df_s[int(0.7*len(df_s)):]

        # Merge
        train = pd.concat([train, train_s])
        test = pd.concat([test, test_s])
    
    df_filled = pd.DataFrame()

    for c in df[group].unique():
        
        # Train
        df_s = train.loc[train[group] == c]
        feat_imp = df_s[vars_input]
        df_MICE = feat_imp.copy(deep=True)
        df_MICE_train_df = df_MICE.interpolate(limit_direction="both")
        
        # Test
        df_s = test.loc[test[group] == c]
        feat_imp = df_s[vars_input]
        df_MICE = feat_imp.copy(deep=True)        
        df_MICE_test_df = df_MICE.interpolate(limit_direction="forward")
        
        # Merge
        df_MICE_trans_df = pd.concat([df_MICE_train_df, df_MICE_test_df])
        df_filled = pd.concat([df_filled, df_MICE_trans_df])
        
    # Merge
    df_filled.columns = vars_input
    df_filled = df_filled.reset_index(drop=True)
    feat_complete = df.drop(columns=vars_input)
    _ = pd.concat([feat_complete.reset_index(drop=True), df_filled.reset_index(drop=True)], axis=1)
    
    return _

#########################
### Prediction models ###        
#########################

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
        min_val = np.min(Y)
        max_val = np.max(Y)
        Y = (Y - min_val) / (max_val - min_val)
        Y=Y.fillna(0) # if seq uniform 
               
    # Initiate test metric
    min_test=np.inf
    for ar in ar_test:
        try:        
            # Get lags
            #ar=ar+1
            def lags(series):
                last = series.iloc[-ar:].fillna(0)
                return last.tolist() + [0] * (ar - len(last))
                    
            data_matrix = []
            for i in range(ar, len(ts) + 1):
                data_matrix.append(lags(ts.iloc[:i]))
                        
            # Columns names
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
                # Validation
                val_train_index = list(y_train[:int(0.5*len(y_train))].index)
                val_test_index = list(y_train[int(0.5*len(y_train)):].index)
                
                splits = np.array([-1] * len(val_train_index) + [0] * len(val_test_index))
                ps = PredefinedSplit(test_fold=splits)
                grid_search = GridSearchCV(estimator=model_pred, param_grid=opti_grid, cv=ps, verbose=0, n_jobs=-1)
                grid_search.fit(x_train, y_train)
                #best_params = grid_search.best_params_
                #model_fit = RandomForestRegressor(**best_params,random_state=0)
                grid_search.fit(x_train, y_train)
                pred = grid_search.predict(x_test)
                
            else: 
                model_pred.fit(x_train, y_train)
                pred = model_pred.predict(x_test)        
           
            if y_test.max()==0:
                error=((y_test.reset_index(drop=True)-pd.Series(pred))**2).mean()
    
            else:   
                # Return final error and order 
                if metric=="mae":
                    # Get MAE
                    error=abs(y_test.reset_index(drop=True)-pd.Series(pred)).mean()
                elif metric=="mse":
                    # Get MSE
                    error=((y_test.reset_index(drop=True)-pd.Series(pred))**2).mean()
                elif metric=="wmse":
                    # Get WMSE
                    error=mean_squared_error(y_test,pred,sample_weight=y_test)
                                
            # If ar_met smaller than min_test
            if error<min_test:
                # Update min_test
                min_test=error
                # Update best parameters
                para=[ar]
                # Update model
                preds_final=pred
                #print(f'Best RF: {para}, with {metric}: {min_test}')
        except:
            continue

    print(f"Final RF {para}, with {metric}: {min_test}")
    if norm==True: 
        y_revert = y_test * (max_val - min_val) + min_val
        y_pred_revert = pd.Series(preds_final) * (max_val - min_val) + min_val
        
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
                    test_clu:list=[3,5,7],
                    test_win:list=[3,5,7,9]):
    
    y_clu=y
    
    if norm==True:
        min_val = np.min(Y)
        max_val = np.max(Y)
        Y = (Y - min_val) / (max_val - min_val)
        Y=Y.fillna(0) # if seq uniform 
    
    ##################################
    ### Clustering hyperparameters ###
    ##################################

    # Initiate test metric
    min_test=np.inf
    # For numer of clusters in test_clus
    for n_clu in test_clu:
        # For window length in test_win
        for number_s in test_win:
            # Update number of clusters in model 
            model.n_clusters=n_clu
            try: 
                #print(f"Test parameters {n_clu,number_s}")
        
                #################
                ## Clustering ###
                #################
                        
                # Training data
                ex=y_clu.iloc[:int(train_test_split*len(y_clu))]
                ts_seq=[]
                    
                ### Training data ###
                # Make list of lists, 
                # each sub-list contains number_s observations
                for i in range(number_s,len(ex)):
                    ts_seq.append(y_clu.iloc[i-number_s:i])
                        
                # Convert into array,
                # each row is a time series of number_s observations     
                ts_seq=np.array(ts_seq)
                    
                # Scaling
                ts_seq_l= pd.DataFrame(ts_seq).T
                ts_seq_l=(ts_seq_l-ts_seq_l.min())/(ts_seq_l.max()-ts_seq_l.min())
                ts_seq_l=ts_seq_l.fillna(0) # if seq uniform 
                ts_seq_l=np.array(ts_seq_l.T)
                                
                # Reshape array,
                # each sub array contains times series of number_s observations
                ts_seq_l=ts_seq_l.reshape(len(ts_seq_l),number_s,1)
                    
                # Clustering and convert into dummy set
                model.n_clusters=n_clu
                m_dba = model.fit(ts_seq_l)
                cl= m_dba.labels_
                cl=pd.Series(cl)
                cl=pd.get_dummies(cl).astype(int)
                    
                # Make sure that length of dummy set is equal to n_clu
                # If not, add empty column 
                cl_b=pd.DataFrame(columns=range(n_clu))
                cl=pd.concat([cl_b,cl],axis=0)   
                cl=cl.fillna(0)
                
                ### Test data ###
                ts_seq=[]
                
                # Make list of lists, 
                # each sub-list contains number_s observations
                for i in range(len(ex),len(y_clu)):
                    ts_seq.append(y_clu.iloc[i-number_s:i])
                        
                # Convert into array,
                # each row is a time series of number_s observations       
                ts_seq=np.array(ts_seq)
                    
                # Sacling
                ts_seq_l= pd.DataFrame(ts_seq).T
                ts_seq_l=(ts_seq_l-ts_seq_l.min())/(ts_seq_l.max()-ts_seq_l.min())
                ts_seq_l=ts_seq_l.fillna(0) # if seq uniform 
                ts_seq_l=np.array(ts_seq_l.T)
                            
                # Reshape array,
                # each sub array contains times series of number_s observations
                ts_seq_l=ts_seq_l.reshape(len(ts_seq_l),number_s,1)
                    
                # Use trained model to predict clusters in test data
                # and convert into dummy set
                y_test = m_dba.predict(ts_seq_l)
                y_test_seq = m_dba.predict(ts_seq_l)
                y_test=pd.Series(y_test)
                y_test=pd.get_dummies(y_test).astype(int)
                    
                # Make sure that length of dummy set is equal to n_clu
                # If not, add empty column 
                y_t=pd.DataFrame(columns=range(n_clu))
                y_test=pd.concat([y_t,y_test],axis=0)   
                y_test=y_test.fillna(0)  
                    
                clusters=pd.concat([cl,y_test],axis=0,ignore_index=True)
                index=list(range(len(y)-len(clusters), len(y)))
                clusters.set_index(pd.Index(index),inplace=True)
                    
                ###################
                ### Predictions ###
                ###################
                            
                for ar in ar_test:
                    
                    try:                            
                        # Get lags
                        #ar=ar+1
                        def lags(series):
                            last = series.iloc[-ar:].fillna(0)
                            return last.tolist() + [0] * (ar - len(last))
                                
                        data_matrix = []
                        for i in range(ar, len(y) + 1):
                            data_matrix.append(lags(y.iloc[:i]))
                                    
                        # Columns names
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
                                
                        # Validation
                        if opti_grid is not None: 
                            val_train_index = list(y_train[:int(0.5*len(y_train))].index)
                            val_test_index = list(y_train[int(0.5*len(y_train)):].index)
                            splits = np.array([-1] * len(val_train_index) + [0] * len(val_test_index))
                            ps = PredefinedSplit(test_fold=splits)
                            grid_search = GridSearchCV(estimator=model_pred, param_grid=opti_grid, cv=ps, verbose=0, n_jobs=-1)
                            grid_search.fit(x_train, y_train.values.ravel())
                            #best_params = grid_search.best_params_
                                    
                            # Fit
                            grid_search.fit(x_train, y_train.values.ravel())
                            pred = grid_search.predict(x_test)
                            
                        else: 
                            # Fit
                            model_pred.fit(x_train, y_train.values.ravel())
                            pred = model_pred.predict(x_test)                           
        
                        if y_test.max()==0:
                            error=((y_test.reset_index(drop=True)-pd.Series(pred))**2).mean()
                                        
                        else:
                           
                            # Return final error and order 
                            if metric=="mae":
                                # Get MAE
                                error=abs(y_test.reset_index(drop=True)-pd.Series(pred)).mean()
                            elif metric=="mse":
                                # Get MSE
                                error=((y_test.reset_index(drop=True)-pd.Series(pred))**2).mean()
                            elif metric=="wmse":
                                # Get WMSE
                                error=mean_squared_error(y_test,pred,sample_weight=y_test)
                                                    
                        # If ar_met smaller than min_test
                        if error<min_test:
                            # Update min_test
                            min_test=error
                            # Update best parameters
                            para=[ar,n_clu,number_s]
                            # Update model
                            preds_final=pred
                            shapes=m_dba.cluster_centers_
                            seq=y_test_seq
                            if y_test_seq.max()==0:
                                s=np.nan
                            else: 
                                s=silhouette_score(ts_seq_l, y_test_seq, metric="dtw") 

                    except:
                        continue
            except: 
                continue

    print(f"Final DRF {para}, with {metric}: {min_test}")
    
    if norm==True:
        y_revert = y_test * (max_val - min_val) + min_val
        y_pred_revert = pd.Series(preds_final) * (max_val - min_val) + min_val
        
        return({'drf_pred':pd.Series(preds_final),'actuals':y_test.reset_index(drop=True),
               'drf_pred_revert':y_pred_revert, 'actuals_revert':y_revert.reset_index(drop=True),
               "shapes":shapes,"s":s,"clusters":seq})
    
    return({'drf_pred':pd.Series(preds_final),'actuals':y_test.reset_index(drop=True),"shapes":shapes,"s":s})
                    

def clustering(y,
               model=TimeSeriesKMeans(n_clusters=5,metric="dtw",max_iter_barycenter=100,verbose=0,random_state=0),
               test_clu:list=[3,5,7],
               test_win:list=[5,6,7,8,9,10,11,12]):

    score_test=-1

    # For numer of clusters in test_clus
    for n_clu in test_clu:
        # For window length in test_win
        for number_s in test_win:
           
            model.n_clusters=n_clu
            ts_seq=[]
            for i in range(number_s,len(y)):
                ts_seq.append(y.iloc[i-number_s:i])  
            ts_seq=np.array(ts_seq)
                    
            ts_seq_l= pd.DataFrame(ts_seq).T
            ts_seq_l=(ts_seq_l-ts_seq_l.min())/(ts_seq_l.max()-ts_seq_l.min())
            ts_seq_l=ts_seq_l.fillna(0) 
            ts_seq_l=np.array(ts_seq_l.T)
                                    
            model.n_clusters=n_clu
            m_dba = model.fit(ts_seq_l)
            cl=m_dba.labels_
            
            s=silhouette_score(ts_seq_l,cl, metric="dtw")

            if s>score_test:
                score_test=s
                s_final=s
                shapes_final=m_dba.cluster_centers_
                seq_final=m_dba.labels_
                                            
    return({"shapes":shapes_final,"s":s_final,"clusters":seq_final})
    
            
        