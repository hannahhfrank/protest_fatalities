This repository contains the replication material for "From Protests to Fatalities: Identifying Dangerous Temporal Patterns in Civil Conflict Transitions".

## Requirements
- The analysis is run in Python 3.10.13 and R version 4.3.1.
- The required python libraries are listed in requirements.txt. The following packages need to be installed in R: sandwich, stargazer, lmtest, and car.

## Description of files 
- /data contains the dataset (df.csv) used for the analysis, as well as the predictions (for the dynamic models: preds_dynamic_nonlinear.csv and preds_dynamic_linear.csv, for the static models: preds_static_nonlinear.csv and preds_static_linear.csv and two merged versions: df_linear.csv and df_nonlinear.csv), extracted protest patterns (ols_shapes_reg.json, ols_shapes.json, rf_shapes.json), and the dataset with the within-country protest patterns (cluster_reg.csv) and the cross-country protest patterns used for the regression analysis (final_shapes_s.csv).
- /out contains the visualizations and tables contained in the paper. 
- data.py creates the dataset used for the analysis data/df.csv. 
- functions.py contains the functions used during the analysis. 
- main_prediction.py obtains predictions within-country. 
- main_regression.py obtains cross-country protest patterns. 
- results_predictions.py creates the outputs for the prediction model. 
- results_regression.R runs the regression model. 

## Replication instructions
First create a virtual environment, activate the environment and install the libraries. 

```
conda create -n protest_fatalities python=3.10.13
conda activate protest_fatalities
pip install -r requirements.txt
```

Then run the main files after each other. This will take approximately 6 hours.

```
python main_prediction.py
python main_regression.py
```

The final results are produced by running the results files. 

 ```
python results_predictions.py
Rscript results_regression.R
```

Adjust the out-paths and working directory as required. 
