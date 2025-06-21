This repository contains the replication material for "From Protests to Fatalities: Identifying Dangerous Temporal Patterns in Civil Conflict Transitions".

## Requirements
- The analysis is run in Python 3.12.3
- The required python libraries are listed in requirements.txt

## Descriptions of files 
- /data contains the datasets (df.csv) used for the analyses, as well as the predictions (df_linear.csv, df_nonlinear.csv) and extracted protest patterns (final_shapes_s.csv, ols_shapes.json, rf_shapes.json).
- /out contains the visualizations and tables contained in the paper. 
- data.py creates the dataset used for the analysis df.csv. 
- functions.py contains some functions used during the analyses. 
- main_prediction.py obtains predictions within-country. This file takes roughly three days to run. 
- main_regression.py obtains across-country protest patterns. 
- results_predictions.py creates the outputs for the prediction model. 
- results_regression.R runs the regression model. 
 


