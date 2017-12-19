import os
import pandas as pd
import sklearn as sk
import math
import os
from sklearn import metrics, model_selection
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut, GridSearchCV
from sklearn import tree, metrics
from scipy.stats import sem


os.chdir("C:\\Users\\Daniel\\Documents\\Git Repos\\big_mart")
train = pd.read_csv(os.path.realpath(".\\Data\\mart_train.csv"))
test_final = pd.read_csv(os.path.realpath(".\\Data\\mart_test.csv"))

#train["Outlet_Size"] = train["Outlet_Size"].fillna("NA")
train["Item_Weight"] = train["Item_Weight"].fillna(train.mean()["Item_Weight"]) 
train, valid = model_selection.train_test_split(train, test_size  = 0.2, random_state = 42)

#test_final["Outlet_Size"] = test_final["Outlet_Size"].fillna("NA")
test_final["Item_Weight"] = test_final["Item_Weight"].fillna(test_final.mean()["Item_Weight"]) 

print(train)

########ONEHOT ENCODING FUNCTIONS#########

def enc_class(row, string):
    if(row == string):
        return(1)
    else: return(0)

def onehot(col, levels):
    foo = pd.DataFrame()
    for lev in levels:
        foo[lev] = col.apply(enc_class, string = lev)
        
    return(foo)


################################################
################# USING h2o ####################
################################################

#Basic Random Forest
import h2o

h2o.init(max_mem_size = "4G")
h2o.remove_all()

from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o import export_file
from h2o.grid.grid_search import H2OGridSearch

train_h2o = h2o.H2OFrame(train) 
valid_h2o = h2o.H2OFrame(valid)
test_final_h2o = h2o.H2OFrame(test_final)

train_h2o_X = train_h2o.col_names[:-1]
train_h2o_y = train_h2o.col_names[-1]

rf_base = H2ORandomForestEstimator(
            model_id = "rf_model_1",
            ntrees = 200,
            stopping_rounds = 2,
            score_each_iteration = True,
            seed = 42)

rf_base.train(train_h2o_X, train_h2o_y, training_frame = train_h2o, validation_frame = valid_h2o)

rf_base

preds = rf_base.predict(test_final_h2o)
preds.col_names = ["Item_Outlet_Sales"]

h2o_submission = test_final_h2o[["Item_Identifier", "Outlet_Identifier"]].cbind(preds)

h2o.export_file(h2o_submission, os.path.realpath(".\\Data\\rfpreds1.csv"))

#gridsearch random forest

rf_params = {'mtries':[3,4,5,6],
             'max_depth':[5,6,7,8,9,10],
             'ntrees':[100,150,200,250,300]
    }

rf_grid1 = H2OGridSearch(model = H2ORandomForestEstimator(stopping_rounds = 2, score_each_iteration = True, seed = 1000),
                         grid_id = "rf_grid1",
                         hyper_params = rf_params)

rf_grid1.train(x = train_h2o_X, 
               y = train_h2o_y, 
               training_frame = train_h2o,
               validation_frame = valid_h2o)

rf_by_rmse = rf_grid1.get_grid(sort_by = 'rmse', decreasing = False)
rf_best = rf_by_rmse.models[0]

preds_grid = rf_best.predict(test_final_h2o)
preds_grid.col_names = ["Item_Outlet_Sales"]

h2o_submission_grid = test_final_h2o[["Item_Identifier", "Outlet_Identifier"]].cbind(preds_grid)
h2o.export_file(h2o_submission_grid, os.path.realpath(".\\Data\\rf_grid_preds2.csv"))

##stop the cluster
h2o.cluster().shutdown()

#Gradient boosting estimator gridsearch
from h2o.estimators import H2OGradientBoostingEstimator

gbe_params = {'max_depth' :  [5,6,7,8,9,10],
              'ntrees' : [20,50,70,100,200],
              'learn_rate' : [0.08,0.09, 0.1, 0.11, 0.12]
            }

gbe_grid = H2OGridSearch(model = H2OGradientBoostingEstimator(stopping_tolerance = 0.01, stopping_rounds = 2, score_each_iteration = True, model_id = "gbe_grid", seed = 42),
                         grid_id = "gbe_grid",
                         hyper_params = gbe_params)

gbe_grid.train(x = train_h2o_X, 
               y = train_h2o_y, 
               training_frame = train_h2o,
               validation_frame = valid_h2o)

gbe_by_mse = gbe_grid.get_grid(sort_by = 'rmse')
gbe_best = gbe_by_mse.models[2]

preds_gbm_grid = gbe_best.predict(test_final_h2o)
preds_gbm_grid.col_names = ["Item_Outlet_Sales"]

h2o_submission_grid = test_final_h2o[["Item_Identifier", "Outlet_Identifier"]].cbind(preds_gbm_grid)
h2o.export_file(h2o_submission_grid, os.path.realpath(".\\Data\\gbe_grid_preds1.csv"))

############average predictions from gbe and rf###############

gbepreds, rfpreds = pd.read_csv(os.path.realpath("./Data/gbe_grid_preds1.csv")), pd.read_csv(os.path.realpath("./Data/rf_grid_preds2.csv"))

avg_preds = pd.concat([gbepreds[[0,1]],(gbepreds["Item_Outlet_Sales"] + rfpreds["Item_Outlet_Sales"])/2], axis = 1) 

avg_preds.to_csv(os.path.realpath("./Data/rf_rbe_avg1.csv"))
