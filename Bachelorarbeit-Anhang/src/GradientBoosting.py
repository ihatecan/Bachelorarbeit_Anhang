from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from azureml.core import Workspace, Dataset
from azureml.core import Run
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder

def calculate_and_log_metrics(data_df, data_name='test'):
    data_df['error'] = np.abs(data_df.ground_truth - data_df.predicted)
    data_df['error_rel'] = data_df.error / data_df.ground_truth
    data_df['success'] = data_df.error_rel <= 0.1 

    run.log('sucess_rate_'+data_name, data_df.success.mean())
    run.log('median_relative_error_'+data_name, data_df.error_rel.median())



    # Count per Status Group
    counts_per_status_group_df = data_df.groupby('STATUS').FINANCIAL_VEHICLE_ID.count().reset_index().rename(columns={'FINANCIAL_VEHICLE_ID': 
    'COUNTS_PER_GROUP'})    
    print(counts_per_status_group_df)
    print(counts_per_status_group_df.to_dict())
    run.log_table(name='Counts_per_Status_'+data_name, value=counts_per_status_group_df.to_dict(orient='list'))

    # Median Success-Rate per Status Group
    sucess_per_status_group_df = data_df.groupby('STATUS').success.mean().reset_index().rename(columns={'success': 'MEAN_SUCESS'})
    run.log_table(name='Sucess_Per_Status_'+data_name, value=sucess_per_status_group_df.to_dict(orient='list'))

    # Median-Error-Rate per Status Group
    error_per_status_group_df = data_df.groupby('STATUS').error_rel.median().reset_index().rename(columns={'error_rel': 'MEDIAN_ERROR'})
    run.log_table(name='Error_Per_Status_'+data_name, value=error_per_status_group_df.to_dict(orient='list'))

    return data_df

def reduce_feature_values(data_df, column, number_of_feature_values):

   all_features = data_df[column].value_counts().index

   data_df[column] = np.where(data_df[column].isin(all_features[0:min(len(all_features), number_of_feature_values)]), data_df[column], 'other')

   return data_df


def one_hot_encode_feature(df, feature_name, feature_list):
    ohe = OneHotEncoder()
    transformed = ohe.fit_transform(df[[feature_name]])
    df[[feature_name + '_' + x for x in ohe.categories_[0]]] = pd.DataFrame(transformed.toarray(),  index=df.index)
    feature_list += [feature_name + '_' + x for x in ohe.categories_[0]]
    feature_list = [x for x in feature_list if x != feature_name]
    feature_list = [x for x in feature_list if x != feature_name+'_encoded']
    return df, feature_list

run = Run.get_context()

def median_rel_error_metric(predt, dtrain):
    ''' Root mean squared log error metric.'''
    y = dtrain.get_label()
    success_rate = np.median((np.abs(predt-y) / y))
    return 'median_rel_error', success_rate

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str)
    args = parser.parse_args()
        
    print('Getting Data')
    ws = run.experiment.workspace
    dataset = Dataset.get_by_id(ws, id=args.input_data)
    data_df = dataset.to_pandas_dataframe()

    data_df = data_df[data_df["CUSTOMER_TYPE"] == "End-Customer"]
    data_df = data_df[data_df["STATUS"] > 1]
    data_df = data_df[data_df["SOLD_PRICE"] > 100]
    
    data_df = reduce_feature_values(data_df,column="SOLD_COUNTRY", number_of_feature_values=10)

    
    features = ['OPERATING_HOURS', 'STATUS', 'AGE', 'CONSTRUCTION_YEAR' , "BLACK_FORX_FLAG","SOLD_COUNTRY", "LEASING_FLAG","KEY_ACCOUNT_FLAG","SOLD_PACKAGE_SIZE",'CAPACITY','WHEEL_QT','MAST_HEIGHT',"MAST_TYPE","MATERIAL_NR"]

    data_df, features = one_hot_encode_feature(data_df, 'MAST_TYPE', features)
    data_df, features = one_hot_encode_feature(data_df, 'SOLD_COUNTRY', features)
    data_df, features = one_hot_encode_feature(data_df, 'SOLD_PACKAGE_SIZE', features)
    data_df, features = one_hot_encode_feature(data_df, 'MATERIAL_NR', features)


    features = [x for x in features if x not in ['SOLD_YEAR', 'TRADER_ID', 'CUSTOMER_TYPE_encoded', 'BLACK_FORX_FLAG']]
    
    X_train , X_test , y_train,y_test = train_test_split(data_df,data_df[["STATUS","SOLD_PRICE"]],test_size=0.2, random_state=42, stratify=data_df[["STATUS","MATERIAL_NR"]])

     
    progress = {}
    train_dmatrix = xgb.DMatrix(data=X_train[features], label=y_train["SOLD_PRICE"])
    test_dmatrix = xgb.DMatrix(data=X_test[features], label=y_test["SOLD_PRICE"])
    model = xgb.train({'tree_method': 'hist',
        'disable_default_eval_metric': 1},
        dtrain=train_dmatrix,
        feval = median_rel_error_metric,
        evals=[(train_dmatrix, 'dtrain'), (test_dmatrix, 'dtest')],
        evals_result=progress
    )
    predictions = model.predict(test_dmatrix)

    print('Training Done')
    # Rate Model
    X_test['ground_truth'] = y_test["SOLD_PRICE"]
    X_test['predicted'] = predictions
    X_test = calculate_and_log_metrics(X_test, data_name='test')

    predictions =  model.predict(train_dmatrix)
    X_train['ground_truth'] = y_train["SOLD_PRICE"]
    X_train['predicted'] = predictions
    X_train = calculate_and_log_metrics(X_train, data_name='train')
    print('Training Done')
    # Rate Model
    
    print('Finished Training')
