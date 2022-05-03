from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from azureml.core import Workspace, Dataset
from azureml.core import Run
import numpy as np
import pandas as pd



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

   
run = Run.get_context()

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



    X_train , X_test , y_train,y_test = train_test_split(data_df,data_df["SOLD_PRICE"],test_size=0.2, random_state=42, stratify=data_df[["STATUS","MATERIAL_NR"]])
 
    print('Training Done')
    # Rate Model
    X = X_train.groupby('STATUS')['SOLD_PRICE'].mean()
    y = pd.DataFrame()
    y["STATUS"] = X.index
    y["predicted"] = X.values

    X_train = X_train.merge(y, on=['STATUS'])
    X_train = X_train.rename(columns={"SOLD_PRICE" : "ground_truth"})
    X_test = X_test.merge(y, on=['STATUS'])
    X_test = X_test.rename(columns={"SOLD_PRICE" : "ground_truth"})

    X_test = calculate_and_log_metrics(X_test, data_name='test')
    X_train = calculate_and_log_metrics(X_train, data_name='train')
    
    print('Finished Training')
