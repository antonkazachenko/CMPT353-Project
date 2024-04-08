# remove useless colomns, modify some features and prepare data file for usage by models
import pandas as pd

data = pd.read_csv('/modified_normalized_data.csv')

data = data.drop(columns=['type','nameOrig','nameDest'])
columns_to_replace = ['type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']
data[columns_to_replace] = data[columns_to_replace].replace({True:1,False:0})
isFraud_col = data.pop('isFraud')
data['isFraud'] = isFraud_col

data.to_csv('clean_data.csv',index=False)