import pandas as pd
import pymongo
import json
from sensor.config import mongo_client

# Provide the mongodb localhost url to connect python to mongodb.
#


Data_file_path='/config/workspace/aps_failure_training_set1.csv'
Database_name='aps'
Collection_name='sensor'


if __name__=="__main__":
    df=pd.read_csv(Data_file_path)
    print(f'Rows,Columns:{df.shape}')
    json_data=list(json.loads(df.T.to_json()).values())
    #storing our data into mongodb
    mongo_client[Database_name][Collection_name].insert_many(json_data)
    
