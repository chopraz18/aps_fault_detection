import pandas as pd
from sensor.config import mongo_client
from sensor.logger import logging
from sensor.exception import SensorException
import os,sys
import yaml




def get_collection_as_dataframe(database_name:str,collection_name:str)->pd.DataFrame:
    """
    Description: This function return collection as dataframe
    =========================================================
    Params:
    database_name: database name
    collection_name: collection name
    =========================================================
    return Pandas dataframe of a collection
    """
    try:
        logging.info(f"reading data from database:{database_name} and collection:{collection_name}")
        df=pd.DataFrame(list(mongo_client[database_name][collection_name].find()))
        logging.info(f"found columns:{df.columns}")
        if "_id" in df.columns:
            logging.info(f"dropping columns: _id")
            df=df.drop("_id",axis=1)
        logging.info(f"columns and rows in df is {df.shape}")    
        return df

    except Exception as e:
        raise SensorException(e, sys)


def write_yaml_file(file_path,data:dict):
    try:
        file_dir=os.path.dirname(file_path)

        os.makedirs(file_dir,exist_ok=True)
        with open(file_path,'w') as file_writer:
            yaml.dump(data,file_writer)
