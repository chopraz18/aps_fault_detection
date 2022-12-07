import sys,os
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.entity import config_entity
import pandas as pd 
from sensor import utils
class DataValidation:

    def __init__(self,data_validation_config:config_entity.DataValidationConfig(training_pipeline_config)):
        try:
            loging.info(f"{'<<'*20} Data Validation {'>>'*20}")
            self.data_validation_config=data_validation_config
            self.validation_error=dict()
        except Exception as e:
            raise SensorException(e, sys)    


    def drop_missing_values_columns(self,df:pd.DataFrame)->pd.DataFrame:
        """
        This function will drop column which contains missing values more than specified threshold

        df: Accepts a pandas dataframe
        threshold: Percentage criteria to drop a column
        =====================================================================
        returns Pandas DataFrame if atleast a single column is available after mssing values columns drop else None
        """

        try:
            threshold=self.data_validation_config.missing_threshold
            null_report=df.isna().sum()/df.shape[0]
            #selecting columns names containing null values
            drop_column_names=null_report[null_report>threshold].index
            self.validation_error['dropped_columns']=drop_column_names
            df.drop(list(drop_column_names),axis=1,inplace=True)

            #return None no columns left
            if len(df.columns)==0:
                return None
            else:
                return df    
            
        except Exception as e:
            raise SensorException(e, sys)

    def is_required_columns_exists(self,base_df:pd.DataFrame,current_df:pd.DataFrame)->bool:
        try:
            base_columns=base_df.columns
            current_columns=current_df.columns
            
            missing_columns=[]
            for base_column in base_columns:
                if base_column not in current_columns:
                    missing_columns.append(base_column)

            if len(missing_columns)>0:
                self.validation_error['missing_columns']=missing_columns
                return False
            else:
                return True        

        except Exception as e:
            raise SensorException(e, sys)            


    def data_drift(self,base_df:pd.DataFrame,current_df:pd.DataFrame):
        try:
            drift_report=dict()

            base_columns=base_df.columns
            current_columns=current_df.columns

            for base_column in base_columns:
                base_data,current_data= base_df[base_column],current_df[base_column]
                #Null hypothese is that both column data drawn is of same distribution
                same_distribution=ks_2samp(base_data,current_data)

                if same_distribution.pvalue>0.05:
                    #we fail to reject null hypothesis
                    drift_report[base_column]={
                        "pvalues":same_distribution.pvalue,
                        "same_distribution":True
                    }
                    #same distribution
                else:
                    pass
                    #different distribution

        except Exception as e:
            raise SensorException(e,sys)



    def initiate_data_validation(self):
        try:
            pass

        except Exception as e:
            raise SensorException(e, sys)