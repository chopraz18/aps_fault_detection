import sys,os
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.entity import artifact_entity,config_entity
from sklearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd 
from sensor import utils
from typing import Optional
from sensor.config import Target_Column


class DataTransformation:
    def __init__(self,data_transformation_config:config_entity.DataTransformationConfig,data_ingestion_artifact:artifact_entity.DataIngestionArtifact):

        try:

            self.data_transformation_config=data_transformation_config
            self.data_ingestion_artifact=data_ingestion_artifact
        except Exception as e:

            raise SensorException(e, sys) 

                     
        

    @classmethod
    def get_data_transformer_object(cls)->Pipeline:
        try:
            simple_imputer = SimpleImputer(strategy="constant", fill_value=0)
            robust_scaler = RobustScaler()

            pipeline = Pipeline(steps=[
                ('Imputer',simple_imputer),
                ('RobustScaler',robust_scaler)
            ]) 
            return pipeline          
        except Exception as e:
            raise SensorException(e, sys)


    def initiate_data_transformation(self)->artifact_entity.DataTransformationArtifact:
        try:
            #reading training and test file
            train_df=pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df=pd.read_csv(self.data_ingestion_artifact.test_file_path)

            #selecting input feature for train and test dataset
            input_feature_train_df=train_df.drop(Target_Column,axis=1)
            input_feature_test_df=test_df.drop(Target_Column,axis=1)

            #selecting target feature for train and test dataset    
            target_feature_train_df=train_df[Target_Column]
            target_feature_test_df=test_df[Target_Column]
            
            #encoding target feature for train and test dataset
            label_encoder=LabelEncoder()
            target_feature_train_arr=label_encoder.fit_transform(target_feature_train_df)
            target_feature_test_arr=label_encoder.transform(target_feature_test_df)

            #transforming input features
            transformation_pipeline = DataTransformation.get_data_transformer_object()
            input_feature_train_arr = transformation_pipeline.fit_transform(input_feature_train_df)
            input_feature_test_arr = transformation_pipeline.transform(input_feature_test_df) 

            #resampling of data
            smt=SMOTETomek(sampling_strategy="minority")
            logging.info(f"Before Resampling, Train_data_input_shape : {input_feature_train_arr.shape} , Train_data_target_shape : {target_feature_train_arr.shape}")
            input_feature_train_arr, target_feature_train_arr=smt.fit_resample(X=input_feature_train_arr, y=target_feature_train_arr)
            logging.info(f"After Resampling, Train_data_input_shape : {input_feature_train_arr.shape} , Train_data_target_shape : {target_feature_train_arr.shape}")

            logging.info(f"Before Resampling, Test_data_input_shape : {input_feature_train_arr.shape} , Test_data_target_shape : {target_feature_train_arr.shape}")
            input_feature_test_arr, target_feature_test_arr=smt.fit_resample(X=input_feature_test_arr, y=target_feature_test_arr)
            logging.info(f"After Resampling, Test_data_input_shape : {input_feature_train_arr.shape} , Test_data_target_shape : {target_feature_train_arr.shape}")

            #concating input and target arrays to train and test array
            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]
            

            #save numpy array
            utils.save_np_array(file_path=self.data_transformation_config.transformed_train_path, array=train_arr)
            utils.save_np_array(file_path=self.data_transformation_config.transformed_test_path,array=test_arr)

            #save pipeline object
            utils.save_object(file_path=self.data_transformation_config.transform_object_path, obj=transformation_pipeline)

            #save encoder object
            utils.save_object(file_path=self.data_transformation_config.target_encoder_object_path, obj=label_encoder)


            data_transformation_artifact=artifact_entity.DataTransformationArtifact(transform_object_path=self.data_transformation_config.transform_object_path, 
                                                                                    transformed_train_path= self.data_transformation_config.transformed_train_path, 
                                                                                    transformed_test_path=self.data_transformation_config.transformed_test_path, 
                                                                                    target_encoder_object_path=self.data_transformation_config.target_encoder_object_path
                                                                                    )

            logging.info(f"Data transformation object {data_transformation_artifact}") 
            return data_transformation_artifact
        except Exception as e:
            raise SensorException(e, sys)