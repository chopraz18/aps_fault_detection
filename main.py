from sensor.logger import logging
from sensor.exception import SensorException
from sensor.utils import get_collection_as_dataframe
import sys,os
from sensor.entity import config_entity,artifact_entity
from sensor.components.data_ingestion import DataIngestion
from sensor.components.data_validation import DataValidation
from sensor.components.data_transformation import DataTransformation
from sensor.components.model_trainer import ModelTrainer
from sensor.components.model_evaluation import ModelEvaluation



if __name__=="__main__":
     try:
          #Data Ingestion
          training_pipeline_config = config_entity.TrainingPipelineConfig()
          data_ingestion_config  = config_entity.DataIngestionConfig(training_pipeline_config=training_pipeline_config)
          print(data_ingestion_config.to_dict())
          data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
          data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
          
          #Data validation
          data_validation_config=config_entity.DataValidationConfig(training_pipeline_config=training_pipeline_config)
          data_validation=DataValidation(data_validation_config=data_validation_config,
                                          data_ingestion_artifact=data_ingestion_artifact)

          data_validation_artifact= data_validation.initiate_data_validation()
          
          #Data Transformation
          data_transformation_config = config_entity.DataTransformationConfig(training_pipeline_config=training_pipeline_config)
          data_tranformation = DataTransformation(data_transformation_config=data_transformation_config , data_ingestion_artifact=data_ingestion_artifact)
          data_transformation_artifact = data_tranformation.initiate_data_transformation()

          #model trainer
          model_trainer_config = config_entity.ModelTrainerConfig(training_pipeline_config=training_pipeline_config)
          model_trainer = ModelTrainer(model_trainer_config=model_trainer_config, data_transformation_artifact=data_transformation_artifact)
          model_trainer_artifact= model_trainer.initiate_model_trainer()

          #model evaluation
          model_eval_config= config_entity.ModelEvaluationConfig(training_pipeline_config=training_pipeline_config)
          model_evaluation = ModelEvaluation(model_eval_config=model_eval_config,
                                             data_ingestion_config=data_ingestion_config,
                                             data_transformation_artifact=data_transformation_artifact,
                                             model_trainer_artifact=model_trainer_artifact)
          model_eval_artifact= model_evaluation.initiate_model_evaluation()                                   

     except Exception as e:
          raise SensorException(e,sys)
