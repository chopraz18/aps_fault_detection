from sensor.predictor import ModelResolver
from sensor.entity import config_entity,artifact_entity
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.utils import load_object
from sklearn.metrics import f1_score
from sensor.config import Target_Column
import pandas as pd
import os,sys


class ModelEvaluation:
    
    def __init__(self,
                model_eval_config:config_entity.ModelEvaluationConfig,
                data_ingestion_config:config_entity.DataIngestionConfig,
                data_transformation_artifact:artifact_entity.DataTransformationArtifact,
                model_trainer_artifact:artifact_entity.ModelTrainerArtifact,
                data_ingestion_artifact:artifact_entity.DataIngestionArtifact):

        try:
            self.model_eval_config=model_eval_config
            self.data_ingestion_config=data_ingestion_config
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_artifact=model_trainer_artifact
            self.model_resolver=ModelResolver()
            self.data_ingestion_artifact=data_ingestion_artifact    
        except Exception as e:
            raise SensorException(e, sys)


    def initiate_model_evaluation(self)->artifact_entity.ModelEvaluationArtifact:

        #if saved model folder has model then we will compare which model is best trained

        try:
            logging.info("if saved model folder has model then we will compare which model is best trained")
            latest_dir_path = self.model_resolver.get_latest_dir_path()
            if latest_dir_path == None:
                model_eval_artifact=artifact_entity.ModelEvaluationArtifact(is_model_selected=True, improved_accuracy=None)
                logging.info(f"Model Evaluation artifact: {model_eval_artifact}")
                return model_eval_artifact

            else:
                #finding location of transformer,model and target_encoder
                logging.info("finding location of transformer,model and target_encoder")
                transformer_path=self.model_resolver.get_latest_transformer_path()
                model_path=self.model_resolver.get_latest_model_path()
                target_encoder_path=self.model_resolver.get_latest_target_encoder_path()  

                #loading previuosly trained objects
                logging.info("#loading previuosly trained objects")
                transformer = load_object(file_path=transformer_path)
                model = load_object(file_path=model_path)
                target_encoder = load_object(file_path=target_encoder)

                #loading current trained model objects
                logging.info("#loading current trained model objects")
                current_transformer = load_object(file_path=self.data_transformation_artifact.transform_object_path)  
                current_model = load_object(file_path=self.model_trainer_artifact.model_path)
                current_target_encoder = load_object(file_path=self.data_transformation_artifact.target_encoder_object_path)

                #loading test df
                logging.info("#loading test df")
                test_df = pd.read_csv(filepath_or_buffer=self.data_ingestion_artifact.test_file_path)
                input_df = test_df[1:]
                target_df = test_df[Target_Column]
                y_true=target_encoder.transform(target_df)

                #accuracy using previuos trained model
                logging.info("accuracy using previuos trained model")
                input_arr=transformer.transform(input_df)
                y_pred = model.predict(input_arr)
                print(f"prediction using previous model:{target_encoder.inverse_transform(y_pred[:5])}")
                previous_model_score = f1_score(y_true=y_true, y_pred=y_pred)
                logging.info(f"accuracy using previous trained model:{previous_model_score}")

                #accuracy using current trained model
                logging.info("accuracy using current trained model")
                y_true=current_target_encoder.transform(target_df)
                input_arr=current_transformer.transform(input_df)
                y_pred = current_model.predict(input_arr)
                print(f"prediction using trained model:{current_target_encoder.inverse_transform(y_pred[:5])}")
                current_model_score = f1_score(y_true=y_true, y_pred=y_pred)
                logging.info(f"accuracy using current trained model:{current_model_score}")


                if current_model_score<=previous_model_score:
                    logging.info("Current Trained model is not better than previous trained model")
                    raise Exception("Current Trained model is not better than previous trained model")
                    

                else:
                    logging.info("current model has more accuracy than previuos model")
                    model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_selected=True,
                    improved_accuracy=current_model_score-previous_model_score)
                    return model_eval_artifact
        except Exception as e:
            raise SensorException(e, sys)        