from sensor import utils
from sensor.entity import config_entity
from sensor.entity import artifact_entity
from sensor.exception import SensorException
from sensor.logger import logging
import os,sys
from xgboost import XGBClassifier
from sklearn.metrics import f1_score



class ModelTrainer:

    def __init__(self,model_trainer_config:config_entity.ModelTrainerConfig,
                data_transformation_artifact:artifact_entity.DataTransformationArtifact
                ):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise SensorException(e, sys)

    def fine_tune(self):
        try:
            #here we can write if we want to fine tune our model using grid_search_cv 
            pass

        except Exception as e:
            raise SensorException(e, sys)        

    def model_training(self,x,y):
        xgb_clf = XGBClassifier()
        xgb_clf.fit(x,y)
        return xgb_clf


    def initiate_model_trainer(self)->artifact_entity.ModelTrainerArtifact:
        try:
            logging.info(f"loading train and test array")
            train_arr = utils.load_np_array(file_path=self.data_transformation_artifact.transformed_train_path)
            test_arr = utils.load_np_array(file_path=self.data_transformation_artifact.transformed_test_path)

            logging.info(f"splitting input and target feature")
            x_train,y_train = train_arr[:,:-1],train_arr[:,-1]
            x_test,y_test = test_arr[:,:-1],test_arr[:,-1]

            logging.info(f'model training')
            model = ModelTrainer.model_training(self, x=x_train, y=y_train)
            
            logging.info(f"calculating train score")
            yhat_train = model.predict(x_train)
            f1_train_score = f1_score(y_true=y_train, y_pred=yhat_train)

            logging.info(f"calculating test score")
            yhat_test = model.predict(x_test)
            f1_test_score = f1_score(y_true=y_test, y_pred=yhat_test)

            logging.info(f'model train score :{f1_train_score}, model test score :{f1_test_score}')
            #checking for overfitting or underfitting 
            logging.info(f"checking for underfitting")
            if f1_test_score<self.model_trainer_config.expected_score :
                logging.info(f"model is underfitted")
                raise Exception(f"modek is not good as it is not able to give \
                                expected accuracy: {self.model_trainer_config.expected_score}: model actual score : {f1_test_score}")


            diff = abs(f1_train_score-f1_test_score)    
            logging.info(f"checking for overfitting")
            if diff > self.model_trainer_config.overfitting_threshold:
                logging.info(f"model is overfitted")
                raise Exception(f"Train and test score diff:{diff} is more than Overfitting threshold {self.model_trainer_config.overfitting_threshold}") 


            else:
                logging.info(f"saving the model object")
                utils.save_object(file_path=self.model_trainer_config.model_path, obj=model)
            
            logging.info(f"preparing the artifact")
            model_trainer_artifact = artifact_entity.ModelTrainerArtifact(model_path=self.model_trainer_config.model_path, f1_train_score=f1_train_score , f1_test_score = f1_test_score) 

            return model_trainer_artifact   

        except Exception as e:
            raise SensorException(e, sys)


