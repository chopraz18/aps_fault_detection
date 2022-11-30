import pymongo
import os
from dataclasses import dataclass

@dataclass
class ENVIRONMENT_VARIABLE:

    mongo_db_url:str=os.getenv("MONGO_DB_URL")


env_var=ENVIRONMENT_VARIABLE()

mongo_client = pymongo.MongoClient(env_var.mongo_db_url)
