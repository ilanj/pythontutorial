import json
from json import JSONEncoder

from bson import json_util
from pymongo import MongoClient

class Connectmongodb:

    client = MongoClient("localhost", 27017)
    mydatabase = client["ila_practice"]  # If there is no previously created database with this name, MongoDB will implicitly create one for the user.
    mycollection = mydatabase['company_names']

    def insert_doc(self):
        rec = {
            'title': 'MongoDB and Python',
            'description': 'MongoDB is no SQL database',
            'tags': ['mongodb', 'database', 'NoSQL'],
            'viewers': 104
        }
        _id = Connectmongodb.mydatabase['company_names'].insert_one(rec)
        print('id=', _id.inserted_id)


    def readdb(self):
        cursor = Connectmongodb.mydatabase['company_names'].find()
        for record in cursor:
            print(record)

if __name__ == "__main__":
    db = Connectmongodb()
    db.insert_doc()
    db.readdb()



