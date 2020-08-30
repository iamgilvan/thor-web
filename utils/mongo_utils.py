import pymongo as pym

def create_client(host, port):
    return pym.MongoClient(host, port,retryWrites=False)


def get_database(client, database_name):
    return client[database_name]


def get_collection(db, collection_name):
    return db[str(collection_name)]


def open_mongo_connection(config): #type='read'
    res = True
    col = None
    try:
        client = create_client(config['mongo_address'], int(config['mongo_port']))
        client.re
        db     = get_database(client, config['mongo_database'])
        db.authenticate(config['mongo_user'], config['mongo_password'])
        col    = get_collection(db, config['mongo_collection'])
    except Exception as ex:
        res = False
    return col

def write_on_mongo(collection, obj):
    return collection.insert_one(obj.__dict__).inserted_id

def get_objects(collection, post_id):
    return collection.find_one({'_id': post_id})

def update(id, document, collection):
    collection.find_one_and_update({'_id': id}, {'$set': document})