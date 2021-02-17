import pymongo as pym

def create_client(host, port):
    return pym.MongoClient(host, port,retryWrites=False)


def get_database(client, database_name):
    return client[database_name]


def get_collection(db, collection_name):
    return db[str(collection_name)]

def connection(collection): #type='read'
    res = True
    col = None
    try:
        client = pym.MongoClient("mongodb+srv://iamgilvan:gil6mec@thor.rtpxq.mongodb.net/thor?retryWrites=true&w=majority")
        client.re
        db     = get_database(client, 'thor')
        col    = get_collection(db, collection)
    except Exception as ex:
        res = False
    return col


def open_mongo_connection(config): #type='read'
    res = True
    col = None
    try:
        client = pym.MongoClient("mongodb+srv://iamgilvan:gil6mec@thor.rtpxq.mongodb.net/thor?retryWrites=true&w=majority")
        client.re
        db     = get_database(client, 'thor')
        col    = get_collection(db, 'projetos')
    except Exception as ex:
        res = False
    return col

def write_on_mongo(collection, obj):
    return collection.insert_one(obj.__dict__).inserted_id

def get_objects(collection, post_id):
    return collection.find_one({'_id': post_id})

def update(id, document, collection):
    collection.find_one_and_update({'_id': id}, {'$set': document})