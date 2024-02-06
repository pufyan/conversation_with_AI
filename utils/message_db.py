
import datetime
import json
import traceback
# from app.settings import sp, get_database
# chat_history = get_database('chat_history')

from app.settings import sp
try:
    chat_history = sp.db['chat_history']
except Exception as e:
    print('CONNECT MONGO ERROR: ', e)
    print(traceback.format_exc())

# connection string to same db:
# mongodb+srv://admin:admin@localhost/admin_panel?retryWrites=true&w=majority

# mongoimport --uri mongodb+srv://admin:admin@localhost/admin_panel --collection chat_history --type json --file data_blocks.json
# mongoimport --uri mongodb://localhost:27017/admin_panel --collection chat_history --type json --file data_blocks.json


# chat_history.delete_many({})
# print('HISTORY REMOVED')


async def get_all_chatids():
    chat_ids = chat_history.distinct('chat_uid')
    return chat_ids


async def load_messages(chat_id, ignore_deleted=False):
    filters = {
        'chat_uid': chat_id,
        # 'time': {
        #     '$gte': datetime.datetime.utcnow() - datetime.timedelta(minutes=20)
        # }
        # why: 'time': {'$gte': datetime.datetime.utcnow() - datetime.timedelta(minutes=20)}
        # because we need to get messages from last 20 minutes
    }
    if not ignore_deleted:
        filters['deleted'] = {'$ne': True}

    msgs = list(chat_history.find(
        filter=filters,
        projection={'_id': False}
    ).sort('time', 1))

    return msgs
    # print(json.dumps(list(msgs), indent=4, ensure_ascii=False))

    return list(msgs)
    # while True:
    #     token_count = sum([count_tokens(m['content']) for m in msgs])
    #     print(chat_id, 'token count: ', token_count)
    #     if token_count > 4097:
    #         msgs = msgs[1:]
    #         msgs = [m for m in msgs if m['content']]
    #     else:
    #         return msgs


async def set_messages(chat_uid, *new_messages: dict()):
    print('saving to database:')
    for msg in new_messages:
        print('msg: ', msg)
        msg['time'] = msg.get('time', datetime.datetime.utcnow())
        msg['bot'] = msg.get('bot', False)
        msg['sender'] = msg.get('sender')
        
        chat_history.insert_one({'chat_uid': chat_uid, **msg})
    print('saved to database')


async def remove_messages(chat_uid):
    # set deleted:
    result = chat_history.update_many(
        filter={
            'chat_uid': chat_uid,
            'deleted': {'$ne': True},  # '$exists': False
        },
        update={'$set': {'deleted': True}}
    )
    return result.modified_count


from pydantic import BaseModel


async def save_entity(source: BaseModel, **kwargs):
    collection_name = f'{source.__class__.__name__}_{source.chat_id}'
    sp.db[collection_name].update_one(kwargs, {'$set': {'source': source.dict()}}, upsert=True)

BaseModelClass = type(BaseModel)


async def load_entities(entity_class: BaseModelClass, **kwargs):
    if 'chat_id' in kwargs:
        collection_name = f'{entity_class.__name__}_{kwargs["chat_id"]}'
    else:
        collection_name = f'{entity_class.__name__}'
    kwargs.pop('chat_id', None)

    entities = list(sp.db[collection_name].find(kwargs, projection={'_id': False}))
    return entities
