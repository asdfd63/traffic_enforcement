import pymongo

myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["mydatabase"]
pos_hist = mydb["pos_hist"]
x = pos_hist.delete_many({})

for i in range(0, 3):
    for ID in range(i+0, i+3):
        cur_frame = ID + i
        cur_pos = [ID+i, ID+i, ID+i, ID+i]
        id_exist = pos_hist.find_one({'_id': ID})
        if not id_exist:
            cur_id = {'_id': ID, 'info': [[cur_frame, cur_pos]]}
            pos_hist.insert_one(cur_id)
        else:
            mydoc = pos_hist.find({"_id": ID})
            for x in mydoc:
                temp = x['info']
                old = {'info': temp.copy()}
                temp.append([cur_frame, cur_pos])
                new = {"$set": {'info': temp}}
                pos_hist.update_one(old, new)

for ID in range(0, 5):
    index_map = {}
    x = pos_hist.find_one({"_id": ID})
    for idx, item in enumerate(x['info']):
        index_map[item[0]] = idx
    print(index_map)
    print(x['info'][0][1])

# cur_id = {}
#
# for i in range(0, 3):
#     for idx in range(i+0, i+3):
#         id_pos = [idx+i, idx+i, idx+i, idx+i]
#         cur_frame = idx+i
#         if idx not in cur_id:
#             cur_id[idx] = []
#         cur_id[idx].append([cur_frame, id_pos])
#
# for idx in range(0, 5):
#     index_map = {}
#     for idx, item in enumerate(cur_id[idx]):
#         index_map[item[0]] = idx
#     print(index_map)
