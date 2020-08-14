import pymongo

dbClient = pymongo.MongoClient('localhost', 27017)
navdataCollection = dbClient['test-db']['combineddatas']

descModificationList = [
    ('aug9_hover10s.json','aug9_0_hover10s.json'),
    ('aug9_hover20s.json','aug9_0_hover20s.json'),
    ('aug9_hover20s_2.json','aug9_0_hover20s_2.json'),
    ('aug9_hover20s_3.json','aug9_0_hover20s_3.json'),
    ('aug9_hover30s_1.json','aug9_0_hover30s_1.json'),
    ('aug9_hover30s_10.json','aug9_0_hover30s_10.json'),
    ('aug9_hover30s_11.json','aug9_0_hover30s_11.json'),
    ('aug9_hover30s_12.json','aug9_0_hover30s_12.json'),
    ('aug9_hover30s_2.json','aug9_0_hover30s_2.json'),
    ('aug9_hover30s_3.json','aug9_0_hover30s_3.json'),
    ('aug9_hover30s_4.json','aug9_0_hover30s_4.json'),
    ('aug9_hover30s_5.json','aug9_0_hover30s_5.json'),
    ('aug9_hover30s_6.json','aug9_0_hover30s_6.json'),
    ('aug9_hover30s_7.json','aug9_0_hover30s_7.json'),
    ('aug9_hover30s_8.json','aug9_0_hover30s_8.json'),
    ('aug9_hover30s_9.json','aug9_0_hover30s_9.json')
]

for desc in descModificationList:
    upd = navdataCollection.update_many(
        {
            "description": desc[0]
        },
        {
            "$set": {
                "description": desc[1]
            }
        }
    )

    print('Desc: {} --> {} updated with {} data points!'.format(desc[0], desc[1], upd.modified_count))