from elasticsearch import Elasticsearch
es=Elasticsearch([{'host':'localhost','port':9200}])
# res=es.get(index='elastic_index', id ="ADTYGnkB5bgT9h-pNKT8")
# print(res) 

res= es.search(index='elastic_index',body={
        "query": {
            "match": {"content":"modulate immunologic cascades"}
        }
    })
print(res)
# 
# import elasticsearch
# from elasticsearch.helpers import scan
# import json
# es = elasticsearch.Elasticsearch('https://localhost:9200')
# es_response = scan(
#     es,
#     index='elastic_index',
#     #doc_type='my_doc_type',
#     query={"query": { "match_all" : {}}}
# )

# for item in es_response:
#     print(json.dumps(item)) 
#     break
