
def vectorize_fullranking_data(config, pipeline):
    query_passage_dict = {}
    max_rank = config.reranker.size
    for elem in pipeline.components[-1].predicted_text:
        query_id = elem[0]
        query_text = elem[1]
        passage_id = elem[2]
        passage_text = elem[3]
        rank = int(elem[4])

        ## Format of the dictionary
        ## key (query_id): value-[query text, list of all doc ids, list of all doc text]
        if query_id not in query_passage_dict.keys():
            query_passage_dict[query_id] = [query_text,['0'] * (max_rank), ['0'] * (max_rank)]
        query_passage_dict[query_id][1][rank-1] = passage_id
        query_passage_dict[query_id][2][rank-1] = passage_text
    
    query_all_ids = []
    query_all_text = []
    doc_all_ids = []
    doc_all_text = []
    count = 1
    for query_id in query_passage_dict.keys():
        query_all_ids += [query_id] * len(query_passage_dict[query_id][2])
        query_all_text += [query_passage_dict[query_id][0]] * len(query_passage_dict[query_id][2])
        doc_all_ids += query_passage_dict[query_id][1]
        doc_all_text += query_passage_dict[query_id][2]
        if count ==10:
            break
        count +=1

    return query_all_ids, query_all_text, doc_all_ids, doc_all_text

