import json
import random
import logging

# def false_entities_list(ent2id, t_m):
#     all_entities = ent2id.keys()
#     false_entities_list = [item for item in all_entities if item not in t_m]
#     return false_entities_list



def train_generate(dataset, batch_size, few, symbol2id, ent2id, e1rel_e2):
    logging.info('Loading Train Data')
    train_tasks = json.load(open(dataset + '/train_tasks.json'))
    logging.info('Loading Candidates')
    task_pool = list(train_tasks.keys())
    num_tasks = len(task_pool)

    rel_idx = 0
    while True:
        task_choice = task_pool[rel_idx]  # choose a rel task
        rel_idx += 1
        task_triples = train_tasks[task_choice]

        # select support set, len = few
        support_triples = task_triples[:few]
        support_pairs = [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in support_triples]
        support_left = [ent2id[triple[0]] for triple in support_triples]
        support_right = [ent2id[triple[2]] for triple in support_triples]

        # select query set, len = batch_size
        other_triples = task_triples[few:]
        if len(other_triples) == 0:
            continue
        if len(other_triples) < batch_size:
            query_triples = [random.choice(other_triples) for _ in range(batch_size)]
        else:
            query_triples = random.sample(other_triples, batch_size)

        query_pairs = [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in query_triples]
        query_left = [ent2id[triple[0]] for triple in query_triples]
        query_right = [ent2id[triple[2]] for triple in query_triples]

        # t_m = [triple[2] for triple in query_triples]
        # all_entities = ent2id.keys()
        # false_entities_list = [item for item in all_entities if item not in t_m]

        yield support_pairs, query_pairs, support_left, support_right, query_left, query_right
