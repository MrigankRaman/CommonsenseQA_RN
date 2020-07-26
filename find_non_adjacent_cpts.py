import networkx as nx
from multiprocessing import Pool
from tqdm import tqdm
import json
import numpy as np
def get_cpnet_simple(nx_graph):
    cpnet_simple = nx.Graph()
    for u, v, data in nx_graph.edges(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        if cpnet_simple.has_edge(u, v):
            cpnet_simple[u][v]['weight'] += w
        else:
            cpnet_simple.add_edge(u, v, weight=w)
    return cpnet_simple

global cpnet, cpnet_simple, concept2id, id2concept
cpnet = nx.read_gpickle('./data/cpnet/conceptnet.en.pruned.graph')
cpnet_simple = get_cpnet_simple(cpnet)
with open('./data/cpnet/concept.txt', "r", encoding="utf8") as fin:
    id2concept = [w.strip() for w in fin]
concept2id = {w: i for i, w in enumerate(id2concept)}

def get_non_adj_cpts_qa_pair(qcs, acs):
    #cpts = qcs + acs
    non_adj_cpts = []
    adj_cpts = []
    for source in qcs:
        for target in acs:
            try:
                all_path = list(nx.all_simple_paths(cpnet_simple, source=concept2id[source], target=concept2id[target], cutoff = 2))
                if len(all_path) == 0 and (source, target) not in non_adj_cpts and (target, source) not in non_adj_cpts and source!=target:
                    non_adj_cpts.append((source, target))
                elif len(all_path) > 0 and (source, target) not in adj_cpts and (target, source) not in adj_cpts and source!=target:
                    adj_cpts.append((source, target))
            except nx.exception.NodeNotFound:
                continue

            # try:
            #     wt = cpnet_simple[concept2id[source]][concept2id[target]]
            #     if (source, target) not in adj_cpts and (target, source) not in adj_cpts and source!=target:
            #         adj_cpts.append((source, target))
            # except KeyError:
            #     if (source, target) not in non_adj_cpts and (target, source) not in non_adj_cpts and source!=target:
            #         non_adj_cpts.append((source, target))
    return non_adj_cpts, adj_cpts
def get_non_adj_cpts(grounded_path, cpnet_graph_path,output_path, num_processes=1):
    sents = []
    answers = []
    # with open(statement_path, 'r') as fin:
    #     lines = [line for line in fin]
    with open(grounded_path, 'r') as fin:
        data = [json.loads(line) for line in fin]
    i = 0
    with open(output_path, 'w') as output_handle:
        for line in tqdm(data):
            # for statement in j["statements"]:
            #     sents.append(statement["statement"])
            # for answer in j["question"]["choices"]:
            #     ans = answer['text']
            #     try:
            #         assert all([i != "_" for i in ans])
            #     except Exception:
            #         print(ans)
            #     answers.append(ans)
            cpt_pairs, cpt_pairs_1 = get_non_adj_cpts_qa_pair(line['qc'], line['ac'])
            output_dict = {'question': line['sent'], 'choice': line['ans'], "non_adj_cp_pair": cpt_pairs, "adj_cp_pair": cpt_pairs_1}
            output_handle.write(json.dumps(output_dict))
            output_handle.write("\n")


def main():
    get_non_adj_cpts('./data/csqa/grounded/dev.grounded.jsonl', './data/cpnet/conceptnet.en.pruned.graph', 'cpt_pairs_2hop_dev.jsonl')

if __name__ == '__main__':
    main()
