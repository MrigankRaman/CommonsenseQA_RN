import networkx as nx
from multiprocessing import Pool
from tqdm import tqdm
import json
import numpy as np
import torch
import pickle
import sys
from scipy.sparse import csr_matrix, coo_matrix
def load_resources(cpnet_vocab_path):
    #global concept2id, id2concept, relation2id, id2relation

    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        id2concept = [w.strip() for w in fin]
    concept2id = {w: i for i, w in enumerate(id2concept)}

    #id2relation = merged_relations
    #relation2id = {r: i for i, r in enumerate(id2relation)}
    return concept2id, id2concept

concept2id, id2concept = load_resources('./data/cpnet/concept.txt')

def create_tensors(cids, json_path, json_path_2, tensor_path, output_path):
    dict = torch.load(tensor_path)
    evidence_tensors = dict['all_evidence_vecs']
    num_tuples = dict['all_evidence_num']
    #evidence = torch.empty(len(num_tuples)//5,5,max_num_tuple,evidence_tensors.shape[1])
    #evidence_num_tuples = torch.empty(len(num_tuples)//5,5)
    #evidence = []

    with open(json_path, 'r') as fin:
        data = [json.loads(line) for line in fin]

    with open(json_path_2, 'r') as fin:
        data_2 = [json.loads(line) for line in fin]
    #i = 0
    j = 0
    ui = 0
    gh=0
    adj_data = []
    for line_1 in tqdm(data):
        #print(line_1)
        node2id = {w: k for k, w in enumerate(cids[ui])}
        n_node = len(cids[ui])
        #print(ui,n_node)
        x = len(data_2[ui]['adj_cp_pair'])
        y = len(list(line_1['generation'].keys()))
        row = np.zeros(((x+y),))
        col = np.zeros(((x+y),))
        data_1 = np.zeros(((x+y),))
        i = 0
        for pairs in data_2[ui]['adj_cp_pair']:
            # pairs[0] = pairs[0].replace(" ", "_")
            # pairs[1] = pairs[1].replace(" ", "_")
            row[i] = node2id[concept2id[pairs[0]]]
            col[i] = node2id[concept2id[pairs[1]]]
            data_1[i] = pairs[2]
            i = i+1
            #j = j+1
        for pair, triple in line_1['generation'].items():
            subj, obj = pair.split(', ')
            subj = subj.replace(" ", "_")
            obj = obj.replace(" ", "_")
            #print(ui,subj, obj)
            try:
                row[i] = node2id[concept2id[subj]]
                col[i] = node2id[concept2id[obj]]
                data_1[i] = j+34
                i = i+1
                # row[i] = node2id[concept2id[pairs[1]]]
                # col[i] = node2id[concept2id[pairs[0]]]
                # data_1[i] = -min(j,evidence_tensors.shape[0]-1)
                # i = i+1
                j = j+1
                gh = gh+1
            except KeyError:
                j = j+1
                gh = gh+1
                continue
        adj = coo_matrix((data_1[:i], (row[:i], col[:i])), shape=(n_node, n_node))
        adj_data.append((adj,cids[ui]))
        ui = ui+1
    print(j, evidence_tensors.shape[0])
    with open(output_path, 'wb') as fout:
        pickle.dump(adj_data, fout)


    # for gf in tqdm(range(0,len(num_tuples),5)):
    #     for hg in range(5):
    #         num_tuple = num_tuples[hg+gf]
    #         f = evidence_tensors[i:i+num_tuple,:]
    #         i = i+num_tuple
    #         if num_tuple < max_num_tuple:
    #             input = torch.empty(max_num_tuple - num_tuple, evidence_tensors.shape[1])
    #             input = torch.zeros_like(input)
    #             f = torch.cat((f,input),0)
    #         elif num_tuple > max_num_tuple:
    #             f = f[:max_num_tuple, :]
    #         evidence[j,hg,:,:] = f
    #         evidence_num_tuples[j, hg] = num_tuple
    #     #evidence.append(f)
    #     j= j+1
    # print(evidence.shape, evidence_num_tuples.shape)
    #return evidence, evidence_num_tuples

    #torch.save(evidence, output_path)

def main(dev_gen_json, train_gen_json, dev_evidences, train_evidences, dev_output_graph='adj_dev_all_pairs_hybrid_reverse.pk', train_output_graph = 'adj_train_all_pairs_hybrid_reverse.pk'):
    #create_tensors('cpt_pairs_dev.gen.jsonl.gpt.layer-1.pt', 100, 'dev_evidences_1.pt')
    with open('./data/csqa/graph/train.graph.adj.pk', 'rb') as handle:
        b = pickle.load(handle)
    cids = [b[i][1] for i in range(len(b))]
    create_tensors(cids, train_gen_json, 'cpt_pairs_1hop_train_reverse_hybrid.jsonl', train_evidences, train_output_graph)
    
    with open('./data/csqa/graph/dev.graph.adj.pk', 'rb') as handle:
        b = pickle.load(handle)
    cids = [b[i][1] for i in range(len(b))]
    create_tensors(cids, dev_gen_json, 'cpt_pairs_1hop_dev_reverse_hybrid.jsonl', dev_evidences, dev_output_graph)
if __name__ == '__main__':
    main(*sys.argv[1:])
