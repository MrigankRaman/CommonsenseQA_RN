import networkx as nx
from multiprocessing import Pool
from tqdm import tqdm
import json
import numpy as np
import torch

def create_tensors(tensor_path, max_num_tuple, output_path):
    dict = torch.load(tensor_path)
    evidence_tensors = dict['all_evidence_vecs']
    num_tuples = dict['all_evidence_num']
    evidence = torch.empty(len(num_tuples)//5,5,max_num_tuple,evidence_tensors.shape[1])
    evidence_num_tuples = torch.empty(len(num_tuples)//5,5)
    #evidence = []
    i = 0
    j = 0
    for gf in tqdm(range(0,len(num_tuples),5)):
        for hg in range(5):
            num_tuple = num_tuples[hg+gf]
            f = evidence_tensors[i:i+num_tuple,:]
            i = i+num_tuple
            if num_tuple < max_num_tuple:
                input = torch.empty(max_num_tuple - num_tuple, evidence_tensors.shape[1])
                input = torch.zeros_like(input)
                f = torch.cat((f,input),0)
            elif num_tuple > max_num_tuple:
                f = f[:max_num_tuple, :]
            evidence[j,hg,:,:] = f
            evidence_num_tuples[j, hg] = num_tuple
        #evidence.append(f)
        j= j+1
    print(evidence.shape, evidence_num_tuples.shape)
    return evidence, evidence_num_tuples

    #torch.save(evidence, output_path)

def main():
    create_tensors('cpt_pairs_dev.gen.jsonl.gpt.layer-1.pt', 100, 'dev_evidences_1.pt')

if __name__ == '__main__':
    main()
