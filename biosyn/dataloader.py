import logging
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import glob
import os



def load_dictionary(dict_path):
    data = []
    with open(dict_path, mode="r", encoding="utf-8") as f:
        lines =  f.readlines()
        for line in tqdm(lines):
            line = line.strip()
            if line == "": continue
            cui, name  = line.split("||")
            data.append((name, cui))
        data = np.array(data)
        return data

def load_queries(data_dir, filter_composite, filter_duplicates, filter_cuiless):
    data = []
    concept_files = glob.glob(os.path.join(data_dir, "*.concept"))
    
    for concept_file in tqdm(concept_files):
        with open(concept_file, "r", encoding='utf-8') as f:
                concepts = f.readlines()
        for concept in concepts:
            concept = concept.split("||")
            mention = concept[3].strip()
            cui = concept[4].strip()
            is_composite = (cui.replace("+","|").count("|") > 0)
            if filter_composite and is_composite:
                continue
            if filter_cuiless and cui == "-1":
                continue
            data.append((mention, cui))
    if filter_duplicates:
        data = list(dict.fromkeys(data))

    data = np.array(data)
    return data

class NamesDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self,idx):
        return {key: val[idx] for key,val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)

class CandidateDataset(torch.utils.data.Dataset):
    def __init__(self, queries, dicts, tokenizer, max_length, topk, pre_tokenize):
        """
        Retrieve top-k candidates based on dense embedding
        Parameters
        ----------
        queries : list
            A list of tuples (name, id)
        dicts : list
            A list of tuples (name, id)
        tokenizer : BertTokenizer
            A BERT tokenizer for dense embedding
        topk : int
            The number of candidates
        """
        self.query_names, self.query_ids = [row[0] for row in queries], [row[1] for row in queries]
        self.dict_names, self.dict_ids = [row[0] for row in dicts], [row[1] for row in dicts]
        self.tokenizer= tokenizer
        self.max_length = max_length
        self.topk = topk
        self.d_cand_idxs = None
        self.dict_id_sets = None    


        self.pre_tokenize = pre_tokenize
        if pre_tokenize:
            all_query_names_tokens = self.tokenizer(self.query_names, max_length=max_length,padding='max_length', truncation=True, return_tensors='pt' )
            self.all_query_names_tokens = [
                {
                    "input_ids": all_query_names_tokens["input_ids"][idx],
                    "attention_mask": all_query_names_tokens["attention_mask"][idx],
                } for  idx in range(len(all_query_names_tokens["input_ids"]))]

            self.all_dict_names_tokens= self.tokenizer(self.dict_names, max_length=max_length,padding='max_length', truncation=True, return_tensors='pt')
        self.cui_to_dict_idx = {}
        for i, cui in enumerate(self.dict_ids):
            toks = cui.split("|") if isinstance(cui, str) else list(cui)
            for t in toks: self.cui_to_dict_idx.setdefault(t, []).append(i)



    def set_dense_candidate_idxs(self, d_cand_idxs):


        self.d_cand_idxs = d_cand_idxs
        self.dict_ids_sets = [set(s.split("|")) if isinstance(s, str) else set(s) for s in self.dict_ids]
        self.query_id_tokens = [tuple(q.split("|")) if isinstance(q, str) else tuple(q) for q in self.query_ids]

        self.gold_idx_per_query = []
        self.labels_per_query = []
        for q_idx, cand_idxs in enumerate(self.d_cand_idxs):
            q_id_tokens = self.query_id_tokens[q_idx]

            possible = set(self.cui_to_dict_idx.get(q_id_tokens[0], []))
            for t in q_id_tokens[1:]:
                possible &= set(self.cui_to_dict_idx.get(t, []))
            gold_idxs = list(possible)
            labels = np.fromiter(
                (1.0 if all(tok in self.dict_ids_sets[i] for tok in q_id_tokens) else 0.0 
                 for i in cand_idxs),
                dtype=np.float32,
                count=len(cand_idxs)
            )
            if labels.max() == 0.0 and gold_idxs:
                replace_pos = -1
                cand_idxs[replace_pos] = gold_idxs[0]
                labels[replace_pos] = 1.0


            self.labels_per_query.append(labels)
            self.gold_idx_per_query.append(gold_idxs[0] if gold_idxs else -1)


    def __getitem__(self, query_idx):
        """
            Return (query_tokens, cand_tokens), labels
            query_tokens: tokenized the query_name (query_name is query_names[query_idx] the specific mention)
            cand_tokens: 
        """
        assert (self.d_cand_idxs is not None)

        if self.pre_tokenize:
            query_tokens = self.all_query_names_tokens[query_idx]
        else:
            query_name = self.query_names[query_idx]
            query_tokens = self.tokenizer(query_name, max_length=self.max_length,padding='max_length', truncation=True, return_tensors='pt' )



        d_cand_idxs = self.d_cand_idxs[query_idx]
        topk_candidate_idx = np.array(d_cand_idxs)

        assert len(topk_candidate_idx) == self.topk
        assert len(topk_candidate_idx) == len(set(topk_candidate_idx))


        if self.pre_tokenize:
            cand_idxs_tensor = torch.as_tensor(topk_candidate_idx, dtype=torch.long)
            cand_tokens = {
                k: v.index_select(0, cand_idxs_tensor)
                for k, v in self.all_dict_names_tokens.items()
                if isinstance(v, torch.Tensor)
            }
        else:
            cand_names = [self.dict_names[cand_idx] for cand_idx in topk_candidate_idx]
            cand_tokens = self.tokenizer(cand_names, max_length=self.max_length, padding="max_length" , truncation=True, return_tensors="pt")

        # labels = self.get_labels(query_idx, topk_candidate_idx).astype(np.float32)
        labels = self.labels_per_query[query_idx]
        return (query_tokens, cand_tokens), labels


    def __len__(self):
        return len(self.query_names)

    def check_label(self, query_id, candidate_id_set):
        """
            check if all q_id in query_id.split("|") exists in candidate_id_set 
        """
        label = 0
        query_ids = query_id.split("|")
        for q_id in query_ids:
            if q_id in candidate_id_set:
                label = 1
                continue
            else:
                label = 0
                break
        return label
    
    def get_labels(self, query_idx, candidate_idxs):
        labels = np.array([])
        query_id = self.query_ids[query_idx]
        candidate_ids = np.array(self.dict_ids)[candidate_idxs]
        for candidate_id in candidate_ids:
            label = self.check_label(query_id,  candidate_id)
            labels = np.append(labels, label)
        return labels

