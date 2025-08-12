
import torch 
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from transformers import default_data_collator, AutoModel, AutoTokenizer


from .dataloader import NamesDataset

import faiss

class BioSyn(object):
    def __init__(self, max_length, use_cuda, topk):
        self.max_length = max_length
        self.use_cuda = use_cuda
        self.topk = topk
        self.encoder = None
        self.tokenizer = None
        self.faiss_index = None


    def get_dense_encoder(self):
        assert (self.encoder is not None)
        
        return self.encoder

    def get_dense_tokenizer(self):
        assert (self.tokenizer is not None)
        
        return self.tokenizer

    def save_model(self, path):
        # save dense encoder
        self.encoder.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_model(self, model_name_or_path):
        self.load_dense_encoder(model_name_or_path)
        return self


    def load_dense_encoder(self, model_name_path):
        self.encoder = AutoModel.from_pretrained(model_name_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_path)
        if self.use_cuda:
            self.encoder = self.encoder.to("cuda")
        return self.encoder, self.tokenizer


    def get_score_matrix(self, query_embeds, dict_embeds):
        """
        Return score matrix
        Parameters
        ----------
        query_embeds : np.array
            2d numpy array of query embeddings
        dict_embeds : np.array
            2d numpy array of query embeddings
        Returns
        -------
        score_matrix : np.array
            2d numpy array of scores
        """
        score_matrix = np.matmul(query_embeds, dict_embeds.T)
        
        return score_matrix
    
    
    def retrieve_candidate(self, score_matrix, topk):
        """
        Return sorted topk idxes (descending order)
        Parameters
        ----------
        score_matrix : np.array
            2d numpy array of scores
        topk : int
            The number of candidates
        Returns
        -------
        topk_idxs : np.array
            2d numpy array of scores [# of query , # of dict]
        """
        
        def indexing_2d(arr, cols):
            rows = np.repeat(np.arange(0,cols.shape[0])[:, np.newaxis],cols.shape[1],axis=1)
            return arr[rows, cols]

        # get topk indexes without sorting
        topk_idxs = np.argpartition(score_matrix,-topk)[:, -topk:]

        # get topk indexes with sorting
        topk_score_matrix = indexing_2d(score_matrix, topk_idxs)
        topk_argidxs = np.argsort(-topk_score_matrix) 
        topk_idxs = indexing_2d(topk_idxs, topk_argidxs)

        return topk_idxs

    def embed_dense(self, names):
        """
        Embedding data into dense representations

        Parameters
        ----------
        names : np.array or list
            An array of names

        Returns
        -------
        dense_embeds : list
            A list of dense embeddings
        """
        self.encoder.eval()
        batch_size = 1024
        dense_embeds = []
        if isinstance(names, np.ndarray):
            names = names.tolist()
        name_encodings = self.tokenizer(names, padding="max_length", max_length=self.max_length, truncation=True, return_tensors="pt")
        if self.use_cuda:
            name_encodings = name_encodings.to('cuda')
        name_dataset = NamesDataset(name_encodings)
        name_dataloader = DataLoader(name_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=batch_size)

        with torch.no_grad():
            for batch in name_dataloader:
                outputs = self.encoder(**batch)
                # [start token]  mention [end token] - mean pooling, using the context of the mention, we do the mean pool for the two tokens surrounding the mention
                batch_dense_embeds = outputs[0][:,0].cpu().detach().numpy() # [CLS] representations
                dense_embeds.append(batch_dense_embeds)
        dense_embeds = np.concatenate(dense_embeds, axis=0)
        return dense_embeds


    def build_faiss_index(self, dict_names):
        dict_embs = self.embed_dense(names=dict_names).astype("float32")
        d= dict_embs.shape[1]
        if self.use_cuda:
            res = faiss.StandardGpuResources()
            gpu_id = torch.cuda.current_device() if self.use_cuda else 0

            cfg = faiss.GpuIndexFlatConfig()
            cfg.device = gpu_id
            cfg.useFloat16 = True
            base = faiss.GpuIndexFlatIP(res, d, cfg)
        else:
            base = faiss.IndexFlatIP(d)
    
        base.add(dict_embs)
        self.faiss_index = base
        return dict_embs

    def update_faiss_index(self, dict_names):
        dict_embs = self.embed_dense(names=dict_names).astype("float32")
        d= dict_embs.shape[1]
        self.faiss_index.reset()
        self.faiss_index.add(dict_embs)
        return dict_embs
    def retreive_cand_idxs_chunks(self, queries_names, mmap_path, chunk_size=768):
        M = len(queries_names)
        shape = (M, self.topk)
        indexes_all = np.memmap(mmap_path, mode="w+", dtype=np.int32, shape=shape)


        #stream queries
        offset = 0
        for start in range(0, M, chunk_size):
            end = min(start + chunk_size, M)
            batch_queries = queries_names[start:end]
            q = self.embed_dense(batch_queries).astype("float32")
            if self.use_cuda:
                q_t = torch.from_numpy(q).to("cuda", non_blocking=True)
                _, I = self.faiss_index.search(q_t, self.topk)
                I = I.cpu().numpy()
            else:
                _, I = self.faiss_index.search(q,self.topk)

            indexes_all[offset:offset + (end-start)] = I.astype(np.int32, copy=False)
            offset = end

        indexes_all.flush()
        return indexes_all