
import os
import torch 
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from transformers import default_data_collator, AutoModel, AutoTokenizer


from .dataloader import NamesDataset
from utils import get_gpu_tier, embed_dense_batch_size

import faiss
import faiss.contrib.torch_utils
import json
import torch.nn.functional as F

 
class BioSyn(object):
    def __init__(self, max_length, use_cuda, topk, model_name_path):
        self.max_length = max_length
        self.use_cuda = use_cuda
        self.topk = topk
        self.encoder = None
        self.tokenizer = None
        self.faiss_index = None

        #know the tier between (cpu, gpu_s, gpu_m, gpu_l, gpu_xl)
        self.gpu_tier, self.has_fp16 = get_gpu_tier()
        self.num_workers=min(4, os.cpu_count())
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.load_dense_encoder(model_name_path)

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
        self.encoder = AutoModel.from_pretrained(model_name_path, use_safetensors=True)
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
        #function to assume the gpu tier and appropriate batch_size based on length of names
        batch_size  = 1024

        dense_embeds = []
        if isinstance(names, np.ndarray):
            names = names.tolist()
        name_encodings = self.tokenizer(names, padding="max_length", max_length=self.max_length, truncation=True, return_tensors="pt")
        name_dataset = NamesDataset(name_encodings)
        name_dataloader = DataLoader(name_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=batch_size)

        with torch.no_grad():
            for batch in name_dataloader:
                if self.use_cuda:
                    batch = {k: v.cuda(non_blocking=True) for k,v in batch.items()}
                outputs = self.encoder(**batch)
                # [start token]  mention [end token] - mean pooling, using the context of the mention, we do the mean pool for the two tokens surrounding the mention
                batch_dense_embeds = outputs[0][:,0].cpu().detach().numpy().astype("float32", copy=False) # [CLS] representations
                dense_embeds.append(batch_dense_embeds)

        if self.use_cuda:
            dense_embeds = np.ascontiguousarray(np.concatenate(dense_embeds, axis=0), dtype="float32")
        else:
            dense_embeds = np.concatenate(dense_embeds, axis=0)
        return dense_embeds



    def embed_dense_optimized(self, names, return_tensor=False):
        batch_size = embed_dense_batch_size(len(names), self.gpu_tier)
        if isinstance(names, np.ndarray):
            names = names.tolist()
        assert isinstance(names, (list, tuple)) and len(names) > 0

        name_tokens = self.tokenizer(names,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            )
        name_dataset = NamesDataset(name_tokens)


        if self.use_cuda:
            data_loader = DataLoader(
                name_dataset,
                shuffle=False,
                collate_fn=default_data_collator,
                batch_size=batch_size,
                pin_memory=True,
                num_workers=0,
                persistent_workers=False
            )
        else:
            data_loader = DataLoader(
                name_dataset,
                shuffle=False,
                collate_fn=default_data_collator,
                batch_size=batch_size
            )


        N, H  = len(names), getattr(getattr(self.encoder, "config", None), "hidden_size", None)
        if return_tensor and self.use_cuda:
            out = torch.empty((N,H), dtype=np.float16, device=self.device)
        else:
            out = torch.empty((N,H), dtype=torch.float32, pin_memory=self.use_cuda)

        idx = 0
        with torch.no_grad():
            if self.use_cuda:
                with torch.amp.autocast('cuda'):
                    for batch in data_loader:
                        batch  = {k: v.cuda(non_blocking=True) for  k, v in batch.items()}
                        enc_out = self.encoder(**batch)
                        embs = enc_out[0][:, 0] # B, H  (CLS)
                        if embs.dtype != torch.float16:
                            embs = embs.to(dtype=torch.float16, copy=False)
                        B = embs.shape[0] #B
                        j = idx + B

                        out[idx:j].copy_(embs, non_blocking=True)
                        idx = j
            else:
                for batch in data_loader:
                    enc_outs = self.encoder(**batch)
                    embs = enc_outs[0][:, 0]
                    B = embs.shape[0]
                    j = idx + B
                    out[idx:j].copy_(embs, non_blocking=True)
                    idx = j
        return out if return_tensor else out.numpy()


    
    def embed_names_mmap(self, names, memap_path_base, batch_size=8192):

        os.makedirs(os.path.dirname(memap_path_base), exist_ok=True)

        #small batch to get d
        names_0 = names[:min(32, len(names))]
        embs_0 = self.embed_dense_optimized(names_0, return_tensor=False)
        assert not torch.is_tensor(embs_0)
        d = int(embs_0.shape[1])

        dtype = np.float16 if self.has_fp16 else np.float32
        dtype_str = "fp16" if self.has_fp16 else "fp32"

        #create meta data file
        mmap_path = memap_path_base + f".{dtype_str}.mmap"
        mm = np.memmap(mmap_path, mode="w+", dtype=dtype, shape=(len(names), d))



        meta = {"N": len(names), "d":d, "dtype": dtype_str, "path": mmap_path}


        with open(memap_path_base + ".json", "w") as f:
            json.dump(meta, f)

        #start writing stream
        offset = 0
        with tqdm(total=len(names), desc="embedding names") as pbar:
            for s in range(0, len(names), batch_size):
                e = min(s+batch_size, len(names) )
                embs = self.embed_dense_optimized(names[s:e], return_tensor=False)
                assert not torch.is_tensor(embs)
                mm[offset:offset+(e-s)] = embs.astype(dtype, copy=False)
                offset = e
                pbar.update(e-s)
                # del embs

        mm.flush()
        del mm
        return mmap_path, meta

    def build_faiss_index(self, dict_names , batch_size=128_000):

        # H is the hidden size of our encoder (usually 768)
        H = getattr(getattr(self.encoder, "config", None), "hidden_size", None)
        assert H is not None


        #build main index
        if self.use_cuda:
            gpu_resources = faiss.StandardGpuResources()
            #Index configurations
            cfg = faiss.GpuIndexFlatConfig()
            cfg.device = torch.cuda.current_device()
            cfg.useFloat16 = bool(self.has_fp16)

            #make the index (this index is on gpu)
            index = faiss.GpuIndexFlatIP(gpu_resources, H, cfg)
        else:
            #make normal cpu index 
            index = faiss.IndexFlatIP(H)

        # disable autograd
        self.encoder.eval()
        with torch.inference_mode():
            dict_names_embs = self.embed_dense_optimized(dict_names, return_tensor=True)

            if self.use_cuda:
                dict_names_embs = dict_names_embs.to("cuda", non_blocking=True).float()
            else:
                dict_names_embs = dict_names_embs.to("cpu").float().numpy()

            N = dict_names_embs.shape[0]
            with tqdm(total = len(dict_names) , desc="embeding & building FAISS index") as pbar:
                #add dict names embeddings into the index in batches
                for start in range(0, N, batch_size):
                    end = min(start + batch_size, N)
                    if self.use_cuda:
                        chunk = dict_names_embs[start:end].contiguous()
                    else:
                        chunk = dict_names_embs[start:end]
                    index.add(chunk)

                    pbar.update(end - start)
                    del chunk
            del dict_names_embs

        self.faiss_index = index
        return

    def build_faiss_index_mmap(self, mmap_base):
        chunk = 12288

        #open meta json
        with open(mmap_base + ".json") as f:
            meta = json.load(f)

        N, d = meta["N"], meta["d"]
        dtype = np.float16 if meta["dtype"] == "fp16" else np.float32

        #open mmap read-only
        mm = np.memmap(meta["path"], mode="r", dtype=dtype, shape=(N,d))

        #build index
        if self.use_cuda:
            res = faiss.StandardGpuResources()
            cfg = faiss.GpuIndexFlatConfig()
            cfg.device = torch.cuda.current_device()
            cfg.useFloat16 = bool(self.has_fp16)
            index = faiss.GpuIndexFlatIP(res, d, cfg)
        else:
            index = faiss.IndexFlatIP(d)


        #stream add to dict in chunks
        with tqdm(total=N, desc="Faiss index build") as pbar:
            for start in range(0, N, chunk):
                e = min(start+chunk, N)
                part = mm[start:e]
                t = torch.from_numpy(part).float()
                if self.use_cuda:
                    t = t.pin_memory().to("cuda", non_blocking=True)
                index.add(t)
                del t
                if self.use_cuda:
                    torch.cuda.synchronize()
                pbar.update(e-start)
        self.faiss_index = index
        return index

    def search_index_mmap(self, queries_mmap_base, save_results_mmap_base):
        batch_size = 1024
        with open(queries_mmap_base + ".json") as f:
            meta = json.load(f)

        M, d = meta["N"], meta["d"]
        dtype = np.float16 if meta["dtype"] == "fp16" else np.float32

        mm = np.memmap(meta["path"], mode="r", dtype=dtype, shape=(M,d))


        mm_I = None
        if save_results_mmap_base:
            mm_path = save_results_mmap_base + ".mmap"
            mm_I = np.memmap(mm_path, mode="w+", dtype=np.int32, shape=(M, self.topk) )
            meta = {"N": M, "d":self.topk, "dtype": "int32", "path": mm_path}
            with open(save_results_mmap_base + ".json", "w") as f:
                json.dump(meta, f)


        I_all = []
        for s in range(0, M, batch_size):
            e = min(s+batch_size, M)
            q_part = mm[s:e]
            t = torch.from_numpy(q_part).float()
            if isinstance(self.faiss_index, faiss.GpuIndex):
                t = t.pin_memory().to("cuda", non_blocking=True)
            _, I = self.faiss_index.search(t, self.topk)

            if torch.is_tensor(I):
                I = I.cpu().numpy()

            if mm_I is not None:
                mm_I[s:e] = I

            I_all.append(I)

        if mm_I is not None:
            mm_I.flush()
        return np.vstack(I_all)


