
import os
import torch 
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from transformers import default_data_collator, AutoModel, AutoTokenizer


from .dataloader import NamesDataset

import faiss
try:
    import faiss.contrib.torch_utils  # <- enables add/search with torch tensors
    _FAISS_TORCH_OK = True
except Exception:
    _FAISS_TORCH_OK = False

def _faiss_ready(x, want_gpu: bool):
    """
    Return x in the optimal form for faiss.{add,search}:
      - GPU index  -> prefer torch.cuda.FloatTensor (requires torch_utils)
      - CPU index  -> NumPy float32 C-contiguous
    """
    if want_gpu and _FAISS_TORCH_OK and torch.is_tensor(x):
        if x.device.type != "cuda":
            x = x.to("cuda", non_blocking=True)
        if x.dtype != torch.float32:
            x = x.to(torch.float32, copy=False)
        return x
    # Fallback: CPU NumPy float32
    if torch.is_tensor(x):
        return x.detach().to("cpu", non_blocking=True).contiguous().numpy().astype("float32", copy=False)
    return np.asarray(x, dtype=np.float32, order="C")


class BioSyn(object):
    def __init__(self, max_length, use_cuda, topk):
        self.max_length = max_length
        self.use_cuda = use_cuda
        self.topk = topk
        self.encoder = None
        self.tokenizer = None
        self.faiss_index = None

        #know the tier between (cpu, gpu_s, gpu_m, gpu_l, gpu_xl)
        self.gpu_tier, self.has_fp16 = self.get_gpu_tier()
        self.num_workers=min(4, os.cpu_count() // 2)
        self.device = torch.device("cuda" if self.use_cuda else "cpu")


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

    def get_gpu_tier(self):
        """
            return (cpu_tier, has_fp16)
            cpu_tier is one of ["cpu", "gpu_s", "gpu_m", "gpu_l", "gpu_xl"]
        """
        if not self.use_cuda:
            return  "cpu", False

        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024 ** 3)

        major, _ = torch.cuda.get_device_capability(0)
        has_fp16= major >= 7

        if vram_gb < 6:
            return "gpu_s", has_fp16
        elif vram_gb < 16:
            return "gpu_m", has_fp16
        elif vram_gb < 32:
            return "gpu_l" , has_fp16
        else:
            return "gpu_xl", has_fp16

    def dense_names_batch_size(self,len_names):
        if self.gpu_tier == "cpu":
            base = 256 if self.max_length <= 128 else 64
        elif self.gpu_tier == "gpu_s":
            base = 512 if self.max_length <=128 else 128
        elif self.gpu_tier == "gpu_m":
            base = 2048 if self.max_length <=128 else 512
        elif self.gpu_tier == "gpu_l":
            base = 2048 if self.max_length <=128 else 512
        else:
            base = 8192 if self.max_length <=128 else 2048

        base = int(base * 1.3)

        base = max(1, min(base, int(len_names) ))
        return base

    def search_faiss_chunk_size(self, M):
        if self.gpu_tier == "cpu":
            base =  8192
        elif self.gpu_tier == "gpu_s":
            base = 1024
        elif self.gpu_tier == "gpu_m":
            base =  4096
        elif self.gpu_tier == "gpu_l":
            base = 8192*2
        else:
            base =  16384
        if self.use_cuda and self.has_fp16:
            base = int(base * 1.3)
        base = min(base, max(M, 1))
        base = max(256, (base //  256) * 256)
        return int(base)


    def embed_dense_opt(self, names, keep_gpu=False, dtype=torch.float32):
        batch_size = self.dense_names_batch_size(len(names))
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
        # add pin memory and num_workers and persistent system = true
        
        data_loader = DataLoader(
            name_dataset,
            shuffle=False,
            collate_fn=default_data_collator,
            batch_size=batch_size,
            pin_memory=self.use_cuda,
            num_workers=self.num_workers,
            persistent_workers=self.use_cuda
        )

        N = len(names)
        H = getattr(getattr(self.encoder, "config", None), "hidden_size", None)
        assert H is not None
        if keep_gpu:
            out_gpu = torch.empty((N,H), dtype=dtype, device=self.device)
        else:
            # pinned cpu tensor and expose it as numpy at the end
            out_cpu = torch.empty((N,H), dtype=torch.float32, pin_memory=True)

        self.encoder.eval()
        idx = 0
        with torch.no_grad():
            if self.use_cuda:
                with torch.amp.autocast('cuda'):
                    for batch in data_loader:
                        batch  = {k: v.cuda(non_blocking=True) for  k, v in batch.items()}
                        outs = self.encoder(**batch)
                        embs = outs[0][:, 0] #cls representation [B, H]
                        if embs.dtype != dtype:
                            embs = embs.to(dtype, copy=False)
                        B = embs.shape[0]
                        j = idx + B

                        if keep_gpu:
                            out_gpu[idx:j].copy_(embs, non_blocking=True)
                        else:
                            out_cpu[idx:j].copy_(embs, non_blocking=True)
                        idx = j
            else:
                for batch in data_loader:
                    outs = self.encoder(**batch)
                    embs = outs[0][:, 0] #cls representation [B, H]
                    if embs.dtype != dtype:
                        embs = embs.to(dtype, copy=False)
                    B = embs.shape[0]
                    j = idx + B

                    if keep_gpu:
                        out_gpu[idx:j].copy_(embs, non_blocking=True)
                    else:
                        out_cpu[idx:j].copy_(embs, non_blocking=True)
                    idx = j
                
        if keep_gpu:
            return out_gpu
        else:
            return out_cpu.numpy()




    def build_faiss_index(self, dict_names):
        dict_embs = self.embed_dense_opt(names=dict_names, keep_gpu=self.use_cuda)
        d= dict_embs.shape[1]
        if self.use_cuda:
            res = faiss.StandardGpuResources()
            cfg = faiss.GpuIndexFlatConfig()
            cfg.device = torch.cuda.current_device()
            cfg.useFloat16 = True
            base = faiss.GpuIndexFlatIP(res, d, cfg)
        else:
            base = faiss.IndexFlatIP(d)
        base.add(_faiss_ready(dict_embs, want_gpu=self.use_cuda ))
        self.faiss_index = base
        return dict_embs

    def update_faiss_index(self, dict_names):
        dict_embs = self.embed_dense_opt(names=dict_names, keep_gpu=self.use_cuda)
        d= dict_embs.shape[1]
        self.faiss_index.reset()
        self.faiss_index.add(_faiss_ready(dict_embs, want_gpu=self.use_cuda))
        return dict_embs


    def retreive_cand_idxs_chunks(self, queries_names, mmap_path):
        M = len(queries_names)
        _topk_here = int(self.topk * 10)
        shape = (M, self.topk)
        indexes_all = np.memmap(mmap_path, mode="w+", dtype=np.int32, shape=shape)

        chunk_size = self.search_faiss_chunk_size(M)
        #stream queries
        offset = 0
        for start in range(0, M, chunk_size):
            end = min(start + chunk_size, M)
            batch_queries = queries_names[start:end]
            q = self.embed_dense_opt(batch_queries, keep_gpu=self.use_cuda)
            _, I = self.faiss_index.search(q,_topk_here)
            if torch.is_tensor(I):
                I = I.to(dtype=torch.int32, device="cpu", non_blocking=True).numpy()
            else:
                I = I.astype(np.int32, copy=False)

            I_topk = np.full((I.shape[0], self.topk), -1, dtype=np.int32)
            for r, row in enumerate(I):
                # indices of first occurrence in the original order
                first_idx = np.unique(row, return_index=True)[1]
                u = row[np.sort(first_idx)][:self.topk]   # preserve FAISS order, trim to topk
                I_topk[r, :u.size] = u

            indexes_all[offset:offset + (end-start)] = I_topk
            offset = end

        indexes_all.flush()
        return indexes_all