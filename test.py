from biosyn.dataloader import load_dictionary, load_queries
from biosyn.biosyn import BioSyn

from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
import os

from utils import save_pkl

TRAIN_DICT_PATH = "./data/data-ncbi-fair/train_dictionary.txt"
TRAIN_DIR = "./data/data-ncbi-fair/traindev"
OUTPUT = "./data/output"
MODEL = 'dmis-lab/biobert-base-cased-v1.1'

train_dictionary  = load_dictionary(dict_path=TRAIN_DICT_PATH)
train_queries  = load_queries(data_dir=TRAIN_DIR, filter_composite=False, filter_duplicates=False, filter_cuiless=True)

train_queries = train_queries[:10]


max_length = 25
topk = 20

query_names, query_ids = [row[0] for row in train_queries], [row[1] for row in train_queries]
dict_names, dict_ids = [row[0] for row in train_dictionary], [row[1] for row in train_dictionary]



mmap_dir = OUTPUT + "/mmap_files"
if not os.path.exists(mmap_dir):
    os.makedirs(mmap_dir)

names_in_train_dict = train_dictionary[:, 0] # N
names_in_train_queries = train_queries[:, 0] # M

biosyn = BioSyn(max_length, torch.cuda.is_available(), topk)
biosyn.load_dense_encoder(MODEL)


epoch = 0
mmap_file = os.path.join(mmap_dir, f"cand_idxs_epoch_{epoch}.mmap")

import faiss
dict_embs = biosyn.embed_dense_opt(names=names_in_train_dict, keep_gpu=False)
d= dict_embs.shape[1]
res = faiss.StandardGpuResources()
cfg = faiss.GpuIndexFlatConfig()
cfg.device = torch.cuda.current_device()
cfg.useFloat16 = True
base = faiss.GpuIndexFlatIP(res, d, cfg)
base.add(dict_embs)
faiss_index = base


queries_names = names_in_train_queries
M = len(queries_names)
_topk_here = int(topk)
shape = (M, topk)
# train_dense_cands_idxs = np.memmap(mmap_path, mode="w+", dtype=np.int32, shape=shape)
# chunk_size = self.search_faiss_chunk_size(M)
train_dense_cands_idxs = np.empty(shape, dtype=np.int32)

chunk_size = 256
offset = 0
for start in range(0, M, chunk_size):
    end = min(start + chunk_size, M)
    batch_queries = queries_names[start:end]
    q = biosyn.embed_dense_opt(batch_queries, keep_gpu=True)
    _, I = faiss_index.search(q,_topk_here)
    if torch.is_tensor(I):
        I = I.to(dtype=torch.int32, device="cpu", non_blocking=True).numpy()
    else:
        I = I.astype(np.int32, copy=False)

    train_dense_cands_idxs[offset:offset + (end-start)] = I
    offset = end

train_dense_cands_idxs.flush()

os.makedirs("./data/pkls", exist_ok=True)
save_pkl(train_dense_cands_idxs, f"./data/pkls/epoch_{epoch}_cands.pkl")



