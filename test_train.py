import json
import numpy as np
import torch
from utils import get_pkl, save_pkl
import faiss


PKLS_DIR = "data/otheres/pkls"
MM_DIR = "data/otheres/mmap_embeds"


train_dictionary = get_pkl(f"{PKLS_DIR}/train_queries.pkl")
train_queries = get_pkl(f"{PKLS_DIR}/train_queries.pkl")

query_names, query_ids = [row[0] for row in train_queries], [row[1] for row in train_queries]
dict_names, dict_ids = [row[0] for row in train_dictionary], [row[1] for row in train_dictionary]

with open(f"{MM_DIR}/dicts_1.json") as f:
    meta_d = json.load(f)
with open(f"{MM_DIR}/queries_1.json") as f:
    meta_q = json.load(f)
with open(f"{MM_DIR}/results_1.json") as f:
    meta_r = json.load(f)


mm_d = np.memmap(f"{MM_DIR}/dicts_1.fp16.mmap", 
                 dtype=np.float16 if meta_d["dtype"] == "fp16" else np.float32, 
                 shape=(meta_d["N"], meta_d["d"]))


mm_q = np.memmap(f"{MM_DIR}/queries_1.fp16.mmap", 
                 dtype=np.float16 if meta_q["dtype"] == "fp16" else np.float32, 
                 shape=((meta_r["N"], meta_q["d"])))

mm_r = np.memmap(f"{MM_DIR}/results_1.mmap", 
                 dtype=np.float16 if meta_r["dtype"] == "fp16" else np.int32, 
                 shape=((meta_r["N"], meta_r["d"])))

dict_ids_sets = [set(s.split("|")) if isinstance(s, str) else set(s) for s in dict_ids]
query_id_tokens = [tuple(q.split("|")) if isinstance(q, str) else tuple(q) for q in query_ids]

cui_to_dict_idx = {}
for i, cui in enumerate(dict_ids):
    toks = cui.split("|") if isinstance(cui, str) else list(cui)
    for t in toks: cui_to_dict_idx.setdefault(t, []).append(i)

chunk = 12288

N, d = meta_d["N"], meta_d["d"]

res = faiss.StandardGpuResources()
cfg = faiss.GpuIndexFlatConfig()
cfg.device = torch.cuda.current_device()
cfg.useFloat16 = bool(True)
index = faiss.GpuIndexFlatIP(res, d, cfg)


#stream add to dict in chunks
for s in range(0, N, chunk):
    e = min(s+chunk, N)
    part = mm_d[s:e]
    t = torch.from_numpy(part).float()
    t = t.pin_memory().to("cuda", non_blocking=True)
    index.add(t)
    del t
    torch.cuda.synchronize()
faiss_index = index


batch_size = 1024

M, d = meta_q["N"], meta_q["d"]
mm = mm_q

I_all = []
for s in range(0, M, batch_size):
    e = min(s+batch_size, M)
    q_part = mm[s:e]
    t = torch.from_numpy(q_part).float()
    if isinstance(faiss_index, faiss.GpuIndex):
        t = t.pin_memory().to("cuda", non_blocking=True)
    _, I = faiss_index.search(t, 20)
    I_all.append(I)

I_all = np.vstack(I_all)

save_pkl(I_all, PKLS_DIR + "/I_all.pkl")
