import logging
import os
import time

import faiss
import torch
from tqdm import tqdm
from biosyn.biosyn import BioSyn
from biosyn.dataloader import CandidateDataset, load_dictionary, load_queries
from utils import get_pkl, init_logging, init_seed, save_pkl
from biosyn.rerank import RerankNet
from torch.utils.data import DataLoader 

MINIMIZE = False
TOP_K = 20
NUM_EPOCHS= 10
TRAIN_BATCH_SIZE  = 16

MAX_LENGTH = 25
USE_CUDA = False
LEARNING_RATE = 1e-5
SAVE_CHKPNT_ALL =True


ENCODER_MODEL_NAME = 'dmis-lab/biobert-base-cased-v1.1' #Dense encoder model nmae
TRAIN_DICT_PATH = r'C:\\Users\\antoi\\Desktop\\thesis\\datasets\\ncbi\\ncbi-disease\\train_dictionary.txt'
TRAIN_DIR = r'C:\\Users\\antoi\\Desktop\\thesis\\datasets\\ncbi\\ncbi-disease\\processed_traindev'
OUTPUT_DIR = "./data/output"
WEIGHT_DECAY = 0.01


LOGGER = logging.getLogger()



def parse_args():
    args = {
        "model_name_or_path": ENCODER_MODEL_NAME,
        "train_dictionary_path": TRAIN_DICT_PATH,
        "train_dir": TRAIN_DIR,
        "output_dir": OUTPUT_DIR,
        "max_length": MAX_LENGTH,
        "seed": 0,
        "use_cuda": USE_CUDA,
        "draft": MINIMIZE,
        "topk": TOP_K,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "epoch": NUM_EPOCHS,
        "save_checkpoint_all": SAVE_CHKPNT_ALL
    }

    return args


def train(data_loader, model):
    LOGGER.info("train!")
    train_loss = 0
    train_steps = 0
    model.train()
    for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        model.optimizer.zero_grad()
        batch_x, batch_y = data
        batch_pred = model(batch_x)  
        loss = model.get_loss(batch_pred, batch_y)
        loss.backward()
        model.optimizer.step()
        train_loss += loss.item()
        train_steps += 1

    train_loss /= (train_steps + 1e-9)
    return train_loss
def main():
    global MINIMIZE, TOP_K, NUM_EPOCHS, TRAIN_BATCH_SIZE, TRAIN_DICT_PATH, ENCODER_MODEL_NAME, TRAIN_DIR, OUTPUT_DIR, WEIGHT_DECAY, MAX_LENGTH, USE_CUDA, LEARNING_RATE, SAVE_CHKPNT_ALL
    
    args = parse_args()
    init_logging(LOGGER)
    init_seed(LOGGER, args["seed"])
    if not os.path.exists(args["output_dir"]):
        os.makedirs(args["output_dir"])

    # np array containing each item is (cui, name)
    # N
    train_dictionary  = load_dictionary(dict_path=TRAIN_DICT_PATH)

    # np array containing each item is (cui, mention)
    # M
    train_queries  = load_queries(data_dir=TRAIN_DIR, filter_composite=False, filter_duplicates=False, filter_cuiless=True)

    if MINIMIZE:
        train_dictionary = train_dictionary[:100]
        train_queries = train_queries[:10]
        OUTPUT_DIR = OUTPUT_DIR + "_min"

    
    # train_dictionary = get_pkl("./data/train_dict.pkl")
    # train_queries = get_pkl("./data/train_queries.pkl")
    

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    names_in_train_dict = train_dictionary[:, 0] # N
    names_in_train_queries = train_queries[:, 0] # M

    biosyn = BioSyn(MAX_LENGTH, USE_CUDA)
    biosyn.load_dense_encoder(ENCODER_MODEL_NAME)

    model = RerankNet(
        encoder=biosyn.encoder,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        use_cuda=USE_CUDA
    )
    train_set = CandidateDataset(
        queries = train_queries, 
        dicts = train_dictionary, 
        tokenizer = biosyn.get_dense_tokenizer(), 
        max_length = MAX_LENGTH, 
        topk= TOP_K
    )



    start = time.time()

    #for epoch

    for epoch in range(1,NUM_EPOCHS+1):
        LOGGER.info("Epoch {}/{}".format(epoch,NUM_EPOCHS))
        dict_embs = biosyn.embed_dense(names=names_in_train_dict)
        # faiss.normalize_L2(dict_embs)
        index = faiss.IndexFlatIP(dict_embs.shape[1])
        index.add(dict_embs)
        model.dict_embs_tensor = torch.from_numpy(dict_embs)
        query_embs = biosyn.embed_dense(names=names_in_train_queries)
        # faiss.normalize_L2(query_embs)
        _, train_dense_candidate_idxs = index.search(query_embs, TOP_K)
        train_set.set_dense_candidate_idxs(train_dense_candidate_idxs)
        train_loader = DataLoader(
            train_set, batch_size=TRAIN_BATCH_SIZE, shuffle=False
        )
        train_loss = train(data_loader=train_loader, model=model)
        LOGGER.info('loss/train_per_epoch={}/{}'.format(train_loss,0))
        if epoch == NUM_EPOCHS:
            biosyn.save_model(OUTPUT_DIR)
    end = time.time()
    training_time = end-start
    training_hour = int(training_time/60/60)
    training_minute = int(training_time/60 % 60)
    training_second = int(training_time % 60)
    LOGGER.info("Training Time!{} hours {} minutes {} seconds".format(training_hour, training_minute, training_second))


if __name__ == "__main__":
    main()