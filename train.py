import argparse
import logging
import os
import time

import faiss
from metric_logger import MetricsLogger
import torch
from tqdm import tqdm
from biosyn.biosyn import BioSyn
from biosyn.dataloader import CandidateDataset, load_dictionary, load_queries
from utils import get_pkl, init_logging, init_seed, save_pkl
from biosyn.rerank import RerankNet
from torch.utils.data import DataLoader 
import numpy as np


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MINIMIZE = False
TOP_K = 20


NUM_EPOCHS= 10
TRAIN_BATCH_SIZE  = 128

MAX_LENGTH = 25
USE_CUDA = torch.cuda.is_available()
LEARNING_RATE = 1e-5

SAVE_CHKPNT_ALL =True


ENCODER_MODEL_NAME = 'dmis-lab/biobert-base-cased-v1.1' #Dense encoder model nmae
TRAIN_DICT_PATH = "./data/data-ncbi-fair/train_dictionary.txt"
TRAIN_DIR = "./data/data-ncbi-fair/traindev"
OUTPUT_DIR = "./data/output"
OUTPUT_OTHERS_DIR = "./data/otheres"
WEIGHT_DECAY = 0.01



PRE_TOKENIZE = True
NOT_USE_FAISS = False


LOGGER = logging.getLogger()



def parse_args():
    global MINIMIZE, TOP_K, NUM_EPOCHS, TRAIN_BATCH_SIZE, TRAIN_DICT_PATH, ENCODER_MODEL_NAME, TRAIN_DIR, OUTPUT_DIR, WEIGHT_DECAY, MAX_LENGTH, USE_CUDA, LEARNING_RATE, SAVE_CHKPNT_ALL
    
    parser = argparse.ArgumentParser(description='Biosyn train')

    parser.add_argument('--model_name_or_path',
                        help='Directory for pretrained model', default=ENCODER_MODEL_NAME)
    parser.add_argument('--train_dictionary_path', type=str,
                    help='train dictionary path', default=TRAIN_DICT_PATH)
    parser.add_argument('--train_dir', type=str, 
                    help='training set directory', default=TRAIN_DIR)
    parser.add_argument('--output_dir', type=str,
                        help='Directory for output', default=OUTPUT_DIR)
    # Tokenizer settings
    parser.add_argument('--max_length', default=MAX_LENGTH, type=int)

    # Train config
    parser.add_argument('--seed',  type=int, 
                        default=0)
    parser.add_argument('--use_cuda',  action="store_true", default=USE_CUDA)
    parser.add_argument('--draft',  action="store_true", default=MINIMIZE)
    parser.add_argument('--topk',  type=int, 
                        default=TOP_K)
    parser.add_argument('--learning_rate',
                        help='learning rate',
                        default=LEARNING_RATE, type=float)
    parser.add_argument('--weight_decay',
                        help='weight decay',
                        default=WEIGHT_DECAY, type=float)
    parser.add_argument('--train_batch_size',
                        help='train batch size',
                        default=TRAIN_BATCH_SIZE, type=int)
    parser.add_argument('--epoch',
                        help='epoch to train',
                        default=NUM_EPOCHS, type=int)

    parser.add_argument('--save_checkpoint_all', action="store_true", default=SAVE_CHKPNT_ALL)


    parser.add_argument('--not_use_faiss',  action="store_true", default=NOT_USE_FAISS)
    parser.add_argument('--pre_tokenize',  action="store_true", default=PRE_TOKENIZE)
    parser.add_argument('--forward_chunk_size', default=64, type=int)


    args = parser.parse_args()
    return args


def train(data_loader, model):
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

def train_new(data_loader, model, has_fp16, epoch_num, metrics_train, pkls_dir=None):
    if model.use_cuda and has_fp16:
        scaler = torch.amp.GradScaler('cuda')
    train_loss = 0
    train_steps = 0
    model.train()

    saved_batches = []

    for i, data in tqdm(enumerate(data_loader), total=len(data_loader), desc="training", unit="epochs"):
        model.optimizer.zero_grad()
        batch_x, batch_y = data
        if has_fp16 and model.use_cuda:
            with torch.amp.autocast('cuda'):
                batch_pred = model(batch_x)
                loss = model.get_loss(batch_pred, batch_y)
            scaler.scale(loss).backward()
            scaler.step(model.optimizer)
            scaler.update()
        else:
            batch_pred = model(batch_x)
            loss = model.get_loss(batch_pred, batch_y)
            loss.backward()
            model.optimizer.step()
        train_loss += loss.item()
        train_steps += 1


        if pkls_dir:
            with torch.no_grad():
                to_save = {
                    "epoch": epoch_num,
                    "y": batch_y.detach().cpu(),
                    "pred": batch_pred.detach().cpu(),
                    "x": batch_x
                }
            saved_batches.append(to_save)


    if pkls_dir:
        save_pkl(saved_batches, pkls_dir + f"/epoch_{epoch_num}_results.pkl")


    LOGGER.info(f"Epoch num: {epoch_num} has did loss: {train_loss}")
    train_loss /= (train_steps + 1e-9)
    return train_loss
    


def main():
    args = parse_args()

    # np array containing each item is (cui, name)
    # N
    train_dictionary  = load_dictionary(dict_path=args.train_dictionary_path)

    # np array containing each item is (cui, mention)
    # M
    train_queries  = load_queries(data_dir=args.train_dir, filter_composite=False, filter_duplicates=True, filter_cuiless=True)



    if args.draft:
        train_dictionary = train_dictionary[:100]
        train_queries = train_queries[:10]
        args.output_dir = args.output_dir + "_min"



    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.exists(OUTPUT_OTHERS_DIR):
        os.makedirs(OUTPUT_OTHERS_DIR)

    mmap_dir = OUTPUT_OTHERS_DIR +  "/mmap_embeds"
    os.makedirs(mmap_dir, exist_ok=True)

    pkls_dir = OUTPUT_OTHERS_DIR +  "/pkls"
    os.makedirs(pkls_dir, exist_ok=True)




    init_logging(LOGGER, base_output_dir= OUTPUT_OTHERS_DIR, logging_folder="train", minimize=args.draft)
    init_seed(LOGGER, args.seed)



    LOGGER.info(f"train_dictionary is loaded from file: {args.train_dictionary_path} with minimize set to: {'True' if args.draft else 'False'}, the length is: {len(train_dictionary)}")
    LOGGER.info(f"train_queries is loaded from file: {args.train_dir} with minimize set to: {'True' if args.draft else 'False'}, the length is: {len(train_queries)}")


    save_pkl(train_dictionary, f"{pkls_dir}/train_dictionary.pkl")
    save_pkl(train_queries, f"{pkls_dir}/train_queries.pkl")
    # train_dictionary = get_pkl("./data/train_dict.pkl")
    # train_queries = get_pkl("./data/train_queries.pkl")



    metrics_train = MetricsLogger(
        logger=LOGGER,
        use_cuda=args.use_cuda,
        tag="train"
    )
    metrics_faiss = MetricsLogger(
        logger=LOGGER,
        use_cuda=args.use_cuda,
        tag="faiss"
    )

    names_in_train_dict = train_dictionary[:, 0] # N
    names_in_train_queries = train_queries[:, 0] # M

    biosyn = BioSyn(args.max_length, args.use_cuda, args.topk, args.model_name_or_path)
    LOGGER.info(f"We are working on tier: {biosyn.gpu_tier}, gpu has native fp16 capability: {biosyn.has_fp16} , num_workers={biosyn.num_workers}")
    LOGGER.info(f"Encoder loaded from: {args.model_name_or_path}")

    model = RerankNet(
        encoder=biosyn.encoder,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        use_cuda=args.use_cuda,
        forward_chunk_size=args.forward_chunk_size,
        metrics_train=metrics_train,
    )
    LOGGER.info(f"Reranknet model is initiated with: learning_rate={args.learning_rate},weight_decay={args.weight_decay}, use_cuda={args.use_cuda} ")

    train_set = CandidateDataset(
        queries = train_queries, 
        dicts = train_dictionary, 
        tokenizer = biosyn.get_dense_tokenizer(), 
        max_length = args.max_length, 
        topk= args.topk,
        pre_tokenize = args.pre_tokenize
    )
    LOGGER.info(f"Candidate DS is initiated with len queries: {len(train_queries)}, len dicts: {len(train_dictionary)}, max_length: {args.max_length}, topk: {args.topk} ")
    LOGGER.info(f"The training will {' not use faiss for score matrix' if  args.not_use_faiss else 'use faiss for score matrix'}")

    start = time.time()
    LOGGER.info(f"Training will start at time: {start} and with {args.epoch} epochs")

    metrics_train.start_run()
    for epoch in tqdm(range(1,args.epoch+1)):
        t_epoch = time.time()


        if args.not_use_faiss:
            query_embs = biosyn.embed_dense(names=names_in_train_queries)
            dict_embs = biosyn.embed_dense(names=names_in_train_dict)
            train_dense_score_matrix = biosyn.get_score_matrix(
                dict_embeds=dict_embs,
                query_embeds=query_embs,
            )
            train_dense_candidate_idxs = biosyn.retrieve_candidate(
                score_matrix=train_dense_score_matrix, 
                topk=args.topk
            )
            # replace dense candidates in the train_set
            train_set.set_dense_candidate_idxs(d_candidate_idxs=train_dense_candidate_idxs)
        else:
            # dict_mmap_base = mmap_dir + f"/dicts_{epoch}"
            # dict_mmap_path, dict_meta = biosyn.embed_names_mmap(names=names_in_train_dict, memap_path_base=dict_mmap_base)
            dict_mmap_base = mmap_dir + f"/dicts_{epoch}"
            metrics_faiss.start_run()
            queries_mmap_base = mmap_dir + f"/queries_{epoch}"
            queries_mmap_path, queries_meta = biosyn.embed_names_mmap(names=names_in_train_queries, memap_path_base=queries_mmap_base)
            metrics_faiss.log_event("embed dense", epoch)
            # save the index if we want after
            #if you want to save the embedings in mmap file, do dict_mmap_base=dict_mmap_base
            build_index_t0 = time.time()
            faiss_index = biosyn.build_faiss_index(dict_names=names_in_train_dict)
            metrics_faiss.log_event("built index", epoch, t0=build_index_t0)


            # dtype = np.float16 if dict_meta["dtype"] == "fp16" else np.float32
            # dict_mm = np.memmap(dict_meta["path"], mode="r", dtype=dtype, shape=(dict_meta["N"], dict_meta["d"]))
            # model.set_dictionary_embedings(dict_mm)

            search_index_t0 = time.time()
            results_mmap_base = mmap_dir + f"/results_{epoch}"
            train_dense_candidate_idxs = biosyn.search_index_mmap(queries_mmap_base=queries_mmap_base , save_results_mmap_base=results_mmap_base)
            metrics_faiss.log_event("search index", epoch, t0=search_index_t0)
            train_set.set_dense_candidate_idxs(train_dense_candidate_idxs)


        train_loader = DataLoader(
            train_set, batch_size=args.train_batch_size, 
            shuffle=False,
            num_workers=min(4, (os.cpu_count() // 2 or 2)),
            pin_memory=args.use_cuda,
            persistent_workers=args.use_cuda
        )
        # train_loss = train(data_loader=train_loader, model=model)
        train_loss = train_new(data_loader=train_loader, model=model, has_fp16=biosyn.has_fp16, epoch_num=epoch, pkls_dir=pkls_dir, metrics_train=metrics_train)
        # LOGGER.info(f'We are in epoch number: {epoch}, we have training loss {train_loss}')
        metrics_train.log_event("epoch_end", epoch=epoch, loss=train_loss, t0=t_epoch)

        if args.save_checkpoint_all:
            checkpoint_dir = os.path.join(args.output_dir, "checkpoint_{}".format(epoch))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            biosyn.save_model(checkpoint_dir)

        if epoch == args.epoch:
            biosyn.save_model(args.output_dir)
    end = time.time()
    training_time = end-start
    training_hour = int(training_time/60/60)
    training_minute = int(training_time/60 % 60)
    training_second = int(training_time % 60)
    LOGGER.info("Training Time!{} hours {} minutes {} seconds".format(training_hour, training_minute, training_second))
    metrics_train.end_run()


if __name__ == "__main__":
    main()




# python train.py \
#     --model_name_or_path ${MODEL_NAME_OR_PATH} \
#     --train_dictionary_path ${DATA_DIR}/train_dictionary.txt \
#     --train_dir ${DATA_DIR}/processed_traindev \
#     --output_dir ${OUTPUT_DIR} \
#     --use_cuda \
#     --topk 20 \
#     --epoch 10 \
#     --train_batch_size 16\
#     --initial_sparse_weight 0\
#     --learning_rate 1e-5 \
#     --max_length 25 \
#     --dense_ratio 0.5
