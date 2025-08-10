import argparse
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


os.environ["KMP_VERBOSE"] = "1"

MINIMIZE = False
TOP_K = 20
NUM_EPOCHS= 10
TRAIN_BATCH_SIZE  = 16

MAX_LENGTH = 25
USE_CUDA = False
LEARNING_RATE = 1e-5

SAVE_CHKPNT_ALL =True
NOT_USE_FAISS = False


ENCODER_MODEL_NAME = 'dmis-lab/biobert-base-cased-v1.1' #Dense encoder model nmae
TRAIN_DICT_PATH = "./data/ncbi-disease/train_dictionary.txt"
TRAIN_DIR = "./data/ncbi-disease/processed_traindev"
OUTPUT_DIR = "./data/output"
WEIGHT_DECAY = 0.01


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
    parser.add_argument('--not_use_faiss',  action="store_true", default=NOT_USE_FAISS)
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
def main():
    args = parse_args()
    init_logging(LOGGER, base_output_dir= args.output_dir, logging_folder="train", minimize=args.draft)
    init_seed(LOGGER, args.seed)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # np array containing each item is (cui, name)
    # N
    train_dictionary  = load_dictionary(dict_path=args.train_dictionary_path)

    # np array containing each item is (cui, mention)
    # M
    train_queries  = load_queries(data_dir=args.train_dir, filter_composite=False, filter_duplicates=False, filter_cuiless=True)

    if args.draft:
        train_dictionary = train_dictionary[:1000]
        train_queries = train_queries[:100]
        args.output_dir = args.output_dir + "_min"

    LOGGER.info(f"train_dictionary is loaded from file: {args.train_dictionary_path} with minimize set to: {'True' if args.draft else 'False'}, the length is: {len(train_dictionary)}")
    LOGGER.info(f"train_queries is loaded from file: {args.train_dir} with minimize set to: {'True' if args.draft else 'False'}, the length is: {len(train_queries)}")

    # train_dictionary = get_pkl("./data/train_dict.pkl")
    # train_queries = get_pkl("./data/train_queries.pkl")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    names_in_train_dict = train_dictionary[:, 0] # N
    names_in_train_queries = train_queries[:, 0] # M

    biosyn = BioSyn(args.max_length, args.use_cuda)
    LOGGER.info(f"Loading encoder from: {args.model_name_or_path}")
    biosyn.load_dense_encoder(args.model_name_or_path)

    model = RerankNet(
        encoder=biosyn.encoder,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        use_cuda=args.use_cuda
    )
    LOGGER.info(f"Reranknet model is initiated with: learning_rate={args.learning_rate},weight_decay={args.weight_decay}, use_cuda={args.use_cuda} ")

    train_set = CandidateDataset(
        queries = train_queries, 
        dicts = train_dictionary, 
        tokenizer = biosyn.get_dense_tokenizer(), 
        max_length = args.max_length, 
        topk= args.topk
    )
    LOGGER.info(f"Candidate DS is initiated with len queries: {len(train_queries)}, len dicts: {len(train_dictionary)}, max_length: {args.max_length}, topk: {args.topk} ")
    LOGGER.info(f"The training will {'will not use faiss for score matrix' if  args.not_use_faiss else 'use faiss for score matrix'}")


    start = time.time()
    LOGGER.info(f"Training will start at time: {start} and with {args.epoch} epochs")
    #for epoch
    for epoch in tqdm(range(1,args.epoch+1)):
        # LOGGER.info("Epoch {}/{}".format(epoch,args.epoch))


        if args.not_use_faiss:
            dict_embs = biosyn.embed_dense(names=names_in_train_dict)
            query_embs = biosyn.embed_dense(names=names_in_train_queries)
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
            dict_embs = biosyn.embed_dense(names=names_in_train_dict)
            index = faiss.IndexFlatIP(dict_embs.shape[1])
            index.add(dict_embs)
            model.dict_embs_tensor = torch.from_numpy(dict_embs)
            query_embs = biosyn.embed_dense(names=names_in_train_queries)

            _, train_dense_candidate_idxs = index.search(query_embs, args.topk)
            train_set.set_dense_candidate_idxs(train_dense_candidate_idxs)





        train_loader = DataLoader(
            train_set, batch_size=args.train_batch_size, shuffle=False
        )
        train_loss = train(data_loader=train_loader, model=model)
        LOGGER.info(f'We are in epoch number: {epoch}, we have training loss {train_loss}')


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
