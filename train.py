import argparse
import logging
import os
import time
from xmlrpc.client import Boolean

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



# use tf32 for float32 ops
torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.allow_tf32 = True

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
    parser.add_argument('--forward_chunk_size', default=128, type=int)
    parser.add_argument('--draft_prcntg', default=.5, type=float)


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

def train_new(data_loader, model,  epoch_num, biosyn,metrics_train, pkls_dir=None):
    # saved_batches = []
    use_cuda = model.use_cuda
    use_fp16 = False

    amp_dtype = torch.float16 if use_fp16 else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    use_amp = use_cuda


    model.train()
    train_loss = 0
    train_steps = 0


    metrics_train.show_gpu_memory(f"Memory at #epoch_{epoch_num} before entering the data loader loop ")

    for i, data in tqdm(enumerate(data_loader), total=len(data_loader), desc="training", unit="epochs"):
        batch_x, batch_y = data
        one_batch_all_start_time = time.time()
        model.optimizer.zero_grad(set_to_none=True)

        #other implementation:
        #  For this, in old forward I was encoding query_embedings for each token, in other suggestion, 
        #   I am computing query_embeddings once before and passing
        # running = 0.0
        # batch_size = batch_y.shape[0]
        # forward_chunk_size = model.forward_chunk_size
        # if has_fp16 and model.use_cuda:
        #     for s in range(0, batch_size, forward_chunk_size):
        #         e = min(s+forward_chunk_size , batch_size)
        #         with torch.amp.autocast("cuda"):
        #             loss_chunk = model.forward_chunk_loss(batch_x, batch_y, s, e)
        #             loss_chunk = loss_chunk * ((e-s) / batch_size)
        #         running += float(loss_chunk.detach())
        #         scaler.scale(loss_chunk).backward()
        #     scaler.step(model.optimizer)
        #     scaler.update() 
        # train_loss += running
        # train_steps += 1


        #this blows oom
            # ...
            #     batch_pred = model(batch_x)
            #     loss = model.get_loss(batch_pred, batch_y)
            # ...
        #embed queries with keeping the grpahs all once
        queries_tokens, cands_tokens = batch_x
        batch_size = batch_y.shape[0]
        chunk_size = model.forward_chunk_size

        loss_chunk_all = 0.0
        query_embs_start = time.time()
        # encode with low-precision but keep loss in fp32 later
        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_cuda):
            query_embeddings = biosyn.encode_queries_with_keeping_graph(queries_tokens)
        metrics_train.log_event("query all embeddings", epoch=epoch_num,  t0=query_embs_start, log_immediate=False,  only_elapsed_time=True, first_iteration_only=True)

        embed_cands_start = time.time()
        #embed candidates in chunks
        for start in range(0, batch_size, chunk_size):
            """
            What gets freed each iteration?
                The candidate activations/graph for the current chunk are released right after its .backward().
                The query graph persists (because of retain_graph=True) until you backprop the final chunk.
            """
            end =min(start+chunk_size, batch_size)
            if use_amp:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    loss_chunk = model.forward_chunk_loss(
                        candidate_tokens=cands_tokens,
                        targets_chunk=batch_y[start:end],
                        start=start,
                        end=end,
                        query_embeddings_chunk=query_embeddings[start:end]
                    )

                chunky_loss_weight = (end - start ) / batch_size
                loss_chunk = loss_chunk * chunky_loss_weight
                # we don't need the graphs just here so we detach and we moved to float32
                loss_chunk_all += float(loss_chunk.detach())
                
            else:
                # loss_chunk = model.chunk
                print("NOT USING AMP!")
                break
            
            #keep the shared query graph alive until last chunk
            #we retain the autograd graphs because graph of query_embeddings is shared by all 
            retain = end < batch_size
            if use_fp16:
                scaler.scale(loss_chunk).backward(retain_graph=retain)
            else: #bf16, no sclaer
                loss_chunk.backward(retain_graph=retain)
 
        metrics_train.log_event("all chunks finished", epoch=epoch_num,  t0=embed_cands_start, log_immediate=False,  only_elapsed_time=True, first_iteration_only=True)
        del embed_cands_start


        stepping_start_time = time.time()


        #step per batch
        if use_fp16:
            scaler.step(model.optimizer)
            scaler.update()
        else:
            model.optimizer.step()

        train_loss += loss_chunk_all
        train_steps += 1
        metrics_train.log_event("stepping batch", epoch=epoch_num, t0=stepping_start_time, first_iteration_only=True, log_immediate=False,  only_elapsed_time=True)
        del stepping_start_time

        # if pkls_dir:
            # with torch.no_grad():
                # to_save = {
                #     "epoch": epoch_num,
                #     "y": batch_y.detach().cpu(),
                #     "pred": batch_pred.detach().cpu(),
                #     "x": batch_x
                # }
            # saved_batches.append(to_save)

        metrics_train.log_event("one batch all", epoch=epoch_num, t0=one_batch_all_start_time, first_iteration_only=True, log_immediate=False,  only_elapsed_time=True)

    # if pkls_dir:
        # save_pkl(saved_batches, pkls_dir + f"/epoch_{epoch_num}_results.pkl")


    LOGGER.info(f"Epoch num: {epoch_num} has did loss: {train_loss}")
    LOGGER.info(f"Epoch {epoch_num} loss (running avg): {train_loss / max(1, train_steps):.6f}")
    train_loss /= max(1, train_steps)
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
        dict_size =int(args.draft_prcntg * len(train_dictionary))
        query_size =int(args.draft_prcntg * len(train_queries))
        train_dictionary = train_dictionary[:dict_size]
        train_queries = train_queries[:query_size]
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
    LOGGER.info(f"train_queries is loaded from file: {args.train_dir} with minimize set to: {'True' if args.draft else 'False'}, the length is: {len(train_queries)}, draft percentage: {args.draft_prcntg}")


    save_pkl(train_dictionary, f"{pkls_dir}/train_dictionary.pkl")
    save_pkl(train_queries, f"{pkls_dir}/train_queries.pkl")
    # train_dictionary = get_pkl("./data/train_dict.pkl")
    # train_queries = get_pkl("./data/train_queries.pkl")



    metrics_train = MetricsLogger(
        logger=LOGGER,
        use_cuda=args.use_cuda,
        tag="train"
    )
    metrics_epoch = MetricsLogger(
        logger=LOGGER,
        use_cuda=args.use_cuda,
        tag="epoch"
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
    LOGGER.info(f"forward_chunk_size: {args.forward_chunk_size}")

    model = RerankNet(
        encoder=biosyn.encoder,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        use_cuda=args.use_cuda,
        forward_chunk_size=args.forward_chunk_size,
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
        metrics_epoch.start_run()
        metrics_epoch.show_gpu_memory(f"Memory at the first of epoch: {epoch}")
        model.metrics =metrics_epoch

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
            metrics_faiss.log_event("Index was built in time specified, and the memory used: ", epoch, t0=build_index_t0)


            # dtype = np.float16 if dict_meta["dtype"] == "fp16" else np.float32
            # dict_mm = np.memmap(dict_meta["path"], mode="r", dtype=dtype, shape=(dict_meta["N"], dict_meta["d"]))
            # model.set_dictionary_embedings(dict_mm)

            search_index_t0 = time.time()
            results_mmap_base = mmap_dir + f"/results_{epoch}"
            train_dense_candidate_idxs = biosyn.search_index_mmap(queries_mmap_base=queries_mmap_base , save_results_mmap_base=results_mmap_base)
            metrics_faiss.log_event("search index", epoch, t0=search_index_t0)
            train_set.set_dense_candidate_idxs(train_dense_candidate_idxs)
            metrics_train.show_gpu_memory("Memory after finishing from faiss index")

        train_loader = DataLoader(
            train_set, batch_size=args.train_batch_size, 
            shuffle=False,
            num_workers=min(8, (os.cpu_count() or 8)),
            pin_memory=args.use_cuda,
            persistent_workers=args.use_cuda,
            prefetch_factor=4
        )
        # train_loss = train(data_loader=train_loader, model=model)
        train_loss = train_new(
            data_loader=train_loader, 
            model=model, 
            epoch_num=epoch, 
            metrics_train=metrics_train,
            biosyn=biosyn)
        # LOGGER.info(f'We are in epoch number: {epoch}, we have training loss {train_loss}')
        metrics_epoch.log_event("epoch_end", epoch=epoch, loss=train_loss, t0=t_epoch)

        if args.save_checkpoint_all:
            checkpoint_dir = os.path.join(args.output_dir, "checkpoint_{}".format(epoch))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            biosyn.save_model(checkpoint_dir)

        if epoch == args.epoch:
            biosyn.save_model(args.output_dir)
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        metrics_epoch.end_run()
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
