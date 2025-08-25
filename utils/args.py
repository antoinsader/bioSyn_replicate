
import torch 
import argparse


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
