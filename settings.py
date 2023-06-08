import torch

class Path:
    # ----- Roots ------#
    ROOT = ""
    ROOT_TRAIN = ROOT + 'train.txt'
    ROOT_VAL = ROOT + 'valid.txt'
    ROOT_TEST = ROOT + 'test.txt'
    ROOT_MODEL = ROOT + 'Model'
    ROOT_PIC_DATASET = ROOT + 'Datas'

    # ----- Logs ------#
    LOG_PROCESSING = ROOT + 'log_processing.txt'
    LOG_TRAIN = ROOT + 'log_train.txt'
    LOG_VAL = ROOT + 'log_valid.txt'
    LOG_TEST = ROOT + 'log_test.txt'

class Config:
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 0.000001
    EPISODE = 50
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    WORKERS_NUM = 2
    BATCH_SIZE = 4
    BATCH_ACCUMULATE = 8
    MIX_UP = 0.1
    STEP = 0
    K_W = [5, 33, 8, 27, 5]
    # K_W = [5, 5, 5, 5, 5]

    CLASS_NUM = 5
    PIC_SIZE = 512

