import time
import numpy as np
import torch
from settings import Path, Config
import utils
from datagenerator import get_data_loader
import CWN

def main(base_model, test_model):
    model = CWN.cwn(base_model, Config.CLASS_NUM, Config.K_W)
    model.load_state_dict(torch.load('Model/' + test_model))
    model.to(Config.DEVICE)
    dataloader_test = get_data_loader(mode='test', file_root=Path.ROOT_TEST)

    print('start testing...')

    '''=====================================  test  ================================'''
    with torch.no_grad():
        print("testing...")
        model.eval()

        time_start = time.time()
        correct = 0
        total = 0
        confusion_matrix = np.zeros((5, 5), dtype=int)

        '''=============  batch  ==========='''
        for idx, (datas, labels) in enumerate(dataloader_test):
            datas = datas.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)

            preds = model(datas)

            predicted = torch.argmax(preds, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i in range(labels.size(0)):
                confusion_matrix[labels[i].item()][predicted[i].item()] += 1

        '''=============  batch  ==========='''

        time_end = time.time()
        accuracy = correct / total
        weight_kappa_score = utils.weight_kappa(confusion_matrix, total, 5)
        utils.print_log_per_epoch(Path.LOG_TEST, 0, accuracy, confusion_matrix, time_end - time_start,
                                  kappa=weight_kappa_score)

        print('test result: ')
        print('time: ', time_end - time_start)
        print('acc: ', accuracy)
        print('weight_kappa_score: ', weight_kappa_score)
        print('\n')

    '''=====================================  test  ================================'''


if __name__ == '__main__':
    main("efficientnet", '18_6_best_efficientnet model.pkl')