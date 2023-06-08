import time
import numpy as np
import torch
from torch.optim import lr_scheduler
from settings import Path, Config
import utils
from utils import LogInfo
from datagenerator import get_data_loader
from myloss import MyWeightedLoss
import CWN


def main(base_model, test_title):
    model = CWN.cwn(base_model, Config.CLASS_NUM, Config.K_W)
    # model.load_state_dict(torch.load('Model/' + '16_19_best_efficientnet model.pkl'))

    model.to(Config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3,
                                               verbose=True)

    # criterion = nn.CrossEntropyLoss().to(Config.DEVICE)
    criterion = MyWeightedLoss().to(Config.DEVICE)

    dataloader_train = get_data_loader(mode='train', file_root=Path.ROOT_TRAIN)
    dataloader_val = get_data_loader(mode='valid', file_root=Path.ROOT_VAL)

    log_info = LogInfo()
    start_episode = 0
    print('start train...')

    best_kappa = 0
    for episode in range(Config.EPISODE):

        '''=====================================  train  ================================'''
        model.train()

        sum_loss = 0
        time_start = time.time()
        correct = 0
        total = 0
        confusion_matrix = np.zeros((Config.CLASS_NUM, Config.CLASS_NUM), dtype=int)
        data_size = dataloader_train.__len__()

        '''=============  batch  ==========='''
        for idx, batch in enumerate(dataloader_train):
            datas, labels = batch[0], batch[1]
            datas = datas.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            preds = model(datas)

            predicted = torch.argmax(preds, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i in range(labels.size(0)):
                confusion_matrix[labels[i].item()][predicted[i].item()] += 1

            loss = criterion(preds, labels)
            sum_loss += loss.item()

            loss = loss / Config.BATCH_ACCUMULATE  # loss regularization
            loss.backward()

            if ((idx + 1) % Config.BATCH_ACCUMULATE) == 0:
                optimizer.step()
                optimizer.zero_grad()

        '''=============  batch  ==========='''

        # g_info = criterion.get_g_info()
        # utils.out_put_g_graph('base_train_g', g_info)

        time_end = time.time()

        sum_loss = sum_loss
        sum_loss = sum_loss / data_size
        log_info.train_losses.append(sum_loss)

        accuracy = correct / total
        weight_kappa_score = utils.weight_kappa(confusion_matrix, total, Config.CLASS_NUM)
        log_info.train_accuracy.append(accuracy)
        log_info.train_kappa.append(weight_kappa_score)

        utils.print_log_per_epoch(Path.LOG_TRAIN, episode + start_episode, accuracy, confusion_matrix,
                                  time_end - time_start, sum_loss, weight_kappa_score)

        print('train episode: ' + str(episode + start_episode))
        print('time: ', time_end - time_start)
        print('loss: ', sum_loss)
        print('acc: ', accuracy)
        print('kappa: ', weight_kappa_score)

        '''=====================================  train  ================================'''

        if (episode + start_episode) % 5 == 0:
            torch.save(model.state_dict(),
                       Path.ROOT_MODEL + '/' + test_title + '_' + base_model + ' %03d model.pkl' % (
                               episode + start_episode))
            print(test_title + '_' + base_model + ' model %03d Saved..' % (episode + start_episode))

        '''=====================================  val  ================================'''
        with torch.no_grad():
            print("validating...")
            model.eval()

            sum_loss = 0
            time_start = time.time()
            correct = 0
            total = 0
            confusion_matrix = np.zeros((Config.CLASS_NUM, Config.CLASS_NUM), dtype=int)
            data_size = dataloader_val.__len__()

            '''=============  batch  ==========='''
            for idx, (datas, labels) in enumerate(dataloader_val):
                datas = datas.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE)

                preds = model(datas)

                predicted = torch.argmax(preds, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                for i in range(labels.size(0)):
                    confusion_matrix[labels[i].item()][predicted[i].item()] += 1

                loss = criterion(preds, labels)
                sum_loss += loss.item()
            '''=============  batch  ==========='''

            time_end = time.time()

            sum_loss = sum_loss
            sum_loss = sum_loss / data_size
            log_info.val_losses.append(sum_loss)

            accuracy = correct / total
            weight_kappa_score = utils.weight_kappa(confusion_matrix, total, Config.CLASS_NUM)
            log_info.val_accuracy.append(accuracy)
            log_info.val_kappa.append(weight_kappa_score)

            utils.print_log_per_epoch(Path.LOG_VAL, episode + start_episode, accuracy, confusion_matrix,
                                      time_end - time_start, sum_loss, weight_kappa_score)

            print('val episode: ' + str(episode + start_episode))
            print('time: ', time_end - time_start)
            print('loss: ', sum_loss)
            print('acc: ', accuracy)
            print('kappa: ', weight_kappa_score)
            print('\n')

            if best_kappa < weight_kappa_score:
                torch.save(model.state_dict(),
                           Path.ROOT_MODEL + '/' + test_title + '_best_' + base_model + ' model.pkl')
                print('######## ' + test_title + '_best_' + base_model + ' model %03d Saved.. ########\n' % (
                        episode + start_episode))
                best_kappa = weight_kappa_score

            scheduler.step(sum_loss)
        '''=====================================  val  ================================'''

    utils.out_put_line_graph(base_model, log_info)


if __name__ == '__main__':
    main("efficientnet", '18_12')
