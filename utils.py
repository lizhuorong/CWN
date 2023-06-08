import numpy as np
import os
import math
import cv2
from matplotlib import pyplot as plt
import itertools

class LogInfo:
    train_losses = []
    val_losses = []
    train_accuracy = []
    val_accuracy = []
    train_kappa = []
    val_kappa = []

def print_log_per_epoch(log_path, episode, accuracy, confusion_matrix, runtime, loss=-1, kappa=-1):
    f = open(log_path, 'a')
    f.write('epoch: ' + str(episode) + '\n')

    if loss >= 0:
        f.write('loss: ' + str(loss) + '\n')
    if kappa >= 0:
        f.write('weight_kappa_score: ' + str(kappa) + '\n')
    f.write('accuracy: ' + '{:.2%}'.format(accuracy) + '\n')

    cow = np.sum(confusion_matrix, axis=0)
    class_num = np.size(confusion_matrix, 0)
    for i in range(class_num):
        f.write('accuracy of ' + str(i) + ': ' + str(confusion_matrix[i][i]) + '/' + str(
            cow[i]) + " " + '{:.2%}'.format(confusion_matrix[i][i] / cow[i]) + '\n')

    f.write("confusion_matrix:\n")
    for i in range(class_num):
        for j in range(class_num):
            f.write('{:10}'.format(confusion_matrix[i][j]))
        f.write('\n')

    f.write("runTime: " + str(runtime) + '\n\n')
    f.close()

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def out_put_g_graph(name, g_list):
    y = g_list
    x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    plt.figure(1)
    plt.plot(x, y)
    plt.ylabel('num')
    plt.xlabel('g')
    plt.savefig(name + '.jpg')

def out_put_line_graph(name, log_info):
    acc = log_info.train_accuracy
    val_acc = log_info.val_accuracy
    loss = log_info.train_losses
    val_loss = log_info.val_losses
    kappa = log_info.train_kappa
    val_kappa = log_info.val_kappa
    epochs = range(1, len(acc) + 1)

    plt.figure(1)
    # plt.plot(epochs, smooth_curve(acc))
    # plt.plot(epochs, smooth_curve(val_acc))
    plt.plot(epochs, acc)
    plt.plot(epochs, val_acc)
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train_acc', 'val_acc'], loc='upper left')
    plt.savefig('acc_' + name + '.jpg')

    plt.figure(2)
    # plt.plot(epochs, smooth_curve(loss))
    # plt.plot(epochs, smooth_curve(val_loss))
    plt.plot(epochs, loss)
    plt.plot(epochs, val_loss)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'val_loss'], loc='upper right')
    plt.savefig('loss_' + name + '.jpg')

    plt.figure(3)
    # plt.plot(epochs, smooth_curve(loss))
    # plt.plot(epochs, smooth_curve(val_loss))
    plt.plot(epochs, kappa)
    plt.plot(epochs, val_kappa)
    plt.ylabel('kappa')
    plt.xlabel('epoch')
    plt.legend(['train_kappa', 'val_kappa'], loc='upper right')
    plt.savefig('kappa_' + name + '.jpg')


def weight_kappa(result, test_num, class_num):
    weight = np.zeros((class_num, class_num), dtype='float')
    for i in range(class_num):
        for j in range(class_num):
            weight[i, j] = (i - j) * (i - j) / ((class_num - 1) * (class_num - 1))
    up = 0
    for i in range(class_num):
        for j in range(class_num):
            up = up + result[i, j] * weight[i, j]
    down = 0
    for i in range(class_num):
        for j in range(class_num):
            down = down + weight[i, j] * result[:, j].sum() * result[i, :].sum()

    weight_kappa_score = 1 - (up / (down / test_num))
    return weight_kappa_score

def crop_image_from_gray(img, tol=20):
    """
    Crop out black borders
    https://www.kaggle.com/ratthachat/aptos-updated-preprocessing-ben-s-cropping
    """

    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def circle_crop_v2(img, resize=512, tol=20):
    """
    Create circular crop around image centre
    """
    # img = cv2.imread(img)
    img = crop_image_from_gray(img, tol)
    height, width, depth = img.shape
    largest_side = np.max((height, width))
    img = cv2.resize(img, (largest_side, largest_side))

    height, width, depth = img.shape

    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))

    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img, tol)
    img = cv2.resize(img, (resize, resize))

    return img


def pre_load_ben_color(image, sigmax=10):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmax), -4, 128)

    return image


def get_image_pixel_mean(img_dir, img_list):
    R_sum = 0
    G_sum = 0
    B_sum = 0
    count = 0
    for img_name in img_list:
        img_path = os.path.join(img_dir, img_name[:-3])
        if not os.path.isdir(img_path):
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image / 255
            r_img = image[:, :, 0]
            g_img = image[:, :, 1]
            b_img = image[:, :, 2]
            R_sum += r_img[r_img > 0].mean()
            G_sum += g_img[g_img > 0].mean()
            B_sum += b_img[b_img > 0].mean()
            count += 1
    R_mean = R_sum / count
    G_mean = G_sum / count
    B_mean = B_sum / count
    print('RGB_mean:{:.5f}, {:.5f}, {:.5f}'.format(R_mean, G_mean, B_mean))
    RGB_mean = [R_mean, G_mean, B_mean]
    return RGB_mean


def get_image_pixel_std(img_dir, img_mean, img_list):
    R_squared_mean = 0
    G_squared_mean = 0
    B_squared_mean = 0
    count = 0
    image_mean = np.array(img_mean)
    for img_name in img_list:
        img_path = os.path.join(img_dir, img_name[:-3])
        if not os.path.isdir(img_path):
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image / 255
            r_img = image[:, :, 0]
            g_img = image[:, :, 1]
            b_img = image[:, :, 2]
            r_img = r_img[r_img > 0] - image_mean[0]
            g_img = g_img[g_img > 0] - image_mean[1]
            b_img = b_img[b_img > 0] - image_mean[2]
            R_squared_mean += np.mean(np.square(r_img))
            G_squared_mean += np.mean(np.square(g_img))
            B_squared_mean += np.mean(np.square(b_img))
            count += 1
    R_std = math.sqrt(R_squared_mean / count)
    G_std = math.sqrt(G_squared_mean / count)
    B_std = math.sqrt(B_squared_mean / count)
    print('RGB_std:{:.5f}, {:.5f}, {:.5f}'.format(R_std, G_std, B_std))
    RGB_std = [R_std, G_std, B_std]
    return RGB_std

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    C = cm
    cm = cm.astype('float') / cm.sum(axis=0)[:, np.newaxis]
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        str = '%.2f' % cm[i, j]
        str = str + '\n' + '%d'%(C[i,j])
        plt.text(j, i, str,
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(title + '.jpg')
    plt.show()



