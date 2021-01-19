import numpy as np
import math, time
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity

#
np.random.seed(348)
show =  not True
show_test = True
fn_train = 'dataTrain.bin'
fn_train_labels = 'labelsTrain.bin'
fn_test = 'dataTest.bin'
fn_test_labels = 'labelsTest.bin'
n_train = 600 # Число рисунков для обучения
n_test = 100 # Число тестовых рисунков
clr_mim, clr_max = 75, 255 # Диапазон оттенков серого цвета
w, h = 64, 64 # Ширина и высота рисунка
w2 = w / 2
border = 1 # Граница
def iss(data): 
    m = 0.5
    s_iss=[]
    cnt = 0
    for i in range(len(data)):
        im1 = data[i].reshape(w*h)
        for j in range(i + 1, len(data)):
            im2 = data[j].reshape(w*h)
            iss = (1-abs(structural_similarity(im1, im2)))/2 
            if iss < m:
                m = iss
                s_iss.append(m)
    '''print(len(s_iss))
    print(int(len(s_iss)/5))
    print(s_iss)
    print(s_iss[len(s_iss)-int(len(s_iss)/5):])
    print(m)'''
    return sum(s_iss[len(s_iss)-int(len(s_iss)/5):])/int(len(s_iss)/5)

def check_sim(data,iss):
    n = len(data)
    cnt = 0
    for i in range(n):
        im1 = data[i].reshape(w*h)
        for j in range(i + 1, n):
            im2 = data[j].reshape(w*h)
            d = (1-structural_similarity(im1, im2))/2
            if d < iss:
                cnt+=1
    return cnt

#
def figure(a,t):
    x = a * ((t)**2 - 1) / (3 * (t)**2 + 1)
    y = a * (t) * ((t)**2 - 1) / (3 * (t)**2 + 1)  
    for i in range(len(x)):
        x_noise = np.random.uniform(-4,4)
        x[i] += x_noise
        y_noise = np.random.uniform(-0.5,0.5)
        y[i] += y_noise

    return x,y
#
def one_class(n, fn, fn2):
    file = open(fn, 'wb')
    file2 = open(fn2, 'wb')
    label = 0
    t = np.linspace(-5, 5, 400)
    for i in range(n): # n - число примеров
        a = np.random.randint(-20,21)
        x,y = figure(a,t)
        x_min = int(min(x))
        x_max = int(max(x))
        y_min = int(min(y))
        y_max = int(max(y))
        dx = int((64 - (x_max - x_min)) / 2) # Половина свободного пространства по x
        dy = int((64 - (y_max - y_min)) / 2) # Половина свободного пространства по y
        shift_x = dx - x_min # Сдвиг по x
        shift_y = dy - y_min # Сдвиг по y
        w = h = 64 # Ширина и высота рисунка
        arrPic = np.zeros((w, h), dtype = np.uint8)
        clr_mim, clr_max = 75, 255 # Диапазон оттенков серого цвета
        for x, y in zip(x, y):
            ix = int(x) + shift_x
            iy = int(y) + shift_y
            clr = np.random.randint(clr_mim, clr_max)
            arrPic[iy, ix] = clr
        file.write(arrPic)
        file2.write(np.uint8(label))
    file.close()
    file2.close()
#
def load_data(fn, fn2):
    with open(fn, 'rb') as read_binary:
        data = np.fromfile(read_binary, dtype = np.uint8)
    with open(fn2, 'rb') as read_binary:
        labels = np.fromfile(read_binary, dtype = np.uint8)
    return data, labels


if not show:
    t0 = time.time()
    print('Поехали')
    one_class(n_train, fn_train, fn_train_labels)
    one_class(n_test, fn_test, fn_test_labels)
    print('Потрачено времени:', round(time.time() - t0, 3))

    def plotData(data, ttl, cls):
        plt.figure(ttl)
        k = 0
        for i in range(10):
            j = np.random.randint(data.shape[0])
            k += 1
            plt.subplot(1, 10, k)
            plt.imshow(data[i], cmap = 'gray')
            plt.title(cls, fontsize = 11)
            plt.axis('off')
        plt.subplots_adjust(hspace = -0.1) # wspace
        plt.show()
    if show_test:
        test_data, _ = load_data(fn_test, fn_test_labels)
        test_data = test_data.reshape(n_test, w, h)
        min_iss_test=iss(test_data)
        print('Минимальный индекс структурного различия в проверочном множестве',min_iss_test)
        print('Количество аналогов', check_sim(test_data, min_iss_test ))
        train_data, _ = load_data(fn_train, fn_train_labels)
        train_data = train_data.reshape(n_train, w, h)
        min_iss_train=iss(train_data)
        print('Минимальный индекс структурного различия в обучающем множестве',min_iss_train)
        print('Количество аналогов', check_sim(train_data, min_iss_train)) 

    ttl = 'Декартов Лист'
    cls = 2
    plotData(test_data, ttl, cls)
