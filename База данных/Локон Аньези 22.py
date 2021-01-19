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
border = 8 # Граница
#
def average(data):
    s = 0
    cnt = 0
    for i in range(len(data)):
        im1 = data[i].reshape(w*h)
        for i in range(len(data)):
            im2 = data[j].reshape(w*h)
            iss = (1-abs(structural_similarity(im1, im2)))/2 
            s+=iss
            cnt+=1
    print(s/cnt)
    return s/cnt
#
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
def similar_pic(data, iss):
    s = 0
    cnt = 0
    pics = []
    for i in range(len(data)):
        im1 = data[i].reshape(w*h)
        for j in range(len(data)):
            im2 = data[j].reshape(w*h)
            iss_1 = (1-abs(structural_similarity(im1, im2)))/2
            if ((iss<iss_1) & (iss_1<0.3)):
                pics.append(im1)
                pics.append(im2)
                break
        if (len(pics)==10):
            return pics

    
def lokon_anezi(a,t):
    x = 2 * a / np.tan(t)
    y = 2 * a * np.sin(t)**2
    for i in range(len(x)):
        x_noise = np.random.uniform(-0.5,0.5)
        x[i] += x_noise
        y_noise = np.random.uniform(-2,2)
        y[i] += y_noise + border
    return x,y
#
def one_class(n, fn, fn2):
    file = open(fn, 'wb')
    file2 = open(fn2, 'wb')
    label = 0
    t = np.linspace(0.1, np.pi - 0.1, 100)
    for i in range(n): # n - число примеров
        sgn = np.random.randint(2) # x или -x
        a = np.random.randint(1,20)
        x,y = lokon_anezi(a,t)
        arrPic = np.zeros((w, h), dtype = np.uint8)
        j=0
        while (x[j]>0):
            ix1 = min(w - 1, int(w2 + x[j]))
            ix2 = max(0, int(w2 - x[j]))
            iy = h - int(y[j])
            iy = max(0, iy)
            iy = min(h - 1, iy) # Уходим из физической системы координат
            clr = np.random.randint(clr_mim, clr_max)
            arrPic[iy, ix1] = clr
            arrPic[iy, ix2] = clr
            j+=1;
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
#
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
            plt.subplot(2, 5, k)
            plt.imshow(data[i], cmap = 'gray')
            plt.title(cls, fontsize = 11)
            plt.axis('off')
        plt.subplots_adjust(hspace = -0.1) # wspace
        plt.show()
    if show_test:
        test_data, _ = load_data(fn_test, fn_test_labels)
        print(test_data.shape)
        test_data = test_data.reshape(n_test, w, h)
        min_iss_test=iss(test_data)
        print('Минимальный индекс структурного различия в проверочном множестве',min_iss_test)
        print('Проверка на неповторяемость', check_sim(test_data, min_iss_test))
        k=0
        pics=np.asarray(similar_pic(test_data, min_iss_test))
        pics= pics.reshape(10, w, h)
        for i in range(10):
            j = np.random.randint(np.asarray(pics).shape[0])
            k += 1
            plt.subplot(5, 2, k)
            plt.imshow(pics[i], cmap = 'gray')
            plt.title(i, fontsize = 8)
            plt.axis('off')
        plt.subplots_adjust(wspace = 0.2) # wspace
        plt.show()
