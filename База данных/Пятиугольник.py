import numpy as np
import math, time
import matplotlib.pyplot as plt
from PIL import Image # Для поворота изображения
from skimage.metrics import structural_similarity

#
np.random.seed(348)
full =  True # Полный прямоугольник, если True
cls = 0 if full else 1
show = not True
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
border = 4 # Граница
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
    '''print(len(s_iss))
    print(int(len(s_iss)/5))
    print(s_iss)
    print(s_iss[len(s_iss)-int(len(s_iss)/5):])
    print(m)'''
    return sum(s_iss[len(s_iss)-int(len(s_iss)/3):])/int(len(s_iss)/3)
    

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
def line(arrPic, x0, y0, x1, y1, hor = True):
    for i in range (int(x0),int(x1)+1):
        y_noise = np.random.uniform(-2,2)
        x_noise = np.random.uniform(-0.5,0.5)
        y = int((y1+y_noise - y0)/(x1 - x0 + x_noise)*(i - x0 + x_noise) + y0 + y_noise) 
        arrPic[i, y] = np.random.randint(100, 255)

def pent(arrPic):
    a = np.random.randint(10, 20)
    coords=[[0.6*a,1.4*a], [1.3*a,0.9*a], [2*a,1.4*a], [1.7*a,2.2*a ], [0.9*a,2.2*a]]
    if full:
        idx = range(5)
    else:
        mis = np.random.randint(5) # Номер отсутствующей стороны
        idx = [i for i in range(5) if i != mis]
    for i in idx:
        if i == 0: line(arrPic, coords[0][0], coords[0][1], coords[1][0], coords[1][1])
        if i == 1: line(arrPic, coords[1][0], coords[1][1], coords[2][0], coords[2][1])
        if i == 2: line(arrPic,  coords[3][0], coords[3][1], coords[2][0], coords[2][1])
        if i == 3: line(arrPic, coords[4][0], coords[4][1],coords[3][0], coords[3][1])
        if i == 4: line(arrPic,  coords[0][0], coords[0][1], coords[4][0], coords[4][1])
#
def prepareData(n, fn, fn2):
    file = open(fn, 'wb')
    file2 = open(fn2, 'wb')
    xs0 = border - w2 + 1
    xe = w2 - border - 1
    dx = 0.1
    for i in range(n):
        if full: # Прямоугольник
            xs = xs0
            label = cls
        else: # Прямоугольник без одной стороны
            xs = xs0
            label = cls
        arrPic = np.zeros((w, h), dtype = np.uint8)
        pent(arrPic)
        ang = np.random.randint(-90, 90)
        arrPic = rot_img(ang, arrPic)
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
def rot_img(ang, img_array):
    # Приводим данные к типу uint8
    img_array = np.array(img_array, dtype = 'uint8')
    # Формируем изображение по массиву img_array
    img = Image.fromarray(img_array, 'L')
    # Поворот изображения на угол ang против часовой стрелки
    img = img.rotate(ang)
    # Переводим изображение в массив
    ix = img.size[0]
    iy = img.size[1]
    img_array_rot = np.array(img.getdata(), dtype = 'uint8').reshape(iy, ix)
    return img_array_rot

if not show:
    t0 = time.time()
    print('Поехали')
    prepareData(n_train, fn_train, fn_train_labels)
    prepareData(n_test, fn_test, fn_test_labels)
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
        print('Проверка на неповторяемость', check_sim(test_data, min_iss_test ))
        train_data, _ = load_data(fn_train, fn_train_labels)
        train_data = train_data.reshape(n_train, w, h)
        #min_iss_train=iss(train_data)
        #print('Минимальный индекс структурного различия в обучающем множестве',min_iss_train)
        #print('Количество аналогов', check_sim(train_data, min_iss_train))
    ttl = 'Декартов Лист'
    cls = 2
    plotData(train_data, ttl, cls)