import numpy as np
import math, time
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
from PIL import Image # Для поворота изображения
import keras
from keras.models import Model
from keras.layers import Input, Dense, Flatten,Conv2D, Reshape,MaxPooling2D, Dropout, BatchNormalization
import keras.callbacks as cb
from sys import exit
np.random.seed(2)
show =  not True
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
#-------------------------------------------------------------------------------------------------------
#Пятиугольник
def line(arrPic, x0, y0, x1, y1, hor = True):
    x = np.linspace(x0, x1, 300)
    y = (y1- y0)/(x1 - x0)*(x - x0) + y0 
    for i in range(len(x)):
        x_noise = np.random.uniform(-0.5,0.5)
        x[i] += x_noise
        y_noise = np.random.uniform(-1,1)
        y[i] += y_noise 
        arrPic[int(x[i]), int(y[i])] = np.random.randint(clr_mim, clr_max)
def pent(arrPic, full=True):
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
def Pentagon(full=True):
	arrPic = np.zeros((w, h), dtype = np.uint8)
	pent(arrPic,full)
	ang = np.random.randint(-90, 90)
	arrPic = rot_img(ang, arrPic)
	return arrPic
#Пятиугольник
def line_(arrPic, x0, y0, x1, y1, hor = True):
    x = np.linspace(x0, x1, 100)
    y = (y1- y0)/(x1 - x0)*(x - x0) + y0 
    for i in range(len(x)):
        x_noise = np.random.uniform(-0.5,0.5)
        x[i] += x_noise
        y_noise = np.random.uniform(-2,2)
        y[i] += y_noise 
        arrPic[int(x[i]), int(y[i])] = np.random.randint(clr_mim, clr_max)
def pent_(arrPic, full=True):
    a = np.random.randint(10, 20)
    coords=[[0.6*a,1.4*a], [1.3*a,0.9*a], [2*a,1.4*a], [1.7*a,2.2*a ], [0.9*a,2.2*a]]
    if full:
        idx = range(5)
    else:
        mis = np.random.randint(5) # Номер отсутствующей стороны
        idx = [i for i in range(5) if i != mis]
    for i in idx:
        if i == 0: line_(arrPic, coords[0][0], coords[0][1], coords[1][0], coords[1][1])
        if i == 1: line_(arrPic, coords[1][0], coords[1][1], coords[2][0], coords[2][1])
        if i == 2: line_(arrPic,  coords[3][0], coords[3][1], coords[2][0], coords[2][1])
        if i == 3: line_(arrPic, coords[4][0], coords[4][1],coords[3][0], coords[3][1])
        if i == 4: line_(arrPic,  coords[0][0], coords[0][1], coords[4][0], coords[4][1])
def Pentagon_1(full=False):
	arrPic = np.zeros((w, h), dtype = np.uint8)
	pent_(arrPic,full)
	ang = np.random.randint(-90, 90)
	arrPic = rot_img(ang, arrPic)
	return arrPic
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
#Локон Аньези
def lokon_anezi(a,t):
    x = 2 * a / np.tan(t)
    y = 2 * a * np.sin(t)**2
    for i in range(len(x)):
        x_noise = np.random.uniform(-0.5,0.5)
        x[i] += x_noise
        y_noise = np.random.uniform(-1,1)
        y[i] += y_noise + 8
    return x,y
def Witch_of_Agnesi():
    t = np.linspace(0.1, np.pi - 0.1, 300)
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
        j+=1
    return arrPic
#Декартов лист
def figure(a,t):
    x = a * ((t)**2 - 1) / (3 * (t)**2 + 1)
    y = a * (t) * ((t)**2 - 1) / (3 * (t)**2 + 1)  
    for i in range(len(x)):
        x_noise = np.random.uniform(-1,1)
        x[i] += x_noise
        y_noise = np.random.uniform(-0.5,0.5)
        y[i] += y_noise
    return x,y
def Folium_of_Descartes():
    t = np.linspace(-5, 5, 400)
    a = np.random.randint(4,21)
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
    for x, y in zip(x, y):
        ix = int(x) + shift_x
        iy = int(y) + shift_y
        clr = np.random.randint(clr_mim, clr_max)
        arrPic[iy, ix] = clr
    arrPic[-ix, -iy] = clr
    return arrPic
#Арксинус
def arcs(x, a,b):
    x_noise = np.random.uniform(-0.5, 0.5)
    y_noise = np.random.uniform(-1, 1)
    y = a*np.arcsin(b*(x)) + y_noise
    return y
def arcs_(x, a,b):
    x_noise = np.random.uniform(-0.5, 0.5)
    y_noise = np.random.uniform(-6, 6)
    y = a*np.arcsin(b*(x)) + y_noise
    return y
def Half_of_Arcsine():
    xe = w2 - border - 1
    dy = 0.01
    ys = 0
    label = 0
    sgn = np.random.randint(2) # y или -y
    a = np.random.randint(5, 10)
    b = np.random.uniform(0.07, 0.1)
    arrPic = np.zeros((w, h), dtype = np.uint8)
    y = ys - dy
    while (y+dy) < 1/b:
            y += dy
            x = arcs(y, a, b)
            iy = min(w - 1, int(w2 + y))
            ix = min(w - 1,int(w2 - x))
            clr = np.random.randint(clr_mim, clr_max)
            if sgn == 1:
                arrPic[ix, iy] = clr
            else:
                arrPic[-ix, -iy] = clr
    return arrPic
def Arcsine():
    xe = w2 - border - 1
    dy = 0.01
    ys = 0
    sgn = np.random.randint(2) # y или -y
    a = np.random.randint(5, 13)
    b = np.random.uniform(0.08, 0.1)
    arrPic = np.zeros((w, h), dtype = np.uint8)
    y = ys - dy
    while (y+dy) < 1/b:
        y += dy
        x = arcs_(y, a, b)
        iy = min(w - 1, int(w2 + y))
        ix = min(w - 1,int(w2 - x))
        clr = np.random.randint(clr_mim, clr_max)
        arrPic[ix, iy] = clr
        arrPic[-ix, -iy] = clr
    arrPic[-ix, -iy] = clr
    return arrPic      
#--------------------------------------------------Создание набора данных-----------------------------------------------------
def clss(c):
	if c==0:
		arrpic = Pentagon()
	elif c==1:
		arrpic = Pentagon_1()
	elif c==2:
 		arrpic = Witch_of_Agnesi()
	elif c==3:
		arrpic = Folium_of_Descartes()
	elif c==4:
		arrpic = Arcsine()
	elif c==5:
		arrpic = Half_of_Arcsine()
	return arrpic
def create_dataset(n, fn, fnl, clss_inst):
	file = open(fn, 'wb')
	file2 = open(fnl, 'wb')
	for i in range(n):
		c = np.random.randint(0, 6)
		clss_inst[c]+=1
		arrpic = clss(c)
		file.write(arrpic)
		file2.write(np.uint8(c))
	file.close()
	file2.close()

class_instance = np.zeros(6)

create_dataset(n_test, fn_test, fn_test_labels, class_instance)
print('Количество экземпляров в классах в тестовом множестве',class_instance)

create_dataset(n_train, fn_train, fn_train_labels, class_instance)
print('Количество экземпляров в классах в обучающем множестве',class_instance)
#--------------------------Примеры загруженных изображений (по одному из каждого класса)-----------------------------------
if show:
	for i in range(6):
		arrpic = clss(i)
		plt.subplot(3, 3, i+1)
		plt.imshow(arrpic, cmap = 'gray')
		plt.title(i, fontsize = 8)
		plt.axis('off')
		plt.subplots_adjust(wspace = 0.2) # wspace
	plt.show()
#--------------------------
show_4 =  not True
pathToData = 'C://Users//user//OneDrive//Рабочий стол//STUDIYNG//КГ//ЛР8//'
img_rows = img_cols = 64
num_classes = 6
#
pathToHistory = '' 
suff = '.txt'
# Имена файлов, в которые сохраняется история обучения
fn_loss = pathToHistory + 'loss_' + suff
fn_acc = pathToHistory + 'acc_' + suff
fn_val_loss = pathToHistory + 'val_loss_' + suff
fn_val_acc = pathToHistory + 'val_acc_' + suff

def show_x(x, img_rows, img_cols):
    print(x[0].shape)
    for k in range(1, 5):
        plt.subplot(2, 2, k)
        # Убираем 3-е измерение
        plt.imshow(x[k].reshape(img_rows, img_cols), cmap = 'gray')
        plt.axis('off')
    plt.show()
# Вывод графиков
def one_plot(n, y_lb, loss_acc, val_loss_acc):
    plt.subplot(1, 2, n)
    if n == 1:
        lb, lb2 = 'loss', 'val_loss'
        yMin = 0
        yMax = 1.05 * max(max(loss_acc), max(val_loss_acc))
    else:
        lb, lb2 = 'acc', 'val_acc'
        yMin = min(min(loss_acc), min(val_loss_acc))
        yMax = 1.0
    plt.plot(loss_acc, color = 'r', label = lb, linestyle = '--')
    plt.plot(val_loss_acc, color = 'g', label = lb2)
    plt.ylabel(y_lb)
    plt.xlabel('Эпоха')
    plt.ylim([0.95 * yMin, yMax])
    plt.legend()
#
def loadBinData(img_rows, img_cols, ft, ftl,ftr,ftrl):
    print('Загрузка данных из двоичных файлов...')
    with open(ftr, 'rb') as read_binary:
        x_train = np.fromfile(read_binary, dtype = np.uint8)
    with open(ftrl, 'rb') as read_binary:
        y_train = np.fromfile(read_binary, dtype = np.uint8)
    with open(ft, 'rb') as read_binary:
        x_test = np.fromfile(read_binary, dtype = np.uint8)
    with open(ftl, 'rb') as read_binary:
        y_test = np.fromfile(read_binary, dtype = np.uint8)
    # Преобразование целочисленных данных в float32 и нормализация; данные лежат в диапазоне [0.0, 1.0]
    x_train = np.array(x_train, dtype = 'float32') / 255
    x_test = np.array(x_test, dtype = 'float32') / 255
    x_train_shape_0 = int(x_train.shape[0] / (img_rows * img_cols))
    x_test_shape_0 = int(x_test.shape[0] / (img_rows * img_cols))
    x_train = x_train.reshape(x_train_shape_0, img_rows, img_cols, 1) # 1 - оттенок серого цвета
    x_test = x_test.reshape(x_test_shape_0, img_rows, img_cols, 1)
    # Преобразование в категориальное представление
    print('Преобразуем массивы меток в категориальное представление')
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return x_train, y_train, x_test, y_test

#----------------------------------------описание мод`ели НС-----------------------------
callbacks = []
callbacks.append(cb.EarlyStopping(monitor = 'val_accuracy', patience = 6))
input_shape = (img_rows, img_cols, 1)
inp = Input(shape = input_shape)
x = inp
x = MaxPooling2D(pool_size = 16, strides = 2, padding = 'same')(x)
Dropout(0.3)
x = Flatten()(x)
x = Dense(units = 128, activation = 'sigmoid')(x)
Dropout(0.5)
x = Dense(units = 64, activation = 'sigmoid')(x)
Dropout(0.5)
x = Dense(units = 32, activation = 'sigmoid')(x)
Dropout(0.5)
output = Dense(num_classes, activation = 'sigmoid')(x)
model = Model(inputs = inp, outputs = output)
model.summary()
model.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])
#---------------------------------------------------------------------------------------------------------------

#Загрузка данных из файлов
x_train, y_train, x_test, y_test = loadBinData(img_rows, img_cols,fn_test, fn_test_labels,  fn_train, fn_train_labels)
if show_4:
    show_x(x_test, img_rows, img_cols)
    exit()
# Обучение нейронной сети
epochs = 95
start = time.time()
history = model.fit(x_train, y_train, batch_size = 128, epochs = epochs,
                        verbose = 2, validation_data = (x_test, y_test))
time7 = time.time() - start
print('Время обучения: %f' % time7)

# Запись истории обучения в текстовые файлы
history = history.history
with open(fn_loss, 'w') as output:
	for val in history['loss']: output.write(str(val) + '\n')
with open(fn_acc, 'w') as output:
    for val in history['accuracy']: output.write(str(val) + '\n')
with open(fn_val_loss, 'w') as output:
	for val in history['val_loss']: output.write(str(val) + '\n')
with open(fn_val_acc, 'w') as output:
	for val in history['val_accuracy']: output.write(str(val) + '\n')
# Вывод графиков обучения
plt.figure(figsize = (9, 4))
plt.subplots_adjust(wspace = 0.5)
one_plot(1, 'Потери', history['loss'], history['val_loss'])
one_plot(2, 'Точность', history['accuracy'], history['val_accuracy'])
plt.suptitle('Потери и точность')
plt.show()


