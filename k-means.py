import numpy as np
import matplotlib.pyplot as plt

entrada = np.array([[-0.4712, 1.7698], [0.1103, 3.1334], [2.0263,3.2474],
                   [1.5697, 0.7579], [1.7254, 4.0834], [2.2676, 0.4092],
                   [-0.4753, 1.1308], [3.2018, 3.1839], [2.0614,  1.6423],
                   [2.4969, 1.6099], [7.1547, 5.4719], [5.8240, 6.5220],
                   [7.0105, 8.9157], [6.3086, 6.0023], [6.1122, 6.2530],
                   [4.1822, 5.1714], [7.5074, 7.1391], [8.5628, 7.4580],
                   [6.9596, 5.4075], [6.7379, 10.1990], [2.1959, 7.1072],
                   [4.9906, 7.8603], [4.0592, 6.7196], [-0.3881, 6.8434],
                   [3.1318, 6.1018], [3.2421, 8.2583], [1.2560, 5.9102],
                   [2.8671, 5.8639], [1.8885, 5.8148], [1.0263, 7.9487],
                   [4.9782, 0.8005], [6.5436, 2.1400], [6.4685, 2.3265],
                   [7.8461, 1.3620], [6.6673, 2.8004], [6.8124, 2.7228],
                   [5.8382, 1.9450], [7.1404, 3.3512], [7.1251, 4.9571],
                   [4.7644, 2.3254],[7.5, 10.4], [-2.23, -1.522], [1.9, -7.2], [13.455, -5.0],[-3, -4],
                    [3.388, 5.239],[0.23, 0.1], [-1, 2.34]])

def plot(idx):
    idx = np.array(idx)
    plt.scatter(entrada[:, 0], entrada[:, 1], c = idx, s=100)
    plt.scatter(Centroide[:, 0], Centroide[:, 1], s=1000, c=np.array([i for i in range(len(Centroide))]), marker='*')
    plt.title('Centroides')
    plt.xlabel('Entrada X0')
    plt.ylabel('Entrada X1')
    plt.axis('equal')
    plt.pause(0.75)


def initList(size):
    mylist = list()
    for i in range(size):
        mylist.append( list() ) #different object reference each time
    return mylist

def init(NCentroides, NEntradas):
     idx = np.random.permutation(entrada.shape[0])[0:NCentroides]
     return entrada[idx, :]

def EuclideanDistance(Centroide, Entrada):
    distance = np.zeros((Entrada.shape[0], Centroide.shape[0]))
    for k in range(Centroide.shape[0]):
        for i in range(Entrada.shape[0]):
            for j in range(Entrada.shape[1]):
                distance[i, k] += (Entrada[i, j] - Centroide[k, j])**2
            distance[i, k] = np.sqrt(distance[i, k])
    return distance


def KMeans(Entrada, Centroide):
    
    oldC = np.zeros((Centroide.shape))
    count = 0
    while True:
        count += 1
        pointCentroides = initList(Centroide.shape[0])
        Distance = EuclideanDistance(Centroide, Entrada)
        Distance = list(Distance)
        idx = []
        for i in range(len(Distance)):
            Distance[i]= list(Distance[i])
        for i in range(Entrada.shape[0]):
            shortest = min(Distance[i])
            index = Distance[i].index(shortest)
            idx.append(index)
            pointCentroides[index].append(Entrada[i, :].tolist())
        nppc = [0]*len(pointCentroides)
        for x in range(len(pointCentroides)):
            nppc[x] = len(pointCentroides[x])
            pointCentroides[x] = np.array(pointCentroides[x])
            pointCentroides[x] = sum(pointCentroides[x])/len(pointCentroides[x])
            Centroide[x] = pointCentroides[x]
        if (Centroide == oldC).all():
            plot(idx)
            break
        else:
            plot(idx)
            plt.gcf().clear()
        oldC = np.copy(Centroide)
        print('Centroides finais: ', Centroide, '\nCount', count)
    plt.show()
    return Centroide, nppc


Centroide = init(6, entrada.shape[1])
KMeans(entrada, Centroide)

# nc = 15
# nmppc = int(0.1*entrada.shape[0])
# for i in range(nc, 0, -1):
#     Centroide = init(i, entrada.shape[1])
#     _, xxx = KMeans(entrada, Centroide)
#     if sum(list(map(lambda x: x >= nmppc, xxx))) == i:
#         print(i)
#         break
