import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def init(NE, NP):
    population = np.zeros((NP,NE))   
    for j in range(NP):
     for i in range(NE):
         x = random.uniform(-5.12,5.12)
         population[j,i] = x

    return population


def Mutacao(population, F):
    popMut = np.zeros((population.shape[0],population.shape[1]))
    for i in range(population.shape[0]):
        idx = np.random.permutation(population.shape[0])[0:3]
        popMut[i,:] = population[idx[0],:] + F * (population[idx[1],:] - population[idx[2],:])
    return popMut


def Cruzamento(CR, population, popmut):
    popcross = np.zeros((population.shape[0],population.shape[1]))
    for i in range(population.shape[0]):
        for j in range(population.shape[1]):
            number = np.random.rand()
            k = random.randrange(0,2,1)
            if number <= CR or j == k:
                popcross[i,j] = popmut[i,j]
    return popcross

def Fitness(iteracoes, population):

    #DEFININDO TAMANHOS
    functionPopC = np.zeros((population.shape[0]))
    functionPopCross = np.zeros((population.shape[0]))
    functionBest = np.zeros((population.shape[0]))
    popBest = np.zeros((population.shape[0],population.shape[1]))
    pi = np.pi
    xj= []
    fit = []
    #Condicao de parada
    for j in range(iteracoes):
        popmut = Mutacao(population, 0.5)
        popcross = Cruzamento(0.9, population, popmut)

        functionPopCross = rastrigin(popcross[:,0],popcross[:,1])
        functionPopC = rastrigin(population[:,0],population[:,1])
        for i in range(population.shape[0]):
            if  functionPopC[i] <= functionPopCross[i]:
                popBest[i,:] =  population[i,:]
            else:
                 popBest[i,:] = popcross[i,:]
        functionBest = rastrigin(popBest[:,0], popBest[:,1])
                
        functionBest = list(functionBest)       
        best = min(functionBest)
        index = functionBest.index(best)
        fitBest = rastrigin(popBest[index,0], popBest[index,1])
        xj.append(j)
        fit.append(fitBest)
        fitold = np.copy(fitBest)
    return popBest, popBest[index,:], xj, fit

def rastrigin(X,Y):
    return 20 + (X**2 - 10 * np.cos(2 * np.pi * X)) + (Y**2 - 10 * np.cos(2 * np.pi * Y))


def plot3d(popBest):
    #plot pontos
    Xp = popBest[:,0]
    Yp= popBest[:,1]
    Zp = rastrigin(Xp, Yp)
    
    #plot superficie
    X = np.linspace(-5.12, 5.12, 200)
    Y = np.linspace(-5.12, 5.12, 200)
    X, Y = np.meshgrid(X, Y)
    Z = rastrigin(X, Y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.plasma, linewidth=0, antialiased=False)
    ax.scatter(Xp,Yp,Zp, 'k^')
    #plt.savefig('rastrigin.png')
    plt.show()


    
population = init(2,100)
popBest, best, X,Y = Fitness(10000, population)
print('Melhor individio: ', best)
print('\nValor do minimo da funcao: ', rastrigin(best[0],best[1]))
'''
plt.plot(X,Y,'b',)
plt.title('Evolucao do fitness por iteracao')
plt.xlabel('Ciclo')
plt.ylabel('Fitness')
plt.show()
'''
#plot3d(popBest)







 
