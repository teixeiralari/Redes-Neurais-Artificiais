import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#from sklearn import metrics
import scipy

#Regressao linear com Adaline e Pseudo-Inversa

entrada = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
target = np.array([2.26, 3.8, 4.43, 5.91, 6.18, 7.26, 8.15, 9.14, 10.87, 11.58, 12.55])


################## TESTE COM ADALINE ############################

def y_liquid(entrada, peso, bias):
    y = entrada * peso + bias
    return y 

def training(entrada, saida, alfa, iteracoes):
    bias = np.random.rand(1)-0.5
    peso = np.random.rand(1)-0.5
    EQT = 100
    EQT2 = 50
    precisao = 0.000001
    count = 0

    #for i in range(iteracoes):
    while abs(EQT - EQT2) > precisao:
        EQT = EQT2
        EQT2 = 0
        count +=1
        for i in range(len(entrada)):
            y = y_liquid(entrada[i], peso, bias)
            EQT2 += 0.5*((saida[i] - y)**2)
            peso += entrada[i] * (saida[i] - y) * alfa
            bias += (saida[i] - y) * alfa
    print("Total de iteracoes: " + str(count))
    print("Variação do Erro Quadratico Total: " + str(abs(EQT2-EQT)))
    return peso, bias

def fit(entrada, peso, bias):
    for i in range(len(entrada)):
        y = y_liquid(entrada[i], peso, bias)
        print("Para a entrada {0}, o Y calculado é {1}".format(entrada[i],y))

def metricas():
    EMQ = 0
    NUM = []
    DEN = []
    PN = []
    PD = []
    PD1 = []
    Y = []
    for i in range(len(entrada)):
        y = y_liquid(entrada[i], peso, bias)
        Y.append(y)
        EMQ += (target[i] - y)**2/len(target)
        NUM.append(((target[i] - y)**2))
        DEN.append((target[i] - np.mean(target))**2)
        PN.append((entrada[i] - np.mean(entrada)) * (target[i] - np.mean(target)))
        PD.append((entrada[i] - np.mean(entrada)) ** 2)
        PD1.append((target[i] - np.mean(target)) ** 2)
    Person = sum(PN) /(np.sqrt(sum(PD)*sum(PD1)))
    R2 = 1 - (sum(NUM)/sum(DEN))
    print("Erro Médio Quadrático: " + str(EMQ)) #"EMQ: " + str(metrics.mean_squared_error(target,Y)))
    print("R-square: " + str(R2))               #"R2: " + str(metrics.r2_score(target,Y)))
    print("Coeficiente de Person: " + str(Person) + '\n')

   
def plot():
    x, y = pd.Series(entrada, name="X"), pd.Series(target, name="Y")
    ax = sns.regplot(x=x, y=y, ci= 68, color = "purple", marker="^")
    plt.show()

###################### TESTE COM PSEUDO-INVERSA #######################

entradaPINV = np.array([[0.0, 1.0], [0.5, 1.0],[1.0, 1.0], [1.5, 1.0], [2.0, 1.0], [2.5, 1.02],
               [3.0, 1.0],[3.5, 1.0], [4.0, 1.0], [4.5, 1.0], [5.0, 1.0]])



def pseudoInversa(entrada, saida):
    coeficientes = np.linalg.pinv(entrada) @ saida
    ycalculado = entrada @ coeficientes
    print("\nPeso: " + str(coeficientes[0]) + "\nBias: " + str(coeficientes[1]))
    EMQa = sum((saida - ycalculado) ** 2) / len(saida)
    R2a = 1 - sum((saida - ycalculado) ** 2) / sum((saida - np.mean(saida)) ** 2)
    print("\nErro Médio Quadrático: " + str(EMQa))
    print("R-square: " + str(R2a) + '\n')
    for i in range(len(saida)):
        print("Para a entrada {0}, o Y calculado é {1}".format(entrada[i,0], ycalculado[i]))

def coefientesFormula():
    entrada2 = np.array(entrada)**2
    b = (sum(entrada2) * sum(target) - sum(entrada*target)*sum(entrada))/(len(entrada) * sum(entrada2) - (sum(entrada))**2)
    a_peso = (len(entrada)*sum(entrada*target) - sum(entrada)*sum(target)) / (len(entrada) * sum(entrada2) - (sum(entrada))**2)
    print("Calculo dos coeficientes pela fórmula\n")
    print("a = {0} e b = {1}".format(a_peso,b))

#def salvarTXT(entrada, target):
#    f = open("dados.txt", 'w')
#    for i in range(len(entrada)):
#        aux, aux2 = entrada[i], target[i]
#        f.write(str(aux) + " " + str(aux2) + "\n")
print("-------------------- TESTE COM ADALINE -----------------\n")
coefientesFormula()
print("==================== TRAINING ====================\n")
peso, bias = training(entrada, target, alfa=0.005, iteracoes = 100)
print("\nPeso adequado encontrado: {0}.\nBias encontrado: {1}".format(peso, bias) + '\n')
print("==================== FIT ==========================\n")
fit(entrada, peso, bias)
print("\nEquação da reta: y = " + str(peso) + " * x + " + str(bias))
print("================= MÉTRICAS DE ERRO ==================\n")
metricas()
print("-------------------- TESTE COM PSEUDO-INVERSA ------------------")
pseudoInversa(entradaPINV, target)
plot()
print("\nFIM")
