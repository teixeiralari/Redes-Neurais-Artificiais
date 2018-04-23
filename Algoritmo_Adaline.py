#ESSE ALGORITMO RECONHECE AS LETRAS X E T

import numpy as np
import pandas as pd


tabela = [
    [1,-1,-1,-1,1,-1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,1,-1,1,-1,-1,-1,1], #X
    [1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1] #T
          ]
target = [1, #TARGET PARA A LETRA X
          -1 #TARGET PARA A LETRA T
          ]


def calcularY(entrada, bias, pesos):
    y = bias
    for i in range(len(entrada)):
        y += entrada[i]*pesos[i]
    return y

def Degrau(entrada, bias, pesos, limiar):
    y = bias
    for i in range(len(entrada)):
        y += entrada[i]*pesos[i]
    if y >= limiar:
        return 1
    else:
        return -1


def treinar(entrada, saidas, TaxaAprendizado, iteracoes):
    pesos = np.random.rand(len(entrada[0]))-0.5
    deltaPesos = np.zeros(len(entrada[0]))
    bias = np.random.rand(1)-0.5
    erroQuadraticoTotal = 100
    erroQuadraticoTotal2 = 50
    count = 0
    while erroQuadraticoTotal-erroQuadraticoTotal2 > 0.00001:
        erroQuadraticoTotal = erroQuadraticoTotal2
        erroQuadratico2 = 0
        count +=1
        for i in range(len(entrada)):
            y = calcularY(entrada[i], bias, pesos)
            erroQuadraticoTotal2 += 0.5*((saidas[i] - y)**2)
            pesos += np.array(entrada[i]) * (saidas[i] - y) * TaxaAprendizado
            bias += (saidas[i] - y) * TaxaAprendizado
        print("Iteracoes: %d, Erro: %0.4f" %(count,abs(erroQuadraticoTotal-erroQuadraticoTotal2)))
    return pesos, bias

def teste(entrada, pesos, bias):
    for i in range(len(entrada)):
        print("Entrada: {0}\n".format(entrada[i]))
        print("Saida: {0}".format(Degrau(entrada[i], bias, pesos, 0)))

def testeRuido(entradaTeste, pesos, bias, saidaDesejada):
    count = 0
    index = np.random.permutation(len(entradaTeste))
    print("Index gerados: " + str(index))
    for i in range(len(entradaTeste)):
        entradaTeste[index[i]] = -entradaTeste[index[i]]
        y = Degrau(entradaTeste, bias, pesos,0)
        if y == saidaDesejada:
            count+=1
    return (count/len(entradaTeste))

print("========================== TREINAMENTO =================================")      
pesos, bias = treinar(tabela, target, TaxaAprendizado = 0.09, iteracoes = 1000)
print("Pesos finais: {0}, bias: {1}\n".format(pesos, bias))
print("============================= TESTE ====================================")
teste(tabela, pesos, bias)
print("======================= TESTE COM RUÍDO ================================")
for i in range(len(tabela)):
    print("Para entrada %d reconhece com até %0.1f%% de pixels modificados\n" %(i,100*testeRuido(tabela[i], pesos, bias, target[i])))

        
        
        
        
        
        
        
