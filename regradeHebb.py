'''Regra de Hebb para das portas lógicas'''


import numpy as np

tabela =[[-1, -1],[-1,1],[1,-1],[1,1]]
saidaOR = [-1,1,1,1]
saidaAND = [-1,-1,-1,1]
saidaNOR = [1,-1,-1,-1]
saidaNAND = [1,1,1,-1]


def neuronioBipolar(entradas, pesos, bias, limiar):
    soma = bias
    for i in range(len(entradas)):
        soma += entradas[i] * pesos[i]
        print(type(soma),type(entradas),type(pesos))
        print(soma)
    if soma >= limiar:
        return 1
    else:
        return -1

def treinamento(entradas,saidas):
    pesos = [0,0]
    b = 0
    for i in range(len(entradas)):
        pesos += np.array(entradas[i]) * saidas[i] 
        b += saidas[i] 
    return pesos, b
    
def teste(entradas, saidas):
    pesos, bias = treinamento(entradas, saidas)
    for i in range(len(entradas)):
        y = neuronioBipolar(entradas[i],pesos, bias, 0)
        print("As entradas sao {0} ".format(tabela[i]))
        print("Saida: %d" %(y))
    print("Pesos: {0}, bias: {1}".format(pesos, bias)) 

print("Porta OR: ")
teste(tabela,saidaOR)
print("\n")
print("Porta AND: ")
teste(tabela,saidaAND)
print("\n")

print("Porta NOR: ")
teste(tabela,saidaNOR)
print("\n")

print("Porta NAND: ")
teste(tabela,saidaNAND)
print("\n")

