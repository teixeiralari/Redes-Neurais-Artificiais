import numpy as np

tabela = [[1,-1,-1,-1,1,-1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,1,-1,1,-1,-1,-1,1],
          [1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1]]
target = [1,-1]


def calcularY(entrada, bias, pesos, limiar):
    soma = bias
    for i in range(len(entrada)):
        soma += entrada[i]*pesos[i]
    if soma >= limiar:
        return 1
    else:
        return -1


def treinar(entrada, saidas, TaxaAprendizado, iteracoes):
    pesos = np.random.rand(len(entrada[0]))
    bias = np.random.rand(1)
    #pesos = np.zeros(len(entrada[0]))
    #bias = 0
    count=0
    erro = True
    while erro:
        erro = False
        for i in range(len(entrada)):
            y = calcularY(entrada[i], bias, pesos, limiar=0)
            if y != saidas[i]:
                pesos += np.array(entrada[i]) * saidas[i] * TaxaAprendizado
                bias += saidas[i] * TaxaAprendizado
                print("Corrigir entrada " + str(i) + " na iteração " + str(count)  + "; Y: " + str(y) + " Target: " + str(target[i]))
                erro = True
                count+=1
        if count >=iteracoes:
            break
    print("Iterações: {0} ".format(count))
    return pesos, bias

def teste(entrada, pesos, bias):
    for i in range(len(entrada)):
        print("Entrada: {0}\n".format(entrada[i]))
        print("Saida: {0}".format(calcularY(entrada[i], bias, pesos,0)))

def testeRuido(entradaTeste, pesos, bias, saidaDesejada):
    count = 0
    index = np.random.permutation(len(entradaTeste))
    print("Index gerados: " + str(index))
    for i in range(len(entradaTeste)):
        entradaTeste[index[i]] = 1 if entradaTeste[index[i]] == -1 else -1
        y = calcularY(entradaTeste, bias, pesos, limiar=0)
        if y == saidaDesejada:
            count+=1
    return (count/len(entradaTeste))
print("========================== TREINAMENTO =================================")      
pesos, bias = treinar(tabela, target, TaxaAprendizado = 0.002, iteracoes = 10000)
print("Pesos finais: {0}, bias: {1}\n".format(pesos, bias))
print("============================= TESTE ====================================")
teste(tabela, pesos, bias)
print("======================= TESTE COM RUÍDO ================================")
for i in range(len(tabela)):
    print("Para entrada %d reconhece com até %0.1f%% de pixels modificados\n" %(i,100*testeRuido(tabela[i], pesos, bias, target[i])))

