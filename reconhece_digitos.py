''' ===================== RECONHECIMENTO DOS 10 DIGITOS ====================='''

import numpy as np


tabela = np.array([[1,-1,-1,-1,1,-1,-1,1,1,-1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1], #ZERO
          [1,-1,1,1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,1,1,1], #UM
          [1,-1,1,1,-1,1,-1,-1,1,-1,-1,1,-1,1,-1,-1,1,-1,1,1,-1], #DOIS
          [1,-1,-1,1,1,-1,1,-1,1,1,1,1,1,-1,-1,-1,1,-1,-1,-1,1], #TRÊS
          [1,1,1,1,1,1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,-1,1,1,-1], #QUATRO
          [1,-1,-1,1,-1,-1,1,-1,-1,1,1,1,-1,1,-1,-1,1,-1,1,1,-1], #CINCO
          [1,1,1,1,1,-1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,-1], #SEIS
          [1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1], #SETE
          [1,-1,1,1,-1,1,-1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,1,-1,-1], # OITO
          [1,-1,1,1,-1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,-1,1,1,-1] #NOVE
                   ]).transpose()


target = np.array([[1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
          [-1,1,-1,-1,-1,-1,-1,-1,-1,-1],
          [-1,-1,1,-1,-1,-1,-1,-1,-1,-1],
          [-1,-1,-1,1,-1,-1,-1,-1,-1,-1],
          [-1,-1,-1,-1,1,-1,-1,-1,-1,-1],
          [-1,-1,-1,-1,-1,1,-1,-1,-1,-1],
          [-1,-1,-1,-1,-1,-1,1,-1,-1,-1],
          [-1,-1,-1,-1,-1,-1,-1,1,-1,-1],
          [-1,-1,-1,-1,-1,-1,-1,-1,1,-1],
          [-1,-1,-1,-1,-1,-1,-1,-1,-1,1]]).transpose()



def calcularY(entrada, pesos, limiar):
    Y = pesos @ entrada
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if Y[i,j] >= limiar:
                Y[i,j] = 1
            else:
                Y[i,j] = -1
    return Y


def treinar(entrada, saidas, TaxaAprendizado, iteracoes):
    pesos = np.random.rand(saidas.shape[0],entrada.shape[0])
    #pesos = np.zeros((saidas.shape[0],entrada.shape[0]))
    count=0
    erro = True
    while erro:
        erro = False
        for i in range(entrada.shape[1]):
            for j in range(saidas.shape[0]):
                y = calcularY(entrada, pesos,0)
                if y[j,i]!=saidas[j,i]:
                    pesos[j,:] += entrada[:,i] * saidas[j,i] * TaxaAprendizado 
                    print("Corrigir entrada do digito " + str(i) + " na iteração " + str(count)  + "; Y: \n" + str(y[:,j]) + " Target: \n" + str(saidas[:,j]))
                    erro = True
                count+=1
        if count >=iteracoes:
            break
    print("Iterações: {0} ".format(count))
    return pesos

def teste(entrada, pesos):
    for i in range(entrada.shape[1]):
        print("Entrada {0} do digito {1}.".format(entrada[:,i],str(i)))
        print("Saida: {0}\n".format(calcularY(entrada, pesos,0)[:,i]))

def testeRuido(entradaTeste, pesos, saidaDesejada):
    for i in range(entradaTeste.shape[1]):
        index = np.random.permutation(len(entradaTeste[0]))
        print("Index gerados: " + str(index))   
        count=0
        for j in range(saidaDesejada.shape[0]):
            entradaTeste[i,index[j]] = 1 if entradaTeste[i,index[j]] == -1 else -1
            y = calcularY(entradaTeste, pesos, limiar=0)
            if y[j,i]==saidaDesejada[j,i]:
               count+=1
        print("Para entrada do digito %d reconhece com até %0.1f%% de pixels modificados\n" %(i,100*(count/entradaTeste.shape[0])))

print("========================== TREINAMENTO =================================")
pesos = treinar(tabela, target, TaxaAprendizado = 0.05, iteracoes = 100000)
print("Pesos finais: {0}\n".format(pesos))
print("============================= TESTE ====================================")
teste(tabela, pesos)
print("======================= TESTE COM RUÍDO ================================")
testeRuido(tabela, pesos, target)
