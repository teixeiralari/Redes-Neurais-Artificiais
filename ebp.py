# IMPLEMENTACAO DA RETROPROPAGACAO DO ERRO
import numpy as np
import matplotlib.pyplot as plt

ENTRADA = np.array([[-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0,
                     0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]).transpose()

TARGET = np.array([[0.048, 0.058, 0.072, 0.093, 0.122, 0.167, 0.238, 0.357, 0.556, 0.833, 1.0, 0.833,
                    0.556, 0.357, 0.238, 0.167, 0.122, 0.093, 0.072, 0.058, 0.048]]).transpose()


# Um neuronio na camada de entrada
# Numero de neuronios na camada escondida aleatorios


def init(NE, NO, NS):
    # ------------------------------ DECLARACAO DOS PESOS --------------------------------

    # PESO DOS NEURONIOS NA CAMADA ESCONDIDA
    WCO = np.random.rand(NE, NO) - 0.5  # ENTRADA.shape[1]
    bCO = np.random.rand(NO) - 0.5
    
    # PESO DOS NEURONIOS DA SAIDA
    WCS = np.random.rand(NO, NS) - 0.5  # TARGET.shape[1]
    bCS = np.random.rand(NS) - 0.5
    return WCO, bCO, WCS, bCS


def ForwardPass(entrada, WCO, bCO, WCS, bCS):
    netOculta = (entrada @ WCO) + bCO
    
    # CALCULO DA SAIDA DOS NEURONIOS ESCONDIDOS
    outputOculta = np.tanh(netOculta)
    
    # CALCULO DAS ENTRADA NOS NEURONIOS DE SAIDA
    netSaida = (outputOculta @ WCS) + bCS
    
    # CALCULO DAS SAIDAS DOS NEURONIOS DE SAIDA
    outputSaida = np.tanh(netSaida)
    
    return outputSaida, outputOculta


def backPropagation(entrada, target, NO, alpha, tolerancia, iteracoes):
    WCO, bCO, WCS, bCS = init(entrada.shape[1], NO, target.shape[1])
    f = open("graficoEQT.txt", 'w')
    for it in range(iteracoes):
        EQTotal = 0
        
        for idx in range(entrada.shape[0]):
            outputSaida, outputOculta = ForwardPass(entrada[idx, :], WCO, bCO, WCS, bCS)
            EQTotal += 0.5 * np.sum(target[idx, :] - outputSaida) ** 2
            
            # PARA OS NEURONIOS ESCONDIDOS EM RELACAO A SAIDA
            deltinhaSaida = (outputSaida - target[idx, :]) * (1 - outputSaida ** 2)
            deltaOS = np.array([outputOculta]).transpose() @ np.array([deltinhaSaida])
            WCS -= alpha * deltaOS
            bCS -= alpha * deltinhaSaida
            
            # PARA A ENTRADA EM RELACAO AO NEURONIOS ESCONDIDO
            deltinhaOculta = (deltinhaSaida @ WCS.transpose()) * (1 - outputOculta ** 2)
            deltaEO = np.array([entrada[idx, :]]).transpose() @ np.array([deltinhaOculta])
            WCO -= alpha * deltaEO
            bCO -= alpha * deltinhaOculta
        print(it, EQTotal)
        f.write(str(EQTotal) + " " + str(it) + "\n")
        if EQTotal <= tolerancia:
            break
    print("Iteracoes: " + str(it))
    f.close()
    return WCO, bCO, WCS, bCS, EQTotal


def plot():
    x = []
    y = []
    dataset = open('graficoEQT.txt', 'r')
    for line in dataset:
        line = line.strip()
        X, Y = line.split(" ")
        x.append(Y)
        y.append(X)
    dataset.close()
    plt.plot(x, y, color='purple', marker='^', linewidth=3.0)
    plt.title('Evolucao do erro por ciclo')
    plt.xlabel('Ciclos')
    plt.ylabel('Erro')
    plt.savefig("Grafico erro vs ciclo", format='eps')


plot()

xerro = []
yerro = []
for i in range(5, 6):
    xin = []
    xcalc = []
    yt = []
    ycalc = []
    WCO, bCO, WCS, bCS, EQTotal = backPropagation(ENTRADA, TARGET, i, 0.1, 0.0001, 1000)
    for idx in range(ENTRADA.shape[0]):
        S, _ = ForwardPass(ENTRADA[idx, :], WCO, bCO, WCS, bCS)
        print("Target: {0} vs Saida: {1}".format(TARGET[idx, :], S))
        xin.append(ENTRADA[idx])
        yt.append(TARGET[idx])
    
    print("Erro final: " + str(EQTotal))
    xerro.append(i)
    yerro.append(EQTotal)
    
    for x in list(np.linspace(-1, 1, 100)):
        S, _ = ForwardPass(np.array([x]), WCO, bCO, WCS, bCS)
        xcalc.append(x)
        ycalc.append(S[0])
    
    plt.plot(xcalc, ycalc, color='r')
    plt.plot(xin, yt, 'b^')  # ,color = 'purple', marker = '^')
    plt.title("Erro final: " + str(EQTotal) + "; Numero de neuronios: " + str(i))
    plt.savefig("Grafico bp com " + str(i) + " neuronios", format='png')
    plt.gcf().clear()

plt.plot(xerro, yerro, color='b')
plt.title('Evolucao do erro por neurônios na camada oculta')
plt.xlabel('Neurônios na camada oculta')
plt.ylabel('Erro quadrático')
plt.savefig("Grafico erro em funcao neuronios", format='png')
plt.gcf().clear()
