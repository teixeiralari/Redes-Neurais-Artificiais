def McCullochPitts(x1,x2,p1,p2,limiar):
    saida = x1*p1+x2*p2
    if saida < limiar:
        saida = 0
    else:
        saida = 1
    return saida


def somador1bit(x1,x2):
    return McCullochPitts(McCullochPitts(x1,x2,0.5,0.5,1),McCullochPitts(x1,x2,1,1,1),-1,1,1)


tabela = [[0, 0],[0,1],[1,0],[1,1]]
for i in range(4):
    x1,x2 = tabela[i][0],tabela[i][1]
    y = somador1bit(x1,x2)
    print("As entradas sao {0} e {1}".format(x1,x2))
    print("Saida: %d" %(y))


'''def RegraHebb(x1,x2,w1_antigo=0,w2_antigo=0,b=0,limiar=0):
    w1_novo = w1_antigo + x1*y
    w2_novo = w2_antigo + x2*y
    b_novo = b_antigo + y    '''