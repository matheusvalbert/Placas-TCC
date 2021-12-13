from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
import cv2
from openalpr import Alpr
from mlxtend.plotting import plot_confusion_matrix

country = 'br'
configFile = '/usr/local/share/openalpr/config/openalpr.defaults.conf'
runtimeDir = '/usr/local/share/openalpr/runtime_data'

path = country + '/'

alpr = Alpr(country, configFile, runtimeDir)
#alpr.set_default_region('base')
#alpr.set_top_n(10)

if not alpr.is_loaded():
    print('Error loading OpenALPR')
    
def cfMatrix(cmCorreto, cmIncorreto):
    tick_label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    cf = confusion_matrix(cmCorreto, cmIncorreto, labels=tick_label)
    fig, ax = plot_confusion_matrix(conf_mat=cf,
                                    colorbar=True,
                                show_absolute=False,
                                show_normed=True,
                                class_names=tick_label)
    plt.xlabel('Predição')
    plt.ylabel('Placa')
    plt.title('Porcentagem de acerto por letra')
    fig = plt.gcf()
    fig.set_size_inches(35, 25)
    fig.savefig('mat_porcentagem_por_letra.png', dpi=300)

def cfMatrixData(prediceted, plate, cmCorreto, cmIncorreto):
    if prediceted == False:
        return cmCorreto, cmIncorreto
    prediceted = list(prediceted[0]['plate'])
    for i in range(7):
        '''
        if prediceted[i] == '0' and i <= 2:
            prediceted[i] = 'O'
        if prediceted[i] == 'O' and i > 2:
            prediceted[i] = '0'
        if prediceted[i] == '1' and i <= 2:
            prediceted[i] = 'I'
        if prediceted[i] == 'I' and i > 2:
            prediceted[i] = '1'
        if prediceted[i] == '8' and i <= 2:
            prediceted[i] = 'B'
        if prediceted[i] == 'B' and i > 2:
            prediceted[i] = '8'
        if prediceted[i] == '5' and i <= 2:
            prediceted[i] = 'S'
        if prediceted[i] == 'S' and i > 2:
            prediceted[i] = '5'
        '''
        cmCorreto.append(plate[i])
        cmIncorreto.append(prediceted[i])
    return cmCorreto, cmIncorreto
    
def pltTopNN(topListSum, title):
    plt.rcParams["figure.figsize"] = [10, 6]
    
    left = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    height = topListSum
    tick_label = ['1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9', '1-10']
    
    col = []
    for val in height:
        if val < 50:
            col.append('red')
        else:
            col.append('green')
            
    _, ax = plt.subplots()

    pps = ax.bar(left, height, tick_label=tick_label, width=0.8, color=col)
    
    for p in pps:
        height = p.get_height()
        ax.text(x=p.get_x()+p.get_width()/2, y=height+1,
                s="{}%".format(int(height)), ha='center')
    
    plt.ylim(0, 100)
    plt.xlabel('TopN-N')
    plt.ylabel('Porcentagem')
    plt.title(title)
    
    plt.show()
    
def pltTopN(topList, title):
    plt.rcParams["figure.figsize"] = [10, 6]
    
    left = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    height = topList
    tick_label = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    
    col = []
    for val in height:
        if val < 50:
            col.append('red')
        else:
            col.append('green')
            
    _, ax = plt.subplots()

    pps = ax.bar(left, height, tick_label=tick_label, width=0.8, color=col)
    
    for p in pps:
        height = p.get_height()
        ax.text(x=p.get_x()+p.get_width()/2, y=height+1,
                s="{}%".format(int(height)), ha='center')
    
    plt.ylim(0, 100)
    plt.xlabel('TopN')
    plt.ylabel('Porcentagem')
    plt.title(title)
    
    plt.show()
    
def avaliarEntradaRecusa(permitidoEntrou, naoPermitidoEntrou, total):
    perm = []
    permSum = []
    nPerm = []
    nPermSum = []
    for i in range(10):
        topN = permitidoEntrou[i]/total
        print('top%d: %.2f%%' % (i+1, topN*100))
        perm.append(topN*100)
    for i in range(10):
        topN += permitidoEntrou[i]/total
        print('top1-%d: %.2f%%' % (i+1, topN*100))
        permSum.append(topN*100)
    for i in range(10):
        topN = naoPermitidoEntrou[i]/total
        print('top%d: %.2f%%' % (i+1, topN*100))
        nPerm.append(topN*100)
    for i in range(10):
        topN += naoPermitidoEntrou[i]/total
        print('top1-%d: %.2f%%' % (i+1, topN*100))
        nPermSum.append(topN*100)
    return perm, permSum, nPerm, nPermSum
    
def entradaAceitaRecusada(permitido, naoPermitido, permitidoEntrou, naoPermitidoEntrou, prediceted, plateString):
    if prediceted == False:
        return permitidoEntrou, naoPermitidoEntrou
    else:
        for i in range(10):
            try:
                '''
                prediceted[i]['plate'] = list(prediceted[i]['plate'])
                for j in range(7):
                    if prediceted[i]['plate'][j] == '0' and j <= 2:
                        prediceted[i]['plate'][j] = 'O'
                    if prediceted[i]['plate'][j] == 'O' and j > 2:
                        prediceted[i]['plate'][j] = '0'
                    if prediceted[i]['plate'][j] == '1' and j <= 2:
                        prediceted[i]['plate'][j] = 'I'
                    if prediceted[i]['plate'][j] == 'I' and j > 2:
                        prediceted[i]['plate'][j] = '1'
                    if prediceted[i]['plate'][j] == '8' and j <= 2:
                        prediceted[i]['plate'][j] = 'B'
                    if prediceted[i]['plate'][j] == 'B' and j > 2:
                        prediceted[i]['plate'][j] = '8'
                    if prediceted[i]['plate'][j] == '5' and j <= 2:
                        prediceted[i]['plate'][j] = 'S'
                    if prediceted[i]['plate'][j] == 'S' and j > 2:
                        prediceted[i]['plate'][j] = '5'
                prediceted[i]['plate'] = ''.join(prediceted[i]['plate'])
                '''
                if prediceted[i]['plate'] == plateString:
                    if plateString in permitido:
                        permitidoEntrou[i] += 1
                        #break
                elif prediceted[i]['plate'] in permitido:
                    naoPermitidoEntrou[i] += 1
            except:
                pass
    return permitidoEntrou, naoPermitidoEntrou
    
def avaliarTopN(detectado, naoDetectado):
    print(detectado, naoDetectado)
    topList = []
    topListSum = []
    for i in range(10):
        topN = detectado[i]/(detectado[i]+naoDetectado[i])
        print('top%d: %.2f%%' % (i+1, topN*100))
        topList.append(topN*100)
    for i in range(10):
        topN += detectado[i]/(detectado[i]+naoDetectado[i])
        print('top1-%d: %.2f%%' % (i+1, topN*100))
        topListSum.append(topN*100)
    return topList, topListSum
    
def topNporcento(prediceted, plateString, detectado, naoDetectado):
    for i in range(10):
        if prediceted == False:
            naoDetectado[i] += 1
            pass
        else:
            try:

                prediceted[i]['plate'] = list(prediceted[i]['plate'])
                for j in range(7):
                    if prediceted[i]['plate'][j] == '0' and j <= 2:
                        prediceted[i]['plate'][j] = 'O'
                    if prediceted[i]['plate'][j] == 'O' and j > 2:
                        prediceted[i]['plate'][j] = '0'
                    if prediceted[i]['plate'][j] == '1' and j <= 2:
                        prediceted[i]['plate'][j] = 'I'
                    if prediceted[i]['plate'][j] == 'I' and j > 2:
                        prediceted[i]['plate'][j] = '1'
                    if prediceted[i]['plate'][j] == '8' and j <= 2:
                        prediceted[i]['plate'][j] = 'B'
                    if prediceted[i]['plate'][j] == 'B' and j > 2:
                        prediceted[i]['plate'][j] = '8'
                    if prediceted[i]['plate'][j] == '5' and j <= 2:
                        prediceted[i]['plate'][j] = 'S'
                    if prediceted[i]['plate'][j] == 'S' and j > 2:
                        prediceted[i]['plate'][j] = '5'
                prediceted[i]['plate'] = ''.join(prediceted[i]['plate'])

                if prediceted[i]['plate'] == plateString:
                    detectado[i] += 1

                    for k in range(10):
                        if k > i:
                            naoDetectado[k] += 1
                    break

                else:
                    naoDetectado[i] += 1
            except:
                naoDetectado[i] += 1
                pass
    return detectado, naoDetectado        
    
def pltPos(pos):
    plt.rcParams["figure.figsize"] = [8, 6]
    
    left = [1, 2, 3, 4, 5, 6, 7]
    height = pos
    tick_label = ['1', '2', '3', '4', '5', '6', '7']
    
    col = []
    for val in height:
        if val < 50:
            col.append('red')
        else:
            col.append('green')
            
    _, ax = plt.subplots()

    pps = ax.bar(left, height, tick_label=tick_label, width=0.8, color=col)
    
    for p in pps:
        height = p.get_height()
        ax.text(x=p.get_x()+p.get_width()/2, y=height+1,
                s="{}%".format(int(height)), ha='center')
    
    plt.ylim(0, 100)
    plt.xlabel('Posicao')
    plt.ylabel('Porcentagem')
    plt.title('Porcentagem de acerto por posicao')
    
    plt.show()
    
def avaliarPosicao(posicoesIncorretas, posicoesCorretas):
    pos = []
    for i in range(7):
        total = posicoesCorretas[i]/(posicoesCorretas[i]+posicoesIncorretas[i])
        print('posicao %d: %.2f%%' % (i+1, total*100))
        pos.append(total*100)
    return pos
    
def posicao(prediceted, plate, posicoesIncorretas, posicoesCorretas):
    if prediceted == False:
        for i in range(7):
            posicoesIncorretas[i] += 1
        return posicoesIncorretas, posicoesCorretas
    prediceted = list(prediceted[0]['plate'])
    for i in range(7):
        '''
        if prediceted[i] == '0' and i <= 2:
            prediceted[i] = 'O'
        if prediceted[i] == 'O' and i > 2:
            prediceted[i] = '0'
        if prediceted[i] == '1' and i <= 2:
            prediceted[i] = 'I'
        if prediceted[i] == 'I' and i > 2:
            prediceted[i] = '1'
        if prediceted[i] == '8' and i <= 2:
            prediceted[i] = 'B'
        if prediceted[i] == 'B' and i > 2:
            prediceted[i] = '8'
        if prediceted[i] == '5' and i <= 2:
            prediceted[i] = 'S'
        if prediceted[i] == 'S' and i > 2:
            prediceted[i] = '5'        
        '''
        if prediceted[i] == plate[i]:
            posicoesCorretas[i] += 1
        else:
            posicoesIncorretas[i] += 1
    return posicoesIncorretas, posicoesCorretas

def pltTotalLetters(total):

    activities = ['Acerto', 'Erro']
    slices = total
    colors = ['green', 'red']
    
    plt.pie(slices, labels = activities, colors=colors, autopct = '%1.2f%%')
    plt.legend()
    plt.title('Acerto total das letras')
    
    plt.show()

def pltLetters(letters):
    
    plt.rcParams["figure.figsize"] = [15, 7]
    
    left = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]
    height = letters
    tick_label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    col = []
    for val in height:
        if val < 50:
            col.append('red')
        else:
            col.append('green')
            
    _, ax = plt.subplots()
    
    pps = ax.bar(left, height, tick_label=tick_label, width=0.8, color=col)
    
    for p in pps:
        height = p.get_height()
        ax.text(x=p.get_x()+p.get_width()/2, y=height+.50,
                s="{}%".format(int(height)), ha='center')
    
    plt.xlabel('Letras')
    plt.ylabel('Porcentagem')
    plt.title('Porcentagem de acerto por letra')
    
    plt.show()
    
def avaliarDigito(corretoDigito, incorretoDigito):
    
    ac = corretoDigito.count('A')
    bc = corretoDigito.count('B')
    cc = corretoDigito.count('C')
    dc = corretoDigito.count('D')
    ec = corretoDigito.count('E')
    fc = corretoDigito.count('F')
    gc = corretoDigito.count('G')
    hc = corretoDigito.count('H')
    ic = corretoDigito.count('I')
    jc = corretoDigito.count('J')
    kc = corretoDigito.count('K')
    lc = corretoDigito.count('L')
    mc = corretoDigito.count('M')
    nc = corretoDigito.count('N')
    oc = corretoDigito.count('O')
    pc = corretoDigito.count('P')
    qc = corretoDigito.count('Q')
    rc = corretoDigito.count('R')
    sc = corretoDigito.count('S')
    tc = corretoDigito.count('T')
    uc = corretoDigito.count('U')
    vc = corretoDigito.count('V')
    wc = corretoDigito.count('W')
    xc = corretoDigito.count('X')
    yc = corretoDigito.count('Y')
    zc = corretoDigito.count('Z')
    c0 = corretoDigito.count('0')
    c1 = corretoDigito.count('1')
    c2 = corretoDigito.count('2')
    c3 = corretoDigito.count('3')
    c4 = corretoDigito.count('4')
    c5 = corretoDigito.count('5')
    c6 = corretoDigito.count('6')
    c7 = corretoDigito.count('7')
    c8 = corretoDigito.count('8')
    c9 = corretoDigito.count('9')
    
    ai = incorretoDigito.count('A')
    bi = incorretoDigito.count('B')
    ci = incorretoDigito.count('C')
    di = incorretoDigito.count('D')
    ei = incorretoDigito.count('E')
    fi = incorretoDigito.count('F')
    gi = incorretoDigito.count('G')
    hi = incorretoDigito.count('H')
    ii = incorretoDigito.count('I')
    ji = incorretoDigito.count('J')
    ki = incorretoDigito.count('K')
    li = incorretoDigito.count('L')
    mi = incorretoDigito.count('M')
    ni = incorretoDigito.count('N')
    oi = incorretoDigito.count('O')
    pi = incorretoDigito.count('P')
    qi = incorretoDigito.count('Q')
    ri = incorretoDigito.count('R')
    si = incorretoDigito.count('S')
    ti = incorretoDigito.count('T')
    ui = incorretoDigito.count('U')
    vi = incorretoDigito.count('V')
    wi = incorretoDigito.count('W')
    xi = incorretoDigito.count('X')
    yi = incorretoDigito.count('Y')
    zi = incorretoDigito.count('Z')
    i0 = incorretoDigito.count('0')
    i1 = incorretoDigito.count('1')
    i2 = incorretoDigito.count('2')
    i3 = incorretoDigito.count('3')
    i4 = incorretoDigito.count('4')
    i5 = incorretoDigito.count('5')
    i6 = incorretoDigito.count('6')
    i7 = incorretoDigito.count('7')
    i8 = incorretoDigito.count('8')
    i9 = incorretoDigito.count('9')
    
    a = ac/(ac+ai)
    print('a: %.2f%%' % (a*100))
    b = bc/(bc+bi)
    print('b: %.2f%%' % (b*100))
    c = cc/(cc+ci)
    print('c: %.2f%%' % (c*100))
    d = dc/(dc+di)
    print('d: %.2f%%' % (d*100))
    e = ec/(ec+ei)
    print('e: %.2f%%' % (e*100))
    f = fc/(fc+fi)
    print('f: %.2f%%' % (f*100))
    g = gc/(gc+gi)
    print('g: %.2f%%' % (g*100))
    h = hc/(hc+hi)
    print('h: %.2f%%' % (h*100))
    i = ic/(ic+ii)
    print('i: %.2f%%' % (i*100))
    j = jc/(jc+ji)
    print('j: %.2f%%' % (j*100))
    k = kc/(kc+ki)
    print('k: %.2f%%' % (k*100))
    l = lc/(lc+li)
    print('l: %.2f%%' % (l*100))
    m = mc/(mc+mi)
    print('m: %.2f%%' % (m*100))
    n = nc/(nc+ni)
    print('n: %.2f%%' % (n*100))
    o = oc/(oc+oi)
    print('o: %.2f%%' % (o*100))
    p = pc/(pc+pi)
    print('p: %.2f%%' % (p*100))
    q = qc/(qc+qi)
    print('q: %.2f%%' % (q*100))
    r = rc/(rc+ri)
    print('r: %.2f%%' % (r*100))
    s = sc/(sc+si)
    print('s: %.2f%%' % (s*100))
    t = tc/(tc+ti)
    print('t: %.2f%%' % (t*100))
    u = uc/(uc+ui)
    print('u: %.2f%%' % (u*100))
    v = vc/(vc+vi)
    print('v: %.2f%%' % (v*100))
    w = wc/(wc+wi)
    print('w: %.2f%%' % (w*100))
    x = xc/(xc+xi)
    print('x: %.2f%%' % (x*100))
    y = yc/(yc+yi)
    print('y: %.2f%%' % (y*100))
    z = zc/(zc+zi)
    print('z: %.2f%%' % (z*100))
    d0 = c0/(c0+i0)
    print('0: %.2f%%' % (d0*100))
    d1 = c1/(c1+i1)
    print('1: %.2f%%' % (d1*100))
    d2 = c2/(c2+i2)
    print('2: %.2f%%' % (d2*100))
    d3 = c3/(c3+i3)
    print('3: %.2f%%' % (d3*100))
    d4 = c4/(c4+i4)
    print('4: %.2f%%' % (d4*100))
    d5 = c5/(c5+i5)
    print('5: %.2f%%' % (d5*100))
    d6 = c6/(c6+i6)
    print('6: %.2f%%' % (d6*100))
    d7 = c7/(c7+i7)
    print('7: %.2f%%' % (d7*100))
    d8 = c8/(c8+i8)
    print('8: %.2f%%' % (d8*100))
    d9 = c9/(c9+i9)
    print('9: %.2f%%' % (d9*100))
    
    total = len(corretoDigito)/(len(corretoDigito)+len(incorretoDigito))
    print('total: %.2f%%' % (total*100))
    
    letters = []
    letters.append(a*100)
    letters.append(b*100)
    letters.append(c*100)
    letters.append(d*100)
    letters.append(e*100)
    letters.append(f*100)
    letters.append(g*100)
    letters.append(h*100)
    letters.append(i*100)
    letters.append(j*100)
    letters.append(k*100)
    letters.append(l*100)
    letters.append(m*100)
    letters.append(n*100)
    letters.append(o*100)
    letters.append(p*100)
    letters.append(q*100)
    letters.append(r*100)
    letters.append(s*100)
    letters.append(t*100)
    letters.append(u*100)
    letters.append(v*100)
    letters.append(w*100)
    letters.append(x*100)
    letters.append(y*100)
    letters.append(z*100)
    letters.append(d0*100)
    letters.append(d1*100)
    letters.append(d2*100)
    letters.append(d3*100)
    letters.append(d4*100)
    letters.append(d5*100)
    letters.append(d6*100)
    letters.append(d7*100)
    letters.append(d8*100)
    letters.append(d9*100)
    
    tt = []
    tt.append(len(corretoDigito))
    tt.append(len(incorretoDigito))
    return letters, tt
    
def digito(prediceted, plate, corretoDigito, incorretoDigito):
    if prediceted == False:
        for i in range(7):
            incorretoDigito.append(plate[i])
        return corretoDigito, incorretoDigito
    prediceted = list(prediceted[0]['plate'])
    for i in range(7):
        '''
        if prediceted[i] == '0' and i <= 2:
            prediceted[i] = 'O'
        if prediceted[i] == 'O' and i > 2:
            prediceted[i] = '0'
        if prediceted[i] == '1' and i <= 2:
            prediceted[i] = 'I'
        if prediceted[i] == 'I' and i > 2:
            prediceted[i] = '1'
        if prediceted[i] == '8' and i <= 2:
            prediceted[i] = 'B'
        if prediceted[i] == 'B' and i > 2:
            prediceted[i] = '8'
        if prediceted[i] == '5' and i <= 2:
            prediceted[i] = 'S'
        if prediceted[i] == 'S' and i > 2:
            prediceted[i] = '5'
        '''
        if prediceted[i] == plate[i]:
            corretoDigito.append(plate[i])
        else:
            incorretoDigito.append(plate[i])
    return corretoDigito, incorretoDigito
    
def detect(image):
    predicted = []
    result = alpr.recognize_ndarray(image)
    if len(result['results']) > 0:
        predicted = result['results'][0]['candidates']
    if len(predicted) > 0:
        return predicted
    else:
        return False
    
def getPlate(photo, file):
    image = cv2.imread(photo)
    with open(file) as f:
        plate_info = [word for line in f for word in line.split()]
    plate = plate_info[5]
    plateString = plate
    plate = list(plate)
    return image, plate, plateString

corretoDigito = []
incorretoDigito = []
posicoesCorretas = [0, 0, 0, 0, 0, 0, 0]
posicoesIncorretas = [0, 0, 0, 0, 0, 0, 0]
detectado = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
naoDetectado = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
cmCorreto = []
cmIncorreto = []

plates = []

for file in os.listdir(path):
    if file.endswith('.jpg'):
        plates.append(file.split('.')[0])
        
permitido, naoPermitido = train_test_split(plates, train_size=0.5, test_size=0.5, random_state=42)
permitidoEntrou = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
naoPermitidoEntrou = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
for file in os.listdir(path):
    if file.endswith('.jpg'):
        image, plate, plateString = getPlate(path + file, path + file.split('.')[0] + '.txt')
        prediceted = detect(image)
        #corretoDigito, incorretoDigito = digito(prediceted, plate, corretoDigito, incorretoDigito)
        #posicao(prediceted, plate, posicoesIncorretas, posicoesCorretas)
        #detectado, naoDetectado = topNporcento(prediceted, plateString, detectado, naoDetectado)
        #permitidoEntrou, naoPermitidoEntrou = entradaAceitaRecusada(permitido, naoPermitido, permitidoEntrou, naoPermitidoEntrou, prediceted, plateString)
        #cmCorreto, cmIncorreto = cfMatrixData(prediceted, plate, cmCorreto, cmIncorreto)
#letters, total = avaliarDigito(corretoDigito, incorretoDigito)
#pltLetters(letters)
#pltTotalLetters(total)
#pos = avaliarPosicao(posicoesIncorretas, posicoesCorretas)
#pltPos(pos)
#topList, topListSum = avaliarTopN(detectado, naoDetectado)
#pltTopN(topList, 'Acerto topN')
#pltTopNN(topListSum, 'Acerto topN-N')
#print(detectado, naoDetectado)
#perm, permSum, nPerm, nPermSum = avaliarEntradaRecusa(permitidoEntrou, naoPermitidoEntrou, len(permitido))
#pltTopN(perm, 'Entrada permitida TopN')
#pltTopNN(permSum, 'Entrada permitida TopN-N')
#pltTopN(nPerm, 'Entrada nao permitida entrou topN')
#pltTopNN(nPermSum, 'Entrada nao permitida entrou topN-N')
#cfMatrix(cmCorreto, cmIncorreto)
