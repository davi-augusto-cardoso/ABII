
from imports import *
from classifier import Classifier


class Player:
    color       = ""
    regions     = []
    reinforces  = 0
    
    def __init__(self, color):
        self.color = color

colors = {  "yellow"    : (numpy.array([ 23, 130, 130]), numpy.array([ 30, 255, 255])),
            "green"     : (numpy.array([ 40, 100, 100]), numpy.array([ 60, 255, 255])),
            "blue"      : (numpy.array([ 80, 200, 160]), numpy.array([100, 255, 255])),
            "red"       : (numpy.array([175, 125, 125]), numpy.array([179, 255, 255])),
            "purple"    : (numpy.array([125,  70,  90]), numpy.array([140, 255, 255])),
            }

class Visualizer:
    imageOriginal       = ''    # -> Imagem original passada em escala HSV
    imageProcessing     = ''    # -> Imagem que está sendo trabalhada de acordo com a sua máscara
    idImage             = 0     # -> ID para nomeação do banco de imagens para treinamento
    
    regionClassifier    = ''
    digitsClassifier    = ''
    
    def __init__(self, path):
        os.chdir(r"C:\Users\Davi Augusto\Desktop\ABII\data")
        self.regionClassifier = Classifier(r"../data/regionsModel.h5", r"../data/regions.txt")
        self.digitsClassifier = Classifier(r"../data/regionsModel.h5", r"../data/regions.txt")
        self.OpenImage(path)
        
    # Captura o frame atual da tela
    def TakePrint(self):
        # Espera pela tecla enter
        # if(pyautogui.KEY == pyautogui.):
        # Faz a leitura do arquivo de imagem
        self.imageOriginal = pyautogui.screenshot()
        # Converte cores da imagem original de BGR para HSV e coloca na imagem de processamento
        self.imageOriginal = cv2.cvtColor(self.imageOriginal, cv2.COLOR_BGR2HSV)
        
    # Abre imagem em array de pixels em BGR
    def OpenImage(self, path): 
        # Faz a leitura do arquivo de imagem
        self.imageOriginal = cv2.imread(path)
        # Converte cores da imagem original de BGR para HSV e coloca na imagem de processamento
        self.imageOriginal = cv2.cvtColor(self.imageOriginal, cv2.COLOR_BGR2HSV)

    # Mostra imagem processada em uam janela
    def ShowImage(self, image):
        cv2.imshow("image", image)
        cv2.waitKey(0)

    # Aplica a mascara com base na coloração
    def CreateMask(self, color):
        colorLower = colors[color][0] # -> Limite superior da cor
        colorUpper = colors[color][1] # -> Limite inferior da cor
        
        # Criando máscara por coloração
        self.imageProcessing = cv2.inRange(self.imageOriginal, colorLower, colorUpper)

    # Faz o pré processamento da imagem para reconhecimento dos contornos das regiões
    def PreProcess(self):
        # Desfoca a imagem (Embassa a imagem de acordo com o ultimo parâmetro)
        self.imageProcessing = cv2.GaussianBlur(self.imageProcessing, (5, 5), 1)
        
        # Filtro de pixels para aumentar foco
        kernelShapenig = numpy.array([  [-1, -1, -1],
                                        [-1, 9, -1],
                                        [-1, -1, -1]])
        # Enfoca a imagem ( Deixa os pixerls das bordas em angulos mais retos)
        self.imageProcessing = cv2.filter2D(self.imageProcessing, -1, kernelShapenig)
    
    # Retorna as imagens para serem classificadas
    def Contourning(self):
        # Extrai todos os contornos da imagem
        contours, hir = cv2.findContours(self.imageProcessing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        
        images = [] # -> Armazena os recortes das imagens contornadas
        
        for countor in contours:
            # Cpaturando as coordenadas das imagens de cada região na figura
            x, y, w, h = cv2.boundingRect(countor)
            # Calculando o tamanho de cada figura
            area = w * h
            # Filtrando pequenos ruídos
            if(area > 600):
                # Recortando imagem
                image = self.imageProcessing[y : y+h, x : x+w]
                # Pega os contornos da imagem
                cntrs, h = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                # Encontra o maior contorno
                largest_contour = max(cntrs, key=cv2.contourArea)
                # Recupera os vétices do maior contorno
                approx = cv2.approxPolyDP(largest_contour, 1, True)
                # Cria uam imagem preta
                mask = numpy.zeros(image.shape[:2], dtype=numpy.uint8)
                # Desenha o maior contorno na imagem preta
                cv2.drawContours(mask, approx, -1, 255, 2)
                # Preenche o contorno criado para criar a mascara
                cv2.fillConvexPoly(mask, approx, 255)
                # Aplica a mascara na imagem
                image = cv2.bitwise_and(image, image, mask=mask)
                # Salva recortes das regioes
                images.append(image)
                self.ShowImage(image)
                
        return images
                
    def ClassifyRegion(self):
        
        player = Player("red")
        self.CreateMask("red")
        self.PreProcess()
        
        images = self.Contourning()
        
        for image in images:
            region = self.regionClassifier.Classify(image)
            
            if(region[1] > 70):
                player.regions.append(region)

    def ShowRegions(self, images):
        for image in images:
            self.ShowImage(image)


