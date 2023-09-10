import cv2 
import numpy
import pyautogui
from keras.models import load_model  # TensorFlow is required for Keras to work
import os

class Player:
    color       = ""
    regions     = []
    reinforces  = 0
    
    def __init__(self, color):
        self.color = color
    

colors = {  "yellow"    : (numpy.array([ 23, 130, 130]), numpy.array([ 30, 255, 255])),
            "green"     : (numpy.array([ 40, 100, 100]), numpy.array([ 60, 255, 255])),
            "blue"      : (numpy.array([ 80, 200, 160]), numpy.array([100, 255, 255])),
            "red"       : (numpy.array([175, 150, 150]), numpy.array([180, 255, 255])),
            "purple"    : (numpy.array([125,  70,  90]), numpy.array([140, 255, 255])),
            }

class RegionClassifier:
    
    model   = ""
    regions = ""
    
    def __init__(self, model, regions) -> None:
        # Disable scientific notation for clarity
        numpy.set_printoptions(suppress=True)
        # Load the model
        self.model = load_model(model, compile=False)
        # Load the labels
        self.regions = open(regions, "r").readlines()
    
    # Retorna o nome da região identificada
    def ClassifyRegion(self, image):
        # Resize the raw image into (224-height,224-width) pixels
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        
        image = numpy.repeat(image[:, :, numpy.newaxis], 3, axis=2)
        
        # Make the image a numpy array and reshape it to the models input shape.
        image = numpy.asarray(image, dtype=numpy.float32).reshape(1, 224, 224, 3)

        # Normalize the image array
        image = (image / 127.5) - 1

        # Predicts the model
        prediction = self.model.predict(image)
        index = numpy.argmax(prediction)
        region = self.regions[index]
        confidenceScore = prediction[0][index]
        
        return (region[2:], confidenceScore *100)
        

class Visualizer:
    imageOriginal   = ''    # -> Imagem original passada em escala HSV
    imageProcessing = ''    # -> Imagem que está sendo trabalhada de acordo com a sua máscara
    idImage         = 0     # -> ID para nomeação do banco de imagens para treinamento
    
    classifier      = ''
    
    def __init__(self, path):
        os.chdir(r"C:\Users\Davi Augusto\Desktop\ABII\data")
        self.classifier = RegionClassifier(r"../data/regionsModel.h5", r"../data/regions.txt")
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
                # Recortando imagemSS
                image = self.imageProcessing[y : y+h, x : x+w]
                
                mask = cv2.rectangle(numpy.zeros_like(image), (y, x), (y+h, x+w), 255, -1)

                # Apply the mask to the image
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
            region = self.classifier.ClassifyRegion(image)
            
            if(region[1] > 70):
                player.regions.append(region)

    def ShowRegions(self, images):
        for image in images:
            self.ShowImage(image)

    def CreatOriginBase(self):
        playersList = ["yellow", "green", "blue", "red", "purple"]
        
        for player in playersList:
            self.CreateMask(player)
            self.PreProcess()
            self.Contourning()

def nothing(x):
    pass

def colorIdentifier(image):
    #Create a window
    cv2.namedWindow('image')
    # image = cv2.imread(r"courtor.png")
    # Create trackbars for color change
    # Hue is from 0-179 for Opencv
    cv2.createTrackbar('HMin', 'image', 0, 179, nothing)
    cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('HMax', 'image', 0, 179, nothing)
    cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
    cv2.createTrackbar('VMax', 'image', 0, 255, nothing)

    # Set default value for Max HSV trackbars
    cv2.setTrackbarPos('HMax', 'image', 179)
    cv2.setTrackbarPos('SMax', 'image', 255)
    cv2.setTrackbarPos('VMax', 'image', 255)

    # Initialize HSV min/max values
    hMin = sMin = vMin = hMax = sMax = vMax = 0
    phMin = psMin = pvMin = phMax = psMax = pvMax = 0

    while(1):
        # Get current positions of all trackbars
        hMin = cv2.getTrackbarPos('HMin', 'image')
        sMin = cv2.getTrackbarPos('SMin', 'image')
        vMin = cv2.getTrackbarPos('VMin', 'image')
        hMax = cv2.getTrackbarPos('HMax', 'image')
        sMax = cv2.getTrackbarPos('SMax', 'image')
        vMax = cv2.getTrackbarPos('VMax', 'image')

        # Set minimum and maximum HSV values to display
        lower = numpy.array([hMin, sMin, vMin])
        upper = numpy.array([hMax, sMax, vMax])

        # Convert to HSV format and color threshold
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(image, image, mask=mask)

        # Print if there is a change in HSV value
        if((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
            print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
            phMin = hMin
            psMin = sMin
            pvMin = vMin
            phMax = hMax
            psMax = sMax
            pvMax = vMax

        # Display result image
        cv2.imshow('image', result)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

path = r"../data/warMap1.png"
vis = Visualizer(path)

colorIdentifier(vis.imageOriginal)

# vis.CreateMask("red")
# vis.PreProcess()
# images = vis.Contourning()
# vis.ShowRegions(images)
# vis.ClassifyRegion()
