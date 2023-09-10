# import PIL.ImageGrab
import cv2 
import pyautogui
import numpy
# from keras.models import load_model

# Captura a tela
# Quando aparecer uma imagem de país pega sua coordenada na imagem
# Reconhece a cor do país
    # Players: Roxo, Cinza, Verde, Vermelho, Amarelo, Azul
# Reconhece o número de tropas

# Retorna o número de países e as tropas que estão neles para cada jogador

# Número de tropas, territórios, número de tropas para reforço, cartas objetivos

class Visualizer():  
    
    # Green
    greenLower = (29, 86, 6)
    greenUpper = (64, 255, 255)

    #Yellow
    yellowLower = numpy.array([22, 93, 0])
    yellowUpper = numpy.array([45, 255, 255])
    
    #Red (173, 20, 45)
    # redLower = numpy.array([350,60,30])
    # redUpper = numpy.array([350,100,55])
    redLower = (173, 15, 40)
    redUpper = (173, 25, 45)
    
    def __main__(self):
        pass
        
    def PrintScreen(self): 
        pass
    
    def TakeObjects(self):
        argeliaId = r"C:\Users\Davi Augusto\Desktop\ABII\Argelia.png"
        # EgitoId = r"C:\Users\Davi Augusto\Desktop\ABII\Egito.png"
        while True:
            if pyautogui.locateOnScreen(argeliaId, confidence = 0.7):
                argeliaRect = pyautogui.locateOnScreen(argeliaId, confidence = 0.8)
                # egitoRect = pyautogui.locateOnScreen(EgitoId, confidence = 0.8)
                
                world = pyautogui.screenshot()
                world = cv2.cvtColor(numpy.array(world), cv2.COLOR_BGR2RGB)
                # world = cv2.cvtColor(numpy.array(world), cv2.COLOR_BGR2HSV)
                cv2.imwrite("world.png", world)
                
                if(argeliaRect != None):
                    
                    argelia = world[argeliaRect.top: argeliaRect.top+argeliaRect.height
                                    ,argeliaRect.left:argeliaRect.left+argeliaRect.width]
                    # egito = world[egitoRect.top: egitoRect.top+egitoRect.height
                    #                 ,egitoRect.left:egitoRect.left+egitoRect.width]

                    maskRed = cv2.inRange(argelia, self.redLower, self.redUpper)
                    cntRed = cv2.findContours(maskRed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

                    
                    if len(cntRed) > 0:
                        cv2.imwrite("screenshotArgelia.png", argelia)
                        print("Argelia, Red")
                    
                    # if(len(cntYellow) > 0):
                    #     cv2.imwrite("screenshotEgito.png", egito)
                    #     print("Egito, Yellow")
vis = Visualizer() 
vis.TakeObjects()    