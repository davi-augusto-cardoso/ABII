from imports import load_model, numpy, cv2

class Classifier:
    
    model   = ""
    labels  = ""
    
    def __init__(self, model, labels) -> None:
        # Disable scientific notation for clarity
        numpy.set_printoptions(suppress=True)
        # Load the model
        self.model = load_model(model, compile=False)
        # Load the labels
        self.labels = open(labels, "r").readlines()
    
    # Retorna o nome da região identificada
    def Classify(self, image):
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
        className = self.labels[index]
        confidenceScore = prediction[0][index]
        
        return (className[2:], confidenceScore *100)