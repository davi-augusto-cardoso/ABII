from pathlib import Path
import requests
import openvino as ov
import openvino.runtime
import cv2
import numpy as np
class text_recognition():
    def __init__(self):
        self.letters = "0123456789abcdefghijklmnopqrstuvwxyz"
        self.model_name = "text-recognition-resnet-fc"
        base_model_dir = Path("../model").expanduser()
        #create folder base_model_dir if not exist
        base_model_dir.mkdir(parents=True, exist_ok=True)

        self.model_xml_name = f'{self.model_name}.xml'
        self.model_bin_name = f'{self.model_name}.bin'

        self.model_xml_path = base_model_dir / self.model_xml_name
        self.model_bin_path = base_model_dir / self.model_bin_name
        

        self.model_xml_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/models_bin/1/" + self.model_name + "/FP32/" + self.model_xml_name
        self.model_bin_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/models_bin/1/" + self.model_name + "/FP32/" + self.model_bin_name
        
        if not self.model_xml_path.exists() or not self.model_bin_path.exists():
            print("Downloading model.....")
            self.download_models()
            print("Downloaded model")


        self.core = openvino.runtime.Core()
        self.model = None
        self.compiled_model = None
        self.load_model()
    def download_models(self):
        r = requests.get(self.model_xml_url)
        with open(self.model_xml_path, 'wb') as f:
            f.write(r.content)
        r = requests.get(self.model_bin_url)
        with open(self.model_bin_path, 'wb') as f:
            f.write(r.content)

    def load_model(self):
                
    
        self.model = self.core.read_model(model=self.model_xml_path, weights=self.model_bin_path)
        self.compiled_model = self.core.compile_model(model=self.model, device_name="CPU")
        

        input_layer_ir = self.compiled_model.input(0)
        print(input_layer_ir)

    def run_image(self, image):
        print("Running image", image.shape)
        
        input_layer_ir = self.compiled_model.input(0)
        output_layer_ir = self.compiled_model.output(0)
        print('input layer shape: ', input_layer_ir.shape)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (128, 32))
        print(image.shape)
        

        

# #        image = np.transpose(image, (2, 0, 1))
#         image = image.reshape(1, 1, 32, 128)

#         result = self.compiled_model(image)[output_layer_ir]

#         #print(result)
        
#         for i in range(result.shape[0]):
#             print(len(result[i][0]))
#             #print index of max value
#             print(np.argmax(result[i][0]) , self.letters[np.argmax(result[i][0])])
#         return image
        