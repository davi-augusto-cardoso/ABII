# from visualizer import *

# if __name__ == "__main__":
#     path = r"../data/warMap1.png"
#     vis = Visualizer(path)
    
#     vis.CreateMask("yellow")
#     vis.PreProcess()
#     images = vis.Contourning()


import cv2
import matplotlib.pyplot as plt
import numpy as np
import openvino as ov
import openvino.runtime
from pathlib import Path
import sys
import requests
import time
import pytesseract

base_model_dir = Path("../model").expanduser()

model_name = "text-detection-0004"
model_xml_name = f'{model_name}.xml'
model_bin_name = f'{model_name}.bin'

model_xml_path = base_model_dir / model_xml_name
model_bin_path = base_model_dir / model_bin_name


model_xml_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/models_bin/1/"+ model_name+"/FP32/text-detection-0004.xml"
model_bin_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/models_bin/1/"+ model_name+"/FP32/text-detection-0004.bin"


model_horizontal_name = "horizontal-text-detection-0001"
model_horizontal_xml_name = f'{model_horizontal_name}.xml'
model_horizontal_bin_name = f'{model_horizontal_name}.bin'

model_horizontal_xml_path = base_model_dir / model_horizontal_xml_name
model_horizontal_bin_path = base_model_dir / model_horizontal_bin_name
#download file from https:
model_horizontal_xml_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/models_bin/1/horizontal-text-detection-0001/FP32/horizontal-text-detection-0001.xml"
model_horizontal_bin_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/models_bin/1/horizontal-text-detection-0001/FP32/horizontal-text-detection-0001.bin"


# For each detection, the description is in the [x_min, y_min, x_max, y_max, conf] format:
# The image passed here is in BGR format with changed width and height. To display it in colors expected by matplotlib, use cvtColor function
def convert_result_to_image(bgr_image, resized_image, boxes, threshold=0.3, conf_labels=True):
    # Define colors for boxes and descriptions.
    colors = {"red": (255, 0, 0), "green": (0, 255, 0)}

    # Fetch the image shapes to calculate a ratio.
    (real_y, real_x), (resized_y, resized_x) = bgr_image.shape[:2], resized_image.shape[:2]
    ratio_x, ratio_y = real_x / resized_x, real_y / resized_y

    # Convert the base image from BGR to RGB format.
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    # Iterate through non-zero boxes.
    for box in boxes:
        # Pick a confidence factor from the last place in an array.
        conf = box[-1]
        if conf > threshold:
            # Convert float to int and multiply corner position of each box by x and y ratio.
            # If the bounding box is found at the top of the image, 
            # position the upper box bar little lower to make it visible on the image. 
            (x_min, y_min, x_max, y_max) = [
                int(max(corner_position * ratio_y, 10)) if idx % 2 
                else int(corner_position * ratio_x)
                for idx, corner_position in enumerate(box[:-1])
            ]

            # Draw a box based on the position, parameters in rectangle function are: image, start_point, end_point, color, thickness.
            rgb_image = cv2.rectangle(rgb_image, (x_min, y_min), (x_max, y_max), colors["green"], 3)

            # Add text to the image based on position and confidence.
            # Parameters in text function are: image, text, bottom-left_corner_textfield, font, font_scale, color, thickness, line_type.
            if conf_labels:
                rgb_image = cv2.putText(
                    rgb_image,
                    f"{conf:.2f}",
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    colors["red"],
                    1,
                    cv2.LINE_AA,
                )

    return rgb_image

def preprocess_image(image):
    #cv2 binary image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    image = cv2.threshold(image, 230, 255, cv2.THRESH_BINARY )[1]
    return image

def model_text_detection(image):

    #download files with url inside model folder
    if not model_xml_path.exists():
        print("Downloading model xml file")
        r = requests.get(model_xml_url)
        model_xml_path.write_bytes(r.content)
        print("Downloading model bin file")
    if not model_bin_path.exists():
        print("Downloading model bin file")
        r = requests.get(model_bin_url)
        model_bin_path.write_bytes(r.content)
        print("Downloaded model bin file")

    core = openvino.runtime.Core()

    model = core.read_model(model=model_xml_path)
    compiled_model = core.compile_model(model=model, device_name="CPU")

    #layers names
    print(model.inputs)

    input_layer_ir = compiled_model.input(0)

   # output_layer_ir0 = compiled_model.output(0)
    #output_layer_ir1 = compiled_model.output(1)


    # Text detection models expect an image in BGR format.
    #image = cv2.imread("../data/warMap.png")

    # N,C,H,W = batch size, number of channels, height, width.
    N, H, W, C = input_layer_ir.shape
    print(f"Model Input shape:", input_layer_ir.shape)
    print("mode width and height", W, H)
    # Resize the image to meet network expected input sizes.
    resized_image = cv2.resize(image, (W, H))

    # Reshape to the network input shape.
    #input_image = np.expand_dims
    print('resized shape', resized_image.shape)
    input_image = np.expand_dims(resized_image, 0)
    #Image, name: Placeholder, shape: 1, 768, 1280, 3 in the format B, H, W, C, where:
    print(f"Input image shape: {input_image.shape}")

    #write image
    #cv2.imwrite("image.png", resized_image)
    # Create an inference request.
    boxes = compiled_model([input_image])
    #boxes1 = compiled_model([input_image])[output_layer_ir1]
    # # Remove zero only boxes.
    #boxes = boxes[~np.all(boxes == 0, axis=1)]
    segm_logits = boxes[0]
    results = segm_logits[0]





    #results shape (192,320,2)
    #100 linha

    # linha = results[30]
    # print(np.mean(linha))

    # linha = results[100]
    # print(np.mean(linha))

    # linha_150 = results[150]
    # print(np.mean(linha_150))

    #create heatmap
    #heatmap = np.zeros((H, W, 3), np.uint8)
    heatmap = np.zeros((192, 320, 3), np.uint8)
    for linha in range(0, results.shape[0]):
        for coluna in range(0, results.shape[1]):
            heatmap[linha][coluna] = results[linha][coluna][0]
            #print(results[linha][coluna][0])
            #print(results[linha][coluna])

    #for coluna in linha:
    #    print(coluna)

    #print(results[0].shape)
    # # Show the image.
    return heatmap
    #plt.imshow(heatmap)
    #plt.show()

    # plt.figure(figsize=(10, 6))
    # plt.axis("off")
    # plt.imshow(convert_result_to_image(image, resized_image, boxes, conf_labels=False));

    # # Show the image.
    # plt.show()


def model_horizontal_detection(image):
    
        #download files with url inside model folder
        if not model_horizontal_xml_path.exists():
            print("Downloading model xml file")
            r = requests.get(model_horizontal_xml_url)
            model_horizontal_xml_path.write_bytes(r.content)
            print("Downloading model bin file")
        if not model_horizontal_bin_path.exists():
            print("Downloading model bin file")
            r = requests.get(model_horizontal_bin_url)
            model_horizontal_bin_path.write_bytes(r.content)
            print("Downloaded model bin file")
    
        core = openvino.runtime.Core()
    
        model = core.read_model(model=model_horizontal_xml_path)
        compiled_model = core.compile_model(model=model, device_name="CPU")
        

        input_layer_ir = compiled_model.input(0)
        output_layer_ir = compiled_model.output("boxes")

        # Text detection models expect an image in BGR format.
        

        # N,C,H,W = batch size, number of channels, height, width.
        N, C, H, W = input_layer_ir.shape

        # Resize the image to meet network expected input sizes.
        resized_image = cv2.resize(image, (W, H))

        # Reshape to the network input shape.
        input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0)

        #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB));

        # Create an inference request.
        boxes = compiled_model([input_image])[output_layer_ir]

        # Remove zero only boxes.
        boxes = boxes[~np.all(boxes == 0, axis=1)]


        return convert_result_to_image(image, resized_image, boxes, conf_labels=False)



start_time = time.time()
image = cv2.imread("../data/warMap.png")

pre_processed_image = preprocess_image(image)
cv2.imwrite("pre_processed_image.png", pre_processed_image)


cnts,new = cv2.findContours(pre_processed_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
image1=image.copy()


cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
#print areas
boxes = []
for c in cnts:
    area = cv2.contourArea(c)
    
    if(area < 20 and area > 3):
        #print(area)
        boxes.append(c)

cv2.drawContours(image1,boxes,-1,(0,255,0),3)
print('len boxes', len(boxes))
cv2.imwrite("contours_results.png", image1)

end_time = time.time()
#plt.imshow(image1)
#plt.show()

cropped_images = []
for box in boxes:
    x,y,w,h = cv2.boundingRect(box)
    #give a little more space
    #print(x,y,w,h)
    x = x - 5
    y = y - 5
    w = w + 10
    h = h + 10
    crop = image[y:y+h,x:x+w]
    cropped_images.append(crop)
    # plt.imshow(crop)
    # plt.show()


desired_width = 128
desired_height = 32
ex_image = cropped_images[1]

image_width = ex_image.shape[1]
image_height = ex_image.shape[0]

color = [0, 0, 0]
delta_w = desired_width - image_width
delta_h = desired_height - image_height
new_left, new_right = delta_w//2, delta_w-(delta_w//2)
new_top, new_bottom = delta_h//2, delta_h-(delta_h//2)
new_im = cv2.copyMakeBorder(ex_image, new_top, new_bottom, new_left, new_right, cv2.BORDER_CONSTANT,
    value=color)


cv2.imwrite("new_im.png", new_im)


import text_recognition_class
text_recognition = text_recognition_class.text_recognition()

new_im = cv2.imread("openvino.jpg")
text_recognition.run_image(new_im)

# plt.figure(figsize=(10, 6))
# plt.axis("off")
# plt.imshow(res);
# plt.show()













#cv2.imwrite("crop.png", crop)









#Funcionou detectar grande
#cv2.imwrite("pre_processed_image.png", pre_processed_image)
#TODO retirar isso e processar a NN com gray mesmo, expandindo o tensor
#pre_processed_image = cv2.cvtColor(pre_processed_image, cv2.COLOR_GRAY2BGR) 
#cv2.imwrite("pre_processed_image.png", pre_processed_image)
# result_image = model_horizontal_detection(pre_processed_image)
# end_time = time.time()
# print("time model_horizontal_detection ", end_time - start_time)

# start_time = time.time()
# result_image = model_text_detection(pre_processed_image)
# print("time model_text_detection ", end_time - start_time)

#60,44 com 
# plt.figure(figsize=(10, 6))
# plt.axis("off")
# plt.imshow(result_image);
# plt.show()

