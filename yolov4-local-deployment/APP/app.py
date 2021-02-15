import cv2
import os
import sys
from flask import Flask, request, Response, jsonify
import io
from PIL import Image
import numpy

# cfg_path = r'C:\Users\cvpra\OneDrive\Desktop\CalPred\YOLOv4Deployment\pre_trained_model\yolov4-obj.cfg'
# weights_path = r'C:\Users\cvpra\OneDrive\Desktop\CalPred\YOLOv4Deployment\pre_trained_model\yolov4-obj_best.weights'


cfg_path = r'/home/pre_trained_model/yolov4-obj.cfg'
weights_path = r'/home/pre_trained_model/yolov4-obj_best.weights'



def tree_printer(root):
    for root, dirs, files in os.walk(root):
        print('ROOT!', root)
        print('Dirs')
        for d in dirs:
            print(os.path.join(root, d))
        print('Files')
        for f in files:
            print(os.path.join(root, f))
print('Printing Folder Strucutre')
tree_printer('.')
print('Folder Structure Done')

def preprocess(input_image):
    input_image = cv2.imread(input_image)
    return input_image

#preprocess request image
# def preprocess(img):
#     o = io.BytesIO(img)
#     pil_image = Image.open(o)
#     input_image = cv2.cvtColor(numpy.array(pil_image), cv2.COLOR_RGB2BGR)
#     return input_image


def load_model(cfg_path=cfg_path,weights_path=weights_path):
    if os.path.isfile(cfg_path) == False:
        print('!!! CFG File Not Found at Path')
        raise Exception('CFG File not Found')

    if os.path.isfile(weights_path) == False:
        print('!!! weights File Not Found at Path')
        raise Exception('Wts File not Found')

    try:
        net = cv2.dnn_DetectionModel(cfg_path, weights_path)
        net.setInputSize(416, 416)
        net.setInputScale(1.0 / 255)
        net.setInputSwapRB(True)

    except Exception as e:
        print('Error while loading model!!', e)
    return net

def inference(model, model_input, conf_thresh=0.1):
    classes = ['SOS']
    confidence = ['SOS']
    try:
        classes, confidence, boxes = model.detect(model_input, confThreshold=conf_thresh, nmsThreshold=0.4, )
    except Exception as e:
        print('Inference Exception!!', e)
        raise Exception('Inference Exception!!',  e)

    print('Classes and Confs', classes, confidence)
    return classes, confidence


#Try getting input_image_path from the arguments cli
# input_image_path = r'C:\Users\cvpra\OneDrive\Desktop\CalPred\YOLOv4Deployment\iv1.jpg'
# input_image_path = sys.argv[1]
#
# input_image = preprocess(input_image_path)
# model = load_model()
# inference(model, input_image)
# print(__name__)
import json
app = Flask(__name__)

# @app.route('/api/test', methods= ['POST'])
# def main():
#     ans = {}
#     input_image_path = r'C:\Users\cvpra\OneDrive\Desktop\CalPred\YOLOv4Deployment\iv1.jpg'
#     # input_image_path = sys.argv[1]
#     #
#     input_image = preprocess(input_image_path)
#     model = load_model()
#     classes, confidences = inference(model, input_image)
#     ans['classes'], ans['confidences'] = classes.tolist(), confidences.tolist()
#     print(ans)
#     return json.dumps(ans)
#     # return Response(response=ans,status=200)

#POST REQUEST
# @app.route('/api/test', methods= ['POST'])
# def main():
#     ans = {}
#     img = request.files['image'].read()
#     # input_image_path = sys.argv[1]
#     # input_image = preprocess(input_image_path)
#     input_image = preprocess(img)
#
#     model = load_model()
#     classes, confidences = inference(model, input_image)
#     ans['classes'], ans['confidences'] = classes.tolist(), confidences.tolist()
#     print(ans)
#     return json.dumps(ans)
#     # return Response(response=ans,status=200)

# @app.route('/api/test', methods= ['GET'])
# def main():
#     ans = {'Its':'Working'}
#     print(ans)
#     return json.dumps(ans)

# if __name__ == '__main__':
#     app.run(debug=True, host = '0.0.0.0')

# app.run(debug=True, host = '0.0.0.0')
# app.run(host = '0.0.0.0')

import boto3
#
# def download_s3_file(bucket_name, s3_file_name, local_file_name):
#     s3 = boto3.client('s3')
#     s3.download_file(bucket_name, s3_file_name, local_file_name)

ACCESS_ID = 'AKIASN6VK6L4LCDN6TXD'
ACCESS_KEY = 'c81JsrzfOfnUg8HMJL2SZ7sTu7HuNGNSlazi7wEW'


#Download with credentials
def download_s3_file(bucket_name, s3_file_name, local_file_name):
    s3 = boto3.resource('s3',
         aws_access_key_id=ACCESS_ID,
         aws_secret_access_key= ACCESS_KEY)
    bucket = s3.Bucket(bucket_name)
    bucket.download_file(s3_file_name, local_file_name)

def download_all_files(bucket_name):
    #initiate resource
    s3 = boto3.resource('s3')
    #select bucket
    bucket = s3.Bucket(bucket_name)
    #download file to current directory
    for s3_object in bucket.objects.all():
        filename = s3_object.key
        bucket.download_file(s3_object.key, filename)

def download_all_objects_in_a_folder(bucket_name, folder_path):
    s3_resource = boto3.resource('s3')
    bucket = s3_resource.Bucket(bucket_name)
    objects = bucket.objects.filter(Prefix = folder_path)
    print(objects)
    for obj in objects:
        path, filename = os.path.split(obj.key)
        if filename != '':
            bucket.download_file(obj.key, filename)
        else:
            print(obj.key, ': did not download this file')



#Run predictions through all those images
#POST REQUEST with sagemaker path
@app.route('/api/test', methods= ['POST'])
def main():
    ans = {}
    data = request.form.to_dict(flat=False)
    uri = data['uri']
    uri = uri[0]
    # print('uri',uri)
    # print(type(uri))
    # print(len(uri))
    uri = uri.split('/')
    bucket_name = uri[2]
    s3_file_name = '/'.join(uri[3:])
    local_file_name = uri[-1]
    cwd = os.getcwd()
    download_s3_file(bucket_name, s3_file_name, os.path.join(cwd, local_file_name))
    input_image_path = os.path.join(cwd, local_file_name)
    input_image = preprocess(input_image_path)
    model = load_model()
    classes, confidences = inference(model, input_image)
    ans['classes'], ans['confidences'] = classes.tolist(), confidences.tolist()
    print(ans)
    return json.dumps(ans)
    # return Response(response=ans,status=200)
    # return json.dumps({'res': uri})

app.run(debug=True, host = '0.0.0.0')
