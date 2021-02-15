import cv2
import io
from PIL import Image
import numpy

##Phase 1
# class_dic = {0:'Chole', 1:'Pohay', 2:'Puri', 3:'Sabudana_Khichidi'}
# cal_dic = {"Puri" : "296 Calories", "Chole" : "130 Calories", "Pohay" : "180 Calories", "Sabudana_Khichidi" : "199 Calories"}

##Phase2
class_dic = {0: 'Aloo_Sabzi',
             1: 'Chole',
             2: 'Coconut_Chutney',
             3: 'Cornflakes',
             4: 'Dosa', '5': 'Egg_Bhurji',
             6: 'Green_Coconut_Chutney',
             7: 'Idli',
             8: 'Medu_Vada',
             9: 'Pohay',
             10: 'Puri',
             11: 'Rasam',
             12: 'Rava_Dosa',
             13: 'Red_Chutney',
             14: 'Rice',
             15: 'Sabudana_Khichidi',
             16: 'Sambar',
             17: 'Semiya_Upma',
             18: 'Set_Dosa',
             19: 'Veg_Kurma'}



cal_dic = {
'Aloo_Sabzi': '100 Calories',
'Chole': '130 Calories',
'Coconut_Chutney': '350 Calories',
'Cornflakes': '378 Calories',
'Dosa': '212 Calories',
'Egg_Bhurji': '82 Calories',
'Green_Coconut_Chutney': '204 Calories',
'Idli': '135 Calories',
'Medu_Vada': '309 Calories',
'Pohay': '180 Calories',
'Puri': '296 Calories',
'Rasam': '19 Calories',
'Rava_Dosa': '169 Calories',
'Red_Chutney': '149 Calories',
'Rice': '129 Calories',
'Sabudana_Khichidi': '199 Calories',
'Sambar': '114 Calories',
'Semiya_Upma': '195 Calories',
'Set_Dosa': '212 Calories',
'Veg_Kurma': '134 Calories'
}




cfg_path = '/opt/ml/model/content/YOLOSM/yolov4-obj.cfg'
weights_path = '/opt/ml/model/content/YOLOSM/yolov4-obj_best.weights'


def load_model(cfg_path=cfg_path,weights_path=weights_path):
    try:
        net = cv2.dnn_DetectionModel(cfg_path, weights_path)
        net.setInputSize(416, 416)
        net.setInputScale(1.0 / 255)
        net.setInputSwapRB(True)
    except Exception as e:
        print('Error while loading model!!', e)
    return net


def result(classes, confidences, class_dic=class_dic, cal_dic=cal_dic):
    res = []
    for i in range(len(classes)):
        s = str(class_dic[classes[i][0]]) + '(' + str(round(confidences[i][0] * 100)) + '%) :' + str(cal_dic[class_dic[classes[i][0]]])
        res.append(s)
    print('Postprocessing Complete!', res, type(res))
    return [res]


class ModelHandler(object):
    def __init__(self):
        self.initialized = False
        self.model = None

    def initialize(self, context):
        self.initialized = True
        try:
            self.model = load_model()
            print('Model Loaded?', self.model)
        except Exception as e:
            print('Model Loading Exception!',e)

    def preprocess(self, request):
        print('REQUEST', request)
        dat = request[0]
        try:
            img_array = dat.get('body')
            o = io.BytesIO(img_array)
            # o.seek(0)
            pil_image = Image.open(o)
            input_image = cv2.cvtColor(numpy.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print('Preprocessing Exception!',e)
        print('Preprocess Completed Without Errors', type(input_image), '!', input_image)
        return  input_image

    def inference(self,model_input, conf_thresh = 0.1):
        try:
            classes, confidence, boxes = self.model.detect(model_input, confThreshold=conf_thresh, nmsThreshold=0.4, )
        except Exception as e:
            print('Inference Excpetion!!',e)
        print('Classes and Confs', classes, confidence)
        return classes, confidence

    def postprocess(self, inference_classes, inference_confidences):
        return result(inference_classes,inference_confidences)

    def handle(self, data, context):

        model_input = self.preprocess(data)
        print('MODEL INPUT', model_input)
        out_classes, out_confidences =self.inference(model_input)
        print('Model Generated output!')
        return self.postprocess(out_classes, out_confidences)

_service = ModelHandler()

def handle(data, context):
    if not _service.initialized:
        print('Service Not Initialized, Initializing...')
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)



















