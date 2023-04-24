from openvino.runtime import Core
import numpy as np
import cv2, time, onnxruntime

# 定义常量
MODEL_NAME = "yolov8n"
# ImageNet RGB mean, Shape:[1,1,C]
IMAGENET_MEAN = np.array([123.675, 116.28 , 103.53])[np.newaxis, np.newaxis, :]  
# ImageNet RGB standard deviation, Shape:[1,1,C]
IMAGENET_STD = np.array([58.395, 57.12 , 57.375])[np.newaxis, np.newaxis, :]

# 载入ImageNet标签
session = onnxruntime.InferenceSession(f'{MODEL_NAME}-cls.onnx', providers=['CPUExecutionProvider'])
meta = session.get_modelmeta().custom_metadata_map  # metadata
imagenet_labels = eval(meta['names'])

# 实例化Core对象
core = Core() 
# 载入并编译模型
net = core.compile_model(f'{MODEL_NAME}-cls.xml', device_name="GPU")
# 获得模型输入输出节点
input_node = net.inputs[0]    # yolov8n-cls只有一个输入节点
N, C, H, W = input_node.shape # 获得输入张量的形状
output_node = net.outputs[0]  # yolov8n-cls只有一个输出节点
ir = net.create_infer_request()
##########################################
#   ---根据模型定义预处理和后处理函数-------
##########################################
# 定义预处理函数
def preprocess(image, new_shape=(W,H)):
    # Preprocess image data from OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   # BGR->RGB
    resized = cv2.resize(image, new_shape)           # Resize Image
    norm = (resized - IMAGENET_MEAN) / IMAGENET_STD  # Normalization
    blob = np.transpose(norm, (2, 0, 1))             # HWC->CHW
    blob = np.expand_dims(blob, 0)                   # CHW->NCHW
    return blob

# 定义后处理函数
def postprocess(outs):
    score = np.max(outs)
    id = np.argmax(outs)
    return score, id, imagenet_labels[id]

##########################################
#   ----- AI同步推理计算 ------------
##########################################
# 采集图像
image = cv2.imread("bus.jpg")
# 数据预处理
blob = preprocess(image)
# 执行推理计算并获得结果
outs = ir.infer(blob)[output_node]
# 对推理结果进行后处理
score, id, label = postprocess(outs)

##########################################
#   ----- 统计带前后预处理的AI推理性能------
##########################################
start = time.time()
N = 4000
for i in range(N):
    blob = preprocess(image)
    outs = ir.infer(blob)[output_node]
    score, id, label = postprocess(outs)
FPS = N / (time.time() - start) 

##########################################
#   ----- 后处理结果集成到AI应用程序 -------
##########################################
# 显示处理结果
msg = f"YOLOv5s-cls Result:{label}, Score:{score:4.2f}, FPS:{FPS:4.2f}"
cv2.putText(image, msg, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 200), 2)
cv2.imshow("YOLOv5s-cls OpenVINO Sync Infer Demo",image)
cv2.waitKey()
cv2.destroyAllWindows()

