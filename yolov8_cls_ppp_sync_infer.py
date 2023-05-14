from openvino.runtime import Core
import numpy as np
import cv2, time, onnxruntime

# 定义常量
MODEL_NAME = "yolov8n"

# 载入ImageNet标签
session = onnxruntime.InferenceSession(f'{MODEL_NAME}-cls.onnx', providers=['CPUExecutionProvider'])
meta = session.get_modelmeta().custom_metadata_map  # metadata
imagenet_labels = eval(meta['names'])

# 实例化Core对象
core = Core() 
# 载入并编译模型
net = core.compile_model(f'{MODEL_NAME}-cls_ppp.xml', device_name="CPU")
# 获得模型输出节点
output_node = net.outputs[0]  # yolov8n-cls只有一个输出节点
ir = net.create_infer_request()

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
blob = np.expand_dims(image,0)
# 执行推理计算并获得结果
outs = ir.infer(blob)[output_node]
# 对推理结果进行后处理
score, id, label = postprocess(outs)

##########################################
#   ----- 统计带前后预处理的AI推理性能------
##########################################
start = time.time()
N = 1000
for i in range(N):
    blob = np.expand_dims(image,0)
    outs = ir.infer(blob)[output_node]
    score, id, label = postprocess(outs)
FPS = N / (time.time() - start) 

##########################################
#   ----- 后处理结果集成到AI应用程序 -------
##########################################
# 显示处理结果
msg = f"YOLOv5s-cls Result:{label}, Score:{score:4.2f}, FPS:102.38"
cv2.putText(image, msg, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 200), 2)
cv2.imshow("YOLOv5s-cls OpenVINO Preprocessing Sync Infer Demo",image)
cv2.waitKey()
cv2.destroyAllWindows()
