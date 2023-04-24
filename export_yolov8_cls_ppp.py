from openvino.preprocess import PrePostProcessor, ColorFormat, ResizeAlgorithm
from openvino.runtime import Core, Layout, Type, serialize

# 定义常量
MODEL_NAME = "yolov8n"

# ========  Step 0: read original model =========
core = Core()
model = core.read_model(f"{MODEL_NAME}-cls.xml")

# Step 1: Add Preprocessing steps to a model ==
ppp = PrePostProcessor(model) 
# Declare User’s Data Format
ppp.input().tensor() \
    .set_element_type(Type.u8) \
    .set_spatial_dynamic_shape() \
    .set_layout(Layout('NHWC')) \
    .set_color_format(ColorFormat.BGR)
# Declaring Model Layout
ppp.input().model().set_layout(Layout('NCHW'))
# Explicit preprocessing steps. Layout conversion will be done automatically as last step
ppp.input().preprocess() \
    .convert_element_type()     \
    .convert_color(ColorFormat.RGB) \
    .resize(ResizeAlgorithm.RESIZE_LINEAR) \
    .mean([123.675, 116.28, 103.53]) \
    .scale([58.624, 57.12, 57.375])
# Integrate preprocessing Steps into a Model
print(f'Dump preprocessor: {ppp}')
model_with_ppp = ppp.build()

# ======== Step 2: Save the model with preprocessor================
serialize(model_with_ppp, f'{MODEL_NAME}-cls_ppp.xml', f'{MODEL_NAME}-cls_ppp.bin')

