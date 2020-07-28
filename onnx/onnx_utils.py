import torch
import torch.onnx
import onnxruntime as ort  # rt: runtime
from model.bisenet import BiSeNet
from utils.misc import load_state_dict
import onnx

"""
https://pytorch.apachecn.org/docs/1.0/onnx.html
    局限、支持的运算符、
https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
"""

"""ONNX faults
ONNX's Upsample/Resize operator did not match Pytorch's Interpolation until opset 11
ONNX export failed: Couldn't export operator aten::upsample_bilinear2d
ONNX export failed on upsample_bilinear2d because align_corners == True not supported
"""


def cvt_onnx(model, onnx_path):
    model.eval()

    # Input to the model, 定义 onnx 模型的可变参数
    batch_size = 1
    input_h, input_w = 512, 512

    # x dummpy input: can be random as long as it is the right type and size.
    # Note: the input size will be fixed in the exported ONNX graph for all the input’s dimensions, unless specified as a dynamic axes.
    # 使用 dynamic axes: B,H,W, 使 ONNX graph 能接受变化 shape 的输入
    x = torch.randn(batch_size, 3, 512, 512, requires_grad=True)
    out = model(x)

    torch.onnx.export(
        model,
        x,  # model input, (or a tuple for multiple inputs)
        onnx_path,  # file path
        verbose=True,  # 输出转换过程

        export_params=True,  # store the trained parameter weights inside the model file, default True
        opset_version=9,  # the ONNX version to export the model to, default 9
        do_constant_folding=True,  # whether to execute constant folding for optimization, default False

        input_names=['input'],  # the model's input names, 可以是多个
        output_names=['output'],  # the model's output names
        dynamic_axes={  # dynamic dimensions
            'input': {  # variable lenght axes
                0: 'batch_size',
                # 2: 'input_h',
                # 3: 'input_w',
            },
            'output': {
                0: 'batch_size',
                #     2: 'output_h',
                #     3: 'output_w',
            }
        }
    )

    check_onnx_model(onnx_path)


def check_onnx_model(onnx_path):
    # Load the ONNX model
    model = onnx.load(onnx_path)  # load the saved model and will output a onnx.ModelProto structure
    print('load', onnx_path)

    # Check that the IR is well formed
    try:
        onnx.checker.check_model(model)  # verify the model’s structure and confirm that the model has a valid schema.
        print('check pass!')
    finally:  # except
        print('check error!')

    # Print a human readable representation of the graph
    onnx.helper.printable_graph(model.graph)  # 没有输出? 因为 check 失败了?


def onnx_infer(onnx_path):
    ort_session = ort.InferenceSession(onnx_path)


def demo_res18():
    from torchvision.models.resnet import resnet18

    model = resnet18(pretrained=False).eval()

    batch_size = 1
    input_h, input_w = 512, 512
    x = torch.rand((batch_size, 3, input_h, input_w))
    out = model(x)
    # print(out.shape)  # 1,1000, 采用 avgpool, 输入 size 无关

    onnx_path = 'onnx/res18.onnx'

    torch.onnx.export(
        model,
        x,
        onnx_path,
        verbose=True,

        export_params=True,  # store the trained parameter weights inside the model file, default True
        opset_version=9,  # the ONNX version to export the model to, default 9
        do_constant_folding=True,  # whether to execute constant folding for optimization, default False

        input_names=['input'],  # the model's input names, 可以是多个
        output_names=['output'],  # the model's output names
        dynamic_axes={  # dynamic dimensions
            'input': {  # variable lenght axes
                0: 'batch_size',
                2: 'input_h',
                3: 'input_w',
            },
            'output': {
                0: 'batch_size',
            }
        }
    )

    check_onnx_model(onnx_path)


if __name__ == '__main__':
    # model = BiSeNet(37, context_path='resnet18', in_planes=32)
    # load_state_dict(model, ckpt_path='runs/SUNRGBD/res18_inp32_deconv_Jul27_100319/checkpoint.pth.tar')
    # onnx_path = 'onnx/res18_inp32_deconv_Jul27_100319.onnx'
    # cvt_onnx(model, onnx_path)

    # todo: demo resnet
    demo_res18()

    # model = BiSeNet(37, context_path='resnet101', in_planes=64)
    # load_state_dict(model, ckpt_path='runs/SUNRGBD/res101_inp64_deconv_Jul26_205859/checkpoint.pth.tar')
    # cvt_onnx(model, onnx_path='onnx/res101_inp64_deconv_Jul26_205859.onnx')
