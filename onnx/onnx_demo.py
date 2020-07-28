# Super Resolution model definition in PyTorch
import torch.nn as nn
import torch.nn.init as init
import torch
import torch.utils.model_zoo as model_zoo
import onnx
import onnxruntime as ort
import numpy as np


class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor, inplace=False):
        super(SuperResolutionNet, self).__init__()

        self.relu = nn.ReLU(inplace=inplace)
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)


def load_torch_model():
    torch_model = SuperResolutionNet(upscale_factor=3)

    # Load pretrained model weights
    model_url = 'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth'

    # Initialize model with the pretrained weights
    map_location = lambda storage, loc: storage
    if torch.cuda.is_available():
        map_location = None
    torch_model.load_state_dict(model_zoo.load_url(model_url, map_location=map_location))
    print('load', model_url)
    return torch_model.eval()  # set the model to inference mode


def cvt_onnx(x, onnx_path):
    # Create the super-resolution model by using the above model definition.
    torch_model = load_torch_model()
    # Input to the model
    torch_out = torch_model(x)

    # Export the model
    torch.onnx.export(
        torch_model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        onnx_path,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=10,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=['img'],  # the model's input names
        output_names=['predict'],  # the model's output names
        dynamic_axes={
            # 这里的名称，与 input_names/output_names 对应
            'img': {
                0: 'batch_size',
                # 2: 'input_h', 3: 'input_w', # 即便输入size改变，也不影响
            },  # 对应 x 中自定义的 dim
            'predict': {
                0: 'batch_size',
                # 2: 'input_h', 3: 'input_w'
            }
        })

    # check
    # onnx_model = onnx.load(onnx_path)
    # onnx.checker.check_model(onnx_model)  # check 仍然会报出 139
    print('convert done!')

    return torch_out


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def onnx_infer(x, onnx_path):
    ort_session = ort.InferenceSession(onnx_path)

    # # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)  # 得到 onnx output

    # print(ort_session.get_inputs()[0].name)  # img, 对应 onnx.export 中 'input_names' 字段
    # print(ort_session.get_outputs()[0].name)  # predict, 对应 onnx.export 中 'output_names' 字段

    return ort_outs  # list


def model_cvt_and_check(onnx_path='onnx/super_resolution.onnx'):
    batch_size = 1
    input_h, input_w = 224, 224
    x = torch.randn(batch_size, 1, input_h, input_w, requires_grad=False)  # 可变参数 batch_size
    torch_out = cvt_onnx(x, onnx_path)
    ort_outs = onnx_infer(x, onnx_path)  # list

    # compare ONNX Runtime and PyTorch results 比较 torch 和 onnx 输出的结果
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)  # 断言 output close to each other
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


def cvt_img_y_to_rgb(img_y):  # y 通道 转 RGB
    img_out_y = Image.fromarray(np.uint8((img_y * 255.0).clip(0, 255)), mode='L')

    # get the output image follow post-processing step from PyTorch implementation
    final_img = Image.merge(
        "YCbCr", [
            img_out_y,
            img_cb.resize(img_out_y.size, Image.BICUBIC),  # 其他通道 上采样
            img_cr.resize(img_out_y.size, Image.BICUBIC),
        ]).convert("RGB")

    return final_img


if __name__ == '__main__':
    from PIL import Image
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt

    img = Image.open("img/cat.jpg")

    resize = transforms.Resize([300, 300])  # 测试可变尺寸
    img = resize(img)

    img_ycbcr = img.convert('YCbCr')
    img_y, img_cb, img_cr = img_ycbcr.split()  # 转成单通道图像

    to_tensor = transforms.ToTensor()
    img_y = to_tensor(img_y)
    img_y.unsqueeze_(0)

    onnx_path = 'onnx/super_resolution.onnx'
    torch_out = cvt_onnx(img_y, onnx_path)
    onnx_out = onnx_infer(img_y, onnx_path)[0]  # 返回 list，元素类型 numpy

    torch_out = cvt_img_y_to_rgb(to_numpy(torch_out).squeeze())
    onnx_out = cvt_img_y_to_rgb(onnx_out.squeeze())  # 颜色略黑

    f, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(torch_out)
    ax[0].set_title('torch_out')
    ax[1].imshow(onnx_out)
    ax[1].set_title('onnx_out')
    plt.show()
