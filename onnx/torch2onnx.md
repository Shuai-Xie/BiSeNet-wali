## Pytroch 上采样 -> ONNX

#### ConvTranspose2d 反卷积
```py
# x8 upsample
self.ffm_upsample = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=8, stride=8, padding=0, output_padding=0)
```
```
%298 : Float(1, 14, 60, 80) = onnx::Add(%297, %291), scope: BiSeNet/FeatureFusionModule[FFM]
%299 : Float(1, 14, 480, 640) = onnx::ConvTranspose[dilations=[1, 1], group=1, kernel_shape=[8, 8], pads=[0, 0, 0, 0], strides=[8, 8]](%298, %187), scope: BiSeNet/DeconvBlock[ffm_upsample]/ConvTranspose2d[deconv]
```

#### Interpolation 插值
```py
# x8 upsample
result = F.interpolate(result, size=(480, 640), mode='bilinear')
# or
result = F.interpolate(result, scale_factor=8, mode='bilinear')
```
设置 `size` or `scale_factor` 其实背后对应同一种插值方式，所以转化成 onnx 时，过程是一样的
```
%278 : Float(1, 14, 60, 80) = onnx::Add(%277, %271), scope: BiSeNet/FeatureFusionModule[FFM]
%279 : Tensor = onnx::Constant[value= 1  1  8  8 [ CPUFloatType{4} ]](), scope: BiSeNet
%280 : Float(1, 14, 480, 640) = onnx::Upsample[mode="linear"](%278, %279), scope: BiSeNet
```
```
%278 : Float(1, 14, 60, 80) = onnx::Add(%277, %271), scope: BiSeNet/FeatureFusionModule[FFM]
%279 : Tensor = onnx::Constant[value= 1  1  8  8 [ CPUFloatType{4} ]](), scope: BiSeNet
%280 : Float(1, 14, 480, 640) = onnx::Upsample[mode="linear"](%278, %279), scope: BiSeNet
```
---
插值方式得到的 onnx 模型在用 tensorRT 构建 engine 时会报错：**Attribute not found: height_scale**
- 原因：`%279 Constant` 定义了放缩因子，而 `%280 Upsample` 并没有得到这个 scale，第一个参数是 height，所以就报错：没有 `height_scale` 这一项
- 修改：重载 onnx 的 upsample


重载最近邻插值：upsample_nearest2d
```py
import torch.onnx.symbolic

# Override Upsample's ONNX export until new opset is supported
@torch.onnx.symbolic.parse_args('v', 'is')
def upsample_nearest2d(g, input, output_size):
    height_scale = float(output_size[-2]) / input.type().sizes()[-2]
    width_scale = float(output_size[-1]) / input.type().sizes()[-1]
    return g.op("Upsample", input,
                scales_f=(1, 1, height_scale, width_scale),
                mode_s="nearest")

# 点进去原始的函数定义，就知道重载函数怎么写了
torch.onnx.symbolic.upsample_nearest2d = upsample_nearest2d
```
重载双线性插值：upsample_bilinear2d
```py
@torch.onnx.symbolic.parse_args('v', 'is', 'i')
def upsample_bilinear2d(g, input, output_size, align_corners):
    height_scale = float(output_size[-2]) / input.type().sizes()[-2]  # 8
    width_scale = float(output_size[-1]) / input.type().sizes()[-1]  # 8
    return g.op("Upsample", input,
                scales_f=(1, 1, height_scale, width_scale),
                mode_s="linear")

torch.onnx.symbolic.upsample_bilinear2d = upsample_bilinear2d
```
这样模型转成 onnx 时 upsample 就能拿到 scale 了
```
%276 : Float(1, 14, 60, 80) = onnx::Add(%275, %269), scope: BiSeNet/FeatureFusionModule[FFM]
%277 : Float(1, 14, 480, 640) = onnx::Upsample[mode="nearest", scales=[1, 1, 8, 8]](%276), scope: BiSeNet
```