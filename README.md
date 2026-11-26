# ComfyUI-Kandinsky-ImageSR

Kandinsky ImageSR 2.0 图片放大节点：输入原图即可，不用提示词。这个 ZIP 包含节点、官方推理代码 wheel 和示例工作流，**不含模型权重**（约 10.4 GB）。

## 安装

1. 解压，把整个 ComfyUI-Kandinsky-ImageSR 文件夹放到 ComfyUI 的 custom_nodes 里。不要多套一层同名文件夹。
2. 到这个插件文件夹，**用你启动 ComfyUI 的那个 Python** 执行下面的命令。Windows 便携版用 python_embeded 目录里的 python.exe 替换 python。

   ~~~bash
   python -m pip install -r requirements.txt
   python -m pip install --no-deps "vendor/kandinsky_image_sr-1.0.0-py3-none-any.whl"
   python download_weights.py
   ~~~

   如果已经把对应权重放到 ComfyUI 的 models/kandinsky_image_sr/Kandinsky-ImageSR-2.0/，可以跳过下载；权重目录应包含 model_index.json、transformer 和 vae。自定义模型目录时可给下载命令加 --models-dir；也可以在 deployment.json 里将 model_directory 改为权重文件夹的绝对路径。

3. 重启 ComfyUI，搜「Kandinsky ImageSR 2.0 (图片超分)」，或拖入 examples/Kandinsky-ImageSR-2.0.json，再上传自己的图片。

需要现有的 CUDA 版 PyTorch（官方要求 torch >= 2.6）和支持 BF16 的 NVIDIA 显卡；依赖安装清单不包含 PyTorch。此包不会在 ComfyUI 启动时自动安装依赖或下载模型。

## 参数

- **upscale_factor**：2 倍、4 倍。
- **creativity**：细节重绘强度，默认 4。
- **overlap**：分块重叠，默认 0.25（25%）。
- **tile_batch_size**：每批处理的块数，默认 1。加大通常会占用更多显存。

## 来源

- [KandinskyLab 官方推理代码](https://github.com/kandinskylab/kandinsky-image-sr/tree/6c2405796cf7576b0cffff27acd103d667b91ca8)，随包附带上游 MIT 许可。
- [官方模型权重](https://huggingface.co/kandinskylab/Kandinsky-ImageSR-2.0/tree/7d5b5ed7def0b45b98f398c0b295a6b1658863db)，下载辅助脚本用官方文件校验值核对完整性。
