from typing import Callable

import numpy as np
import pynvml
import torch
import torch.nn.functional as F
from torch import Tensor, nn, optim
from torchinfo import summary
from tqdm import tqdm


def print_memory_torch(prefix: str):
    """Print memory usage.
    """    
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)    
    memory_al = torch.cuda.memory_allocated()
    memory_res = torch.cuda.memory_reserved()
    memory_maxal = torch.cuda.max_memory_allocated()

    print(f"{prefix}: allocated = {memory_al/1024**2:.1f} MiB, "
        f"reserved = {memory_res/1024**2:.1f}MiB, "
        f"max allocated = {memory_maxal/1024**2:.1f} MiB, "
        f"used = {int(info.used)/1024**2:.1f} MiB")
    

def is_memoryless(class_name: str) -> bool:
    ''' Return True if the class is memoryless type.
    Activations, normalizations and dropouts perform in-place updates by default
    and does not require additional memory.
    '''
    return any((class_name == "ReLU",
                class_name == "LeakyReLU",
                class_name == "Sigmoid",
                class_name == "Tanh",
                class_name == "ELU",
                class_name == "GLU",
                class_name == "PReLU",
                class_name == "GELU",
                class_name == "Mish",
                class_name == "Softmin",
                class_name == "Softmax",
                class_name == "Softmax2d"))


def print_memory_estimate2(
    model: nn.Module, 
    dim_input: list[int], 
    moment: int, 
    ddp: int=1, 
    mixed_pre: float = 1):
    '''Print theoretical memory usage.
    
    Parameters
    ----------
    model: 
    dim_input: Shape of input data including batch size. e.g. [batch size, channel, width, height]
    moment: Moment use for optimization. SGD: 0, Adagrad, RMSprop: 1, Adam: 2
    ddp: Multiple GPU use. Distributed data parallel: 2, Not: 1
    mixed_pre: Forward outputs memory saving by Mixed precision: 0.5, Not: 1
    '''
    info = summary(model, dim_input, verbose=0)
    dim_output = info.summary_list[-1].output_size[1:]

    num_param = 0
    num_output_shape = 0
    last_layer = len(info.summary_list) -1
    print("#, Class, Leaf, Memoryless, Output")
    for i, layer in enumerate(info.summary_list):
        print(f"{i}, {layer.class_name}, {layer.is_leaf_layer}, {is_memoryless(layer.class_name)}, {layer.output_size}")
        if layer.is_leaf_layer:
            num_param += layer.trainable_params
            if i != last_layer and not is_memoryless(layer.class_name):
                num_output_shape += np.prod(layer.output_size)
    
    mem_data = (np.prod(dim_input) + np.prod(dim_output)) * 4
    mem_weight = num_param * 4
    mem_weight_grad = mem_weight * (ddp + moment)
    mem_forward_output = num_output_shape * 4 * mixed_pre
    mem_output_gradient = mem_forward_output + mem_data
    mem_training = mem_data + mem_weight + mem_forward_output + mem_weight_grad + mem_output_gradient
    mem_inference = mem_data + mem_weight + mem_forward_output

    print(f"Data(MiB): {mem_data/1024**2:.1f}")
    print(f"Weight(MiB): {mem_weight/1024**2:.1f}")
    print(f"Forward output(MiB): {mem_forward_output/1024**2:.1f}")
    print(f"Weight gradient(MiB): {mem_weight_grad/1024**2:.1f}")
    print(f"Output gradient(MiB): {mem_output_gradient/1024**2:.1f}")
    print(f"Total for training(MiB): {mem_training/1024**2:.1f}")
    print(f"Total for inference(MiB): {mem_inference/1024**2:.1f}")


def train(
    model:nn.Module, 
    dim_input: list[int], 
    dim_output: list[int],
    batchsize: int, 
    epoch: int, 
    criterion: Callable[..., Tensor],
    optimizer = optim.SGD, 
    device: str = "cuda"):
    """Train model using random dataset.
    
    Parameters
    ----------
    model: 
    dim_input: Shape of input data including data size. e.g. [data size, channel, width, height]
    dim_output: 
    batchsize:
    epoch:
    optimizer:
    device: 
    """
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print_memory_torch("Initial")

    model.to(device)
    print_memory_torch("Model")
    
    data = [[torch.randn([batchsize] + dim_input[1:]), 
             torch.randn([batchsize] + dim_output)] 
             for _ in range(dim_input[0]//batchsize)]

    criterion = F.cross_entropy
    opt = optimizer(model.parameters(), lr=0.01)
    for ep in range(epoch):
        model.train()
        with tqdm(data) as pbar:
            pbar.set_description(f'[Epoch {ep + 1}]')
            for x, y in pbar:
                x = x.to(device)
                y = y.to(device)
                
                opt.zero_grad()
                y_pred = model(x)
                loss = criterion(y_pred, y)
                loss.backward()
                opt.step()
            
        print_memory_torch("Train")
    print_memory_torch("Final")


class Config:
    def __init__(self):
        self.dim_input = [3,224,224]
        self.dim_output = [10]
        self.datasize = 512
        self.batchsize = 128
        self.num_epochs = 3
        self.lr = 1e-2
        self.device = 'cuda'
        self.criterion = F.cross_entropy
        self.optim = optim.SGD
        self.moment = 0         # SGD: 0, Adagrad, RMSprop: 1, Adam: 2
        self.ddp = 1            # Distributed data parallel: 2, Not: 1
        self.mixed_pre = 1      # Mixed precision: 0.5, Not: 1


class CNN(nn.Module):
    def __init__(self, dim_c: int, dim_h: int, dim_w: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=dim_c, out_channels=8, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(num_features=8)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(inplace=True)
        self.fc = nn.Linear(8*((dim_h-2)//2)*((dim_w-2)//2), 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


if __name__=="__main__":

    conf = Config()
    model_cnn = CNN(conf.dim_input[0], conf.dim_input[1], conf.dim_input[2])
    
    result = summary(model_cnn, [conf.batchsize] + conf.dim_input,
                depth=6,
                col_names=["input_size",
                            "output_size",
                            "num_params",
                            "params_percent",
                            "kernel_size",
                            "mult_adds",
                            "trainable"])
    print(result)

    print("=== Estimated ===")
    print_memory_estimate2(model_cnn, [conf.batchsize] + conf.dim_input, 
                        conf.moment, conf.ddp, conf.mixed_pre)

    print("=== Real ===")
    train(model_cnn, [conf.datasize] + conf.dim_input, conf.dim_output, conf.batchsize, conf.num_epochs,
        conf.criterion, conf.optim, conf.device)