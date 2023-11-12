# Forward Neural Networkのメモリ見積もりの試し
# ref: https://github.com/py-img-recog/python_image_recognition


from collections import deque
import copy
from tqdm import tqdm
from typing import Callable

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torchinfo import summary
import numpy as np
from PIL import Image
import random
import pynvml


def print_memory_torch(prefix: str) -> None:
# Print memory usage

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)    
    memory_al = torch.cuda.memory_allocated()
    memory_res = torch.cuda.memory_reserved()
    memory_maxal = torch.cuda.max_memory_allocated()

    print(f"{prefix}: allocated = {memory_al/1024**2:.3f} MiB, "
        f"reserved = {memory_res/1024**2:.3f}MiB, "
        f"max allocated = {memory_maxal/1024**2:.3f} MiB, "
        f"used = {info.used/1024**2:.3f} MiB")

class FNN(nn.Module):
    '''
    順伝播型ニューラルネットワーク
    dim_input        : 入力次元
    dim_hidden       : 特徴量次元
    num_hidden_layers: 隠れ層の数
    num_classes      : 分類対象の物体クラス数
    '''
    def __init__(self, dim_input: int, dim_hidden: int,
                 num_hidden_layers: int, num_classes: int):
        super().__init__()

        ''''' 層の生成 '''''
        self.layers = nn.ModuleList()

        # 入力層 -> 隠れ層
        self.layers.append(self._generate_hidden_layer(
            dim_input, dim_hidden))

        # 隠れ層 -> 隠れ層
        for _ in range(num_hidden_layers - 1):
            self.layers.append(self._generate_hidden_layer(
                dim_hidden, dim_hidden))

        # 隠れ層 -> 出力層
        self.linear = nn.Linear(dim_hidden, num_classes)
        ''''''''''''''''''''

    '''
    隠れ層生成関数
    dim_input : 入力次元
    dim_output: 出力次元
    '''
    def _generate_hidden_layer(self, dim_input: int, dim_output: int):
        layer = nn.Sequential(
            nn.Linear(dim_input, dim_output, bias=False),
            nn.BatchNorm1d(dim_output),
            nn.ReLU(inplace=True)
        )

        return layer

    '''
    順伝播関数
    x           : 入力, [バッチサイズ, 入力次元]
    return_embed: 特徴量を返すかロジットを返すかを選択する真偽値
    '''
    def forward(self, x: torch.Tensor, return_embed: bool=False):
        h = x
        for layer in self.layers:
            h = layer(h)

        # return_embedが真の場合、特徴量を返す
        if return_embed:
            return h

        y = self.linear(h)

        # return_embedが偽の場合、ロジットを返す
        return y

    '''
    モデルパラメータが保持されているデバイスを返す関数
    '''
    def get_device(self):
        return self.linear.weight.device

    '''
    モデルを複製して返す関数
    '''
    def copy(self):
        return copy.deepcopy(self)


class Config:
    '''
    ハイパーパラメータとオプションの設定
    '''
    def __init__(self):
        self.val_ratio = 0.2       # 検証に使う学習セット内のデータの割合
        self.dim_hidden = 512      # 隠れ層の特徴量次元
        self.num_hidden_layers = 2 # 隠れ層の数
        self.num_epochs = 3       # 学習エポック数
        self.lr = 1e-2             # 学習率
        self.moving_avg = 20       # 移動平均で計算する損失と正確度の値の数
        self.batch_size = 32       # バッチサイズ
        self.num_workers = 2       # データローダに使うCPUプロセスの数
        self.device = 'cuda'        # 学習に使うデバイス
        self.num_samples = 200     # t-SNEでプロットするサンプル数


def train_eval():
    config = Config()

    # 入力データ正規化のために学習セットのデータを使って
    # 各次元の平均と標準偏差を計算
    dataset = torchvision.datasets.CIFAR10(
        root='data', train=True, download=True,
        transform=transform)
    channel_mean, channel_std = get_dataset_statistics(dataset)

    # 正規化を含めた画像整形関数の用意
    img_transform = lambda x: transform(
        x, channel_mean, channel_std)

    # 学習、評価セットの用意
    train_dataset = torchvision.datasets.CIFAR10(
        root='data', train=True, download=True,
        transform=img_transform)
    test_dataset = torchvision.datasets.CIFAR10(
        root='data', train=False, download=True,
        transform=img_transform)

    # 学習・検証セットへ分割するためのインデックス集合の生成
    val_set, train_set = generate_subset(
        train_dataset, config.val_ratio)

    print(f'学習セットのサンプル数　: {len(train_set)}')
    print(f'検証セットのサンプル数　: {len(val_set)}')
    print(f'テストセットのサンプル数: {len(test_dataset)}')

    # インデックス集合から無作為にインデックスをサンプルするサンプラー
    train_sampler = SubsetRandomSampler(train_set)

    # DataLoaderを生成
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        num_workers=config.num_workers, sampler=train_sampler)
    val_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        num_workers=config.num_workers, sampler=val_set)
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size,
        num_workers=config.num_workers)

    # 目的関数の生成
    loss_func = F.cross_entropy

    # 検証セットの結果による最良モデルの保存用変数
    val_loss_best = float('inf')
    model_best = None

    # FNNモデルの生成
    model = FNN(32 * 32 * 3, config.dim_hidden,
                config.num_hidden_layers,
                len(train_dataset.classes))

    # モデルを指定デバイスに転送(デフォルトはCPU)
    torch.cuda.reset_peak_memory_stats()
    pre = torch.cuda.memory_allocated()
    print_memory_torch("Initial")

    model.to(config.device)
    
    print(f"Define {(torch.cuda.memory_allocated() - pre)/1024**2:.6f}MiB")
    print_memory_torch("Define")

    # 最適化器の生成
    optimizer = optim.SGD(model.parameters(), lr=config.lr)

    for epoch in range(config.num_epochs):
        model.train()

        pre = torch.cuda.memory_allocated()

        with tqdm(train_loader) as pbar:
            pbar.set_description(f'[エポック {epoch + 1}]')

            # 移動平均計算用
            losses = deque()
            accs = deque()
            for x, y in pbar:
                # データをモデルと同じデバイスに転送
                x = x.to(model.get_device())
                y = y.to(model.get_device())

                # 既に計算された勾配をリセット
                optimizer.zero_grad()

                # 順伝播
                y_pred = model(x)

                # 学習データに対する損失と正確度を計算
                loss = loss_func(y_pred, y)
                accuracy = (y_pred.argmax(dim=1) == \
                            y).float().mean()

                # 誤差逆伝播
                loss.backward()

                # パラメータの更新
                optimizer.step()

                # 移動平均を計算して表示
                losses.append(loss.item())
                accs.append(accuracy.item())
                if len(losses) > config.moving_avg:
                    losses.popleft()
                    accs.popleft()
                pbar.set_postfix({
                    'loss': torch.Tensor(losses).mean().item(),
                    'accuracy': torch.Tensor(accs).mean().item()})
                
        print(f"Train {(torch.cuda.memory_allocated() - pre)/1024**2:.6f}MiB")
        print_memory_torch("Train")

        # 検証セットを使って精度評価
        torch.cuda.reset_peak_memory_stats()
        pre = torch.cuda.max_memory_allocated()

        val_loss, val_accuracy = evaluate(
            val_loader, model, loss_func)
        
        print(f'検証　: loss = {val_loss:.3f}, '
                f'accuracy = {val_accuracy:.3f}')
        print(f"Valid {(torch.cuda.max_memory_allocated() - pre)/1024**2:.6f}MiB")
        print_memory_torch("Valid")

        # より良い検証結果が得られた場合、モデルを記録
        if val_loss < val_loss_best:
            val_loss_best = val_loss
            model_best = model.copy()

    # テスト
    pre = torch.cuda.memory_allocated()

    test_loss, test_accuracy = evaluate(
        test_loader, model_best, loss_func)
    
    print(f'テスト: loss = {test_loss:.3f}, '
          f'accuracy = {test_accuracy:.3f}')
    print(f"Test {(torch.cuda.memory_allocated() - pre)/1024**2:.6f}MiB")
    print_memory_torch("Test")

    # モデルサマリー（VRAMを利用するかもなので、計測に影響しないように最後に実行する）
    summary(model, (config.batch_size, 32*32*3),
            col_names=("output_size", "num_params", "kernel_size"))

    '''
data_loader: 評価に使うデータを読み込むデータローダ
model      : 評価対象のモデル
loss_func  : 目的関数
'''
def evaluate(data_loader: Dataset, model: nn.Module,
             loss_func: Callable):
    model.eval()

    losses = []
    preds = []
    for x, y in data_loader:
        with torch.no_grad():
            x = x.to(model.get_device())
            y = y.to(model.get_device())

            y_pred = model(x)

            losses.append(loss_func(y_pred, y, reduction='none'))
            preds.append(y_pred.argmax(dim=1) == y)

    loss = torch.cat(losses).mean()
    accuracy = torch.cat(preds).float().mean()

    return loss, accuracy


'''
データセットを分割するための2つの排反なインデックス集合を生成する関数
dataset    : 分割対象のデータセット
ratio      : 1つ目のセットに含めるデータ量の割合
random_seed: 分割結果を不変にするためのシード
'''
def generate_subset(dataset: Dataset, ratio: float,
                    random_seed: int=0):
    # サブセットの大きさを計算
    size = int(len(dataset) * ratio)

    indices = list(range(len(dataset)))

    # 二つのセットに分ける前にシャッフル
    random.seed(random_seed)
    random.shuffle(indices)

    # セット1とセット2のサンプルのインデックスに分割
    indices1, indices2 = indices[:size], indices[size:]

    return indices1, indices2


'''
各次元のデータセット全体の平均と標準偏差を計算する関数
dataset: 平均と標準偏差を計算する対象のPyTorchのデータセット
'''
def get_dataset_statistics(dataset: Dataset):
    data = []
    for i in range(len(dataset)):
        # 3072次元のベクトルを取得
        img_flat = dataset[i][0]
        data.append(img_flat)
    # 第0軸を追加して第0軸でデータを連結
    data = np.stack(data)

    # データ全体の平均と標準偏差を計算
    channel_mean = np.mean(data, axis=0)
    channel_std = np.std(data, axis=0)

    return channel_mean, channel_std


'''
img         : 整形対象の画像
channel_mean: 各次元のデータセット全体の平均, [入力次元]
channel_std : 各次元のデータセット全体の標準偏差, [入力次元]
'''
def transform(img: Image.Image, channel_mean: np.ndarray=None,
              channel_std: np.ndarray=None):
    # PIL to numpy array, PyTorchでの処理用に単精度少数を使用
    img = np.asarray(img, dtype='float32')

    # [32, 32, 3]の画像を3072次元のベクトルに平坦化
    x = img.flatten()

    # 各次元をデータセット全体の平均と標準偏差で正規化
    if channel_mean is not None and channel_std is not None:
        x = (x - channel_mean) / channel_std

    return x


train_eval()


num_input = 32*32*3*32
num_param = 32*32*3*512 + 512*512 + 512*10+10 + 512*2*2
print("Trainable prams", num_param)
print("Input size(byte)", num_input * 4)
print("Prams size(byte)", num_param * 4)
print("Forward/backward(byte)", 2*(32*10+32*512*4+32*10)*4) # https://github.com/sksq96/pytorch-summary/issues/51