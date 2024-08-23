import torch as t
import torch.nn as nn
from PIL import Image
import numpy as np
import pandas 
from torchvision.transforms import transforms
import tqdm as tq
from tqdm import trange
#################################################################################
if t.cuda.is_available():
    t.set_default_tensor_type(t.cuda.FloatTensor)
    print("使用",t.cuda.get_device_name(0))
    pass
device=t.device("cuda" if t.cuda.is_available() else "cpu" )
#################################################################################
def dr(self):
    image = Image.open('sjji/anime-faces/{0}.png'.format(self))
    image_array = np.array(image)
    image_1v=(image_array.reshape(-1))
    tensor = t.from_numpy(image_1v).float() 
    tensor=tensor.to('cuda')
    return  tensor
def seed1(size):
    random_data = t.rand(100)
    random_data=random_data.to('cuda')
    return random_data
def seed2(size):
    random_data = t.randn(100)
    random_data=random_data.to('cuda')
    return random_data
def san_one(s):                                                                  #1转3
    q = s.reshape(64,64,3)
    return q
def png(a):
    im=Image.fromarray(a)
    im.save("out.png")
    print("导出成功")
def zh (e):
    print(type(e))
    e=t.tensor(e)
    print(type(e))
    return e
class View(nn.Module):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)
###################################################################################














#————————————————————————————神经网络部分————————————————————————————#
class D(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model= nn.Sequential(
            nn.Linear(12288,1000),
            nn.LeakyReLU(0.05),
            nn.LayerNorm(1000),

            nn.Linear(1000,100),
            nn.LeakyReLU(0.05),
    
            nn.LayerNorm(100),

            nn.Linear(100,1),
            nn.Sigmoid(),

        )
        self.loss_function=nn.MSELoss()
        self.optimiser=t.optim.Adam(self.parameters(),)
        self.counter = 0;
        self.progress = []
        pass
    def forward(self, inputs):
        return self.model(inputs)
    
    
    def train(self, inputs, targets):
        outputs = self.forward(inputs)
        loss = self.loss_function(outputs, targets)
        self.counter += 1;
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
            pass
        if (self.counter % 10000 == 0):
            print("counter = ", self.counter)
            pass
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        pass
    
    
    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0, 1.0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))
        pass
    
    pass

#————————————————————————————生成器部分————————————————————————————#
class G(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model= nn.Sequential(
            nn.Linear(100,1000),
            nn.LeakyReLU(0.05),
            nn.LayerNorm(1000),

            nn.Linear(1000,10000),
            nn.LeakyReLU(0.05),
            nn.LayerNorm(10000),

            nn.Linear(10000,12288),
            nn.Sigmoid(),

        )
        self.optimiser=t.optim.SGD(self.parameters(),lr=0.02)
        self.counter = 0;
        self.progress = []
        pass
    def forward(self, inputs):
        return self.model(inputs)
    
    
    def train(self, D , inputs, targets):

        outputs = self.forward(inputs)
        d_output=D.forward(outputs)
        loss = D.loss_function(d_output,targets)
        if loss.dim() > 0:  # 如果loss不是一个标量
            loss = loss.mean()  # 将其转换为标量
        self.counter += 1;
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
            pass
        if (self.counter % 10000 == 0):
            print("counter = ", self.counter)
            pass
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        pass
    
    
    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0, 1.0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))
        pass
    
    pass


#————————————————————————————训练部分————————————————————————————#
G=G()
D=D()
G.to(device)
D.to(device)
a=0
b=0           #循环训练次数（不使用全部数据）
c=0
for q in range(2):
    a=0
    for i in trange(8000):
        a=a+1
        b=b+1
        D.train(dr(a),t.cuda.FloatTensor([1.0]))
        D.train(G.forward(seed1(100)),t.cuda.FloatTensor([0.0]))
        G.train(D,seed2(100),t.cuda.FloatTensor([1.0]))
    c=c+1
    print("周期",c)
seed=seed1(100)
out=G.forward(seed)
out_3=san_one(out)

out_3=out_3*255
out_3=t.tensor(out_3,dtype=t.int64)
out_3=t.tensor(out_3,dtype=t.uint8)
# 定义转换
out_4=t.Tensor.cpu(out_3)
out_5=np.transpose(out_4, (2, 0, 1))
to_pil = transforms.ToPILImage()
height, width, channels = out_5.shape
print("通道:", height)
print("长度:", width)
print("宽度:", channels)
# 将 tensor 转换为 PIL 图像
pil_image = to_pil(out_5)
t.save(G.state_dict(), 'mx.pth')
# 显示图像
pil_image.show()
pil_image.save('out/1.png')