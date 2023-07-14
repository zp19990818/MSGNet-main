import torch
from model.MSGNet import MSGNet
import time

inputRes = (384, 384)

torch.cuda.set_device(device=0)

MSGNet = MSGNet().cuda()
MSGNet.train(False)
input = torch.randn(1, 3, 384, 384).cuda()
flow = torch.randn(1, 3, 384, 384).cuda()
start_time = time.time()
for i in range(1000):
    with torch.no_grad():
        a, _, _, _, _, _ = MSGNet.forward(input, flow)
end_time = time.time()
print(end_time-start_time)
print('MSGNet-time:', (end_time-start_time)/1000)


