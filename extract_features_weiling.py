import torch, torchvision
import ffmpeg, time, os
import numpy as np
from PIL import Image
from torch.autograd import Variable
from pathlib import Path
from resnet import i3_res50


def run(i3d, frequency, temppath, batch_size, sample_mode, last_segment):
    tmp_path = os.path.join(temppath, "..")
    dataloader = create_dataloader(tmp_path, frequency, 0)
    frames = []
    full_features = [[] for i in range(10)]
    for i, (inputs, _) in enumerate(dataloader):  # i is the index of the frame
        if i == len(dataloader) - 1:
            if inputs.shape[0] < frequency:
                inputs = torch.cat([inputs, inputs[-1].repeat(frequency - inputs.shape[0], 1, 1, 1, 1)], dim=0)
        frames.append(inputs)
        if (i + 1) % batch_size == 0 or i == len(dataloader) - 1:  # 每320个feed一次model or 数据全部取出
            print(i + 1)
            frames_batch = torch.cat(frames, dim=0)
            frames_batch = frames_batch.view(-1, frequency, 10, 3, 224, 224)  # [20, 16, 10, 3, 224, 224]
            frames_batch = frames_batch.permute(2, 0, 3, 1, 4, 5)  # [10, 20, 16, 3, 224, 224]
            for j in range(10):
                with torch.no_grad():
                    b_data = frames_batch[j].cuda()
                    inp = {'frames': b_data}  # bsx3x16x224x224
                    # print(i, j)
                    outputs = i3d(inp)
                full_features[j].append(outputs.data.cpu().numpy())
            frames = []

    # full_features: {list: 10, batch_cnt}, each shape = [20,2048, 1,1 1], 20 is batch size, batch_cnt is no. of batches
    full_features = [np.concatenate(i, axis=0) for i in
                     full_features]  # {list: 10}, each shape = [batch_cnt * bs, 2048, 1,1,1]
    full_features = [np.expand_dims(i, axis=0) for i in
                     full_features]  # {list: 10}, each shape = [1,batch_cnt * bs,2048, 1,1,1]
    full_features = np.concatenate(full_features, axis=0)  # [10,batch_cnt * bs,2048, 1,1,1]
    full_features = full_features[:, :, :, 0, 0, 0]  # [10,batch_cnt * bs,2048]
    full_features = np.array(full_features).transpose([1, 0, 2])  # [batch_cnt * bs,10,2048]
    print(full_features.shape)
    return full_features


def create_dataloader(input_dir, batch_size, num_workers):
    dataset = torchvision.datasets.ImageFolder(input_dir, transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.TenCrop(224),
        torchvision.transforms.Lambda(
            lambda crops: torch.stack([torchvision.transforms.ToTensor()(crop) for crop in crops])),
        torchvision.transforms.Normalize(mean=[114.75, 114.75, 114.75], std=[57.375, 57.375, 57.375]),
    ]))
    # create a dataloader filling the last batch with last frames
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloader
