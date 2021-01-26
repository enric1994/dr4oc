import argparse
import torch
import os
import numpy as np
# import datasets.crowd as crowd
from src.resnet import resnet50

parser = argparse.ArgumentParser(description='Test ')
parser.add_argument('--device', default='0', help='assign device')
parser.add_argument('--crop-size', type=int, default=512,
                    help='the crop size of the train image')
parser.add_argument('--model-path', type=str, default='pretrained_models/model_qnrf.pth',
                    help='saved model path')
parser.add_argument('--data-path', type=str,
                    default='data/QNRF-Train-Val-Test',
                    help='saved model path')
parser.add_argument('--dataset', type=str, default='qnrf',
                    help='dataset name: qnrf, nwpu, sha, shb')
parser.add_argument('--constant', type=float, default=False,
                    help='Dummy classifier')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.device  # set vis gpu
device = torch.device('cuda')

model_path = args.model_path
crop_size = args.crop_size
data_path = args.data_path
# if args.dataset.lower() == 'qnrf':
#     dataset = crowd.Crowd_qnrf(os.path.join(data_path, 'test'), crop_size, 8, method='val')
# elif args.dataset.lower() == 'nwpu':
#     dataset = crowd.Crowd_nwpu(os.path.join(data_path, 'val'), crop_size, 8, method='val')
# elif args.dataset.lower() == 'sha' or args.dataset.lower() == 'shb':
#     dataset = crowd.Crowd_sh(os.path.join(data_path, 'test'), crop_size, 8, method='val')
if args.dataset.lower() == 'apples':
    from src.datasets.crowd import Crowd_apples
    dataset = Crowd_apples(data_path, crop_size, 8, method='test')
elif args.dataset.lower() == 'vehicles':
    from src.datasets.crowd import Crowd_trancos
    dataset = Crowd_trancos(data_path, crop_size, 10000,8, method='test')
elif args.dataset.lower() == 'people':
    from src.datasets.crowd import Crowd_sh
    dataset = Crowd_sh(data_path, crop_size, 8, method='test')
elif args.dataset.lower() == 'penguins':
    from src.datasets.crowd import Crowd_penguins
    dataset = Crowd_penguins(data_path, crop_size, 8, method='test')
else:
    raise NotImplementedError
dataloader = torch.utils.data.DataLoader(dataset, 1, shuffle=False,
                                         num_workers=1, pin_memory=True)

model = resnet50()
model.to(device)
model.eval()
model.load_state_dict(torch.load(model_path, device))
image_errs = []
for inputs, count, name in dataloader:
    inputs = inputs.to(device)
    assert inputs.size(0) == 1, 'the batch size should equal to 1'
    with torch.set_grad_enabled(False):
        outputs, _ = model(inputs)
    if args.constant == False:
        img_err = count[0].item() - torch.sum(outputs).item()
        print(name, img_err, count[0].item(), torch.sum(outputs).item())
    else:
        img_err = count[0].item() - args.constant
        print(name, img_err, count[0].item(), args.constant)

    
    image_errs.append(img_err)

image_errs = np.array(image_errs)
mse = np.sqrt(np.mean(np.square(image_errs)))
mae = np.mean(np.abs(image_errs))
print('{}: mae {}, mse {}\n'.format(model_path, mae, mse))
