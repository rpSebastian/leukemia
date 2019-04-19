import yaml
import torch
from data import CellDataset, showSample
from torch.utils.data import DataLoader
from utils import calc_mIoU, Logger, create_path, import_mod

### load hyper parameter
with open("./config/config.yaml")as f:
    args = yaml.load(f)
    args['batch_size'] = 1
    log_file = "logs/" + args["model"] + "/log.txt"
    params_file = "params/" + args["model"] + "/params.ckpt"
    logger = Logger(log_file)

    
### load eval dataset
eval_dataset = CellDataset(train=False, size=args["image_size"])
eval_loader = DataLoader(dataset=eval_dataset, batch_size=args['batch_size'], shuffle=False)

### load device --- cpu or gpu
device = torch.device('cuda:0') if args["device"]=="gpu" else torch.device('cpu')

### load model
model = import_mod("model." + args["model"])
model = model().to(device)
model.load_state_dict(torch.load(params_file))
model.eval()

for i, (inputs, targets) in enumerate(eval_loader):
    inputs = inputs.to(device)
    targets = targets.to(device)
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    mIoU = calc_mIoU(outputs, targets)
    logger.write("{}/{}, eval_mIoU: {:.4f}".format(i, len(eval_loader), mIoU))
    showSample(inputs, targets, preds)
