import yaml
import torch
from data import CellDataset
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from progressbar import ProgressBar, Widget, Percentage, Bar, Timer, ETA
from utils import calc_dice, calc_params, import_mod, Logger, create_path

### load hyper parameter
with open("./config/config.yaml")as f:
    args = yaml.load(f)
    log_file = "logs/" + args["model"] + "/log.txt"
    params_file = "params/" + args["model"] + "/params.ckpt"
    logger = Logger(log_file)
    create_path(params_file)
    for key, value in args.items():
        logger.write(str(key) + ": " + str(value))

### load train and eval dataset
train_dataset = CellDataset(train=True)
eval_dataset = CellDataset(train=False)
train_loader = DataLoader(dataset=train_dataset, batch_size=args['batch_size'], shuffle=False)
eval_loader = DataLoader(dataset=eval_dataset, batch_size=args['batch_size'], shuffle=False)

### load device --- cpu or gpu
device = torch.device('cuda:0') if args["device"]=="gpu" else torch.device('cpu')

### load model
model = import_mod("model." + args["model"])
model = model().to(device)
calc_params(model, logger)

### load criterion and assign weight
weight = torch.tensor([0.01, 0.20]).to(device)
criterion = CrossEntropyLoss(weight=weight)
weight_p, bias_p = [],[]
for name, p in model.named_parameters():
    if 'bias' in name:
         bias_p += [p]
    else:
        weight_p += [p]

### load optimizer used weight decay in order to avoid overfitting
optimizer = Adam([{'params': weight_p, 'weight_decay': args["weight_decay"]},
                  {'params': bias_p, 'weight_decay':0}], lr=args["learning_rate"])

### train model in one epoch
def train():
    model.train()
    widgets = ['Epoch [' + str(epoch+1) + '/' + str(args["num_epochs"]) + '], Training: ', Percentage(), ' ', Bar('*'),' ', Timer(),  ' ', ETA()]  
    pbar = ProgressBar(widgets=widgets, maxval=len(train_loader)).start()      
    total_loss, total_dice = 0, 0
    for i, (inputs, targets) in enumerate(train_loader):
        pbar.update(i)
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_dice += calc_dice(outputs, targets)
    pbar.finish()
    torch.save(model.state_dict(), params_file)
    total_loss /= len(train_loader)
    total_dice /= len(train_dataset)
    return total_loss, total_dice

### eval model in one epoch
def eval():
    model.eval()
    total_loss, total_dice = 0, 0
    for i, (inputs, targets) in enumerate(eval_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()
        total_dice += calc_dice(outputs, targets)
    total_loss /= len(eval_loader)
    total_dice /= len(eval_dataset)
    return total_loss, total_dice

### train model
for epoch in range(args["num_epochs"]):
    train_loss, train_dice = train()
    eval_loss, eval_dice = eval()
    logger.write("Epoch[{}/{}], train_loss:{:.4f}, train_dice:{:.4f}, eval_loss:{:.4f}, eval_dice:{:.4f}"
        .format(epoch, args["num_epochs"], train_loss, train_dice, eval_loss, eval_dice))


