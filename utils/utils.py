
# def calc_dice(outputs, targets):
#     import torch
#     dice = 0
#     _, outputs = torch.max(outputs, 1)
#     batch_size = outputs.size(0)
#     for i in range(batch_size):
#         preds, labels = outputs[i], targets[i]
#         A = (preds != 0).sum().item()
#         B = (labels != 0).sum().item()
#         C = ((preds == 1) & (labels == 1)).sum().item() + ((preds == 2) & (labels == 2)).sum().item()
#         if (A + B == 0):
#             dice += 1
#         else:
#             dice += 2 * C / (A + B)
#     # print((labels == 0).sum().item(), (labels == 1).sum().item())
#     # print((preds == 0).sum().item(), (preds == 1).sum().item())
#     return dice
def calc_mIoU(outputs, targets):
    import torch
    IoU = 0
    _, outputs = torch.max(outputs, 1)
    batch_size = outputs.size(0)
    for i in range(batch_size):
        preds, labels = outputs[i], targets[i]
        A = ((preds == 1) & (labels == 1)).sum().item()
        B = ((preds == 1) & (labels == 0)).sum().item()
        C = ((preds == 0) & (labels == 1)).sum().item()
        # print(A, B, C)
        IoU += A / (A + B + C)
    return IoU

def calc_params(model, logger=None):
    sum = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            sum = sum + param.reshape(-1).size(0)
    func = logger.write if logger else print
    func("Total Model Parameters: {:.2f}MB".format(sum * 4 / 1024 / 1024))

def import_mod(name):
    from importlib import import_module
    components = name.split('.')
    mod = import_module(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def create_path(file):
    import os
    path, file = os.path.split(file)
    if not os.path.exists(path):
        os.makedirs(path)

class Logger():
    def __init__(self, log_file):
        create_path(log_file)
        self.log_file = log_file

    def write(self, str):
        import time
        current_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        str = '['+current_time+'] '+str
        print(str)   
        with open(self.log_file, 'a') as f:
            print(str, file=f)        
    
