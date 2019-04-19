def test_unet():
    from utils import calc_params
    from model import unet
    import torch
    model = unet()
    calc_params(model)
    x = torch.randn(1, 3, 128, 128)
    print(model(x).shape)

def test_unetpp():
    from utils import calc_params
    from model import unetpp
    import torch
    model = unetpp()
    calc_params(model)
    x = torch.randn(1, 3, 128, 128)
    print(model(x).shape)

def test_mIoU():
    import torch
    from utils import calc_mIoU
    preds = torch.tensor([1, 1, 0, 0, 0, 1, 1])
    targets = torch.tensor([0, 1, 1, 1, 0, 0, 0])
    A = ((preds == 1) & (targets == 1)).sum().item()
    B = ((preds == 1) & (targets == 0)).sum().item()
    C = ((preds == 0) & (targets == 1)).sum().item()
    print(A, B, C)
    
if __name__ == "__main__":
    test_mIoU()