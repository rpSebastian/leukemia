def test_unet():
    from utils import calc_params
    from model import unet
    import torch
    model = unet()
    calc_params(model)
    x = torch.randn(1, 3, 192, 128)
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

def test_data():
    for i in range(81):
        image_path = "./data/image/" + str(i) + ".jpg"
        label_path = "./data/label/" + str(i) + ".jpg"
        image_save_path = "./data/image0/" + str(i) + ".jpg"
        label_save_path = "./data/label0/" + str(i) + ".jpg"
        from PIL import Image
        image = Image.open(image_path)
        label = Image.open(label_path)
        # from torchvision.transforms import ToTensor
        # image = ToTensor()(image)
        # label = ToTensor()(label)
        # print(image.shape, label.shape)
        from torchvision.transforms import Scale
        image = Scale((450, 300))(image)
        label = Scale((450, 300))(label)
        image.save(image_save_path)
        label.save(label_save_path)

if __name__ == "__main__":
    test_unet()
