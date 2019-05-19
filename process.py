def model():
    from PIL import Image
    import torch
    import yaml
    import numpy as np
    from utils import import_mod
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
    from data import showLabel
    from utils import count

    with open("./config/config.yaml")as f:
        args = yaml.load(f)
        params_file = "params/" + args["model"] + "/params.ckpt"
    
    img = Image.open("data/image/12.jpg")
    transform = transforms.Compose([
                transforms.Resize((128, 192)),
                transforms.ToTensor()
    ])
    inputs = transform(img)
    inputs = inputs.unsqueeze(dim=0)
    model = import_mod("model." + args["model"])()
    model.load_state_dict(torch.load(params_file),strict=False)
    outputs = model(inputs)
    _, outputs = torch.max(outputs, 1)
    outputs = outputs.squeeze()
    outputs = showLabel(outputs, show=True)
    plt.imshow(outputs)
    plt.show()
    num = count(outputs)
    return outputs, num

model()
