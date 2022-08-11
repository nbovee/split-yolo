import warnings
from functools import partial
from typing import Callable, Any, Optional, List

import torch
from torch import Tensor
from torch import nn

from torchvision.models import mobilenetv2
from torchinfo import summary
torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

class SplitModel(mobilenetv2.MobileNetV2):
    def __init__(
        self,
        pretrained = False,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.2,
    ):
        super(SplitModel, self).__init__()
        temp = nn.ModuleList()
        for i, mod in enumerate(self.children()):
            for q in mod:
                temp.append(q)
        self.model = temp
        del self.features # these deletes should not impact inherited methods from what I observed
        del self.classifier
        if pretrained:
            self.load_weights_of_model(pretrained)

    def load_weights_of_model(self, input_model):
        self.model.load_state_dict(input_model.features.state_dict(), False)
        self.model.load_state_dict(input_model.classifier.state_dict(), False)

    def _forward_impl(self, x: Tensor, start, end) -> Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        # x = self.features(x) # 19 layers
        # Cannot use "squeeze" as batch-size can be 1
        for i, layer in enumerate(self.model):
            if i > end:
                break
            if i >= start:
                if i == 19:
                    x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
                    x = torch.flatten(x, 1)
                x = layer(x)
        return x

    def forward(self, x: Tensor, start = 0, end = 21) -> Tensor:
        return self._forward_impl(x, start, end)

pretrained = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
model = SplitModel(pretrained)
model.eval()
from PIL import Image
from torchvision import transforms
from time import time as time_synchronized
filename = '../yolov7/inference/images/horses.jpg'
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

import pickle
t = time_synchronized()
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)
t1 = time_synchronized()
print(t1 - t)
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')
for i in range(10):
    model(input_batch)
print(len(pickle.dumps(input_batch)))
print(len(pickle.dumps(input_image)))
for i in range(0, 21):
    with torch.no_grad():
        t1, t2, t22, t3, output = 0, 0, 0, 0, 0
        loops = 50
        for f in range(loops):
            t1 += time_synchronized()
            x = model(input_batch, start = 0, end = i)
            t2 += time_synchronized()
            out = model(x, start = i + 1, end = 21)
            t22 += time_synchronized()
            probabilities = torch.nn.functional.softmax(out[0], dim=0)
            t3 += time_synchronized()
            output += len(pickle.dumps(out))
        print(f"{i}\tt1:{(t2-t1)/loops:0.4f}\tt2:{(t22-t2)/loops:0.4f}\t{(t3-t22)/loops}")