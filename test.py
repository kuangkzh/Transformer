import Transformer
import Utils
from matplotlib import pyplot as plt
import torch
import numpy as np


transformer = Transformer.Transformer(10000, 1000, 10000, 1000)
print([list(p.size()) for p in transformer.parameters()])
para = sum([np.prod(list(p.size())) for p in transformer.parameters()])
print(para)
transformer = torch.nn.Transformer()
print([list(p.size()) for p in transformer.parameters()])
para = sum([np.prod(list(p.size())) for p in transformer.parameters()])
print(para)
