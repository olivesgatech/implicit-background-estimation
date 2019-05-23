import torch
import torch.nn as nn
import torch.nn.functional as F

class ImplicitBackground(nn.Module):
    '''
    This will take a tensor and extend the indicated dimension with the negative log sum exponential.
    '''
    def __init__(self, dim=1):
        super(ImplicitBackground, self).__init__()
        self.dim = dim

    def forward(self, input):
        non_bg_x = torch.logsumexp(input, dim=self.dim, keepdim=True)
        x = torch.cat([-non_bg_x, input], self.dim)
        return x

