import torch
import torch.nn as nn

class ExpectedNonDistinctiveness(nn.Module):
    '''
    Assumes the input is NCHW and that the background class
    is at index 0
    '''
    def __init__(self, mode=1):
        super(ExpectedNonDistinctiveness, self).__init__()
        self.running_membership = 0
        self.num_pixels = 0
        self.mode = mode

    def reset(self):
        self.running_membership = 0
        self.num_pixels = 0

    def forward(self, input):
        N, num_classes, H, W = input.shape
        num_non_bg_classes = num_classes-1
        self.num_pixels += N*H*W
        # SoftMax here minimum is 1/num_non_bg_classes and must be normalized
        id_conf = (torch.softmax(input[:,1:,:,:],1).max(1, keepdim=True)[0]-1/num_non_bg_classes)/(1-1/num_non_bg_classes)
        if self.mode == 1:
            bg_conf = torch.softmax(input,1)[:,0,:,:].unsqueeze(1)
        elif self.mode == 2:
            id_act = input[:,1:,:,:].max(1, keepdim=True)[0]
            bg_act = input[:,0,:,:].unsqueeze(1)
            id_bg_conf = torch.softmax(torch.cat([id_act,bg_act], 1),1)
            bg_conf = id_bg_conf[:,1,:,:].unsqueeze(1)
        ineq_plot = bg_conf*id_conf
        self.running_membership += ineq_plot.sum()
        mem = self.membership()
        return mem

    def membership(self):
        return self.running_membership/self.num_pixels
