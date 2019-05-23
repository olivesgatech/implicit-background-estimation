import torch
import IBE
from IBE import ExpectedNonDistinctiveness as END
x = torch.ones(10,20,32,32)
x[:,0,:,:] += 10
x[:,1:,:,:] -= 10
x = torch.cat([10*torch.ones(10,1,32,32),x],1)
end1 = END(mode=1)
y1 = end1(x)
end2 = END(mode=2)
y2 = end2(x)
print(y1,y2)
print(y1.dtype==torch.float, y2.dtype==torch.float)

from IBE import ImplicitBackground

ibe = ImplicitBackground(dim=1)
y = ibe(x)
print(y.size(1)-1 == x.size(1))

