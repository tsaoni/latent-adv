from lib import *

""" add perturbation on adapter input """
class PerturbLayer(nn.Module):
    def __init__(self, config: Namespace, layer):
        super().__init__()
        config = Namespace(
            
        ) 
        self.layer = layer

    def forward(self, x):
        noise = torch.randn(*x.shape).to(x.device)
        out = self.layer(x + noise)
        return out
