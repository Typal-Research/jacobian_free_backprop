import torch
import torch.nn as nn


class MNIST_FCN(nn.Module):
    def __init__(self, dim_d, dim_out, dim_hid, hidden_layers, device):
        super().__init__()
        self.layers = hidden_layers
        self.device = device
        self.dim_d = dim_d
        self.dim_out = dim_out
        self.dim_hid = dim_hid
        self.fc_mid = nn.ModuleList([torch.nn.Linear(dim_hid, dim_hid,
                                    bias=False) for i in range(self.layers)])
        self.fc_d = torch.nn.Linear(dim_d, dim_hid, bias=True)        
        self.fc_y = torch.nn.Linear(dim_hid, dim_out, bias=True)
        self.fc_u = torch.nn.Linear(dim_out, dim_out, bias=False)
        self.relu = nn.ReLU()
        # Initialize fully connected (fc) layers to have unit singular values
        self.fc_d.weight.data = self.proj(self.fc_d.weight.data, low_s_val=1.0)
        self.fc_y.weight.data = self.proj(self.fc_y.weight.data, low_s_val=1.0)        
        self.fc_u.weight.data = self.proj(self.fc_u.weight.data, low_s_val=1.0)
        for i in range(self.layers):
            self.fc_mid[i].weight.data = self.proj(self.fc_mid[i].weight.data,
                                                   low_s_val=1.0)

    def forward(self, u, d=None):
        y = self.relu(self.fc_d(d.float()))
        for i in range(self.layers):
            y = self.relu(self.fc_mid[i](y))
        return torch.clamp(0.1 * self.fc_u(u.float()) + 0.9 * self.fc_y(y),
                           min=0, max=1.0)

    def project_weights(self):
        self.fc_d.weight.data = self.proj(self.fc_d.weight.data)
        self.fc_u.weight.data = self.proj(self.fc_u.weight.data)
        for i in range(self.layers):
            self.fc_mid[i].weight.data = self.proj(self.fc_mid[i].weight.data)
        self.fc_y.weight.data = self.proj(self.fc_y.weight.data)

    def proj(self, Ak, low_s_val=0.1):
        u, s, v = torch.svd(Ak)
        s[s > 1.0] = 1.0
        s[s < low_s_val] = low_s_val
        return torch.mm(torch.mm(u, torch.diag(s)), v.t())
