import torch
import torch.nn as nn


class MNIST_FCN(nn.Module):
    def __init__(self, dim_out, dim_hid, device):
        super().__init__()
        self.fc_d = nn.Linear(784,     dim_hid, bias=True)
        self.fc_m1 = nn.Linear(dim_hid, dim_hid, bias=True)
        self.fc_m2 = nn.Linear(dim_hid, dim_hid, bias=True)
        self.fc_y = nn.Linear(dim_hid, dim_out, bias=False)
        self.fc_u = nn.Linear(dim_out, dim_out, bias=False)
        self.relu = nn.ReLU()

    def name(self):
        return 'MNIST_FCN'

    def forward(self, u, d=None):
        y = self.relu(self.fc_d(d.float()))
        y = self.relu(self.fc_m1(y))
        y = 0.5 * y + 0.5 * self.relu(self.fc_m2(y))
        return 0.1 * self.fc_u(u.float()) + 0.9 * self.fc_y(y)

    def project_weights(self, s_lo=0.9):
        """ All linear maps must yield 1-Lipschitz operators,
            which is accomplished by bounding all singular values
            by unity.
        """
        self.fc_d.weight.data = self.proj_sing_val(self.fc_d.weight.data, s_lo)
        self.fc_m1.weight.data = self.proj_sing_val(self.fc_m1.weight.data, s_lo)
        self.fc_m2.weight.data = self.proj_sing_val(self.fc_m2.weight.data, s_lo)
        self.fc_u.weight.data = self.proj_sing_val(self.fc_u.weight.data, s_lo)
        self.fc_y.weight.data = self.proj_sing_val(self.fc_y.weight.data, s_lo)

    def proj_sing_val(self, Ak, s_lo=0.0):
        """ Project singular values of matrices onto interval [s_low, 1.0].
        """
        u, s, v = torch.svd(Ak)
        s[s > 1.0] = 1.0
        s[s < s_lo] = s_lo
        return torch.mm(torch.mm(u, torch.diag(s)), v.t())


class MNIST_CNN(nn.Module):
    def __init__(self, sig_dim):
        super(MNIST_CNN, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.fc_u = nn.Linear(sig_dim, sig_dim, bias=False)
        self.fc_f = nn.Linear(sig_dim, sig_dim, bias=False)
        self.fc_y = nn.Linear(1500,    sig_dim, bias=True)

        #self.drop_out = nn.Dropout(p=0.001)

        self.conv1 = torch.nn.utils.spectral_norm(nn.Conv2d(in_channels=1,
                                                            out_channels=90,
                                                            kernel_size=3,
                                                            stride=1),
                                                            n_power_iterations=20)
        self.conv2 = torch.nn.utils.spectral_norm(nn.Conv2d(in_channels=90,
                                                            out_channels=60,
                                                            kernel_size=3,
                                                            stride=1),
                                                            n_power_iterations=20)

    def name(self):
        return 'MNIST_CNN'

    def forward(self, u, d):
        y = self.maxpool(self.relu(self.conv1(d)))
        #y = self.maxpool(self.drop_out(self.relu(self.conv2(y))))
        y = self.maxpool(self.relu(self.conv2(y)))
        y = y.view(d.shape[0], -1)
        y = self.relu(self.fc_y(y))
        return 0.1 * self.fc_u(u) + 0.9 * self.fc_f(y)

    def project_weights(self, s_lo=0.5):
        """ All linear maps must yield 1-Lipschitz operators,
            which is accomplished by bounding all singular values
            by unity.
        """
        self.fc_u.weight.data = self.proj_sing_val(self.fc_u.weight.data, s_lo)
        self.fc_y.weight.data = self.proj_sing_val(self.fc_y.weight.data, s_lo)
        self.fc_f.weight.data = self.proj_sing_val(self.fc_f.weight.data, s_lo)

    def proj_sing_val(self, Ak, s_lo=0.0):
        """ Project singular values of matrices onto interval [s_low, 1.0].
        """
        u, s, v = torch.svd(Ak)
        s[s > 1.0] = 1.0
        s[s < s_lo] = s_lo
        return torch.mm(torch.mm(u, torch.diag(s)), v.t())
