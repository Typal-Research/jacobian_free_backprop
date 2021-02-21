import torch
import torch.nn as nn
from FPN import FPN


class CIFAR10_CNN(FPN):
    def __init__(self, lat_dim, device, s_hi=1.0, inf_dim=10):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self._lat_dim = lat_dim
        self._inf_dim = inf_dim
        self._s_hi = s_hi
        self._device = device
        self.dropout = nn.Dropout(p=0.5)

        self._dim_out1 = 80
        self._dim_out2 = 90
        self._dim_out3 = 100

        self.fc_u = nn.Linear(lat_dim, lat_dim, bias=False)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self._dim_out1,
                               kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=self._dim_out1,
                               out_channels=self._dim_out2, kernel_size=3,
                               stride=1)
        self.conv3 = nn.Conv2d(in_channels=self._dim_out2,
                               out_channels=self._dim_out3, kernel_size=3,
                               stride=1)
        self.fc_v = nn.Linear(in_features=self._dim_out3 * 4,
                              out_features=lat_dim, bias=True)

    def name(self):
        return 'CIFAR10_CNN'

    def device(self):
        return self._device

    def lat_dim(self):
        return self._lat_dim

    def s_hi(self):
        return self._s_hi

    def latent_space_forward(self, u, v):
        return self.fc_u(0.99 * self.relu(u) + v)

    def data_space_forward(self, d):
        current_batch_size = d.shape[0]
        v = self.maxpool(self.relu(self.conv1(d)))
        v = self.maxpool(self.relu(self.dropout(self.conv2(v))))
        v = self.maxpool(self.relu(self.conv3(v)))
        v = v.view(current_batch_size, -1)
        return self.relu(self.fc_v(v))

    def map_latent_to_inference(self, u):
        return u[:, 0:10]


class MNIST_FCN(FPN):
    def __init__(self, lat_dim, device, s_hi=1.0, inf_dim=10):
        super().__init__()
        self.fc_d = nn.Linear(784,          95, bias=True)
        self.fc_v = nn.Linear(95,      lat_dim, bias=True)
        self.fc_u = nn.Linear(lat_dim, lat_dim, bias=False)
        self.relu = nn.ReLU()
        self._lat_dim = lat_dim
        self._inf_dim = inf_dim
        self._s_hi = s_hi
        self._device = device
        self.dropout = nn.Dropout(p=0.02)

    def name(self):
        return 'MNIST_FCN'

    def device(self):
        return self._device

    def lat_dim(self):
        return self._lat_dim

    def inf_dim(self):
        return self._inf_dim

    def s_hi(self):
        return self._s_hi

    def data_space_forward(self, d):
        v = self.relu(self.dropout(self.fc_d(d.float())))
        v = self.relu(self.fc_v(v))
        return v

    def latent_space_forward(self, u, v):
        return 0.8 * self.fc_u(self.relu(u) + v)

    def map_latent_to_inference(self, u):
        return u[:, 0:10]


class MNIST_CNN(FPN):
    def __init__(self, lat_dim, device, s_hi=1.0):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.fc_y = nn.Linear(1250,    lat_dim, bias=False)
        self.fc_u = nn.Linear(lat_dim, lat_dim, bias=False)
        self.fc_y = nn.Linear(lat_dim, lat_dim, bias=False)
        self._lat_dim = lat_dim
        self._device = device
        self._s_hi = s_hi
        self.drop_out = nn.Dropout(p=0.4)
        self.conv1 = torch.nn.utils.spectral_norm(nn.Conv2d(in_channels=1,
                                                            out_channels=85,
                                                            kernel_size=3,
                                                            stride=1),
                                                  n_power_iterations=10)
        self.conv2 = torch.nn.utils.spectral_norm(nn.Conv2d(in_channels=85,
                                                            out_channels=50,
                                                            kernel_size=3,
                                                            stride=1),
                                                  n_power_iterations=10)

    def name(self):
        return 'MNIST_CNN'

    def device(self):
        return self._device

    def lat_dim(self):
        return self._lat_dim

    def s_hi(self):
        return self._s_hi

    def latent_space_forward(self, u, v):
        u = 0.9 * self.fc_u(self.relu(u) + v)
        return u

    def data_space_forward(self, d):
        v = self.maxpool(self.relu(self.conv1(d)))
        v = self.maxpool(self.relu(self.drop_out(self.conv2(v))))
        v = v.view(d.shape[0], -1)
        v = self.relu(self.fc_y(v))
        return v

    def map_latent_to_inference(self, u):
        return self.relu(self.fc_f(u))
