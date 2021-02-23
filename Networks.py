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
    def __init__(self, lat_dim, s_hi=1.0, inf_dim=10):
        super().__init__()
        self.fc_d = nn.Linear(784,          95, bias=True)
        self.fc_v = nn.Linear(95,      lat_dim, bias=True)
        self.fc_u = nn.Linear(lat_dim, lat_dim, bias=False)
        self.fc_f = nn.Linear(lat_dim,      10, bias=False)
        self.relu = nn.ReLU()
        self._lat_dim = lat_dim
        self._inf_dim = inf_dim
        self._s_hi = s_hi

    def name(self):
        return 'MNIST_FCN'

    def lat_dim(self):
        return self._lat_dim

    def inf_dim(self):
        return self._inf_dim

    def s_hi(self):
        return self._s_hi

    def data_space_forward(self, d):
        v = self.relu(self.fc_d(d.float()))
        return self.relu(self.fc_v(v))

    def latent_space_forward(self, u, v):
        return 0.9 * self.relu(self.fc_u(self.relu(u) + v))

    def map_latent_to_inference(self, u):
        return self.fc_f(u)


class MNIST_CNN(FPN):
    def __init__(self, lat_dim, device, s_hi=1.0):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.fc_y = nn.Linear(1000,    lat_dim, bias=True)
        self.fc_u = nn.Linear(lat_dim, lat_dim, bias=False)
        self.fc_f = nn.Linear(lat_dim, 10, bias=False)
        self._lat_dim = lat_dim
        self._device = device
        self._s_hi = s_hi
        self.drop_out = nn.Dropout(p=0.5)
        self.soft_max = nn.Softmax(dim=1)
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=96,
                               kernel_size=3,
                               stride=1)
        self.conv2 = nn.Conv2d(in_channels=96,
                               out_channels=40,
                               kernel_size=3,
                               stride=1)

    def name(self):
        return 'MNIST_CNN'

    def device(self):
        return self._device

    def lat_dim(self):
        return self._lat_dim

    def s_hi(self):
        return self._s_hi

    def latent_space_forward(self, u, v):
        return self.relu(self.fc_u(0.99 * u + v))

    def data_space_forward(self, d):
        v = self.maxpool(self.relu(self.drop_out(self.conv1(d))))
        v = self.maxpool(self.relu(self.drop_out(self.conv2(v))))
        v = v.view(d.shape[0], -1)
        return self.relu(self.fc_y(v))

    def map_latent_to_inference(self, u):
        return self.fc_f(u)


class SVHN_CNN(FPN):
    def __init__(self, lat_dim, device, s_hi=1.0, inf_dim=10):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self._lat_dim = lat_dim
        self._inf_dim = inf_dim
        self._s_hi = s_hi
        self._device = device
        self.relu = nn.LeakyReLU(0.1)
        self.fc_u = nn.Linear(lat_dim, lat_dim, bias=False)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=30,
                               kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=30, out_channels=48,
                               kernel_size=5, stride=1)
        self.fc_input1 = nn.Linear(in_features=1200, out_features=lat_dim)

        self.fc_final = nn.Linear(lat_dim, 10)

    def name(self):
        return 'SVHN_CNN'

    def device(self):
        return self._device

    def lat_dim(self):
        return self._lat_dim

    def s_hi(self):
        return self._s_hi

    def latent_space_forward(self, u, v):
        u = self.fc_u(self.relu(u))
        output = 0.5*u + v
        return output

    def data_space_forward(self, d):
        current_batch_size = d.shape[0]

        # ------------------------
        # First Convolution Block
        # ------------------------
        v = self.conv1(d)
        v = self.relu(v)
        v = self.maxpool(v)

        # ------------------------
        # Second Convolution Block
        # ------------------------
        v = self.conv2(v)
        v = self.relu(v)
        v = self.maxpool(v)

        # ------------------------
        # Map back to 10-dim space
        # ------------------------
        v = v.view(current_batch_size, -1)
        v = self.fc_input1(v)
        v = self.relu(v)
        return v

    def map_latent_to_inference(self, u):
        return self.fc_final(u)
