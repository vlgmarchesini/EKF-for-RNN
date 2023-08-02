import torch.nn as nn

class model_x(nn.Module):
    def __init__(self, y_dim, x_dim, u_dim):
        '''
        :param y_dim: y(k) dimension, Hout in torch
        :param x_dim: x(k) dimension (number of hidden states), Hn of the GRU
        :param u_dim: u(k) dimension (number of control commands), input of the GRU
        '''
        super().__init__()

        self.fc_in = nn.Linear(in_features=y_dim, out_features=x_dim, bias=False)
        self.model = nn.GRU(input_size=u_dim, hidden_size=x_dim, num_layers=1, batch_first=True)

    def get_x0(self, y0):
        x0 = self.fc_in(y0)
        return x0

    def forward(self, u, x):
        x_next, _ = self.model(u, x)
        return x_next

class model_y(nn.Module):
    def __init__(self, y_dim, x_dim, u_dim):
        '''
        :param y_dim: y(k) dimension, Hout in torch
        :param x_dim: x(k) dimension (number of hidden states), Hn of the GRU
        :param u_dim: u(k) dimension (number of control commands), input of the GRU
        '''
        super().__init__()
        self.fc_out = nn.Linear(in_features=x_dim, out_features=y_dim, bias=False)

    def get_x0(self, y0):
        x0 = self.fc_in(y0)
        return x0

    def forward(self, u, x):
        y = self.fc_out(x)
        return y