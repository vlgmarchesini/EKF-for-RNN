import numpy as np
import torch
import argparse

# Local Imports
import models
from datasets import TrajectoryGRU,  process_data, normalization_parameters
from train import train_with_EKF

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--path', metavar='DATA_PATH',
        default='./*.csv',
        help='path to the dataset directory, e.g. "/mundus/vgomesma005/Indentifier20Steps/"'
    )
    parser.add_argument(
        '--stats', metavar='DATA_PATH',
        default='',
        help='path to the dataset directory, e.g. "/mundus/vgomesma005/Indentifier20Steps/"'
    )
    parser.add_argument('--epochs', help='number of epochs for training', type=int, default=5)
    parser.add_argument('--qx', help='covariance matrix of the state variables', type=float, default=1)
    parser.add_argument('--qtheta', help='covariance matrix of the process', type=float, default=1)
    parser.add_argument('--qy', help='covariance matrix of the process', type=float, default=1)
    parser.add_argument('--px', help='covariance matrix of the process', type=float, default=1)
    parser.add_argument('--ptheta', help='covariance matrix of the process', type=float, default=1)
    parser.add_argument('--noise_level', help='Hidden dim for NN', type=int, default=1)

    torch.manual_seed(0)


    parser.add_argument(
        '--cuda', action='store_true',
        help='train the model on GPU (may crash if cuda is not available)'
    )
    args = parser.parse_args()

    print("Reading data file", flush=True)
    xtraining, tseries, state, u, dt, num_features = process_data(args.path)

    print("Normalizing data", flush=True)
    if args.stats == '':
        stats = normalization_parameters(state, u)
    else:
        stats = np.load(args.stats, allow_pickle=True)
    mean_states, std_states, mean_cmd, std_cmd = stats

    print(f'meanX={mean_states}, stdX={std_states}, meanU={mean_cmd}, stdU={std_cmd}')

    X_norm = (state - mean_states) / std_states
    u_norm = (u - mean_cmd) / std_cmd

    noise = args.noise_level * np.array([0, 0, 0, 0.01, 0.01, 0.017, 0.01]);
    noise = noise[3:3+len(std_states)]/std_states
    qy = noise**2
    n_items, nstates = X_norm.shape
    noise = np.random.randn(n_items, nstates)*noise
    X_norm = X_norm + noise

    f_name = str(args.path).replace('.csv', '_').replace('*', '') + '.npy'
    if args.stats == '':
        np.save(f_name, (mean_states, std_states, mean_cmd, std_cmd))

    print("Creating dataset and dataloader", flush=True)
    train_set = TrajectoryGRU(X_norm, u_norm)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False)

    u_dim = len(std_cmd)
    y_dim = len(std_states)
    x_dim = 6*y_dim

    Model_x = models.model_x(y_dim, x_dim, u_dim)
    Model_y = models.model_y(y_dim, x_dim, u_dim)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    with torch.no_grad():
        x0 = Model_x.get_x0(train_set[0][1].unsqueeze(0))

    train_with_EKF(Model_x,
          Model_y,
          train_loader,
          x0,
          device,
          args.epochs,
          Qx=args.qx,
          Qy=args.qy,
          Qthetax=args.qtheta,
          Qthetay=args.qtheta
          )