import torch
import torch.nn as nn
from tqdm import tqdm
from calc_jacobian import calc_jacobian
import torch.optim as optim


def train(model_x,
          model_y,
          dataloader,
          x0,
          device,
          epochs,
          Qx=1,
          Qy=1,
          Qthetax=1,
          Qthetay=1,
          Q0=0,
          Px=1,
          Pthetax=1,
          Pthetay=1,
          P0=0,
          criterion=nn.MSELoss(),
          print_rate=1000
          ):

    '''
    :param model_x: x(k+1) = f(x(k), u(k))
    :param model_y: y(k) = h(x(k), u(k))
    :param dataloader: torch dataloader with batch_size == 1, and should not be shuffled, unpacks u(k), y(k+1)
    :param x0: initial value for the state variables x
    :param device:
    :param Qx:
    :param Qy:
    :param Qthetax:
    :param Qthetay:
    :param Q0:
    :param Px:
    :param Pthetax:
    :param Pthetay:
    :param P0:
    :param criterion:
    :param print_rate: number of iterations it prints the avg running loss
    :return:
    '''


    # Kalman Filter Parameter initialization
    P, Q, Qy, n_thetay, n_thetax, ny, nx, model_x, model_y, criterion = initialize_parameters(model_x, model_y,
                           dataloader, device,
                           Qthetay, Qthetax,
                           Qx, Qy,
                           Q0,
                           Px, Pthetax, P0,
                           criterion, x0)

    # Variables to track the running average of the loss
    running_loss = 0.
    loss = 0
    running_samples = 0

    for epoch in range(epochs):
        for it, data in tqdm(enumerate(dataloader)):
            # Unpack the data
            if it == 0:
                cmds, y_k = data
                x0 = x0.to(device)
                cmds = cmds.to(device)
                y_k = y_k.to(device)
                # First iteration
                with torch.no_grad():
                    x_hat_k_k1 = model_x(cmds, x0)  # next_estimatedStates = x(k+1|k)
            else:
                # Forward pass of the model
                cmds, y_k_1 = data
                cmds = cmds.to(device)
                y_k_1 = y_k_1.to(device)

                P, x_hat_k1_k = update_par(model_x, model_y, y_k, x_hat_k_k1, P, cmds,
                                               device,
                                               Qy, Q, n_thetax, n_thetay)

                y_k = y_k_1
                x_hat_k_k1 = x_hat_k1_k

                with torch.no_grad():
                    loss = criterion(y_k, model_y(x_hat_k_k1))

                    running_loss += loss
                    running_samples += 1

                if (running_samples % print_rate) == 0:
                    print(f'ep: {epoch}, it: {it}, Kloss : {running_loss / (running_samples):.6f}',
                          flush=True)
                    running_loss = 0.
                    running_samples = 0


def initialize_parameters(model_x, model_y,
                           dataloader, device,
                           Qthetay, Qthetax,
                           Qx, Qy,
                           Q0,
                           Px, Pthetax, P0,
                           criterion, x0):
    '''
    Initializes Q, P, asserts sizes and move matrices, criterion and models to device.
    :param model_x: x(k+1) = f(x(k), u(k))
    :param model_y: y(k) = h(x(k), u(k))
    :param dataloader: torch dataloader with batch==1
    :param device: torch.device
    :param Qthetay: covariance matrix (can be a torch 2D array or a single value)
    :param Qthetax: covariance matrix (can be a torch 2D array or a single value)
    :param Qx: covariance matrix (can be a torch 2D array or a single value)
    :param Qy: covariance matrix (can be a torch 2D array or a single value)
    :param Q0: covariance matrix, if 2D, ignores Qx and Qy
    :param Px: Initial value for Px matrix (can be a torch 2D array or a single value)
    :param Pthetax: Initial value for Px matrix (can be a torch 2D array or a single value)
    :param P0: initial P matrix, if 2D, ignores Px and Pthetax
    :param criterion: loss function
    :param x0: initial value for state variables
    :return: P0, Q0, Qy, n_thetay, n_thetax, ny, nx, model_x, model_y, criterion
    '''
    if dataloader.batch_size != 1:
        raise Exception("Batchsize must be 1 for EKF")

    nx = x0.shape[-1]
    if hasattr(Qx, "__len__"):
        if len(Qx) != nx:
            raise Exception("Mismatch between the size of Qx and x0")
    else:
        Qx = Qx * torch.eye(nx)

    for dt in dataloader:
        y0, u0, y1 = dt
        break
    ny = y0.shape[-1]
    if hasattr(Qy, "__len__"):
        if len(Qy) != ny:
            raise Exception("Mismatch between the size of Qx and x0")
    else:
        Qy = Qy * torch.eye(ny)

    pars = model_x.parameters()
    par_per_item = [p.numel() for p in pars]
    n_thetax = sum(par_per_item)

    pars = model_y.parameters()
    par_per_item = [p.numel() for p in pars]
    n_thetay = sum(par_per_item)

    if hasattr(Qthetay, "__len__"):
        if len(Qthetay) != n_thetay:
            raise Exception("Mismatch between the size of Qthetay and the number of parameters of model_y")
    else:
        Qthetay = Qthetay*torch.eye(n_thetay)

    if hasattr(Q0, "__len__"):
        if len(Q0) != (nx + n_thetax + n_thetay):
            raise Exception("Mismatch between the size of Q0 and (nx + n_thetax + n_thetay)")
    else:
        if hasattr(Qx, "__len__"):
            if len(Qx) != nx:
                raise Exception("Mismatch between the size of Qx and x0")
        else:
            Qx = Qx * torch.eye(nx)
        if hasattr(Qthetax, "__len__"):
            if len(Qthetax) != n_thetax:
                raise Exception("Mismatch between the size of Qthetax and n_thetax")
        else:
            Qthetax = Qthetax * torch.eye(n_thetax)
        Q0 = (torch.zeros(nx + n_thetax + n_thetay))
        Q0[0:nx, 0:nx] = Qx
        Q0[nx:, nx:] = Qthetax


    if hasattr(P0, "__len__"):
        if len(P0) != (nx + n_thetax + n_thetay):
            raise Exception("Mismatch between the size of P0 and (nx + n_thetax + n_thetay)")
    else:
        if hasattr(Px, "__len__"):
            if len(Px) != nx:
                raise Exception("Mismatch between the size of Px and x0")
        else:
            Px = Px * torch.eye(nx)
        if hasattr(Pthetax, "__len__"):
            if len(Pthetax) != n_thetax:
                raise Exception("Mismatch between the size of Pthetax and n_thetax")
        else:
            Pthetax = Pthetax * torch.eye(n_thetax)
        P0 = (torch.zeros(nx + n_thetax + n_thetay))
        P0[0:nx, 0:nx] = Px
        P0[nx:, nx:] = Pthetax



    # Move model and criterion to the device
    model_x = model_x.to(device)
    model_y = model_y.to(device)
    criterion = criterion.to(device)
    P0 = P0.to(device)
    Q0 = Qx.to(device)
    Qy = Qy.to(device)

    return P0, Q0, Qy, n_thetay, n_thetax, ny, nx, model_x, model_y, criterion


def update_par(model_x, model_y, y_k, x_hat_k_k1, P, u_k,
               device, Qy, Q, optimizer, n_thetax, n_thetay, nx, ny
               ):


    C = (x_hat_k_k1, u_k, model_y, ny, device, n_thetax, n_thetay)
    with torch.no_grad():

        err = (y_k - model_y(x_hat_k_k1)).flatten()
        x_hat_k_k = torch.empty(x_hat_k_k1.shape).to(device)
        M1 = torch.matmul(torch.matmul(C, P), C.T)
        M = torch.inverse(M1 + Qy)
        del M1
        M = torch.matmul(torch.matmul(P, C.T), M)
        delta = torch.matmul(M, err)

        x_hat_k_k = x_hat_k_k1.clone() + delta[0:nx]

        torch.cuda.empty_cache()
        # Calculate P(k | k)
        M = - torch.matmul(M, C)
        M[range(len(M)), range(len(M))] = M[range(len(M)), range(len(M))] + 1
        P = torch.matmul(M, P)
        del M

        delta_thetax = delta[nx:nx+n_thetax]
        state_dict = model_x.state_dict()
        curr_index = 0
        for st in state_dict:
            par_size = int(torch.prod(torch.tensor(state_dict[st].shape)))
            dth = delta_thetax[curr_index:curr_index + par_size]
            state_dict[st] = torch.reshape(dth, state_dict[st].shape) + state_dict[st]
            curr_index += par_size
        model_x.load_state_dict(state_dict)

        delta_thetay = delta[nx+n_thetax:]
        state_dict = model_y.state_dict()
        curr_index = 0
        for st in state_dict:
            par_size = int(torch.prod(torch.tensor(state_dict[st].shape)))
            dth = delta_thetay[curr_index:curr_index + par_size]
            state_dict[st] = torch.reshape(dth, state_dict[st].shape) + state_dict[st]
            curr_index += par_size
        model_y.load_state_dict(state_dict)

    x_hat_k1_k = model_x(u_k, x_hat_k_k)
    #print(x_hat_k1_k.shape)
    dfdx, dfdthetax = calc_jacobian(model_x, u_k, x_hat_k_k, x_hat_k1_k, device, n_thetax)

    with torch.no_grad():
        A = torch.zeros((nx + n_thetax + n_thetay, nx + n_thetax + n_thetay))  # .to(device)
        A[0:nx, 0:nx] = dfdx[:].clone()  # .detach().numpy()
        del dfdx
        A[0:nx, nx:nx + n_thetax] = dfdthetax[:].clone()  # .detach().numpy()
        del dfdthetax
        A[nx:, nx:] = torch.eye(n_thetax + n_thetay)  # np.eye(n_theta)

        A = A.to(device)
        P = torch.matmul(A, P)
        torch.cuda.empty_cache()
        P = torch.matmul(P, A.T)  # np.matmul(A, P)
        del A
        P[range(len(P)), range(len(P))] = P[range(len(P)), range(len(P))] + Q

    return P, x_hat_k1_k

def buildC(x_hat_k_k1, u_k, model_y, ny, device, n_thetax, n_thetay):
    y_hat_1 = model_y(u_k, x_hat_k_k1)
    dfy_dx, dfydthetay = calc_jacobian(model_y, u_k, x_hat_k_k1, y_hat_1, device, n_thetay)
    Zeros = torch.zeros((ny, n_thetax)).to(device)
    C = torch.cat((dfy_dx.to(device), Zeros, dfydthetay.to(device)), dim=1)
    return C