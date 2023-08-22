import torch



def calc_jacobian(model, cmds, x_k, y_hat, device, n_theta, optimizer):
    n = y_hat.shape[-1]
    df_dtheta = torch.zeros((n, n_theta)).to(device)
    print(f'y_hat.shape={y_hat.shape}')
    for j, est in enumerate(y_hat[0, 0, :]):
        optimizer.zero_grad()
        est.backward(retain_graph=True)
        idx = 0
        pars = model.parameters()
        for par in pars:

            #print(f'par.shape={par.shape}')
            #print(f'par.grad.shape={par.grad.shape}')
            if not(par.grad is None):
                grad_par = torch.flatten(par.grad, start_dim=0, end_dim=- 1)
                n_par = len(grad_par)
                df_dtheta[j, idx:idx + n_par] = grad_par
                idx = idx + n_par
    df_dtheta = df_dtheta
    dfdx_aux = torch.autograd.functional.jacobian(model, (cmds, x_k), create_graph=False, strict=False,
                                                  vectorize=False,
                                                  strategy='reverse-mode')

    dfdx = dfdx_aux[1][0, 0, :, 0, 0, :]
    return dfdx, df_dtheta
