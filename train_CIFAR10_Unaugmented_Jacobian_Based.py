import torch
import torch.nn as nn
import torch.optim as optim
from prettytable import PrettyTable
import time
from BatchCG import cg_batch
from time import sleep
from tqdm import tqdm

from Networks import CIFAR10_FPN_Unaugmented_Jacobian_Based, BasicBlock
from utils import cifar_loaders, compute_fixed_point

device = "cuda:0"
print('device = ', device)
seed = 43
torch.manual_seed(seed)

# -----------------------------------------------------------------------------
# Load dataset
# -----------------------------------------------------------------------------
batch_size = 100
test_batch_size = 400

train_loader, test_loader = cifar_loaders(train_batch_size=batch_size,
                                          test_batch_size=400, augment=True)

# -----------------------------------------------------------------------------
# Compute testing statistics
# -----------------------------------------------------------------------------


def get_test_stats(net, data_loader, criterion, eps, max_depth):
    test_loss = 0
    correct = 0
    net.eval()
    with torch.no_grad():
        for d_test, labels in test_loader:
            labels = labels.to(device)
            d_test = d_test.to(device)
            batch_size = d_test.shape[0]

            Qd = net.data_space_forward(d_test)
            y, depth = compute_fixed_point(T, Qd, max_depth, device, eps=eps)
            S_Ru = net.map_latent_to_inference(y)
            test_loss += batch_size * criterion(S_Ru, labels).item()

            pred = S_Ru.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct/len(test_loader.dataset)

    net.train()

    return test_loss, test_acc, correct, depth

# -----------------------------------------------------------------------------
# Network setup
# -----------------------------------------------------------------------------


num_blocks = [1, 1, 1]
contract_factor = 0.9
res_layers = 1
T = CIFAR10_FPN_Unaugmented_Jacobian_Based(block=BasicBlock,
                                           num_blocks=num_blocks,
                                           res_layers=res_layers,
                                           num_channels=64,
                                           contraction_factor=contract_factor)
T = T.to(device)
eps = 1.0e-1
max_depth = 50

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
max_epochs = 200
learning_rate = 1.0e-4  # 1e-3 for others
weight_decay = 1e-3
optimizer = optim.Adam(T.parameters(), lr=learning_rate,
                       weight_decay=weight_decay)

lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=1.0)
checkpt_path = './models/'
criterion = nn.CrossEntropyLoss()
batch_size = 100
test_batch_size = 400


def num_params(model):
    table = PrettyTable(["Modules", "Parameters"])
    num_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        table.add_row([name, parameter.numel()])
        num_params += parameter.numel()
    print(table)
    print(f"Total Trainable Params: {num_params}")
    return num_params


avg_time = 0.0         # saves average time per epoch
total_time = 0.0
time_hist = []
n_Umatvecs = []
max_iter_cg = max_depth
tol_cg = eps
JTJ_shift = 1e-3
save_dir = './'

# save histories for testing data set
test_loss_hist = []  # test loss history array
test_acc_hist = []  # test accuracy history array
depth_test_hist = []  # test depths history array
train_loss_hist = []  # train loss history array
train_acc_hist = []  # train accuracy history array

# start_time_epoch = time.time() # timer for display execution time per
# epoch multiple
fmt = '[{:4d}/{:4d}]: train acc = {:5.2f}% | train_loss = {:7.3e} | '
fmt += ' test acc = {:5.2f}% | test loss = {:7.3e} | '
fmt += 'depth = {:5.1f} | lr = {:5.1e} | time = {:4.1f} sec | '
fmt += 'n_Umatvecs = {:4d}'
print(T)        # display Tnet configuration
num_params(T)   # display Tnet parameters
print('\nTraining Fixed Point Network')

best_test_acc = 0.0
train_acc = 0.0
# -----------------------------------------------------------------------------
# Execute Training
# -----------------------------------------------------------------------------
for epoch in range(max_epochs):

    sleep(0.5)  # slows progress bar so it won't print on multiple lines
    tot = len(train_loader)
    temp_n_Umatvecs = 0
    start_time_epoch = time.time()
    temp_max_depth = 0
    loss_ave = 0.0
    with tqdm(total=tot, unit=" batch", leave=False, ascii=True) as tepoch:

        tepoch.set_description("[{:3d}/{:3d}]".format(epoch+1, max_epochs))
        for idx, (d, labels) in enumerate(train_loader):
            labels = labels.to(device)
            d = d.to(device)

            # -----------------------------------------------------------------
            # Find Fixed Point
            # -----------------------------------------------------------------
            train_batch_size = d.shape[0]  # re-define if batch size changes
            with torch.no_grad():
                Qd = T.data_space_forward(d)
                u, depth = compute_fixed_point(T, Qd, max_depth, device,
                                               eps=eps)
                temp_max_depth = max(depth, temp_max_depth)

            # -----------------------------------------------------------------
            # Jacobian_Based Backprop
            # -----------------------------------------------------------------
            T.train()
            optimizer.zero_grad()  # Initialize gradient to zero

            # compute output for backprop
            u.requires_grad = True
            Qd = T.data_space_forward(d)

            Ru = T.latent_space_forward(u, Qd)

            S_Ru = T.map_latent_to_inference(Ru)
            loss = criterion(S_Ru, labels)
            train_loss = loss.detach().cpu().numpy() * train_batch_size
            loss_ave += train_loss

            # compute rhs: = J * dldu = J * dS/du * dl/dS
            dldu = torch.autograd.grad(outputs=loss, inputs=Ru,
                                       retain_graph=True,
                                       create_graph=True,
                                       only_inputs=True)[0]
            dldu = dldu.detach()  # note: dldu here = dS/du * dl/dS

            # -----------------------------------------------------------------
            # rhs = J^T * dldu = (I - dRdu^T) * dldu
            # -----------------------------------------------------------------
            dRduT_dldu = torch.autograd.grad(outputs=Ru, inputs=u,
                                             grad_outputs=dldu,
                                             retain_graph=True,
                                             create_graph=True,
                                             only_inputs=True)[0]
            rhs = dldu - dRduT_dldu
            # -----------------------------------------------------------------

            rhs = rhs.detach()
            # vectorize channels (when R is a CNN)
            rhs = rhs.view(train_batch_size, -1)
            # unsqueeze for number of rhs. CG requires it
            # to have dimensions n_samples x n_features x n_rh
            rhs = rhs.unsqueeze(2)

            # -----------------------------------------------------------------
            # Define JTJ matvec function
            # -----------------------------------------------------------------
            def JTJ_matvec(v, u=u, Ru=Ru):
                # inputs:
                # v = vector to be multiplied by
                #     U = I - alpha*DS - (1-alpha)*DT) requires grad
                # u = fixed point vector u
                # (should be untracked and vectorized!) requires grad
                # Ru = R applied to u (requires grad)

                # assumes one rhs:
                # x (n_samples, n_dim, n_rhs) -> (n_samples, n_dim)

                v = v.squeeze(2)      # squeeze number of RHS
                v = v.view(Ru.shape)  # reshape to filter space
                v.requires_grad = True

                # (dRdu)^T * v = JT * v
                dRduT_mv = torch.autograd.grad(outputs=Ru, inputs=u,
                                               grad_outputs=v,
                                               retain_graph=True,
                                               create_graph=True,
                                               only_inputs=True)[0]
                JTv = v - dRduT_mv

                # compute J * JTv
                # trick for computing J * (JTv):
                # # take take derivative d(JTv)/dv * JTv = J * JTv
                dRdu_JTv = torch.autograd.grad(outputs=JTv,
                                               inputs=v,
                                               grad_outputs=JTv.detach(),
                                               retain_graph=True,
                                               create_graph=True,
                                               only_inputs=True)[0]
                JJTv = JTv - dRdu_JTv  # J = I - dRdu
                v = v.detach()
                JTv = JTv.detach()
                Amv = JJTv.detach()
                Amv = Amv.view(Ru.shape[0], -1)
                Amv = Amv.unsqueeze(2).detach()
                return Amv + JTJ_shift*v.view(Amv.shape)

            JTJinv_rhs, info = cg_batch(JTJ_matvec, rhs, M_bmm=None, X0=None,
                                        rtol=0, atol=tol_cg,
                                        maxiter=max_iter_cg, verbose=False)
            # JTJinv_v has size (batch_size x n_hidden_features),
            # n_rhs is squeezed
            JTJinv_rhs = JTJinv_rhs.squeeze(2)
            JTJinv_rhs = JTJinv_rhs.view(Ru.shape)

            temp_n_Umatvecs += info['niter'] * train_batch_size

            if info['optimal']:

                Ru.backward(JTJinv_rhs)

                u.requires_grad = False

                # update T parameters
                optimizer.step()

            # -------------------------------------------------------------
            # Output training stats
            # -------------------------------------------------------------
            pred = S_Ru.argmax(dim=1, keepdim=True)
            correct = pred.eq(labels.view_as(pred)).sum().item()
            train_acc = 0.99 * train_acc + 1.00 * correct / train_batch_size
            tepoch.update(1)
            tepoch.set_postfix(train_loss="{:5.2e}".format(train_loss
                               / train_batch_size),
                               train_acc="{:5.2f}%".format(train_acc),
                               depth="{:5.1f}".format(temp_max_depth))

    loss_ave /= len(train_loader.dataset)

    # update optimization scheduler
    lr_scheduler.step()

    # compute test loss and accuracy
    test_loss, test_acc, correct, depth_test = get_test_stats(T, test_loader,
                                                              criterion, eps,
                                                              max_depth)

    end_time_epoch = time.time()
    time_epoch = end_time_epoch - start_time_epoch

    # ---------------------------------------------------------------------
    # Compute costs and statistics
    # ---------------------------------------------------------------------
    time_hist.append(time_epoch)
    total_time += time_epoch
    avg_time /= total_time/(epoch+1)
    n_Umatvecs.append(temp_n_Umatvecs)

    test_loss_hist.append(test_loss)
    test_acc_hist.append(test_acc)
    train_loss_hist.append(loss_ave)
    train_acc_hist.append(train_acc)
    depth_test_hist.append(depth_test)

    # ---------------------------------------------------------------------
    # Print outputs to console
    # ---------------------------------------------------------------------

    print(fmt.format(epoch+1, max_epochs, train_acc, loss_ave,
                     test_acc, test_loss, temp_max_depth,
                     optimizer.param_groups[0]['lr'],
                     time_epoch, temp_n_Umatvecs))

    # ---------------------------------------------------------------------
    # Save weights
    # ---------------------------------------------------------------------
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        state = {
            'test_loss_hist': test_loss_hist,
            'test_acc_hist': test_acc_hist,
            'T_state_dict': T.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler
        }
        file_name = save_dir + T.name() + '_weights.pth'
        torch.save(state, file_name)
        print('Model weights saved to ' + file_name)

    # ---------------------------------------------------------------------
    # Save history at last epoch
    # ---------------------------------------------------------------------
    if epoch+1 == max_epochs:
        state = {
            'test_loss_hist': test_loss_hist,
            'test_acc_hist': test_acc_hist,
            'train_loss_hist': train_loss_hist,
            'train_acc_hist': train_acc_hist,
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler,
            'avg_time': avg_time,
            'n_Umatvecs': n_Umatvecs,
            'time_hist': time_hist,
            'tol_cg': tol_cg,
            'eps': eps,
            'T_state_dict': T.state_dict(),
            'test_loss_hist': test_loss_hist,
            'test_acc_hist': test_acc_hist,
            'depth_test_hist': depth_test_hist
        }
        file_name = save_dir + T.name() + '_history.pth'
        torch.save(state, file_name)
        print('Training history saved to ' + file_name)
