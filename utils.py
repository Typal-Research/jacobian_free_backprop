import torch
from prettytable import PrettyTable

from time import sleep
import time
import tqdm


def display_model_params(model: torch.tensor):
    table = PrettyTable(["Network Component", "# Parameters"])
    num_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        table.add_row([name, parameter.numel()])
        num_params += parameter.numel()
    print(table)
    print(f"Total Trainable Paramseters: {num_params}")


# -----------------------------------------------------------------------------
# returns testing loss, accuracy, # correct, depth (test statistics)
# -----------------------------------------------------------------------------
def test_statistics(KM, eps, device, test_loader, batch_size, n, loss,
                    hidden_size):
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for d_test, labels in test_loader:
            labels = labels.to(device)
            d_test = d_test.to(device)
            ut = torch.zeros((batch_size, n)).to(device)
            for i in range(batch_size):
                ut[i, labels[i].cpu().numpy()] = 1.0

            u0_test = 0.1 * torch.zeros((batch_size, hidden_size)).to(device)
            u0_test[:, 0:n] = 1.0 / float(n)
            u, depth = KM(u0_test, d_test, eps)
            y = u[:, 0:n].to(device)
            # test_loss += loss(y, labels).item() # sum up batch loss
            test_loss += loss(y.double(), ut.double()).item()
            pred = y.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct/len(test_loader.dataset)

    return test_loss, test_acc, correct, depth


def train_network(S, T, KM, num_epochs, lr_scheduler, train_loader,
                  test_loader, device, batch_size, n, hidden_size, eps,
                  optimizer, alpha, loss):

    fmt = '[{:3d}/{:3d}]: train - ({:6.2f}%, {:6.2e}), test - ({:6.2f}%, '
    fmt += '{:6.2e}) | depth = {:4.1f} | lr = {:5.1e} | time = {:4.1f} sec'

    test_loss_hist = []  # test loss history array
    test_acc_hist = []  # test accuracy history array
    depth_test_hist = []  # test depths history array
    loss_ave = 0.0         # displays moving average of training loss
    depth_ave = 0.0         # displays moving average of iterations to converge
    train_acc = 0.0
    print(T, '\n', KM, '\n')
    display_model_params(T)
    print('\nTraining Fixed Point Network')

    for epoch in range(num_epochs):
        sleep(1.0)  # slows progress bar so it won't print on multiple lines
        epoch_start_time = time.time()

        with tqdm(total=len(train_loader), unit=" batch",
                  leave=False, ascii=True) as tepoch:

            tepoch.set_description("[{:3d}/{:3d}]".format(epoch+1, num_epochs))

            for idx, (d, labels) in enumerate(train_loader):
                labels = labels.to(device)
                d = d.to(device)

                ut = torch.zeros((batch_size, n)).to(device)
                for i in range(batch_size):
                    ut[i, labels[i].cpu().numpy()] = 1.0

                # --------------------------------------------------------------
                # Forward prop to fixed point
                # --------------------------------------------------------------
                u0 = 0.1 * torch.zeros((batch_size, hidden_size)).to(device)
                u0[:, 0:n] = 1.0 / float(n)
                KM.assign_ops(S, T)
                u, depth = KM(u0, d, eps)
                depth_ave = 0.95 * depth_ave + 0.05 * depth
                # -------------------------------------------------------------
                # Step with fixed point and then backprop
                # -------------------------------------------------------------
                optimizer.zero_grad()
                y = KM.apply_T(u.float().to(device), d)[:, 0:n]
                output = loss(y.double(), ut.double())
                # output = loss(y, labels)
                loss_val = output.detach().cpu().numpy() / batch_size
                loss_ave = 0.99 * loss_ave + 0.01 * loss_val
                output.backward()
                optimizer.step()
                # -------------------------------------------------------------
                # Output training stats
                # -------------------------------------------------------------
                if idx % 50 == 0:
                    T.project_weights()

                pred = y.argmax(dim=1, keepdim=True)
                correct = pred.eq(labels.view_as(pred)).sum().item()
                train_acc = 0.99 * train_acc + 1.00 * correct / batch_size
                tepoch.update(1)
                tepoch.set_postfix(train_loss="{:5.2e}".format(loss_ave),
                                   train_acc="{:5.2f}%".format(train_acc),
                                   depth="{:5.1f}".format(depth_ave))
        # ---------------------------------------------------------------------
        # Save weights every 10 epochs
        # ---------------------------------------------------------------------
        if epoch % 10 == 0:
            # create dictionary saving all required parameters:
            state = {
                'alpha': alpha,
                'eps': eps,
                'Tnet_state_dict': T.state_dict(),
                'test_loss_hist': test_loss_hist,
                'test_acc_hist': test_acc_hist,
                'depth_test_hist': depth_test_hist
            }
            torch.save(state, 'MNIST-CNN_weights.pth')

        test_loss, test_acc, correct, depth_test = test_statistics(KM, eps,
                                                                   device,
                                                                   test_loader)
        print(fmt.format(epoch+1, num_epochs, train_acc, loss_ave,
                         test_acc, test_loss, depth_ave,
                         optimizer.param_groups[0]['lr'], eps,
                         time.time() - epoch_start_time))
        lr_scheduler.step()
        epoch_start_time = time.time()
        return T
