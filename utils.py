import torch
from prettytable import PrettyTable
from time import sleep
import time
from tqdm import tqdm


def get_stats(net, test_loader, batch_size, n, loss,
              hid_size):
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for d_test, labels in test_loader:
            labels = labels.to(net.device)
            d_test = d_test.to(net.device)
            if net.name() == "MNIST_FCN":
                d_test = d_test.view(d_test.size()[0], 784).to(net.device)

            ut = torch.zeros((d_test.size()[0], n)).to(net.device)
            for i in range(d_test.size()[0]):
                ut[i, labels[i].cpu().numpy()] = 1.0

            y = net(d_test)[:, 0:n]

            if str(loss) == "MSELoss()":
                test_loss += loss(y.double(), ut.double()).item()
            elif str(loss) == "CrossEntropyLoss()":
                test_loss += loss(y, labels).item()  # sum up batch loss
            else:
                print("Error: Invalid Loss Function")

            pred = y.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct/len(test_loader.dataset)

    return test_loss, test_acc, correct


def model_params(net):
    table = PrettyTable(["Network Component", "# Parameters"])
    num_params = 0
    for name, parameter in net.named_parameters():
        if not parameter.requires_grad:
            continue
        table.add_row([name, parameter.numel()])
        num_params += parameter.numel()
    table.add_row(['TOTAL', num_params])
    return table


def train_class_net(net, num_epochs, lr_scheduler, train_loader,
                    test_loader, batch_size, sig_dim, op_dim,
                    optimizer, loss, alg_params=None, save_dir='./'):

    fmt = '[{:3d}/{:3d}]: train - ({:6.2f}%, {:6.2e}), test - ({:6.2f}%, '
    fmt += '{:6.2e}) | depth = {:4.1f} | lr = {:5.1e} | time = {:4.1f} sec'

    loss_ave = 0.0
    depth_ave = 0.0
    train_acc = 0.0
    best_test_acc = 0.0
    
    test_loss_hist  = []
    test_acc_hist   = []
    
    print(net)
    print(model_params(net))
    print('\nTraining Fixed Point Network')

    for epoch in range(num_epochs):
        sleep(0.5)  # slows progress bar so it won't print on multiple lines
        epoch_start_time = time.time()
        tot = len(train_loader)
        with tqdm(total=tot, unit=" batch", leave=False, ascii=True) as tepoch:

            tepoch.set_description("[{:3d}/{:3d}]".format(epoch+1, num_epochs))

            for idx, (d, labels) in enumerate(train_loader):
                labels = labels.to(net.device)
                d = d.to(net.device)

                if net.name() == "MNIST_FCN":
                    d = d.view(d.size()[0], 784).to(net.device)

                ut = torch.zeros((d.size()[0], sig_dim)).to(net.device)
                for i in range(d.size()[0]):
                    ut[i, labels[i].cpu().numpy()] = 1.0
                # -------------------------------------------------------------
                # Apply network to get fixed point and then backprop
                # -------------------------------------------------------------
                optimizer.zero_grad()
                y = net(d, alg_params)
                y = y[:, 0:sig_dim]
                depth_ave = net.depth
                output = None
                if str(loss) == "MSELoss()":
                    output = loss(y.double(), ut.double())
                elif str(loss) == "CrossEntropyLoss()":
                    output = loss(y, labels)
                else:
                    print("Error: Invalid Loss Function")
                loss_val = output.detach().cpu().numpy() / batch_size
                loss_ave = 0.99 * loss_ave + 0.01 * loss_val
                output.backward()
                optimizer.step()
                # -------------------------------------------------------------
                # Output training stats
                # -------------------------------------------------------------
                pred = y.argmax(dim=1, keepdim=True)
                correct = pred.eq(labels.view_as(pred)).sum().item()
                train_acc = 0.99 * train_acc + 1.00 * correct / batch_size
                tepoch.update(1)
                tepoch.set_postfix(train_loss="{:5.2e}".format(loss_ave),
                                   train_acc="{:5.2f}%".format(train_acc),
                                   depth="{:5.1f}".format(depth_ave))

        test_loss, test_acc, correct = get_stats(net,
                                                 test_loader,
                                                 batch_size,
                                                 sig_dim, loss,
                                                 op_dim)
        
        test_loss_hist.append(test_loss)
        test_acc_hist.append(test_acc)

        print(fmt.format(epoch+1, num_epochs, train_acc, loss_ave,
                         test_acc, test_loss, depth_ave,
                         optimizer.param_groups[0]['lr'],
                         time.time() - epoch_start_time))
        # ---------------------------------------------------------------------
        # Save weights every 10 epochs
        # ---------------------------------------------------------------------
        if epoch % 10 == 0 and test_acc > best_test_acc:
            state = {
                'eps': alg_params.eps,
                'max depth': alg_params.depth,
                'alpha': alg_params.alpha,
                'gamma': alg_params.gamma,
                'test_loss_hist': test_loss_hist,
                'test_acc_hist': test_acc_hist, 
                'net_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler
            }
            file_name = save_dir + 'KM_' + net.name() + '_weights.pth'
            torch.save(state, file_name)
            print('Model weights saved to ' + file_name)

        lr_scheduler.step()
        epoch_start_time = time.time()
    return net
