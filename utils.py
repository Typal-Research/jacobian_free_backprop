import torch
from prettytable import PrettyTable
from time import sleep
import time
from tqdm import tqdm


def get_stats(net, test_loader, loss, num_classes, eps, depth):
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for d_test, labels in test_loader:
            labels = labels.to(net.device())
            d_test = d_test.to(net.device())
            batch_size = d_test.shape[0]
            if net.name() == "MNIST_FCN":
                d_test = d_test.view(d_test.size()[0], 784).to(net.device())

            ut = torch.zeros((d_test.size()[0], num_classes))
            ut = ut.to(net.device())
            for i in range(d_test.size()[0]):
                ut[i, labels[i].cpu().numpy()] = 1.0

            y = net(d_test, eps=eps, max_depth=depth)

            if str(loss) == "MSELoss()":
                test_loss += batch_size * loss(y.double(), ut.double()).item()
            elif str(loss) == "CrossEntropyLoss()":
                test_loss += batch_size * loss(y, labels).item()
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
                    test_loader, optimizer, loss,
                    num_classes, eps, depth, save_dir='./'):

    fmt = '[{:3d}/{:3d}]: train - ({:6.2f}%, {:6.2e}), test - ({:6.2f}%, '
    fmt += '{:6.2e}) | depth = {:4.1f} | lr = {:5.1e} | time = {:4.1f} sec'

    depth_ave       = 0.0
    train_acc       = 0.0
    best_test_acc   = 0.0

    total_time    = 0.0
    time_hist     = []

    test_loss_hist  = []
    test_acc_hist   = []
    train_loss_hist = []
    train_acc_hist  = []


    print(net)
    print(model_params(net))
    print('\nTraining Fixed Point Network')

    for epoch in range(num_epochs):
        sleep(0.5)  # slows progress bar so it won't print on multiple lines
        loss_ave        = 0.0
        epoch_start_time = time.time()
        tot = len(train_loader)
        with tqdm(total=tot, unit=" batch", leave=False, ascii=True) as tepoch:

            tepoch.set_description("[{:3d}/{:3d}]".format(epoch+1, num_epochs))

            for idx, (d, labels) in enumerate(train_loader):
                labels = labels.to(net.device())
                d = d.to(net.device())

                batch_size = d.shape[0]

                if net.name() == "MNIST_FCN":
                    d = d.view(d.size()[0], 784).to(net.device())

                # -------------------------------------------------------------
                # Apply network to get fixed point and then backprop
                # -------------------------------------------------------------
                optimizer.zero_grad()
                y = net(d, eps=eps, max_depth=depth)

                depth_ave = max(net.depth, depth_ave)
                output = None
                if str(loss) == "MSELoss()":
                    
                    ut = torch.zeros((d.size()[0], num_classes)).to(net.device())
                    for i in range(d.size()[0]):
                        ut[i, labels[i].cpu().numpy()] = 1.0

                    output = loss(y.double(), ut.double())
                elif str(loss) == "CrossEntropyLoss()":
                    output = loss(y, labels)
                else:
                    print("Error: Invalid Loss Function")
                loss_val = output.detach().cpu().numpy() * batch_size
                loss_ave += loss_val
                output.backward()
                optimizer.step()
                # -------------------------------------------------------------
                # Output training stats
                # -------------------------------------------------------------
                pred = y.argmax(dim=1, keepdim=True)
                correct = pred.eq(labels.view_as(pred)).sum().item()
                train_acc = 0.99 * train_acc + 1.00 * correct / batch_size
                tepoch.update(1)
                tepoch.set_postfix(train_loss="{:5.2e}".format(loss_val
                                   / batch_size),
                                   train_acc="{:5.2f}%".format(train_acc),
                                   depth="{:5.1f}".format(net.depth))

        #  divide by total number of training samples
        loss_ave = loss_ave / len(train_loader.dataset)

        test_loss, test_acc, correct = get_stats(net,
                                                 test_loader,
                                                 loss,
                                                 num_classes,
                                                 eps,
                                                 depth)

        test_loss_hist.append(test_loss)
        test_acc_hist.append(test_acc)
        train_loss_hist.append(loss_ave)
        train_acc_hist.append(train_acc)

        epoch_end_time = time.time()
        time_epoch = epoch_end_time - epoch_start_time

        time_hist.append(time_epoch)
        total_time += time_epoch 

        print(fmt.format(epoch+1, num_epochs, train_acc, loss_ave,
                         test_acc, test_loss, depth_ave,
                         optimizer.param_groups[0]['lr'],
                         time_epoch))
        # ---------------------------------------------------------------------
        # Save weights
        # ---------------------------------------------------------------------
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            state = {
                'test_loss_hist': test_loss_hist,
                'test_acc_hist': test_acc_hist,
                'net_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler
            }
            file_name = save_dir + 'FPN_' + net.name() + '_weights.pth'
            torch.save(state, file_name)
            print('Model weights saved to ' + file_name)

        # ---------------------------------------------------------------------
        # Save history at last epoch
        # ---------------------------------------------------------------------

        if epoch+1 == num_epochs:
            state = {
                'test_loss_hist': test_loss_hist,
                'test_acc_hist': test_acc_hist,
                'train_loss_hist': train_loss_hist,
                'train_acc_hist': train_acc_hist,
                'lr_scheduler': lr_scheduler,
                'time_hist': time_hist,
                'eps': eps,
            }
            file_name = save_dir + 'FPN_' + net.name() + '_history.pth'
            torch.save(state, file_name)
            print('Training history saved to ' + file_name)

        lr_scheduler.step()
        epoch_start_time = time.time()
    return net
