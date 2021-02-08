import torch


from time import sleep
import time
from tqdm import tqdm


def test_statistics(KM, device, test_loader, batch_size, n, loss,
                    hid_size):
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for d_test, labels in test_loader:
            labels = labels.to(device)
            d_test = d_test.to(device)
            if KM.T.name() == "MNIST_FCN":
                d_test = d_test.view(d_test.size()[0], 784).to(device)

            ut = torch.zeros((d_test.size()[0], n)).to(device)
            for i in range(d_test.size()[0]):
                ut[i, labels[i].cpu().numpy()] = 1.0

            u0_test = 0.1 * torch.zeros((d_test.size()[0], hid_size)).to(device)
            u0_test[:, 0:n] = 1.0 / float(n)
            u, depth = KM(u0_test, d_test)
            y = u[:, 0:n].to(device)

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

    return test_loss, test_acc, correct, depth


def train_class_net(KM, num_epochs, lr_scheduler, train_loader,
                    test_loader, device, batch_size, n, hid_size,
                    optimizer, loss):

    fmt = '[{:3d}/{:3d}]: train - ({:6.2f}%, {:6.2e}), test - ({:6.2f}%, '
    fmt += '{:6.2e}) | depth = {:4.1f} | lr = {:5.1e} | time = {:4.1f} sec'

    loss_ave = 0.0
    depth_ave = 0.0
    train_acc = 0.0

    print(KM)
    print('\nTraining Fixed Point Network')

    for epoch in range(num_epochs):
        sleep(1.0)  # slows progress bar so it won't print on multiple lines
        epoch_start_time = time.time()
        tot = len(train_loader)
        with tqdm(total=tot, unit=" batch", leave=False, ascii=True) as tepoch:

            tepoch.set_description("[{:3d}/{:3d}]".format(epoch+1, num_epochs))

            for idx, (d, labels) in enumerate(train_loader):
                labels = labels.to(device)
                d = d.to(device)

                if KM.T.name() == "MNIST_FCN":
                    d = d.view(d.size()[0], 784).to(device)

                ut = torch.zeros((d.size()[0], n)).to(device)
                for i in range(d.size()[0]):
                    ut[i, labels[i].cpu().numpy()] = 1.0
                # --------------------------------------------------------------
                # Forward prop to fixed point
                # --------------------------------------------------------------
                u0 = 0.1 * torch.zeros((d.size()[0], hid_size)).to(device)
                u0[:, 0:n] = 1.0 / float(n)
                u, depth = KM(u0, d)
                depth_ave = 0.95 * depth_ave + 0.05 * depth
                # -------------------------------------------------------------
                # Step with fixed point and then backprop
                # -------------------------------------------------------------
                optimizer.zero_grad()
                y = KM.apply_T(u.float().to(device), d)[:, 0:n]

                output = 0
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
                KM.T.project_weights()
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

        test_loss, test_acc, correct, depth_test = test_statistics(KM,
                                                                   device,
                                                                   test_loader,
                                                                   batch_size,
                                                                   n, loss,
                                                                   hid_size)

        print(fmt.format(epoch+1, num_epochs, train_acc, loss_ave,
                         test_acc, test_loss, depth_ave,
                         optimizer.param_groups[0]['lr'],
                         time.time() - epoch_start_time))
        # ---------------------------------------------------------------------
        # Save weights every 10 epochs
        # ---------------------------------------------------------------------
        if (epoch + 1) % 10 == 0:
            state = {
                'eps': KM.eps_tol,
                'max depth': KM.max_depth,
                'alpha': KM.alpha,
                'T_state_dict': KM.T.state_dict(),
            }
            torch.save(state, 'KM_' + KM.T.name() + '_weights.pth')
            print('Model weights saved to KM_' + KM.T.name() + '_weights.pth')

        lr_scheduler.step()
        epoch_start_time = time.time()
    return KM
