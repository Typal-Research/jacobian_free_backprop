import torch
from prettytable import PrettyTable
from time import sleep
import time
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data         import Dataset, TensorDataset, DataLoader
from torchvision              import datasets
import numpy as np
from BatchCG import cg_batch


def get_stats(net, test_loader, criterion, num_classes, eps, max_depth):
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

            y = net(d_test, eps=eps, max_depth=max_depth)

            if str(criterion) == "MSELoss()":
                test_loss += batch_size * criterion(y.double(), ut.double()).item()
            elif str(criterion) == "CrossEntropyLoss()":
                test_loss += batch_size * criterion(y, labels).item()
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


def train_class_net(net, max_epochs, lr_scheduler, train_loader,
                    test_loader, optimizer, criterion,
                    num_classes, eps, max_depth, save_dir='./'):

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

    for epoch in range(max_epochs):
        sleep(0.5)  # slows progress bar so it won't print on multiple lines
        loss_ave        = 0.0
        epoch_start_time = time.time()
        tot = len(train_loader)
        with tqdm(total=tot, unit=" batch", leave=False, ascii=True) as tepoch:

            tepoch.set_description("[{:3d}/{:3d}]".format(epoch+1, max_epochs))

            for _, (d, labels) in enumerate(train_loader):
                labels = labels.to(net.device())
                d = d.to(net.device())

                batch_size = d.shape[0]

                if net.name() == "MNIST_FCN":
                    d = d.view(d.size()[0], 784).to(net.device())

                # -------------------------------------------------------------
                # Apply network to get fixed point and then backprop
                # -------------------------------------------------------------
                optimizer.zero_grad()
                y = net(d, eps=eps, max_depth=max_depth)

                depth_ave = 0.99 * depth_ave + 0.01 * net.depth
                output = None
                if str(criterion) == "MSELoss()":
                    
                    ut = torch.zeros((d.size()[0], num_classes)).to(net.device())
                    for i in range(d.size()[0]):
                        ut[i, labels[i].cpu().numpy()] = 1.0

                    output = criterion(y.double(), ut.double())
                elif str(criterion) == "CrossEntropyLoss()":
                    output = criterion(y, labels)
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
                                                 criterion,
                                                 num_classes,
                                                 eps,
                                                 max_depth)

        test_loss_hist.append(test_loss)
        test_acc_hist.append(test_acc)
        train_loss_hist.append(loss_ave)
        train_acc_hist.append(train_acc)

        epoch_end_time = time.time()
        time_epoch = epoch_end_time - epoch_start_time

        time_hist.append(time_epoch)
        total_time += time_epoch 

        print(fmt.format(epoch+1, max_epochs, train_acc, loss_ave,
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
            file_name = save_dir + net.name() + '_weights.pth'
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
                'lr_scheduler': lr_scheduler,
                'time_hist': time_hist,
                'eps': eps,
            }
            file_name = save_dir + net.name() + '_history.pth'
            torch.save(state, file_name)
            print('Training history saved to ' + file_name)

        lr_scheduler.step()
        epoch_start_time = time.time()
    return net

def mnist_loaders(train_batch_size, test_batch_size=None):
    if test_batch_size is None:
        test_batch_size = train_batch_size

    train_loader = train_loader = torch.utils.data.DataLoader(
                        datasets.MNIST('data',
                                    train=True,
                                    download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                    ])),
                        batch_size=train_batch_size,
                        shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(
                        datasets.MNIST('data',
                                    train=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                    ])),
                        batch_size=test_batch_size,
                        shuffle=False)
    return train_loader, test_loader


def svhn_loaders(train_batch_size, test_batch_size=None):
    if test_batch_size is None:
        test_batch_size = train_batch_size

    normalize = transforms.Normalize(mean=[0.4377, 0.4438, 0.4728],
                                      std=[0.1980, 0.2010, 0.1970])
    train_loader = torch.utils.data.DataLoader(
            datasets.SVHN(
                root='data', split='train', download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    normalize
                ]),
            ),
            batch_size=train_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.SVHN(
            root='data', split='test', download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])),
        batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader

def cifar_loaders(train_batch_size, test_batch_size=None, augment=True):
    if test_batch_size is None:
        test_batch_size = train_batch_size
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if augment:
        transforms_list = [transforms.RandomHorizontalFlip(),                                                                    
                           transforms.ToTensor(),
                           normalize,
                           transforms.RandomCrop(32, 2, fill=0.449),   # 3 
                           transforms.RandomErasing(p=0.95, scale=(0.1, 0.25), 
                                                    ratio=(0.2, 5.0), 
                                                    value=[0.485, 0.456, 0.406])  
                           ]
    else:
        transforms_list = [transforms.ToTensor(),
                            normalize]
    train_dataset = datasets.CIFAR10('data',
                                train=True,
                                download=True,
                                transform=transforms.Compose(transforms_list))
    test_dataset = datasets.CIFAR10('data',
                                train=False,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    normalize
                                ]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size,
                                                shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size,
                                                shuffle=False, pin_memory=True)
    return train_loader, test_loader


#-------------------------------------------------------------------------------
# Jacobian-based functions
#-------------------------------------------------------------------------------
def compute_fixed_point(T, Qd, max_depth, device, eps=1e-4):

    depth = 0.0
    u = torch.zeros(Qd.shape, device=T.device())
    u_prev = np.Inf*torch.ones(u.shape, device=T.device()) 
    indices = np.array(range(len(u[:, 0])))

    # approximately normalize weights by lipschitz constant before computing fixed point
    T.normalize_lip_const(u, Qd)
    
    with torch.no_grad():
        all_samp_conv = False
        while not all_samp_conv and depth < max_depth:
            u_prev = u.clone()
            u = T.latent_space_forward(u, Qd)
            depth += 1.0
            all_samp_conv = torch.max(torch.norm(u - u_prev, dim=1)) <= eps
            
    return u.detach(), depth

def train_Jacobian_based_net(net, max_epochs, lr_scheduler, train_loader,
                    test_loader, optimizer, criterion,
                    num_classes, eps, max_depth, save_dir='./', JTJ_shift=0.0):

    avg_time      = 0.0         
    total_time    = 0.0
    time_hist     = []
    n_Umatvecs    = []           
    max_iter_cg   = max_depth
    tol_cg        = eps

    depth_ave     = 0.0
    best_test_acc = 0.0
    train_acc     = 0.0

    test_loss_hist   = [] # test loss history array
    test_acc_hist    = [] # test accuracy history array
    depth_test_hist  = [] # test depths history array
    train_loss_hist  = [] # train loss history array
    train_acc_hist   = [] # train accuracy history array

    fmt        = '[{:4d}/{:4d}]: train acc = {:5.2f}% | train_loss = {:7.3e} | ' 
    fmt       += ' test acc = {:5.2f}% | test loss = {:7.3e} | '
    fmt       += 'depth = {:5.1f} | lr = {:5.1e} | time = {:4.1f} sec | n_Umatvecs = {:4d} | cg = {:7.3e}'
    print(net)                 # display Tnet configuration
    print(model_params(net))   # display Tnet parameters
    print('\nTraining Jacobian-based Network')

    for epoch in range(max_epochs): 

        sleep(0.5)  # slows progress bar so it won't print on multiple lines
        tot = len(train_loader)
        temp_n_Umatvecs = 0
        cg_iters    = 0
        start_time_epoch = time.time() 
        temp_max_depth = 0
        loss_ave = 0.0
        with tqdm(total=tot, unit=" batch", leave=False, ascii=True) as tepoch:

            tepoch.set_description("[{:3d}/{:3d}]".format(epoch+1, max_epochs))
          
            for idx, (d, labels) in enumerate(train_loader):         
                labels  = labels.to(net.device()); d = d.to(net.device())

                #-----------------------------------------------------------------------
                # Find Fixed Point
                #----------------------------------------------------------------------- 
                train_batch_size = d.shape[0] # re-define if batch size changes
                # u0 = torch.zeros((train_batch_size, lat_dim)).to(device)
                with torch.no_grad():
                    Qd = net.data_space_forward(d)
                    u, depth = compute_fixed_point(net, Qd, max_depth, net.device(), eps=eps)

                    depth_ave = 0.99 * depth_ave + 0.01 * net.depth

                    temp_max_depth = max(depth, temp_max_depth)

                #-----------------------------------------------------------------------
                # Jacobian_Based Backprop 
                #-----------------------------------------------------------------------  
                net.train()
                optimizer.zero_grad() # Initialize gradient to zero

                # compute output for backprop
                u.requires_grad=True
                Qd = net.data_space_forward(d)

                Ru = net.latent_space_forward(u, Qd); 
                S_Ru = net.map_latent_to_inference(Ru)
                loss  = criterion(S_Ru, labels)
                train_loss = loss.detach().cpu().numpy() * train_batch_size
                loss_ave += train_loss

                # compute rhs = J * dldu
                dldu    = torch.autograd.grad(outputs = loss, inputs = Ru, 
                                            retain_graph=True, create_graph = True, only_inputs=True)[0];
                dldu    = dldu # note: dldu here = dS/du * dl/dS

                #-----------------------------------------------------------------------
                # trick for computing J * (JTv): # take take derivative d(JTv)/dv * JTv = J * JTv
                #-----------------------------------------------------------------------
                # compute dldu_JT:
                dldu_dRduT = torch.autograd.grad(outputs=Ru, inputs = u_fixed_pt, grad_outputs=dldu, retain_graph=True, create_graph = True, only_inputs=True)[0];
                dldu_JT = dldu - dldu_dRduT 

                # compute J * dldu: take derivative of d(JT*v)/v * v = J*v
                dldu_J = torch.autograd.grad(outputs = dldu_JT, inputs = dldu, grad_outputs = dldu, retain_graph=True, create_graph = True, only_inputs=True)[0];
                rhs = dldu_J

                rhs = rhs.detach()
                rhs = rhs.view(train_batch_size, -1) # vectorize channels (when R is a CNN)
                rhs = rhs.unsqueeze(2) # unsqueeze for number of rhs. CG requires it to have dimensions n_samples x n_features x n_rh

                #-----------------------------------------------------------------------
                # Define JTJ matvec function
                #-----------------------------------------------------------------------
                def v_JJT_matvec(v, u=u, Ru=Ru):
                    # inputs:
                    # v = vector to be multiplied by U = I - alpha*DS - (1-alpha)*DT) requires grad
                    # u = fixed point vector u (should be untracked and vectorized!) requires grad
                    # Ru = R applied to u (requires grad)

                    # assumes one rhs: x (n_samples, n_dim, n_rhs) -> (n_samples, n_dim)

                    v = v.squeeze(2)      # squeeze number of RHS
                    v = v.view(Ru.shape)  # reshape to filter space 
                    v.requires_grad=True

                    # compute v*J = v*(I - dRdu)
                    # trick for computing J * (v): # take take derivative d(JTv)/dv * v = J * v
                    v_dRduT = torch.autograd.grad(outputs = Ru, inputs = u, grad_outputs = v, retain_graph=True, create_graph = True, only_inputs=True)[0]
                    v_dRdu  = torch.autograd.grad(outputs = v_dRduT, inputs = v, grad_outputs = v, retain_graph=True, create_graph = True, only_inputs=True)[0]
                    v_J      = v - v_dRdu

                    # compute v_JJT
                    v_JJT     = torch.autograd.grad(outputs = v_J, inputs = v, grad_outputs= v_J, retain_graph=True, create_graph = True, only_inputs=True)[0]

                    v = v.detach()
                    v_J = v_J.detach()
                    Amv = v_JJT.detach() 
                    Amv = Amv.view(Ru.shape[0], -1)
                    Amv = Amv.unsqueeze(2).detach()
                    return Amv 

                JJTinv_rhs, info = cg_batch(v_JJT_matvec, rhs, M_bmm=None, X0=None, rtol=0, atol=tol_cg, maxiter=max_iter_cg, verbose=False)
                JJTinv_rhs = JJTinv_rhs.squeeze(2) # JTJinv_v has size (batch_size x n_hidden_features), n_rhs is squeezed
                JJTinv_rhs = JJTinv_rhs.view(Ru.shape)

                temp_n_Umatvecs += info['niter'] * train_batch_size
                cg_iters    += info['niter']


                if info['optimal'] == True:
                    # avoid updating "bad batches", i.e., update only when CG converges

                    # compute dTdtheta
                    # Ru = Ru.view(train_batch_size, -1) # reshape in case Ru is a CNN
                    # computes v_JTJinv_dRdTheta = dSdu * dldS * Jinv * dRdTheta
                    u.requires_grad=False
                    Ru.backward(JTJinv_rhs)

                    S_Ru = net.map_latent_to_inference(Ru.detach())
                    loss  = criterion(S_Ru, labels)
                    loss.backward()

                    u.requires_grad=False

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
                                    depth="{:5.1f}".format(temp_max_depth),
                                    cgiters="{:5.1f}".format(info['niter']))
            
        loss_ave /= len(train_loader.dataset)

        # update optimization scheduler
        lr_scheduler.step()

        # compute test loss and accuracy
        test_loss, test_acc, correct = get_stats(net, test_loader, criterion, 10, eps, max_depth)
        # test_loss, test_acc, correct = get_stats_Jacobian(net, test_loader, criterion, eps, max_depth)

        end_time_epoch = time.time()
        time_epoch = end_time_epoch - start_time_epoch

        #---------------------------------------------------------------------------
        # Compute costs and statistics
        #---------------------------------------------------------------------------
        time_hist.append(time_epoch)
        total_time += time_epoch 
        avg_time /= total_time/(epoch+1)
        n_Umatvecs.append(temp_n_Umatvecs)

        test_loss_hist.append(test_loss)
        test_acc_hist.append(test_acc)
        train_loss_hist.append(loss_ave)
        train_acc_hist.append(train_acc)
        depth_test_hist.append(net.depth)

        #---------------------------------------------------------------------------
        # Print outputs to console
        #---------------------------------------------------------------------------

        print(fmt.format(epoch+1, max_epochs, train_acc, loss_ave,
                        test_acc, test_loss, temp_max_depth,
                        optimizer.param_groups[0]['lr'],
                        time_epoch, temp_n_Umatvecs, cg_iters))

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
            file_name = save_dir + net.name() + '_weights.pth'
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
                'net_state_dict': net.state_dict(),
                'test_loss_hist': test_loss_hist,
                'test_acc_hist': test_acc_hist,
                'depth_test_hist': depth_test_hist
            }
            file_name = save_dir  + net.name() + '_history.pth'
            torch.save(state, file_name)
            print('Training history saved to ' + file_name)

    return net      


def train_Neumann_FPN_net(net, max_epochs, lr_scheduler, train_loader,
                    test_loader, optimizer, criterion,
                    num_classes, eps, max_depth, save_dir='./', neumann_order=0):

    avg_time      = 0.0         
    total_time    = 0.0
    time_hist     = []
    n_Umatvecs    = []           

    depth_ave     = 0.0
    best_test_acc = 0.0
    train_acc     = 0.0

    test_loss_hist   = [] # test loss history array
    test_acc_hist    = [] # test accuracy history array
    depth_test_hist  = [] # test depths history array
    train_loss_hist  = [] # train loss history array
    train_acc_hist   = [] # train accuracy history array

    fmt        = '[{:4d}/{:4d}]: train acc = {:5.2f}% | train_loss = {:7.3e} | ' 
    fmt       += ' test acc = {:5.2f}% | test loss = {:7.3e} | '
    fmt       += 'depth = {:5.1f} | lr = {:5.1e} | time = {:4.1f} sec | n_Umatvecs = {:4d}'
    print(net)                 # display Tnet configuration
    print(model_params(net))   # display Tnet parameters
    print('\nTraining Neumann-based Network')

    for epoch in range(max_epochs): 

        sleep(0.5)  # slows progress bar so it won't print on multiple lines
        tot = len(train_loader)
        temp_n_Umatvecs = 0
        cg_iters    = 0
        start_time_epoch = time.time() 
        temp_max_depth = 0
        loss_ave = 0.0
        with tqdm(total=tot, unit=" batch", leave=False, ascii=True) as tepoch:

            tepoch.set_description("[{:3d}/{:3d}]".format(epoch+1, max_epochs))
          
            for idx, (d, labels) in enumerate(train_loader):         
                labels  = labels.to(net.device()); d = d.to(net.device())

                #-----------------------------------------------------------------------
                # Find Fixed Point
                #----------------------------------------------------------------------- 
                train_batch_size = d.shape[0] # re-define if batch size changes

                with torch.no_grad():
                    Qd = net.data_space_forward(d)
                    u, depth = compute_fixed_point(net, Qd, max_depth, net.device(), eps=eps)

                    depth_ave = 0.99 * depth_ave + 0.01 * net.depth

                    temp_max_depth = max(depth, temp_max_depth)

                #-----------------------------------------------------------------------
                # Jacobian_Based Backprop 
                #-----------------------------------------------------------------------  
                net.train()
                optimizer.zero_grad() # Initialize gradient to zero

                # compute output for backprop
                u.requires_grad=True
                Qd = net.data_space_forward(d)

                Ru = net.latent_space_forward(u, Qd); 
                S_Ru = net.map_latent_to_inference(Ru)
                loss  = criterion(S_Ru, labels)
                train_loss = loss.detach().cpu().numpy() * train_batch_size
                loss_ave += train_loss

                dldS_dSdu    = torch.autograd.grad(outputs = loss, inputs = Ru, 
                            retain_graph=True, create_graph = True, only_inputs=True)[0];
                dldS_dSdu    = dldS_dSdu.detach() # note: dldu here = dS/du * dl/dS

                dldS_dSdu_Jinv = dldS_dSdu.clone().detach()
                v_dRdu_k = dldS_dSdu.clone().detach()
            
                # Approximate Jacobian inverse with Neumann series expansion upto neumann_order terms
                for i in range(1, neumann_order+1):
                  # trick for computing dRdu * v
                  # compute dRdu^T * v, then dRdu * v = d( dRdu^T * v )/dv * v
                  v = v_dRdu_k.clone()
                  v.requires_grad = True
                  v_dRdu       = torch.autograd.grad(outputs=Ru, inputs = u, grad_outputs=v, retain_graph=True, create_graph = True, only_inputs=True)[0]
                  v_dRdu_k      = torch.autograd.grad(outputs=v_dRdu, inputs = v, grad_outputs=v.detach(), retain_graph=True, create_graph = True, only_inputs=True)[0].detach()
                  dldS_dSdu_Jinv = dldS_dSdu_Jinv + v_dRdu_k.detach()
            
                temp_n_Umatvecs += int(neumann_order*(neumann_order+1)/2)

                Ru.backward(dldS_dSdu_Jinv)


                # compute dl/dS * dS/dTheta
                # Qd = net.data_space_forward(d)
                Ru = net.latent_space_forward(u, Qd); 
                S_Ru = net.map_latent_to_inference(Ru.detach())
                loss  = criterion(S_Ru, labels)
                loss.backward()

                u.requires_grad=False

                # update net parameters
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
        test_loss, test_acc, correct = get_stats(net, test_loader, criterion, 10, eps, max_depth)
        # test_loss, test_acc, correct = get_stats_Jacobian(net, test_loader, criterion, eps, max_depth)

        end_time_epoch = time.time()
        time_epoch = end_time_epoch - start_time_epoch

        #---------------------------------------------------------------------------
        # Compute costs and statistics
        #---------------------------------------------------------------------------
        time_hist.append(time_epoch)
        total_time += time_epoch 
        avg_time /= total_time/(epoch+1)
        n_Umatvecs.append(temp_n_Umatvecs)

        test_loss_hist.append(test_loss)
        test_acc_hist.append(test_acc)
        train_loss_hist.append(loss_ave)
        train_acc_hist.append(train_acc)
        depth_test_hist.append(net.depth)

        #---------------------------------------------------------------------------
        # Print outputs to console
        #---------------------------------------------------------------------------

        print(fmt.format(epoch+1, max_epochs, train_acc, loss_ave,
                        test_acc, test_loss, temp_max_depth,
                        optimizer.param_groups[0]['lr'],
                        time_epoch, temp_n_Umatvecs, cg_iters))

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
            file_name = save_dir + net.name() + '_weights.pth'
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
                'eps': eps,
                'net_state_dict': net.state_dict(),
                'test_loss_hist': test_loss_hist,
                'test_acc_hist': test_acc_hist,
                'depth_test_hist': depth_test_hist
            }
            file_name = save_dir  + net.name() + '_history.pth'
            torch.save(state, file_name)
            print('Training history saved to ' + file_name)

    return net

