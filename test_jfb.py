import torch

def test_addition():
    assert 1 + 1 == 2


def v_JJT_matvec(v, u, Ru):
    # inputs:
    # v = vector to be multiplied by JJT
    # u = fixed point vector u (requires grad)
    # Ru = R applied to u (requires grad)

    # assumes one rhs:
    # x (n_samples, n_dim, n_rhs) -> (n_samples, n_dim)

    v = v.view(Ru.shape)  # reshape to filter space
    v.requires_grad = True

    # compute v*J = v*(I - dRdu)
    v_dRdu = torch.autograd.grad(outputs=Ru, inputs=u,
                                 grad_outputs=v,
                                 retain_graph=True,
                                 create_graph=True,
                                 only_inputs=True)[0]
    v_J = v - v_dRdu

    # compute v_JJT
    v_JJT = torch.autograd.grad(outputs=v_J, inputs=v,
                                grad_outputs=v_J,
                                retain_graph=True,
                                create_graph=True,
                                only_inputs=True)[0]

    v = v.detach()
    v_J = v_J.detach()
    Amv = v_JJT.detach()
    return Amv


def test_symmetry_of_Jacobians():

    n_features = 100
    u = torch.randn(1, n_features)
    u.requires_grad = True
    fc = torch.nn.Linear(n_features, n_features)
    relu = torch.nn.ReLU()
    Ru = relu(fc(u))

    JJT_mat = torch.zeros(n_features, n_features)
    for i in range(n_features):
        temp_vec = torch.zeros(n_features)
        temp_vec[i] = 1.0

        JJT_mat[i, :] = v_JJT_matvec(temp_vec, u, Ru)

    assert(torch.norm(JJT_mat - JJT_mat.transpose(1, 0)) < 1e-6)

# XXX - Symmmetric Jacobian
# XXX - Fixed Point Error
# XXX - Relative Error of CG vs Explicit
# XXX - Dimension size input to networks
