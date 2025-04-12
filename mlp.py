# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Stefan Pócoš, Iveta Bečková 2017-2025

from util import *


class MLP:
    """
    Multi-Layer Perceptron (abstract base class)
    """

    def __init__(self, dim_in, dim_hid, dim_out, weight_init='normal_dist', sparsity=None, weight_scale=1.0):
        """
        Initialize model, set initial weights
        """
        self.dim_in = dim_in
        self.dim_hid = dim_hid
        self.dim_out = dim_out

        if weight_init == 'normal_dist':
            self.W_hid = np.random.randn(dim_hid, dim_in + 1) * weight_scale
            self.W_out = np.random.randn(dim_out, dim_hid + 1) * weight_scale

        elif weight_init == 'uniform':
            self.W_hid = np.random.uniform(-weight_scale, weight_scale, (dim_hid, dim_in + 1))
            self.W_out = np.random.uniform(-weight_scale, weight_scale, (dim_out, dim_hid + 1))

        elif weight_init == 'sparse':
            self.W_hid = np.random.randn(dim_hid, dim_in + 1) * weight_scale
            self.W_out = np.random.randn(dim_out, dim_hid + 1) * weight_scale
            mask_hid = np.random.rand(*self.W_hid.shape) < sparsity
            mask_out = np.random.rand(*self.W_out.shape) < sparsity
            self.W_hid *= mask_hid
            self.W_out *= mask_out

        else:
            raise ValueError(f"Unknown weight initialization type: {weight_init}")


    # Activation functions & derivations
    # (not implemented, to be overridden in derived classes)
    def f_hid(self, x):
        raise NotImplementedError

    def df_hid(self, x):
        raise NotImplementedError

    def f_out(self, x):
        raise NotImplementedError

    def df_out(self, x):
        raise NotImplementedError

    # Back-propagation
    def forward(self, x):
        """
        Forward pass - compute output of network
        x: single input vector (without bias, size=dim_in)
        """
        a = self.W_hid @ add_bias(x)
        h = self.f_hid(a)
        b = self.W_out @ add_bias(h)
        y = self.f_out(b)

        return a, h, b, y

    def backward(self, x, a, h, b, y, d):
        """
        Backprop pass - compute dW for given input and activations
        x: single input vector (without bias, size=dim_in)
        a: net vector on hidden layer (size=dim_hid)
        h: activation of hidden layer (without bias, size=dim_hid)
        b: net vector on output layer (size=dim_out)
        y: output vector of network (size=dim_out)
        d: single target vector (size=dim_out)
        """
        """g_out = np.zeros(self.dim_out)
        for i in range(self.dim_out):
            g_out[i] = (d[i] - y[i]) * self.df_out(b[i])
        g_hid = np.zeros(self.dim_hid)
        for k in range(self.dim_hid):
            g_hid[k] = (np.sum(self.W_out[:, k] * g_out)) * self.df_hid(a[k])

        dW_out = np.outer(g_out, add_bias(h))
        dW_hid = np.outer(g_hid, add_bias(x))"""
        # Compute output layer gradient vector (dim_out,)
        g_out = (d - y) * self.df_out(b)
        
        # Compute hidden layer gradient vector (dim_hid,), ignoring bias weights in W_out
        g_hid = (self.W_out[:, :-1].T @ g_out) * self.df_hid(a)
        
        # Compute weight gradient matrices with outer products
        dW_out = np.outer(g_out, add_bias(h))
        dW_hid = np.outer(g_hid, add_bias(x))

        return dW_hid, dW_out

