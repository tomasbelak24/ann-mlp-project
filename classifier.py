# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Stefan Pócoš, Iveta Bečková 2017-2025

from mlp import *
from util import *


class MLPClassifier(MLP):
    def __init__(self, dim_in, dim_hid, n_classes, weight_init='normal_dist', hidden_activation='sigmoid', output_activation='softmax', sparsity=None, weight_scale=1.0):
        self.n_classes = n_classes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        super().__init__(dim_in, dim_hid, dim_out=n_classes, weight_init=weight_init, sparsity=sparsity, weight_scale=weight_scale)

    def error(self, targets, outputs):
        """
        Cost / loss / error function
        """

        return np.sum((targets - outputs)**2, axis=0)

    def get_function(self, activation, derivative=False):
            if activation == 'sigmoid':
                s = lambda x: 1 / (1 + np.exp(-x))
                if derivative:
                    return lambda x: s(x) * (1 - s(x))
                else:
                    return lambda x: np.where(x >= 0, s(x), np.exp(x) / (1 + np.exp(x)))
            elif activation == 'tanh':
                if derivative:
                    return lambda x: 1 - x**2
                else:
                    return lambda x: np.tanh(x)
            elif activation == 'relu':
                if derivative:
                    return lambda x: (x > 0).astype(float)
                else:
                    return lambda x: np.maximum(0, x)
            elif activation == 'softmax':
                if derivative:
                    raise NotImplementedError("Use softmax with cross-entropy combo instead.")
                else:
                    return lambda x: np.exp(x) / np.sum(np.exp(x), axis=0)
            elif activation == 'linear':
                if derivative:
                    return lambda x: 1
                else:
                    return lambda x: x
            else:
                raise ValueError(f"Unknown activation function: {activation}")
    
    # @override
    def f_hid(self, x):
        return self.get_function(self.hidden_activation)(x)

    # @override
    def df_hid(self, x):
        return self.get_function(self.hidden_activation, derivative=True)(x)

    # @override
    def f_out(self, x):
        return self.get_function(self.output_activation)(x)
      
    # @override
    def df_out(self, x):
        return self.get_function(self.output_activation, derivative=True)(x)
        
    def predict(self, inputs):
        """
        Prediction = forward pass
        """

        # If self.forward() can process only one input at a time
        #outputs = np.stack([self.forward(x)[-1] for x in inputs.T]).T
        # # If self.forward() can take a whole batch
        *_, outputs = self.forward(inputs)
        return outputs, onehot_decode(outputs)

    def test(self, inputs, labels):
        """
        Test model: forward pass on given inputs, and compute errors
        """
        targets = onehot_encode(labels, self.n_classes)
        outputs, predicted = self.predict(inputs)
        CE = np.mean(predicted != labels)
        RE = np.mean(self.error(targets, outputs))
        return CE, RE

    def train(self, inputs, labels, val_inputs = None, val_labels = None, alpha=0.1, eps=100, early_stopping = False, patience = 10, live_plot=False, live_plot_interval=10):
        """
        Training of the classifier
        inputs: matrix of input vectors (each column is one input vector)
        labels: vector of labels (each item is one class label)
        alpha: learning rate
        eps: number of episodes
        live_plot: plot errors and data during training
        live_plot_interval: refresh live plot every N episodes
        """
        (_, count) = inputs.shape
        targets = onehot_encode(labels, self.n_classes)

        if live_plot:
            interactive_on()

        CEs = []
        REs = []
        val_CEs = []

        best_val_CE = float('inf')
        best_weights = None
        no_improve_count = 0

        for ep in range(eps):
            CE = 0
            RE = 0

            for idx in np.random.permutation(count):
                x = inputs[:, idx]
                d = targets[:, idx]

                a, h, b, y = self.forward(x)
                dW_hid, dW_out = self.backward(x, a, h, b, y, d)

                self.W_hid += alpha * dW_hid
                self.W_out += alpha * dW_out

                CE += labels[idx] != onehot_decode(y)
                RE += self.error(d, y)

            CE /= count
            RE /= count
            CEs.append(CE)
            REs.append(RE)
            if (ep+1) % 5 == 0: print('Epoch {:3d}/{}, CE = {:6.2%}, RE = {:.5f}'.format(ep+1, eps, CE, RE))

            if live_plot and ((ep+1) % live_plot_interval == 0):
                _, predicted = self.predict(inputs)
                plot_dots(inputs, labels, predicted, block=False)
                plot_both_errors(CEs, REs, block=False)
                plot_areas(self, inputs, block=False)
                redraw()

            if early_stopping:
                val_CE, _ = self.test(val_inputs, val_labels)
                val_CEs.append(val_CE)

                if val_CE < best_val_CE:
                    best_val_CE = val_CE
                    no_improve_count = 0
                    best_weights = (self.W_hid.copy(), self.W_out.copy())
                else:
                    no_improve_count += 1
            
                if no_improve_count >= patience:
                    print(f"stopped early at epoch {ep+1} (patience = {patience} epochs)")
                    break

        
        if early_stopping and best_weights is not None:
            self.W_hid, self.W_out = best_weights

        if live_plot:
            interactive_off()

        print()

        return CEs, REs, val_CEs if early_stopping else None
