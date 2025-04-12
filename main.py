import random
from itertools import product
from classifier import *
from util import *
import json

random.seed(24)

label_map = {'A': 0,
             'B': 1,
             'C': 2}

data = np.genfromtxt('data/2d.trn.dat', dtype=[('x', float), ('y', float), ('label', 'U1')])[1:]
#print(data.shape)
#print(data)

inputs = np.array([[row[0], row[1]] for row in data]).T
labels = np.array([label_map[row[2]] for row in data]).T
mean = np.mean(inputs, axis=1, keepdims=True)
std = np.std(inputs, axis=1, keepdims=True)
#print('mean', mean.shape)
#print('std', std.shape)

(dim, count) = inputs.shape
#print(inputs.shape)
#print(type(inputs[0]))
#print(labels.shape)
 
# # Split training & validation set
indices = list(range(count))
random.shuffle(indices)   

# 80/20 split
split = int(0.8 * count)
train_indices = indices[:split]
val_indices = indices[split:]

train_inputs = inputs[:, train_indices]
train_labels = labels[train_indices]

val_inputs = inputs[:, val_indices]
val_labels = labels[val_indices]

print(inputs)
print("Train inputs shape:", train_inputs.shape) # (6400, 2)
print("Train labels shape:", train_labels.shape) # (6400, 3)

#plot_dots(train_inputs, train_labels, None, None, None, None)

ACTIVATION_FUNCTIONS = ['sigmoid', 'tanh', 'relu', 'linear']

hyperparams = {
    'dim_hid': [2, 3, 4, 5, 10, 15],
    'alpha': [0.01,],
    'eps': [100, 150, 200, 250],
    'normalize': [True, False],
    'weight_init': ['sparse', 'normal_dist', 'uniform'],
    'hidden_activation': ACTIVATION_FUNCTIONS,
    'output_activation': ACTIVATION_FUNCTIONS,
    'sparsity': [0.1, 0.2, 0.3],
    'weight_scale': [0.1, 0.5, 1.0, 2.0],
    'patience': [5, 10, 15, None]
}

best_val_CE = float('inf')
best_model = None
best_params = None
results = []
seen_configs = set()

keys = list(hyperparams.keys())
for i_model, values in enumerate(product(*hyperparams.values())):
    hp = dict(zip(keys, values))

    if hp['weight_init'] != 'sparse':
        hp['sparsity'] = None

    hp_key = json.dumps(hp, sort_keys=True)
    if hp_key in seen_configs:
        continue
    seen_configs.add(hp_key)

    use_early_stopping = hp['patience'] is not None
    patience = hp['patience'] if use_early_stopping else 0
    
    x_train = (train_inputs - mean) / std if hp['normalize'] else train_inputs
    x_val = (val_inputs - mean) / std if hp['normalize'] else val_inputs

    print(f"{i_model+1}. Training with hyperparameters: {hp}")
    model = MLPClassifier(dim_in=x_train.shape[0], dim_hid=hp['dim_hid'], n_classes=np.max(train_labels)+1, weight_init=hp['weight_init'],
                          hidden_activation=hp['hidden_activation'], output_activation=hp['output_activation'], sparsity=hp['sparsity'], weight_scale=hp['weight_scale'])
    
    train_CEs, train_REs = model.train(x_train, train_labels, val_inputs=x_val if use_early_stopping else None, val_labels=val_labels if use_early_stopping else None, alpha=hp['alpha'], eps=hp['eps'],early_stopping=use_early_stopping, patience=patience, live_plot=False)

    val_CE, val_RE = model.test(x_val, val_labels)
    results.append((hp, train_CEs[-1], train_REs[-1], val_CE, val_RE))

    if val_CE < best_val_CE:
        best_val_CE = val_CE
        best_model = model
        best_params = hp

print(results)