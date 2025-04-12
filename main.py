import random
from itertools import product
from classifier import *
from util import *
import json
import csv

random.seed(24)
np.random.seed(24)
show_graphs = True

inputs, labels = load_data('data/2d.trn.dat')
#print(labels)
mean = np.mean(inputs, axis=1, keepdims=True)
std = np.std(inputs, axis=1, keepdims=True)

(dim, count) = inputs.shape
#print(inputs.shape)
#print(type(inputs[0]))
#print(labels.shape)
 
# # Split training & validation set
indices = list(range(count))
random.shuffle(indices)
#print(indices[:20])

# 80/20 split
split = int(0.8 * count)
train_indices = indices[:split]
val_indices = indices[split:]

train_inputs = inputs[:, train_indices]
train_labels = labels[train_indices]

val_inputs = inputs[:, val_indices]
val_labels = labels[val_indices]

#print(inputs)
#print("Train inputs shape:", train_inputs.shape) # (6400, 2)
#print("Train labels shape:", train_labels.shape) # (6400, 3)

#plot_dots(train_inputs, int2str_labels(train_labels), None, None, None, None, block=False, filename='plots/train_data.png', show=False)
#plot_dots(None, None, None, val_inputs, int2str_labels(val_labels), None, block=False, filename='plots/val_data.png', show=False)


hyperparams = {
    'dim_hid': [10,],
    'alpha': [0.01,],
    'eps': [200,],
    'normalize': [True, False],
    'weight_init': ['normal_dist', 'uniform', 'sparse'],
    'hidden_activation': ['sigmoid', 'tanh', 'relu'],
    'output_activation': ['softmax'],
    'sparsity': [0.1, 0.2],
    'weight_scale': [1.0, 2.0],
    'patience': [None, 10],
    'delta':[0, 0.001]
}

""""
hyperparams = {
    'dim_hid': [10,],
    'alpha': [0.01,],
    'eps': [50,],
    'normalize': [True],
    'weight_init': ['normal_dist',],
    'hidden_activation': ['sigmoid',],
    'output_activation': ['softmax',],
    'sparsity': [0.2],
    'weight_scale': [1.0, ],
    'patience': [10,],
    'delta':[0.01,]
}
"""



best_val_CE = float('inf')
best_val_RE = float('inf')
best_model = None
best_params = None
results = []
seen_configs = set()
best_model_i = None

keys = list(hyperparams.keys())
grid_search_start_time = time.time()

with open("all_model_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    header = ["model_id", ] + list(keys) + ["train_CE", "train_RE", "val_CE", "val_RE"]
    writer.writerow(header)
    for i_model, values in enumerate(product(*hyperparams.values()), start=1):
        hp = dict(zip(keys, values))

        if hp['weight_init'] != 'sparse':
            hp['sparsity'] = None

        if hp['patience'] is None:
            hp['delta'] = None

        hp_key = json.dumps(hp, sort_keys=True)
        if hp_key in seen_configs:
            continue
        seen_configs.add(hp_key)

        use_early_stopping = hp['patience'] is not None
        patience = hp['patience'] if use_early_stopping else 0
        
        x_train = (train_inputs - mean) / std if hp['normalize'] else train_inputs
        x_val = (val_inputs - mean) / std if hp['normalize'] else val_inputs

        print(f"{i_model}. Training with hyperparameters: {hp}")
        model = MLPClassifier(dim_in=x_train.shape[0], dim_hid=hp['dim_hid'], n_classes=np.max(train_labels)+1, weight_init=hp['weight_init'],
                            hidden_activation=hp['hidden_activation'], output_activation=hp['output_activation'], sparsity=hp['sparsity'], weight_scale=hp['weight_scale'])
        
        train_CEs, train_REs, val_CEs = model.train(x_train, train_labels, val_inputs=x_val if use_early_stopping else None, val_labels=val_labels if use_early_stopping else None, alpha=hp['alpha'], eps=hp['eps'],early_stopping=use_early_stopping, patience=patience, delta=hp['delta'], live_plot=False)

        best_epoch_i = hp['eps'] - 1

        if val_CEs is not None:
            best_epoch_i = int(np.argmin(val_CEs))
            #print(f"Best epoch: {best_epoch_i+1}")
            

        train_CE_final = train_CEs[best_epoch_i]
        train_RE_final = train_REs[best_epoch_i]

        val_CE, val_RE = model.test(x_val, val_labels)
        print(f"Val CE: {val_CE * 100:.2f}%, Val RE: {val_RE:.5f}")
        
        row = [i_model, ] + [hp[k] for k in keys] + [train_CE_final, train_RE_final, val_CE, val_RE]
        writer.writerow(row)

        if val_CE < best_val_CE:
            best_val_CE = val_CE
            best_val_RE = val_RE
            best_model = model
            best_params = hp
            best_model_i = i_model

grid_search_elapsed_time = time.time() - grid_search_start_time
print(f"Total grid search time: {grid_search_elapsed_time:.3f} seconds")

with open("best_model_params.json", "w") as f:
    json.dump({best_model_i: best_params}, f, indent=4)


print(f"Best model: {best_model_i} with val CE: {best_val_CE * 100:.2f}% and val RE: {best_val_RE:.5f}")
print(f"Best hyperparameters: {best_params}")


test_inputs, test_labels = load_data('data/2d.tst.dat')
#plot_dots(test_inputs, int2str_labels(test_labels), None, None, None, None, title="test data distribution", block=False, filename='plots/test_data.png', show=show_graphs)

model = best_model
if best_params['normalize']:
    inputs = (inputs - mean) / std
    test_inputs = (test_inputs - mean) / std

train_CEs, train_REs, _ = model.train(inputs, labels, None, None, alpha=best_params['alpha'], eps=best_params['eps'], early_stopping=False, patience=None, live_plot=False)

test_CE, test_RE = model.test(test_inputs, test_labels)

print(f"Test CE: {test_CE * 100:.2f}%, Test RE: {test_RE:.5f}")
_, train_predicted = model.predict(inputs)
train_predicted = int2str_labels(train_predicted)
_, test_predicted  = model.predict(test_inputs)
test_predicted = int2str_labels(test_predicted)

plot_errors(title="Error (%) vs. Time (Training CE)",errors=train_CEs, test_error=test_CE, block=False, filename='plots/train_CE.png', show=show_graphs)

plot_confusion_matrix(test_labels, test_predicted, num_classes=3, block=False, filename='plots/confusion_matrix_test.png', show=show_graphs)


plot_dots(inputs = test_inputs, labels = int2str_labels(test_labels), block=False, filename='plots/test_data.png', show=show_graphs)
plot_dots(inputs = inputs, labels = int2str_labels(labels), predicted = train_predicted, test_inputs=test_inputs, test_labels=int2str_labels(test_labels), test_predicted=test_predicted, block=False, filename='plots/train_data_predicted.png', show=show_graphs)
plot_dots(inputs = None, test_inputs=test_inputs, test_labels=int2str_labels(test_labels), test_predicted=test_predicted, title='Test data only', block=False, filename='plots/test_data_predicted.png', show=show_graphs)
print(len(seen_configs))