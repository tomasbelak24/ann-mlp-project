import random
from itertools import product
from classifier import *
from util import *
import json
import csv

# set random seed for reproducibility
random.seed(24)
np.random.seed(24)
show_graphs = True

inputs, labels = load_data('data/2d.trn.dat')
#plot_dots(inputs = inputs, labels = int2str_labels(labels), block=False, filename='plots/train_data.png', show=show_graphs)
mean = np.mean(inputs, axis=1, keepdims=True)
std = np.std(inputs, axis=1, keepdims=True)

(dim, count) = inputs.shape
 
# Split training & validation set
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

# First search with hyperparameters that could have big impact on the model
hyperparams_part1 = {
    'dim_hid': [5, 10, 20, 50],
    'alpha': [0.01, 0.005, 0.001],
    'eps': [50, 150, 250],
    'normalize': [True,],
    'early_stopping': [{'stop-early': True, 'patience': 15, 'delta': 0}],
    'hidden_activation': ['sigmoid', 'tanh', 'relu'],
    'output_activation': ['softmax']
}


# Experimental hyperparameters for the second search to fine tune the model
early_stopping_options = [{'stop-early': True, 'patience': 15, 'delta': 0}, {'stop-early': True, 'patience': 15, 'delta': 0.001},{'stop-early': False}]
lr_schedule_options = [{'decay': 'exponential_decay', 'params': {'decay_rate': 0.01}}, {'decay': 'step_decay', 'params': {'drop': 0.8, 'epochs_drop': 15}}, {'decay': None}]


hyperparams_part2 = {
    'lr_schedule': lr_schedule_options,
    'weight_init': ['normal_dist', 'uniform', 'sparse'],
    'sparsity': [0.1, 0.2],
    'weight_scale': [1.0, 2.0],
    'early_stopping': early_stopping_options
}


# hyperparams for quick testing
""" hyperparams_part1 = {
    'dim_hid': [10,],
    'alpha': [0.01,],
    'eps': [20,],
    'normalize': [True,],
    'hidden_activation': ['sigmoid'],
    'output_activation': ['softmax']
}

hyperparams_part2 = {
    'lr_schedule': [{'decay': 'exponential_decay', 'params': {'decay_rate': 0.01}}],
    'weight_init': ['normal_dist',],
    'sparsity': [0.1,],
    'weight_scale': [1.0,],
    'early_stopping': early_stopping_options,
} """


# Function to perform grid search
# logs results to a CSV file
# returns the best model and its parameters

def perform_grid_search(hyperparams, fixed_params=None, filename="model_results.csv"):
    best_val_CE = float('inf')
    best_model = None
    best_params = None
    seen_configs = set() # to avoid duplicate configurations


    keys = list(hyperparams.keys())
    all_keys = list(fixed_params.keys()) + list(hyperparams.keys()) if fixed_params else list(keys)

    header = ["Model Index"] + all_keys + ["Train CE", "Train RE", "Val CE", "Val RE", "Duration"]

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for i_model, values in enumerate(product(*hyperparams.values()), start=1):

            hp = dict(zip(keys, values))
            if fixed_params:
                hp.update(fixed_params)

            # Remove sparsity if weight_init is not sparse
            if hp.get('weight_init') != 'sparse':
                try:
                    del hp['sparsity']
                except KeyError:
                    pass
            
            # creates json out of the hyperparameters to use as a key for the seen_configs set
            # this is to avoid duplicate configurations
            hp_key = json.dumps(hp, sort_keys=True)
            if hp_key in seen_configs:
                continue
            seen_configs.add(hp_key)


            # Normalize inputs if wanted
            x_train = (train_inputs - mean) / std if hp['normalize'] else train_inputs
            x_val = (val_inputs - mean) / std if hp['normalize'] else val_inputs

            print(f"{i_model}. Training with hyperparameters: {hp}")
            
            #Model initialization
            model = MLPClassifier(
                dim_in=x_train.shape[0],
                dim_hid=hp['dim_hid'],
                n_classes=np.max(train_labels) + 1,
                weight_init=hp.get('weight_init', 'normal_dist'),
                hidden_activation=hp['hidden_activation'],
                output_activation=hp['output_activation'],
                sparsity=hp.get('sparsity'),
                weight_scale=hp.get('weight_scale', 1.0)
            )

            # reading lr_schedule and early_stopping from hyperparameters
            lr_schedule = hp.get('lr_schedule', {'decay': None})
            use_early_stopping = hp.get('early_stopping', {}).get('stop-early', False)

            # train the model with the given hyperparameters
            train_CEs, train_REs, val_CEs, duration = model.train(
                x_train, train_labels,
                val_inputs=x_val if use_early_stopping else None,
                val_labels=val_labels if use_early_stopping else None,
                alpha=hp['alpha'], eps=hp['eps'],
                early_stopping=hp.get('early_stopping', {'stop-early': False}), lr_schedule=lr_schedule,
                live_plot=False
            )

            best_epoch_i = hp['eps'] - 1
            
            if val_CEs is not None:
                #improvements = [val_CEs[i] - val_CEs[i + 1] for i in range(len(val_CEs) - 1)]
                #print(improvements)
                #plot_val_errors(val_CEs, f"{i_model}.model validation error vs time", block=False)
                best_epoch_i = int(np.argmin(val_CEs))

            train_CE_final = train_CEs[best_epoch_i]
            train_RE_final = train_REs[best_epoch_i]

            val_CE, val_RE = model.test(x_val, val_labels)
            print(f"Val CE: {val_CE * 100:.2f}%, Val RE: {val_RE:.5f}\n")

            row = [i_model, ] + [hp.get(k) for k in all_keys] + [train_CE_final, train_RE_final, val_CE, val_RE, duration]
            writer.writerow(row)

            if val_CE < best_val_CE:
                best_val_CE = val_CE
                best_model = model
                best_params = hp

    return best_model, best_params

# Perform the first grid search
print("Starting first grid search...\n")
best_model_part1, best_params_part1 = perform_grid_search(hyperparams_part1, filename="results/model_results_part1.csv")

try:
    del best_params_part1['early_stopping']
except KeyError:
    pass

# Perform the second grid search using the best parameters from the first search
print("Starting second grid search...\n")
best_model_part2, best_params_part2 = perform_grid_search(hyperparams_part2, fixed_params=best_params_part1, filename="results/model_results_part2.csv")

print("Grid search completed.\n")

# Save the best parameters to json
with open("best_model_params.json", "w") as f:
    json.dump({"part1": best_params_part1, "part2": best_params_part2}, f, indent=4)

#print(f"Best parameters from part 1: {best_params_part1}")
#print(f"Best parameters from part 2: {best_params_part2}")


# Testing the best model on the test set
test_inputs, test_labels = load_data('data/2d.tst.dat')

model = best_model_part2
if best_params_part2['normalize']:
    inputs = (inputs - mean) / std
    test_inputs = (test_inputs - mean) / std

lr_schedule = best_params_part2.get('lr_schedule', {'decay': None})

print("Training final model...")
train_CEs, train_REs, _, duration = model.train(inputs, labels, None, None, alpha=best_params_part2['alpha'], eps=best_params_part2['eps'], early_stopping={'stop-early': False}, lr_schedule=lr_schedule, live_plot=False)

plot_both_errors(train_CEs, train_REs,block=False)

test_CE, test_RE = model.test(test_inputs, test_labels)
print(f"Test CE: {test_CE * 100:.2f}%, Test RE: {test_RE:.5f}")

_, train_predicted = model.predict(inputs)
train_predicted = int2str_labels(train_predicted)
_, test_predicted  = model.predict(test_inputs)
test_predicted = int2str_labels(test_predicted)


plot_confusion_matrix(test_labels, test_predicted, num_classes=3, block=False, filename='plots/confusion_matrix_test.png', show=show_graphs)
plot_dots(inputs = test_inputs, labels = int2str_labels(test_labels), block=False, filename='plots/test_data.png', show=show_graphs)
plot_dots(inputs = inputs, labels = int2str_labels(labels), predicted = train_predicted, test_inputs=test_inputs, test_labels=int2str_labels(test_labels), test_predicted=test_predicted, block=False, filename='plots/train_data_predicted.png', show=show_graphs)
plot_dots(inputs = None, test_inputs=test_inputs, test_labels=int2str_labels(test_labels), test_predicted=test_predicted, title='Test data only', block=False, filename='plots/test_data_predicted.png', show=show_graphs)
plot_areas(model, test_inputs, test_labels, 'Decision Boundaries of final model', filename='plots/decision_boundary.png', show=show_graphs)