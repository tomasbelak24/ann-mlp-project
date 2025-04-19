import json
import numpy as np
from classifier import MLPClassifier
import random
from util import load_data, onehot_encode, int2str_labels, plot_errors, plot_dots, plot_confusion_matrix, plot_both_errors, plot_areas

random.seed(24)
np.random.seed(24)

# Load best parameters
with open("best_model_params.json", "r") as f:
    best_params = json.load(f)["part2"]

# Load and optionally normalize training data
inputs, labels = load_data('data/2d.trn.dat')
mean = np.mean(inputs, axis=1, keepdims=True)
std = np.std(inputs, axis=1, keepdims=True)

if best_params.get("normalize", False):
    inputs = (inputs - mean) / std

# Create and train the model
model = MLPClassifier(
    dim_in=inputs.shape[0],
    dim_hid=best_params["dim_hid"],
    n_classes=np.max(labels) + 1,
    weight_init=best_params["weight_init"],
    weight_scale=best_params["weight_scale"],
    hidden_activation=best_params["hidden_activation"],
    output_activation=best_params["output_activation"]
)

print(f"Training with hyperparameters: {best_params}")
train_CEs, train_REs, _, duration = model.train(
    inputs, labels,
    alpha=best_params["alpha"],
    eps=best_params["eps"],
    early_stopping={"stop-early": False},
    lr_schedule=best_params.get("lr_schedule", {"decay": None}),
    live_plot=False
)
#print(train_CEs)
#print(train_REs)


plot_both_errors(
    train_CEs, train_REs,
    block=False,
)
# Plot error vs. time
#plot_errors(errors=train_CEs, title="Error vs. Time (Cross-Entropy)", block=False, filename="/plots/train_CE_maybe.png")

# Predict and plot outputs in 2D
_, predicted = model.predict(inputs)
predicted_str = int2str_labels(predicted)
plot_dots(inputs, labels=int2str_labels(labels), predicted=predicted_str, title="2D Output - Training Data", block=False)

# Plot confusion matrix
plot_confusion_matrix(labels, predicted_str, num_classes=3, block=False)
test_inputs, test_labels = load_data('data/2d.tst.dat')


if best_params['normalize']:
    test_inputs = (test_inputs - mean) / std
plot_areas(model, test_inputs, test_labels)
