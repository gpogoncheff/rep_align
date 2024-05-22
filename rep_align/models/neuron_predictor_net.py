import torch
import torch.nn as nn

class NeuronNet(nn.Module):
    def __init__(self, base_model, activation_layer_id, img_shape=(3,32,32), n_neurons_out=255):
        super(NeuronNet, self).__init__()
        self.base_model = base_model

        self.activation_layer_id = activation_layer_id
        self.activations = {}
        self.hook = self.base_model.get_submodule(self.activation_layer_id)\
                                   .register_forward_hook(self.get_activation(self.activation_layer_id))
        _ = self.base_model(torch.rand((1,*img_shape)))
        _, act_c, _, _ = self.activations[self.activation_layer_id].shape
        self.activations = {}

        self.output_layer = nn.Linear(act_c, n_neurons_out)
        self.elu = nn.ELU()

    def get_activation(self, layer):
        def hook(model, input, output):
            self.activations[layer] = output
        return hook

    def forward(self, x):
        clf_out = self.base_model(x)
        neuron_out = self.activations[self.activation_layer_id]
        neuron_out = torch.mean(neuron_out, dim=(-2,-1))
        neuron_out = self.output_layer(neuron_out)
        neuron_out = self.elu(neuron_out) + 1.
        return clf_out, neuron_out