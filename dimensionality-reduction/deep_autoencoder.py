import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepAutoencoder(nn.Module):
    def __init__(self, models):
        super(DeepAutoencoder, self).__init__()
        encoders = []
        encoder_biases = []
        decoders = []
        decoder_biases = []
        for model in models:
            encoders.append(nn.Parameter(model.W.clone()))
            encoder_biases.append(nn.Parameter(model.h_bias.clone()))
            decoders.append(nn.Parameter(model.W.clone()))
            decoder_biases.append(nn.Parameter(model.v_bias.clone()))
        self.encoders = nn.ParameterList(encoders)
        self.encoder_biases = nn.ParameterList(encoder_biases)
        self.decoders = nn.ParameterList(reversed(decoders))
        self.decoder_biases = nn.ParameterList(reversed(decoder_biases))

    def forward(self, v):
        p_h = self.encode(v)
        return self.decode(p_h)

    def encode(self, v):
        p_v = v
        activation = v
        for i in range(len(self.encoders)):
            W = self.encoders[i]
            h_bias = self.encoder_biases[i]
            activation = torch.mm(p_v, W) + h_bias
            p_v = torch.sigmoid(activation)
        return activation

    def decode(self, h):
        p_h = h
        for i in range(len(self.encoders)):
            W = self.decoders[i]
            v_bias = self.decoder_biases[i]
            activation = torch.mm(p_h, W.t()) + v_bias
            p_h = torch.sigmoid(activation)
        return p_h


class NaiveDeepAutoencoder(nn.Module):
    def __init__(self, layers):
        super(NaiveDeepAutoencoder, self).__init__()
        self.layers = layers
        encoders = []
        decoders = []
        prev_layer = layers[0]
        for layer in layers[1:]:
            encoders.append(
                nn.Linear(in_features=prev_layer, out_features=layer))
            decoders.append(
                nn.Linear(in_features=layer, out_features=prev_layer))
            prev_layer = layer
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))

    def forward(self, x):
        x_encoded = self.encode(x)
        return self.decode(x_encoded)

    def encode(self, x):
        for i, enc in enumerate(self.encoders):
            if i == len(self.encoders) - 1:
                x = enc(x)
            else:
                x = torch.sigmoid(enc(x))
        return x
    
    def decode(self, x):
        for dec in self.decoders:
            x = torch.sigmoid(dec(x))
        return x