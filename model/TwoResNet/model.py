import torch
import torch.nn as nn


from model.HighResNet.model import HighResNet
from model.TwoResNet.cell import GRUCell


class TwoResNet(nn.Module):
    def __init__(self, adj_mx, cluster_handler, HighResNet_kwargs, LowResNet_kwargs, horizon, input_dim):
        super().__init__()
        self.high_resolution_block = HighResNet(
            adj_mx, **HighResNet_kwargs, horizon=horizon, input_dim=input_dim)
        self.low_resolution_block = LowResNet(
            **LowResNet_kwargs, horizon=horizon, input_dim=input_dim)
        self.cluster_handler = cluster_handler

    def encoder(self, inputs, labels=None, rnn_teacher_forcing=lambda: False):
        """
        encoder forward pass on t time steps
        :param inputs: shape (batch_size, input_dim, num_sensors, seq_len)
        :return: encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size, num_sensors)
        """
        rnn_labels = None if labels is None else self.cluster_handler.downscale(
            labels)
        rnn_in = self.cluster_handler.downscale(inputs)

        low_resol_outs = self.low_resolution_block(
            rnn_in, rnn_labels, rnn_teacher_forcing)
        highresnet_enc_hs = self.high_resolution_block.encoder(
            inputs[..., -self.high_resolution_block.seq_len:])
        low_resol_in_last = rnn_in[:, [0], ...][..., [-1]]  # trick to keep dim

        return highresnet_enc_hs, low_resol_outs, low_resol_in_last

    def decoder(self, high_resol_enc_hs, low_resol_outs, low_resol_in_last, labels=None,
                teacher_forcing_p=0):

        batch_size, _, num_sensors = high_resol_enc_hs[0].size()
        go_symbol = high_resol_enc_hs[0].new_zeros(batch_size, 1, num_sensors)
        high_resol_dec_hs = high_resol_enc_hs

        high_resol_outs = []
        supports = self.high_resolution_block._supports

        diff = torch.cat([low_resol_in_last, low_resol_outs],
                         dim=-1).diff(dim=-1)
        diff = self.cluster_handler.upscale(diff)

        high_resol_dec_in = go_symbol

        if labels is not None:
            true_mean = self.cluster_handler.downscale(labels)
            true_diff = torch.cat(
                [low_resol_in_last, true_mean], dim=-1).diff(dim=-1)
            true_diff = self.cluster_handler.upscale(true_diff)

        for t in range(self.high_resolution_block.horizon):

            pass_label = torch.rand_like(high_resol_dec_in) < teacher_forcing_p
            high_resol_dec_in = high_resol_dec_in + diff[..., t] if labels is None else (~pass_label) * (
                high_resol_dec_in + diff[..., t]) + pass_label * (labels[..., t] + true_diff[..., t])

            high_resol_dec_hs = self.high_resolution_block.decoder_model(
                high_resol_dec_in, supports, high_resol_dec_hs)
            high_resol_dec_in = self.high_resolution_block.projector(
                high_resol_dec_hs[-1])
            high_resol_outs.append(high_resol_dec_in)

        high_resol_outs = torch.stack(high_resol_outs, dim=-1)
        return high_resol_outs, low_resol_outs

    def forward(self, inputs, labels=None, teacher_forcing_p=0, LowResNet_tf=lambda: False):
        """
        seq2seq forward pass
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :param labels: shape (horizon, batch_size, num_sensor * output)
        :param batches_seen: batches seen till now
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """

        highresnet_enc_hs, low_resol_outs, low_resol_in_last = self.encoder(
            inputs, labels, LowResNet_tf)
        outputs, out_agg = self.decoder(
            highresnet_enc_hs, low_resol_outs, low_resol_in_last, labels, teacher_forcing_p)
        return outputs, out_agg


class GRUModel(nn.Module):
    def __init__(self, in_dim, hid_dim, num_rnn_layers, dropout):
        super().__init__()
        gru_layers = nn.ModuleList(
            [GRUCell(in_dim, hid_dim, dropout)])
        for _ in range(num_rnn_layers - 1):
            gru_layers.append(
                GRUCell(
                    hid_dim,
                    hid_dim, dropout))
        self.gru = GRU(gru_layers, hid_dim)

    def forward(self, inputs, hidden_state=None):
        hidden_states = self.gru(inputs, hidden_state)
        return hidden_states


class GRU(nn.Module):
    def __init__(self, gru_layers, hid_dim):
        super().__init__()
        self.gru_layers = gru_layers
        self.hid_dim = hid_dim
        self.num_rnn_layers = len(self.gru_layers)

    def forward(self, inputs, hidden_state):
        """
        Encoder forward pass.

        :param inputs: shape (batch_size, input_dim, num_nodes)
        :param hidden_state: (num_layers, batch_size, self.hid_dim, num_nodes)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.hidden_state_size)
                 hidden_state # shape (num_layers, batch_size, self.hid_dim, num_nodes)
                 (lower indices mean lower layers)
        """

        hidden_states = []
        next_hidden_state = inputs
        for hs_layer, gru_layer in zip(hidden_state, self.gru_layers):
            next_hidden_state = gru_layer(
                next_hidden_state, hs_layer)
            hidden_states.append(next_hidden_state)
        # runs in O(num_layers) so not too slow
        return hidden_states


class LowResNet(nn.Module):
    def __init__(self, input_dim, output_dim, num_rnn_units, num_rnn_layers, horizon, seq_len, dropout):
        super().__init__()
        self.encoder_model = GRUModel(
            input_dim, num_rnn_units, num_rnn_layers, dropout)
        self.decoder_model = GRUModel(
            output_dim, num_rnn_units, num_rnn_layers, dropout)
        self.out_dim = output_dim
        self.horizon = horizon
        self.num_rnn_layers = num_rnn_layers
        self.num_rnn_units = num_rnn_units
        self.seq_len = seq_len
        self.projector = nn.Conv1d(num_rnn_units, output_dim, 1)

    def encoder(self, inputs):
        """
        encoder forward pass on t time steps
        :param inputs: shape (batch_size, input_dim, num_sensors, seq_len)
        :return: encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size, num_sensors)
        """
        batch_size, _, num_sensors, _ = inputs.size()
        encoder_hidden_state = [inputs.new_zeros(batch_size,
                                                 self.num_rnn_units,
                                                 num_sensors) for _ in range(self.num_rnn_layers)]

        for seq in range(self.seq_len):
            encoder_hidden_state = self.encoder_model(
                inputs[..., seq], encoder_hidden_state)

        return encoder_hidden_state

    def decoder(self, encoder_hidden_state, labels=None,
                teacher_forcing=lambda: False):
        """
        Decoder forward pass
        :param encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        :param labels: (self.horizon, batch_size, self.num_nodes * self.output_dim) [optional, not exist for inference]
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        batch_size, _, num_sensors = encoder_hidden_state[0].size()
        go_symbol = encoder_hidden_state[0].new_zeros(
            batch_size,
            self.out_dim,
            num_sensors
        )
        decoder_hidden_state = encoder_hidden_state

        decoder_input = go_symbol
        outputs = []
        for t in range(self.horizon):
            decoder_hidden_state = self.decoder_model(
                decoder_input, decoder_hidden_state)
            decoder_input = self.projector(decoder_hidden_state[-1])
            outputs.append(decoder_input)
            if teacher_forcing():
                decoder_input = labels[..., t]
        outputs = torch.stack(outputs, dim=-1)
        return outputs

    def forward(self, inputs, labels=None, teacher_forcing=lambda: False):
        """
        seq2seq forward pass
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :param labels: shape (horizon, batch_size, num_sensor * output)
        :param batches_seen: batches seen till now
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        encoder_hidden_state = self.encoder(inputs)
        outputs = self.decoder(encoder_hidden_state,
                               labels, teacher_forcing)

        return outputs
