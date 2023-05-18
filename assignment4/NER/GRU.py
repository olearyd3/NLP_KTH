import torch
from torch import nn
import numpy as np


class GRUCellV2(nn.Module):
    """
    GRU cell implementation
    """
    def __init__(self, input_size, hidden_size, activation=torch.tanh):
        """
        Initializes a GRU cell

        :param      input_size:      The size of the input layer
        :type       input_size:      int
        :param      hidden_size:     The size of the hidden layer
        :type       hidden_size:     int
        :param      activation:      The activation function for a new gate
        :type       activation:      callable
        """
        super(GRUCellV2, self).__init__()
        self.activation = activation
        # I ADDED THIS IN -- NOT IN OG -- size of hidden layer
        self.hidden_size = hidden_size
        # initialize weights by sampling from a uniform distribution between -K and K
        K = 1 / np.sqrt(hidden_size)
        # weights
        self.w_ih = nn.Parameter(torch.rand(3 * hidden_size, input_size) * 2 * K - K)
        self.w_hh = nn.Parameter(torch.rand(3 * hidden_size, hidden_size) * 2 * K - K)
        self.b_ih = nn.Parameter(torch.rand(3 * hidden_size) * 2 * K - K)
        self.b_hh = nn.Parameter(torch.rand(3 * hidden_size) * 2 * K - K)

        
    def forward(self, x, h):
        """
        Performs a forward pass through a GRU cell


        Returns the current hidden state h_t for every datapoint in batch.
        
        :param      x:    an element x_t in a sequence
        :type       x:    torch.Tensor
        :param      h:    previous hidden state h_{t-1}
        :type       h:    torch.Tensor
        """
        #
        # YOUR CODE HERE
        #

        # r_t = sigmoid(W_ir *  x_t + b_ir + W_hr * h_(t-1) + b_hr) (reset gate)
        # z_t = sigmoid(W_iz *  x_t + b_iz + W_hz * h_(t-1) + b_hz) (update gate)
        # n_t = tanh(W_in *  x_t + b_in + r_t(W_hn * h_(t-1) + b_hn)) (new gate)
        # h_t = (1 - z_t) * n_t + z_t * h_(t-1)

        # calculate the first part of the r z and n parts of the GRU by doing component-wise mult. of W with x and and adding b
        rzn_t_in_non_act = self.w_ih @ x + self.b_ih
        # 2 times the size of hidden layer
        rz_max_index = 2*self.hidden_size
        # calculate the second parts of the reset and update gates as per the formulae and concatenate it to the first part
        rz_t_non_act = rzn_t_in_non_act[:rz_max_index] + self.w_hh[:rz_max_index, :] @ h + self.b_hh[:rz_max_index]
        # use sigmoid on the reset and update gates
        rz_t = torch.sigmoid(rz_t_non_act)
        # split into two chunks - one for reset, one for update
        r_t, z_t = torch.chunk(rz_t, 2)
        # calculate the non-activation part of n by concatenating the first part with the rest of the formula as above
        n_t_non_act = rzn_t_in_non_act[rz_max_index:] + r_t * (self.w_hh[rz_max_index:, :] @ h + self.b_hh[rz_max_index:])
        # get the tanh of the internal bit 
        n_t = self.activation(n_t_non_act)
        # return h_t as in the docs
        return (1 - z_t) * n_t + z_t * h


class GRU2(nn.Module):
    """
    GRU network implementation
    """
    def __init__(self, input_size, hidden_size, bias=True, activation=torch.tanh, bidirectional=False):
        super(GRU2, self).__init__()
        self.bidirectional = bidirectional
        self.fw = GRUCellV2(input_size, hidden_size, activation=activation) # forward cell
        if self.bidirectional:
            self.bw = GRUCellV2(input_size, hidden_size, activation=activation) # backward cell
        self.hidden_size = hidden_size
        
    def forward(self, x):
        """
        Performs a forward pass through the whole GRU network, consisting of a number of GRU cells.
        Takes as input a 3D tensor `x` of dimensionality (B, T, D),
        where B is the batch size;
              T is the sequence length (if sequences have different lengths, they should be padded before being inputted to forward)
              D is the dimensionality of each element in the sequence, e.g. word vector dimensionality

        The method returns a 3-tuple of (outputs, h_fw, h_bw), if self.bidirectional is True,
                           a 2-tuple of (outputs, h_fw), otherwise
        `outputs` is a tensor containing the output features h_t for each t in each sequence (the same as in PyTorch native GRU class);
                  NOTE: if bidirectional is True, then it should contain a concatenation of hidden states of forward and backward cells for each sequence element.
        `h_fw` is the last hidden state of the forward cell for each sequence, i.e. when t = length of the sequence;
        `h_bw` is the last hidden state of the backward cell for each sequence, i.e. when t = 0 (because the backward cell processes a sequence backwards)
        
        :param      x:    a batch of sequences of dimensionality (B, T, D)
        :type       x:    torch.Tensor
        """
        #
        # YOUR CODE HERE
        #
        # compute a forward pass through the GRU network
        outputs, h_fw = self.compute_forward_pass(x)
        # if bidirectional is true
        if self.bidirectional:
            # do a backwards pass and return the outputs
            h_bw = self.compute_backward_pass(x)
            return (outputs, h_fw, h_bw)
        # return the outputs
        return (outputs, h_fw)

    # function to compute the forward pass for the GRU
    def compute_forward_pass(self, x):
        # set the outputs and the h values for the forward pass to the appropriate sizes
        outputs = torch.zeros(x.shape[0], x.shape[1], self.hidden_size)
        h_fw = torch.zeros(1, x.shape[0], self.hidden_size)
        # for each batch of x in x
        for batch_index, batch_x in enumerate(x):
            # let the h be all zeroes of correct size
            h_t = torch.zeros(self.hidden_size)
            # for each t and input in the batch
            for t, x_t in enumerate(batch_x):
                # call the forward function from before and set it to the output 
                h_t = self.fw.forward(x_t, h_t)
                # let the correct slot of outputs be h_t
                outputs[batch_index][t] = h_t
                # let the correct slot of h forward be the h_t value and return the outputs and h_fw
            h_fw[0][batch_index] = h_t
        # return outputs tensor and hidden state of forward cell tensor
        return outputs, h_fw
    
    # function to compute the backward pass for the GRU
    def compute_backward_pass(self, x):
        # let h of the backward pass be all zeros of the size of the hidden layer
        h_bw = torch.zeros(1, x.shape[0], self.hidden_size)
        # for each batch of inputs in x
        for batch_index, batch_x in enumerate(x):
            # let the output be zeroes
            h_t = torch.zeros(self.hidden_size)
            # for each output
            for t in range(batch_x.shape[0]):
                # let the x part be the appropriate slot of batch_x
                x_t = batch_x[batch_x.shape[0] - 1 - t]
                # call forward again using the x and h values 
                h_t = self.bw.forward(x_t, h_t)
            # let the current slot of h_bw be h_t and return
            h_bw[0][batch_index] = h_t
        # return the tensor with the hidden states of backward cell
        return h_bw


def is_identical(a, b):
    return "Yes" if np.all(np.abs(a - b) < 1e-6) else "No"


if __name__ == '__main__':
    torch.manual_seed(100304343)
    x = torch.randn(5, 3, 10)
    gru = nn.GRU(10, 20, bidirectional=False, batch_first=True)
    outputs, h = gru(x)

    torch.manual_seed(100304343)
    x = torch.randn(5, 3, 10)
    gru2 = GRU2(10, 20, bidirectional=False)
    outputs, h_fw = gru2(x)
    
    print("Checking the unidirectional GRU implementation")
    print("Same hidden states of the forward cell?\t\t{}".format(
        is_identical(h[0].detach().numpy(), h_fw.detach().numpy())
    ))

    torch.manual_seed(100304343)
    x = torch.randn(5, 3, 10)
    gru = GRU2(10, 20, bidirectional=True)
    outputs, h_fw, h_bw = gru(x)

    torch.manual_seed(100304343)
    x = torch.randn(5, 3, 10)
    gru2 = nn.GRU(10, 20, bidirectional=True, batch_first=True)
    outputs, h = gru2(x)
    
    print("Checking the bidirectional GRU implementation")
    print("Same hidden states of the forward cell?\t\t{}".format(
        is_identical(h[0].detach().numpy(), h_fw.detach().numpy())
    ))
    print("Same hidden states of the backward cell?\t{}".format(
        is_identical(h[1].detach().numpy(), h_bw.detach().numpy())
    ))