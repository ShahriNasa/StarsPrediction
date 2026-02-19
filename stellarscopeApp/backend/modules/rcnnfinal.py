import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CustomLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Define the layers for the input, forget, cell, and output gates
        self.i2h = nn.Linear(input_size, 4 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size)
        
    def forward(self, x, hidden):
        h, c = hidden
        
        # Compute all gate activations in a single matrix multiplication
        gates = self.i2h(x) + self.h2h(h)
        
        # Split the gate activations into their components
        i_gate, f_gate, c_gate, o_gate = gates.chunk(4, 1)
        
        # Apply non-linearities
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        c_gate = torch.tanh(c_gate)
        o_gate = torch.sigmoid(o_gate)
        
        # Compute the new cell state
        c_new = (f_gate * c) + (i_gate * c_gate)
        
        # Compute the new hidden state
        h_new = o_gate * torch.tanh(c_new)
        
        return h_new, c_new


class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(CustomLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = nn.ModuleList([CustomLSTMCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])
        
    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = [(torch.zeros(x.size(0), self.hidden_size).to(x.device),
                       torch.zeros(x.size(0), self.hidden_size).to(x.device)) for _ in range(self.num_layers)]
        
        seq_len = x.size(1)
        for t in range(seq_len):
            inp = x[:, t, :]
            for l in range(self.num_layers):
                hidden[l] = self.cells[l](inp, hidden[l])
                inp = hidden[l][0]
                
        return hidden[-1][0].unsqueeze(1), hidden


class RCNN(nn.Module):
    def __init__(self,OUT_CHANNELS_1,OUT_CHANNELS_2,OUT_CHANNELS_3,poolsize1,poolsize2,poolsize3, num_in, log, kernel1, kernel2,kernel3, stride1, stride2,stride3, \
                 padding1, padding2,padding3, dropout, hidden1, hidden2, \
                 hidden3, rnn_hidden_size, rnn_num_layers,hidden4=128,ThreeLayers=False,hiddenthree=False):
        self.threelayers=ThreeLayers
        self.hiddenthree=hiddenthree
        if OUT_CHANNELS_3!=0:
            self.threelayers=True
        if hidden3!=0:
            self.hiddenthree=True
        super(RCNN, self).__init__()
        
        self.log = log
        self.dropout = torch.nn.Dropout(p=dropout)
        self.num_in = num_in
        print(self.num_in, file=log)

        OUT_CHANNELS_1 = OUT_CHANNELS_1
        dilation1 = 1
        poolsize1 = poolsize1
        
        OUT_CHANNELS_2 = OUT_CHANNELS_2
        dilation2 = 1
        poolsize2 = poolsize2


        OUT_CHANNELS_3 = OUT_CHANNELS_3
        dilation3 = 1
        poolsize3 = poolsize3

        # first convolution
        self.conv1 = nn.Conv1d(in_channels=1,
                               out_channels=OUT_CHANNELS_1,
                               kernel_size=kernel1,
                               dilation=dilation1,
                               stride=stride1,
                               padding=padding1)
        self.num_out = ((self.num_in+2*padding1-dilation1* \
                         (kernel1-1)-1)/stride1)+1
        assert str(self.num_out)[-1] == '0'
        print(self.num_out, file=log)
        self.bn1 = nn.BatchNorm1d(num_features=OUT_CHANNELS_1)
        self.pool1 = nn.AvgPool1d(kernel_size=poolsize1)
        self.num_out = (self.num_out/poolsize1)
        assert str(self.num_out)[-1] == '0'
        print(self.num_out, file=log)
        print(self.num_out)
        # second convolution
        self.conv2 = nn.Conv1d(in_channels=OUT_CHANNELS_1,
                               out_channels=OUT_CHANNELS_2,
                               kernel_size=kernel2,
                               stride=stride2,
                               padding=padding2)
        self.num_out = ((self.num_out+2*padding2-dilation2* \
                         (kernel2-1)-1)/stride2)+1
        assert str(self.num_out)[-1] == '0'
        print(self.num_out, file=log)
        self.bn2 = nn.BatchNorm1d(num_features=OUT_CHANNELS_2)
        self.pool2 = nn.AvgPool1d(kernel_size=poolsize2)
        self.num_out = (self.num_out/poolsize2)
        assert str(self.num_out)[-1] == '0'
        print(self.num_out, file=log)
        print(self.num_out)
         # Third convolution

        
        if self.threelayers==True:
            self.conv3 = nn.Conv1d(in_channels=OUT_CHANNELS_2,
                                out_channels=OUT_CHANNELS_3,
                                kernel_size=kernel3,
                                stride=stride3,
                                padding=padding3)
            
            self.num_out = ((self.num_out + 2*padding3 - dilation3 * (kernel3 - 1) - 1) / stride3) + 1
            assert str(self.num_out)[-1] == '0'
            print(self.num_out, file=log)
            print(self.num_out)
            self.bn3 = nn.BatchNorm1d(num_features=OUT_CHANNELS_3)
            self.pool3 = nn.AvgPool1d(kernel_size=poolsize3)
            self.num_out = self.num_out / poolsize3
            assert str(self.num_out)[-1] == '0'
            print(self.num_out, file=log)

        
        # Custom LSTM layer
        self.num_out = int(self.num_out)
        self.lstm = CustomLSTM(input_size=OUT_CHANNELS_2,
                            hidden_size=rnn_hidden_size,
                            num_layers=rnn_num_layers)
        
        # fully-connected network
        self.fc_input_dim = rnn_hidden_size + 1
        
        self.linear1 = nn.Linear(self.fc_input_dim, hidden1)
        self.linear2 = nn.Linear(hidden1, hidden2)
        if self.hiddenthree==True:
            self.linear3 = nn.Linear(hidden2, hidden3)
        # self.linear4 = nn.Linear(hidden3, hidden4)
        
        # output prediction
            self.predict = nn.Linear(hidden3, 1)
        else:
            self.predict = nn.Linear(hidden2, 1)
    


    def forward(self, x, s):
        x = self.bn1(self.pool1(F.relu(self.conv1(x))))
        x = self.bn2(self.pool2(F.relu(self.conv2(x))))
        if self.threelayers==True:
            x = self.bn3(self.pool3(F.relu(self.conv3(x))))
       
        x = x.permute(0, 2, 1)  # Reshape for LSTM (batch, seq_len, channels)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # Take the last output of the LSTM
        
        x = torch.cat((x, s), 1)

        x = self.dropout(F.relu(self.linear1(x)))
        if  self.hiddenthree:
            x = self.dropout(F.relu(self.linear2(x)))
            x = F.relu(self.linear3(x))
        else:
            x = F.relu(self.linear2(x))
        # x = self.dropout(F.relu(self.linear3(x)))

        x = F.softplus(self.predict(x))
        
        return x
