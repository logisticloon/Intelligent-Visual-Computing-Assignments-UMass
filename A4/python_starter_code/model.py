import torch.nn as nn
import torch
from torch.nn.utils import weight_norm as wn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(
        self,
        args,
        dropout_prob=0.1,
    ):
        super(Decoder, self).__init__()

        # **** YOU SHOULD IMPLEMENT THE MODEL ARCHITECTURE HERE ****
        # Define the network architecture based on the figure shown in the assignment page.
        # Read the instruction carefully for layer details.
        # Pay attention that your implementation should include FC layers, weight_norm layers,
        # PReLU layers, Dropout layers and a tanh layer.
        # self.fc = nn.Linear(3, 1)
        self.dropout_prob = dropout_prob
        self.th = nn.Tanh()

        self.fc1_linear = wn(nn.Linear(3,512))
        self.fc1_pr = nn.PReLU()
        self.fc1_drp = nn.Dropout(p=dropout_prob)
        self.fc1 = nn.Sequential(self.fc1_linear,self.fc1_pr,self.fc1_drp)

        self.fc2_linear = wn(nn.Linear(512,512))
        self.fc2_pr = nn.PReLU()
        self.fc2_drp = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Sequential(self.fc2_linear,self.fc2_pr,self.fc2_drp)

        self.fc3_linear = wn(nn.Linear(512,512))
        self.fc3_pr = nn.PReLU()
        self.fc3_drp = nn.Dropout(p=dropout_prob)
        self.fc3 = nn.Sequential(self.fc3_linear,self.fc3_pr,self.fc3_drp)

        self.fc4_linear = wn(nn.Linear(512,509))
        self.fc4_pr = nn.PReLU()
        self.fc4_drp = nn.Dropout(p=dropout_prob)
        self.fc4 = nn.Sequential(self.fc4_linear,self.fc4_pr,self.fc4_drp)

        self.fc5_linear = wn(nn.Linear(512,512))
        self.fc5_pr = nn.PReLU()
        self.fc5_drp = nn.Dropout(p=dropout_prob)
        self.fc5 = nn.Sequential(self.fc5_linear,self.fc5_pr,self.fc5_drp)

        self.fc6_linear = wn(nn.Linear(512,512))
        self.fc6_pr = nn.PReLU()
        self.fc6_drp = nn.Dropout(p=dropout_prob)
        self.fc6 = nn.Sequential(self.fc6_linear,self.fc6_pr,self.fc6_drp)
        
        self.fc7_linear = wn(nn.Linear(512,512))
        self.fc7_pr = nn.PReLU()
        self.fc7_drp = nn.Dropout(p=dropout_prob)
        self.fc7 = nn.Sequential(self.fc7_linear,self.fc7_pr,self.fc7_drp)
        
        self.fc8_linear = nn.Linear(512,1)


        # ***********************************************************************

    # input: N x 3
    def forward(self, input):

        # **** YOU SHOULD IMPLEMENT THE FORWARD PASS HERE ****
        # Based on the architecture defined above, implement the feed forward procedure
        # x = self.fc(input)
        # x = self.th(x)
        #++++++++++++++++++++++++++
        x = self.fc4(self.fc3(self.fc2(self.fc1(input))))
        x = torch.cat((x,input),dim=1)
        x = self.th(self.fc8_linear(self.fc7(self.fc6(self.fc5(x)))))

        # ***********************************************************************


        return x
