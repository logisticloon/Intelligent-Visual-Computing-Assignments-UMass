import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        """
        
        DEFINE YOUR NETWORK HERE
        
        """
        self.layer1 = nn.Conv2d(1,8,7,stride=1,padding=0,bias=False)
        torch.nn.init.kaiming_uniform_(self.layer1.weight)
        self.norm1 = nn.LayerNorm([8,106,106])
        self.leakyReLU1 = nn.LeakyReLU(0.01)
        self.maxPooling1 = nn.MaxPool2d(2,2)

        ####################################
        self.deepConv = nn.Conv2d(in_channels=8,out_channels=8,kernel_size=7,stride=2,padding=0,groups=8,bias=False)
        torch.nn.init.kaiming_uniform_(self.deepConv.weight)
        self.norm2 = nn.LayerNorm([8,24,24])
        self.leakyReLU2 = nn.LeakyReLU(0.01)
        self.maxPooling2 = nn.MaxPool2d(2,2)
        self.pointWiseConv = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=1,stride=1,padding=0,bias=True)
        torch.nn.init.xavier_uniform(self.pointWiseConv.weight)

        ########################################

        self.deepConv2 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=7,stride=1,padding=0,groups=16,bias=False)
        torch.nn.init.kaiming_uniform_(self.deepConv2.weight)
        self.norm3 = nn.LayerNorm([16,6,6])
        self.leakyReLU3 = nn.LeakyReLU(0.01)
        self.maxPooling3 = nn.MaxPool2d(2,2)
        self.pointWiseConv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1,stride=1,padding=0,bias=True)
        torch.nn.init.xavier_uniform(self.pointWiseConv2.weight)

        

        self.fullyConnectedConv = nn.Conv2d(in_channels=32,out_channels=10,kernel_size=3,bias=True)
        torch.nn.init.xavier_uniform(self.fullyConnectedConv.weight)


        # change this obviously!
        # self.naive = nn.Conv2d(in_channels=1, out_channels=num_classes, kernel_size=112, stride=1, padding=0, bias=True)
        
    def forward(self, x):
        """

        DEFINE YOUR FORWARD PASS HERE

        """
       
        out = self.maxPooling1(self.leakyReLU1(self.norm1(self.layer1(x))))
        
        seq2 = nn.Sequential(
            self.deepConv,
            self.norm2,
            self.leakyReLU2,
            self.maxPooling2,
            self.pointWiseConv
        )
        out = seq2(out)

        seq3 = nn.Sequential(
            self.deepConv2,
            self.norm3,
            self.leakyReLU3,
            self.maxPooling3,
            self.pointWiseConv2
        )
        out = seq3(out)
        out = self.fullyConnectedConv(out)

        


        
        return out