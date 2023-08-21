#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

class My_Net(torch.nn.Module):
    def __init__(self, out_channel):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, out_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(out_channel, out_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(out_channel, out_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(out_channel, out_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(out_channel, out_channel, 5, 1, 2),
            torch.nn.Sigmoid()
        )

    def forward(self, data):
        data = self.conv1(data)
        data = self.conv2(data)
        data = self.conv3(data)
        data = self.conv4(data)
        data = self.final(data)
        return data

class My_Net2(torch.nn.Module):
    def __init__(self,in_channel ,out_channel):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, in_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, in_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, in_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, in_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, out_channel, 5, 1, 2),
            torch.nn.Sigmoid()
        )

    def forward(self, data):
        data = self.conv1(data)
        data = self.conv2(data)
        data = self.conv3(data)
        data = self.conv4(data)
        data = self.final(data)
        return data

class My_Net3(torch.nn.Module):
    def __init__(self,in_channel ,out_channel):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, in_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, in_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, in_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, in_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, out_channel, 5, 1, 2),
            torch.nn.Sigmoid()
        )
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, in_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, in_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, in_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, in_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(6, out_channel, 5, 1, 2),
            torch.nn.Sigmoid()
        )

        self.conv11 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, in_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv12 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, in_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv13 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, in_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv14 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, in_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.final1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, out_channel, 5, 1, 2),
            torch.nn.Sigmoid()
        )

        self.conv21 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, in_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv22 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, in_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv23 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, in_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv24 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, in_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        # self.final2 = torch.nn.Sequential(
        #     torch.nn.Conv2d(in_channel, out_channel, 5, 1, 2),
        #     torch.nn.Sigmoid()
        # )

    def forward(self, data1,data2,data3):
        data1 = self.conv1(data1)
        data1 = self.conv2(data1)
        data1 = self.conv3(data1)
        data1 = self.conv4(data1)
        data = self.final(data1)
        
        

        data2 = self.conv11(data2)
        data2 = self.conv12(data2)
        data2 = self.conv13(data2)
        data2 = self.conv14(data2)

        data3 = self.conv21(data3)
        data3 = self.conv22(data3)
        data3 = self.conv23(data3)
        data3 = self.conv24(data3)
        data=torch.cat((data1,data2,data3),1)
        data = self.final(data)


        return data



class My_Net4(torch.nn.Module):
    def __init__(self,in_channel ,out_channel):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, in_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, in_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, in_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, in_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, out_channel, 5, 1, 2),
            torch.nn.Sigmoid()
        )
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, in_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, in_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, in_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, in_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(6, out_channel, 5, 1, 2),
            torch.nn.Sigmoid()
        )

        self.conv11 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, in_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv12 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, in_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv13 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, in_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv14 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, in_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        # self.final1 = torch.nn.Sequential(
        #     torch.nn.Conv2d(in_channel, out_channel, 5, 1, 2),
        #     torch.nn.Sigmoid()
        # )

        self.conv21 = torch.nn.Sequential(
            torch.nn.Conv2d(4, 4, 5, 1, 2),
            torch.nn.BatchNorm2d(4),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv22 = torch.nn.Sequential(
            torch.nn.Conv2d(4, 4, 5, 1, 2),
            torch.nn.BatchNorm2d(4),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv23 = torch.nn.Sequential(
            torch.nn.Conv2d(4, 4, 5, 1, 2),
            torch.nn.BatchNorm2d(4),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv24 = torch.nn.Sequential(
            torch.nn.Conv2d(4, 4, 5, 1, 2),
            torch.nn.BatchNorm2d(4),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.final2 = torch.nn.Sequential(
            torch.nn.Conv2d(4, out_channel, 5, 1, 2),
            torch.nn.Sigmoid()
        )

    def forward(self, data1,data2):
        data1 = self.conv1(data1)
        data1 = self.conv2(data1)
        data1 = self.conv3(data1)
        data1 = self.conv4(data1)
        data = self.final(data1)
        netfeature=data

        # data2 = self.conv11(data2)
        # data2 = self.conv12(data2)
        # data2 = self.conv13(data2)
        # data2 = self.conv14(data2)

        data3=torch.cat((data*2,data2),1)

        data3 = self.conv21(data3)
        data3 = self.conv22(data3)
        data3 = self.conv23(data3)
        data3 = self.conv24(data3)
        # data=torch.cat((data1,data2,data3),1)
        data = self.final2(data3)


        return netfeature,data




class My_Net5(torch.nn.Module):
    def __init__(self,in_channel ,out_channel):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, in_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, in_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, in_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, in_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, out_channel, 5, 1, 2),
            torch.nn.Sigmoid()
        )
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, in_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, in_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, in_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, in_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(6, out_channel, 5, 1, 2),
            torch.nn.Sigmoid()
        )

        self.conv11 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, in_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv12 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, in_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv13 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, in_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv14 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, in_channel, 5, 1, 2),
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        # self.final1 = torch.nn.Sequential(
        #     torch.nn.Conv2d(in_channel, out_channel, 5, 1, 2),
        #     torch.nn.Sigmoid()
        # )

        self.conv21 = torch.nn.Sequential(
            torch.nn.Conv2d(4, 4, 5, 1, 2),
            torch.nn.BatchNorm2d(4),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv22 = torch.nn.Sequential(
            torch.nn.Conv2d(4, 4, 5, 1, 2),
            torch.nn.BatchNorm2d(4),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv23 = torch.nn.Sequential(
            torch.nn.Conv2d(8, 4, 5, 1, 2),
            torch.nn.BatchNorm2d(4),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv24 = torch.nn.Sequential(
            torch.nn.Conv2d(8, 4, 5, 1, 2),
            torch.nn.BatchNorm2d(4),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.final2 = torch.nn.Sequential(
            torch.nn.Conv2d(8, out_channel, 5, 1, 2),
            torch.nn.Sigmoid()
        )

    def forward(self, data1,data2):
        data1 = self.conv1(data1)
        data1 = self.conv2(data1)
        data1 = self.conv3(data1)
        data1 = self.conv4(data1)
        data = self.final(data1)
        netfeature=data
        

        # data2 = self.conv11(data2)
        # data2 = self.conv12(data2)
        # data2 = self.conv13(data2)
        # data2 = self.conv14(data2)

        data3=torch.cat((data*2,data2),1)

        data31 = self.conv21(data3)
        data32 = self.conv22(data31)
        data32c=torch.cat((data31,data32),1)

        data33 = self.conv23(data32c)
        data33c=torch.cat((data32,data33),1)

        data34 = self.conv24(data33c)
        data34c=torch.cat((data33,data34),1)
        # data=torch.cat((data1,data2,data3),1)
        data = self.final2(data34c)


        return data