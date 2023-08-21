"""
Modified on 2021/5/20
@author: Yingjie Shi
"""
import argparse

parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--name', type=str, default="SOTS_GT")
parser.add_argument('--datasets', type=str, default="real-world")
parser.add_argument('--clip', type=bool, default=True)
parser.add_argument('--loadmodel', type=bool, default=True)
parser.add_argument('--epoch', type=int, default=801)
parser.add_argument('--learning_rate', type=float, default=0.005)
parser.add_argument('--outpath', type=str, default="/opt/data/private/syj/code/PSDNet/0821/")
parser.add_argument('--priomask1path', type=str, default="/opt/data/private/syj/code/PSDNet/FeatureMap/priomask1")
parser.add_argument('--maskpath', type=str, default="/opt/data/private/syj/code/PSDNet/FeatureMap/mask")
parser.add_argument('--ambientpath', type=str, default="/opt/data/private/syj/code/PSDNet/FeatureMap/ambient")
parser.add_argument('--savemodel', type=str, default="/opt/data/private/syj/code/PSDNet/model/")
parser.add_argument('--pretrainmodel', type=str, default="/opt/data/private/syj/code/PSDNet/model/")
options = parser.parse_args()
