#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified on 2021/5/20
@author: Yingjie Shi
"""

def write_log(file_name, title, psnr, ssim):
    fp = open(file_name, "a+")
    fp.write(title+ ':\n')
    fp.write('PSNR:%0.6f\n'%psnr)
    fp.write('SSIM:%0.6f\n'%ssim)
    fp.close()
