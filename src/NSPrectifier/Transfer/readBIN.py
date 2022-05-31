# coding:utf-8
import time, serial
from struct import *
import binascii

file = open('E:\Dataset\mixed_corpus_bert_base_model.bin', 'rb')
i = 0
while 1:
    c = file.read(1)
    # 将字节转换成16进制；
    ssss = str(binascii.b2a_hex(c))[2:-1]
    print(str(binascii.b2a_hex(c))[2:-1])
    if not c:
        break
    ser = serial.Serial('COM3', 57600, timeout=1)
    ser.write(bytes().fromhex(ssss))# 将16进制转换为字节
    if i % 16 == 0:
        time.sleep(0.001)
    #写每一行等待的时间

    i += 1
    ser.close()
file.close()