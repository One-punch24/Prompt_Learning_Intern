import os

for i in range(-1,24):
    os.system("python Prefix_tune.py --bsz 6 --epochs 10 --lr 0.00001 --sel "+ str(i))