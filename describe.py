#!/usr/bin/python3


import sys
from create_classes import create_classe

print(sys.argv)

nbr_arg = len(sys.argv)

if nbr_arg >= 2:
    print(sys.argv[1])
    dataset = create_classe(sys.argv[1])

    print(dataset.nbr)
    print(dataset.house)
