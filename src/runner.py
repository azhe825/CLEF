from __future__ import division, print_function

from pdb import set_trace
from demos import cmd

from mar import MAR





##### general

def one_run():
    topics = [1,4,6,9,11,14,19,23,28,33,35,37,38,43,44,45,50,53,54,55]
    topics = map(str,topics)
    records = {}
    for topic in topics:
        read = START_AUTO(topic)
        records[topic]=read.record
    set_trace()


##### basic

def START_AUTO_abs(filename):
    lifes=5
    life=lifes
    poslast=0

    read = MAR()
    read = read.create(filename)
    read.restart()
    while True:
        pos, neg, total, pos_true = read.get_numbers()
        print("%d, %d, %d" %(pos,pos+neg, pos_true))
        if pos==0:
            for id in read.random():
                read.auto_code(id)
        else:
            if poslast==pos:
                life=life-1
            else:
                life=lifes
            if life<=0:
                break
            ids,c =read.show(pne=True)
            for id in ids:
                read.auto_code(id)
        poslast = pos
    return read

def START_AUTO(filename):
    lifes=3
    life=lifes
    poslast=0

    read = MAR()
    read = read.create(filename)
    read.restart()
    a,b=read.get_allpos()
    print("%d, %d"%(a,b))
    while True:
        pos, neg, total, pos_true = read.get_numbers()
        print("%d, %d, %d" %(pos,pos+neg, pos_true))
        if pos==0:
            for id in read.random():
                read.auto_code(id)
        else:
            if pos_true == 0:
                ids,c =read.show(pne=True)
                for id in ids:
                    read.auto_code(id)
            else:
                if poslast==pos_true:
                    life=life-1
                else:
                    life=lifes
                if life<=0:
                    break
                ids,c =read.show_true(pne=True)
                for id in ids:
                    read.auto_code(id)
        poslast = pos_true
    return read



def START(filename):
    stop=0.90

    read = MAR()
    read = read.create(filename)
    read.restart()
    a,b = read.get_allpos()
    print("%d, %d"%(a,b))
    target = int(a*stop)
    while True:
        pos, neg, total, pos_true = read.get_numbers()
        print("%d, %d, %d" % (pos, pos + neg, pos_true))
        if pos > target:
            break
        if pos==0:
            for id in read.random():
                read.auto_code(id)
        else:
            ids,c =read.show(pne=True)
            for id in ids:
                read.auto_code(id)
    return read


if __name__ == "__main__":
    eval(cmd())
