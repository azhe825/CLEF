from __future__ import division, print_function

from pdb import set_trace
from demos import cmd
import pickle
from mar import MAR
import os



#### stat
def stat(what):
    import numpy as np
    with open('./dump/' + what + '.pickle', 'rb') as f:
        record = pickle.load(f)
    n=len(record)
    m=len(record[record.keys()[0]])
    out = "\\begin{tabular}{ |l|"+'c|'*(m*2+1)+" }\n \\hline \n"
    out = out + "& \\multicolumn{"+str(m)+"}{|c|}{MEDIAN} & & \\multicolumn{"+str(m)+"}{|c|}{IQR} \\\\ \n \\cline{2-"+str(2*m+2)+"} \n"
    header = ""
    result=""
    for i,topic in enumerate(record):
        med = "Topic"+str(topic)
        iqr = ""
        for method in record[topic]:
            if i==0:
                header = header + " & " + method
            tmp = record[topic][method]['x']
            med = med+' & '+str(int(np.median(tmp)))
            iqr = iqr + ' & ' + str(int(np.percentile(tmp,75)-np.percentile(tmp,25)))
        med=med+" & "
        iqr=iqr+"\\\\\n\\hline \n"
        result=result+med+iqr

    header=header+'&'+header+'\\\\ \n \\hline \n'
    out=out+header+result+'\\end{tabular}'
    print(out)


##### general

def one_run(runid,path="train"):
    if path=="train":
        topics = [1, 4, 9, 11, 14, 19, 23, 28, 33, 35, 37, 38, 43, 44, 45, 50, 53, 54, 55, 6]
    else:
        topics = [2, 5, 7, 8, 10, 12, 15, 16, 17, 18, 21, 22, 25, 26, 27, 29, 31, 32, 34, 36, 39, 40, 41, 42, 47, 48,
                  49, 51, 56, 57]
        # topics = [18, 21, 22, 25, 26, 27, 29, 31, 32, 34, 36, 39, 40, 41, 42, 47, 48,
        #           49, 51, 56, 57]

    topics = map(str,topics)


    # if os._exists("../workspace/output/"+path+"/"+runid):
    #     os.remove("../workspace/output/"+path+"/"+runid)
    for topic in topics:
        read = START_AUTO_abs(topic,cl='SVM',runid=runid,path=path)

def repeat_run(path="train"):
    repeats=10
    record={}
    if path == "train":
        topics = [1, 4, 9, 11, 14, 19, 23, 28, 33, 35, 37, 38, 43, 44, 45, 50, 53, 54, 55, 6]
    else:
        topics = [2, 5, 7, 8, 10, 12, 15, 16, 17, 18, 21, 22, 25, 26, 27, 29, 31, 32, 34, 36, 39, 40, 41, 42, 47,
                  48,  49, 51, 56, 57]
        # topics = [18, 21, 22, 25, 26, 27, 29, 31, 32, 34, 36, 39, 40, 41, 42, 47, 48,
        #           49, 51, 56, 57]

    topics = map(str, topics)

    # if os._exists("../workspace/output/"+path+"/"+runid):
    #     os.remove("../workspace/output/"+path+"/"+runid)
    for topic in topics:
        record[topic]={'AU':{'x':[],'pos':[]},'CAL':{'x':[],'pos':[]},'AU+RW':{'x':[],'pos':[]},'RW':{'x':[],'pos':[]}}
        for i in xrange(repeats):
            print(str(topic) + " : " + str(i))
            read = START(topic, cl='SVM', runid='AU'+str(i), path=path)
            record[topic]['AU']['x'].append(read.record['x'][-1])
            record[topic]['AU']['pos'].append(read.record['pos'][-1])
            read = START_imb(topic, cl='SVM', runid='CAL' + str(i), path=path)
            record[topic]['CAL']['x'].append(read.record['x'][-1])
            record[topic]['CAL']['pos'].append(read.record['pos'][-1])
            read = START_full(topic, cl='SVM', runid='AU+RW' + str(i), path=path)
            record[topic]['AU+RW']['x'].append(read.record['x'][-1])
            record[topic]['AU+RW']['pos'].append(read.record['pos'][-1])
            read = START_rw(topic, cl='SVM', runid='RW' + str(i), path=path)
            record[topic]['RW']['x'].append(read.record['x'][-1])
            record[topic]['RW']['pos'].append(read.record['pos'][-1])
            with open('./dump/' + path + '.pickle', 'wb') as f:
                pickle.dump(record, f)






##### basic

def START_AUTO_abs(filename,cl='SVM',runid="abs",path="train"):
    lifes=5
    life=lifes
    poslast=0
    starting=40
    rank=1

    read = MAR()
    read = read.create(filename,path=path)
    read.restart()
    while True:
        pos, neg, total, pos_true = read.get_numbers()
        print("%d, %d, %d" %(pos,pos+neg, pos_true))
        if pos == 0 or pos + neg < starting:
            for id in read.random():
                read.auto_code(id, 0, rank)
        else:
            if poslast==pos:
                life=life-1
            else:
                life=lifes
            if life<=0 or pos+neg==total:
                break
            ids,c =read.show(pne=False,cl=cl)
            for i, id in enumerate(ids):
                read.auto_code(id, c[i], rank)
        poslast = pos
        rank = rank + 1
    read.export(runid=runid)
    return read

def START_AUTO(filename,cl='SVM',runid="full",path="train"):
    lifes=5
    life=lifes
    poslast=0
    rank=1
    starting=40

    read = MAR()
    read = read.create(filename, path=path)
    read.restart()
    a,b=read.get_allpos()
    print("%d, %d"%(a,b))
    while True:
        pos, neg, total, pos_true = read.get_numbers()
        print("%d, %d, %d" %(pos,pos+neg, pos_true))
        if pos==0 or pos+neg<starting:
            for id in read.random():
                read.auto_code(id, 0, rank)
        else:
            if pos_true == 0:
                ids,c =read.show(pne=False,cl=cl)
                for i, id in enumerate(ids):
                    read.auto_code(id, c[i], rank)
            else:
                if poslast==pos_true:
                    life=life-1
                else:
                    life=lifes
                if life<=0 or pos+neg==total:
                    break
                ids,c =read.show(pne=False,cl=cl)
                for i,id in enumerate(ids):
                    read.auto_code(id, c[i], rank)
        poslast = pos_true
        rank=rank+1
    read.export(runid=runid)
    return read



def START(filename,cl='SVM',runid="simple",path="train"):
    # stop=1
    rank = 1
    starting = 40

    read = MAR()
    read = read.create(filename, path=path)
    read.restart()
    a,b = read.get_allpos()
    # print("%d, %d"%(a,b))

    # target = int(a*stop)
    while True:
        pos, neg, total, pos_true = read.get_numbers()
        # print("%d, %d, %d" % (pos, pos + neg, pos_true))
        if pos_true >= b or pos+neg==total:
            break
        if pos==0 or pos+neg< starting:
            for id in read.random():
                read.auto_code(id, 0, rank)
        else:
            ids,c =read.show(pne=False,cl=cl)
            for i, id in enumerate(ids):
                read.auto_code(id, c[i], rank)
        rank=rank+1
    read.export(runid=runid)
    return read


def START_imb(filename,cl='SVM',runid="imb",path="train"):
    # stop=1
    rank = 1
    starting = 40

    read = MAR()
    read = read.create(filename, path=path)
    read.restart()
    a,b = read.get_allpos()
    read.enough=a
    # print("%d, %d"%(a,b))

    # target = int(a*stop)
    while True:
        pos, neg, total, pos_true = read.get_numbers()
        # print("%d, %d, %d" % (pos, pos + neg, pos_true))
        if pos_true >= b or pos+neg==total:
            break
        if pos==0 or pos+neg< starting:
            for id in read.random():
                read.auto_code(id, 0, rank)
        else:
            ids,c =read.show(pne=False,cl=cl)
            for i, id in enumerate(ids):
                read.auto_code(id, c[i], rank)
        rank=rank+1
    read.export(runid=runid)
    return read


def START_full(filename,cl='SVM',runid="simple_full",path="train"):
    # stop=1
    rank = 1
    starting = 40

    read = MAR()
    read = read.create(filename, path=path)
    read.restart()
    read.weight=9
    a,b = read.get_allpos()
    # print("%d, %d"%(a,b))
    # target = int(a*stop)
    while True:
        pos, neg, total, pos_true = read.get_numbers()
        # print("%d, %d, %d" % (pos, pos + neg, pos_true))
        if pos_true >= b:
            break
        if pos==0 or pos+neg< starting:
            for id in read.random():
                read.auto_code(id, 0, rank)
        else:
            ids,c =read.show(pne=False,cl=cl)
            for i, id in enumerate(ids):
                read.auto_code(id, c[i], rank)
    read.export(runid=runid)
    return read

def START_rw(filename,cl='SVM',runid="simple_full",path="train"):
    # stop=1
    rank = 1
    starting = 40

    read = MAR()
    read = read.create(filename, path=path)
    read.restart()
    read.weight=9
    read.enough=1000
    a,b = read.get_allpos()
    # print("%d, %d"%(a,b))
    # target = int(a*stop)
    while True:
        pos, neg, total, pos_true = read.get_numbers()
        # print("%d, %d, %d" % (pos, pos + neg, pos_true))
        if pos_true >= b:
            break
        if pos==0 or pos+neg< starting:
            for id in read.random():
                read.auto_code(id, 0, rank)
        else:
            ids,c =read.show(pne=False,cl=cl)
            for i, id in enumerate(ids):
                read.auto_code(id, c[i], rank)
    read.export(runid=runid)
    return read

if __name__ == "__main__":
    eval(cmd())
