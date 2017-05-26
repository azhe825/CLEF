from __future__ import print_function, division
import urllib2
import pickle
from pdb import set_trace
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from sklearn import svm


class MAR(object):
    def __init__(self):
        self.fea_num = 4000
        self.step = 10
        self.enough = 30
        self.downrate = 2
        self.atleast=100
        self.true_rate=1
        self.weight=0


    def create(self,filename,path="train"):
        self.filename=str(filename)
        self.name=self.filename.split(".")[0]
        self.record={"x":[],"pos":[]}
        self.body={}
        self.abs=[]
        self.full=[]
        try:
            ## if model already exists, load it ##
            return self.load()
        except:
            ## otherwise read from file ##
            self.loadfile()
            self.preprocess()
            self.save()


        return self

    ## save model ##
    def save(self):
        with open("memory/"+str(self.name)+".pickle","w") as handle:
            pickle.dump(self,handle)

    ## load model ##
    def load(self):
        with open("memory/" + str(self.name) + ".pickle", "r") as handle:
            tmp = pickle.load(handle)
        return tmp



    def loadfile(self):

        ## load all candidates
        with open("../workspace/"+self.path+"_data/topics_"+self.path+"/" + str(self.filename), "r") as f:
            content = f.read()
        self.topic=content.split('\n')[0].split('Topic:')[1].strip()
        self.body['Pid']=map(str.strip,content.split('Pids: \n')[1].strip().split('\n'))
        self.body['Pid']=list(np.unique(self.body['Pid']))
        self.body['text']=[]
        print("%d pids to query" %len(self.body['Pid']))
        pidstr=[]
        for i,pid in enumerate(self.body['Pid']):
            pidstr.append(pid)
            if i%10==9 or i==len(self.body['Pid'])-1:
                qref = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=' + ','.join(pidstr) + '&rettype=abstract&retmode=text'
                req = urllib2.Request(qref)
                req.add_header('User-agent', 'Mozilla/5.0 (Linux i686)')
                response = urllib2.urlopen(req)
                texts = response.read()
                for the_pid in pidstr:
                    self.body['text'].append(texts.split('PMID: '+the_pid)[0].strip())
                    try:
                        texts=''.join(texts.split('PMID: '+the_pid)[1:])
                    except:
                        set_trace()
                        exit()
                pidstr=[]
        self.body['code'] = ['undetermined']*len(self.body['Pid'])
        self.body['true_code'] = ['undetermined'] * len(self.body['Pid'])
        self.body['cost'] = ['NS']*len(self.body['Pid'])
        self.body['cost2'] = ['NFN'] * len(self.body['Pid'])
        self.body['score'] = [0] * len(self.body['Pid'])
        self.body['rank'] = [-1] * len(self.body['Pid'])

        ## load review results
        with open("../workspace/"+self.path+"_data/qrel_abs_"+self.path, "r") as f:
            abs = f.readlines()
        with open("../workspace/"+self.path+"_data/qrel_content_"+self.path, "r") as f:
            full = f.readlines()
        for ab in abs:
            tmp = ab.split()
            if self.topic in tmp and tmp[-1].strip() == '1':
                self.abs.append(tmp[-2].strip())
        for ful in full:
            tmp = ful.split()
            if self.topic in tmp and tmp[-1].strip() == '1':
                self.full.append(tmp[-2].strip())
        exc = []
        for ab in self.abs:
            if not ab in self.body["Pid"]:
                exc.append(ab)
        for ex in exc:
            self.abs.remove(ex)
        exc = []
        for ful in self.full:
            if not ful in self.body["Pid"]:
                exc.append(ful)
        for ex in exc:
            self.full.remove(ex)
        return

    def preprocess(self):

        ### Feature selection by tfidf in order to keep vocabulary ###
        tfidfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=None, use_idf=True, smooth_idf=False,
                                  sublinear_tf=False, decode_error="ignore")
        tfidf = tfidfer.fit_transform(self.body['text'])
        weight = tfidf.sum(axis=0).tolist()[0]
        kept = np.argsort(weight)[-self.fea_num:]
        self.voc = np.array(tfidfer.vocabulary_.keys())[np.argsort(tfidfer.vocabulary_.values())][kept]
        ##############################################################

        ### Term frequency as feature, L2 normalization ##########
        tfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=u'l2', use_idf=False,
                               vocabulary=self.voc, decode_error="ignore")
        # tfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=None, use_idf=False,
        #                 vocabulary=self.voc,decode_error="ignore")
        self.csr_mat = tfer.fit_transform(self.body['text'])
        ########################################################
        return

    def get_numbers(self):
        total = len(self.body["code"])
        pos = Counter(self.body["code"])["yes"]
        neg = Counter(self.body["code"])["no"]
        pos_true = Counter(self.body["true_code"])["yes"]
        try:
            tmp=self.record['x'][-1]
        except:
            tmp=-1
        if int(pos+neg)>tmp:
            self.record['x'].append(int(pos+neg))
            self.record['pos'].append(int(pos))
        self.pool = np.where(np.array(self.body['code']) == "undetermined")[0]
        self.labeled = list(set(range(len(self.body['code']))) - set(self.pool))
        return pos, neg, total, pos_true

    ## Get suggested exmaples
    def show(self,pne=False,cl="SVM"):
        clf=self.train(pne,cl=cl)
        certain_id, certain_prob = self.certain(clf)
        return np.array(self.body['Pid'])[certain_id], certain_prob

    ## Train model ##
    def train(self, pne=False, cl='SVM'):
        if cl == 'SVM':
            clf = svm.SVC(kernel='linear', probability=True)
        elif cl == "CART":
            from sklearn import tree
            clf = tree.DecisionTreeClassifier()
        elif cl == "RF":
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier()
        elif cl == "LR":
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression()
        elif cl == "NB":
            from sklearn.naive_bayes import GaussianNB
            clf = GaussianNB()
        # clf = svm.SVC(kernel='linear', probability=True)
        poses = np.where(np.array(self.body['code']) == "yes")[0]
        negs = np.where(np.array(self.body['code']) == "no")[0]
        true_pos = np.where(np.array(self.body['true_code']) == "yes")[0]
        left = poses
        decayed = list(left) + list(negs)
        if pne:
            unlabeled = self.pool
            try:
                unlabeled = np.random.choice(unlabeled, size=np.max((len(decayed), self.atleast)), replace=False)
            except:
                pass
        else:
            unlabeled = []

        labels = np.array([x if x != 'undetermined' else 'no' for x in self.body['code']])
        all_neg = list(negs) + list(unlabeled)
        all = list(decayed) + list(unlabeled)

        clf.fit(self.csr_mat[all + list(true_pos) * self.weight], labels[all + list(true_pos) * self.weight])
        ## aggressive undersampling ##
        if len(poses) >= self.enough and len(poses) * self.downrate < len(negs):
            pos_at = list(clf.classes_).index('yes')
            train_prob = clf.predict_proba(self.csr_mat[all_neg])[:, pos_at]
            negs_sel = np.argsort(train_prob)[:len(left)]
            sample = list(left) + list(np.array(all_neg)[negs_sel])
            clf.fit(self.csr_mat[sample + list(true_pos) * self.weight], labels[sample + list(true_pos) * self.weight])
        return clf

    ## Get certain ##
    def certain(self,clf):
        pos_at = list(clf.classes_).index("yes")
        prob = clf.predict_proba(self.csr_mat[self.pool])[:,pos_at]
        order = np.argsort(prob)[::-1][:self.step]
        return np.array(self.pool)[order],np.array(prob)[order]





    ## Get suggested exmaples based on true code
    def show_true(self,pne=False,cl="SVM"):
        clf=self.train(pne,cl=cl)
        clf_true=self.train_true(pne,cl=cl)
        certain_id, certain_prob = self.certain_true(clf,clf_true)
        return np.array(self.body['Pid'])[certain_id], certain_prob

    ## Train model on true_code ##
    def train_true(self,pne=False,cl='SVM'):
        if cl == 'SVM':
            clf = svm.SVC(kernel='linear', probability=True)
        elif cl=="CART":
            from sklearn import tree
            clf = tree.DecisionTreeClassifier()
        elif cl=="RF":
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier()
        elif cl=="LR":
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression()
        elif cl=="NB":
            from sklearn.naive_bayes import GaussianNB
            clf = GaussianNB()
        # clf = svm.SVC(kernel='linear', probability=True)
        poses = np.where(np.array(self.body['true_code']) == "yes")[0]
        negs = np.where(np.array(self.body['true_code']) == "no")[0]
        left = poses
        decayed = list(left) + list(negs)
        if pne:
            unlabeled = self.pool
            try:
                unlabeled = np.random.choice(unlabeled,size=np.max((len(decayed),self.atleast)),replace=False)
            except:
                pass
        else:
            unlabeled = []

        labels=np.array([x if x!='undetermined' else 'no' for x in self.body['code']])
        all_neg=list(negs)+list(unlabeled)
        all = list(decayed)+list(unlabeled)

        clf.fit(self.csr_mat[all], labels[all])
        ## aggressive undersampling ##
        if len(poses)>=self.enough and len(poses)*self.downrate<len(negs):
            pos_at = list(clf.classes_).index('yes')
            train_prob = clf.predict_proba(self.csr_mat[all_neg])[:, pos_at]
            negs_sel = np.argsort(train_prob)[:len(left)]
            # train_dist = clf.decision_function(self.csr_mat[all_neg])
            # pos_at = list(clf.classes_).index("yes")
            # if pos_at:
            #     train_dist=-train_dist
            # negs_sel = np.argsort(train_dist)[::-1][:len(left)]
            sample = list(left) + list(np.array(all_neg)[negs_sel])
            clf.fit(self.csr_mat[sample], labels[sample])
        return clf

    ## Export results
    def export(self,runid="first-run"):
        order=np.argsort(self.body['rank'])
        max=self.body['rank'][order[-1]]
        content=[]
        notread=[]
        for o in order:
            if self.body['rank'][o]<1:
                line = [self.topic, self.body['cost'][o], self.body['Pid'][o], max+1,
                        "%.2f" % self.body['score'][o], runid]
                notread.append('\t'.join(map(str, line)) + '\n')
            else:
                line = [self.topic, self.body['cost'][o], self.body['Pid'][o], self.body['rank'][o], "%.2f" %self.body['score'][o], runid]
                content.append('\t'.join(map(str,line))+'\n')
        content.extend(notread)
        with open("../workspace/output/"+self.path+"/" + self.name+'_'+runid, "w") as f:
            f.writelines(content)
        with open("../workspace/output/"+self.path+"/" + runid, "a") as f:
            f.writelines(content)


    ## Get certain ##
    def certain_true(self,clf,clf_true):
        pos_at = list(clf.classes_).index("yes")
        prob = clf.predict_proba(self.csr_mat[self.pool])[:,pos_at]
        pos_at_true = list(clf_true.classes_).index("yes")
        prob_true = clf_true.predict_proba(self.csr_mat[self.pool])[:,pos_at_true]
        prob_all = prob*(1-self.true_rate)+prob_true*self.true_rate

        order = np.argsort(prob_all)[::-1][:self.step]
        return np.array(self.pool)[order],np.array(prob_all)[order]

    ## Get random ##
    def random(self):
        return np.array(self.body['Pid'])[np.random.choice(self.pool,size=np.min((self.step,len(self.pool))),replace=False)]

    ## Code candidate studies by abstract ##
    def code(self,Pid,label):
        id=self.body['Pid'].index(Pid)
        self.body["code"][id] = label

    ## Code candidate studies by full content ##
    def true_code(self, Pid, label):
        id = self.body['Pid'].index(Pid)
        self.body["true_code"][id] = label

    ## Code candidate studies with qref_train##
    def auto_code(self, Pid, score, rank):
        id = self.body['Pid'].index(Pid)

        self.body['score'][id] = score
        self.body['rank'][id] = rank
        self.body['cost'][id] = 'AF'
        if Pid in self.abs:
            self.body['cost2'][id] = 'AFS'
            self.body['code'][id] = 'yes'
        else:
            self.body['cost2'][id] = 'AFN'
            self.body['code'][id] = 'no'
        if Pid in self.full:
            self.body['true_code'][id] = 'yes'
        else:
            self.body['true_code'][id] = 'no'

    def get_allpos(self):
        return len(self.abs),len(self.full)

    ## Restart ##
    def restart(self,path="train"):

        self.body['code'] = ['undetermined'] * len(self.body['Pid'])
        self.body['true_code'] = ['undetermined'] * len(self.body['Pid'])
        self.body['cost'] = ['NS'] * len(self.body['Pid'])
        self.body['cost2'] = ['NFN'] * len(self.body['Pid'])
        self.body['score'] = [0] * len(self.body['Pid'])
        self.body['rank'] = [-1] * len(self.body['Pid'])
        self.weight = 0
        self.path = path

        # self.abs=[]
        # self.full=[]
        # ## load review results
        # with open("../workspace/training_data/qrel_abs_train", "r") as f:
        #     abs = f.readlines()
        # with open("../workspace/training_data/qrel_content_train", "r") as f:
        #     full = f.readlines()
        # for ab in abs:
        #     tmp = ab.split()
        #     if self.topic in tmp and tmp[-1].strip() == '1':
        #         self.abs.append(tmp[-2].strip())
        # for ful in full:
        #     tmp = ful.split()
        #     if self.topic in tmp and tmp[-1].strip() == '1':
        #         self.full.append(tmp[-2].strip())
        # exc=[]
        # for ab in self.abs:
        #     if not ab in self.body["Pid"]:
        #         exc.append(ab)
        # for ex in exc:
        #     self.abs.remove(ex)
        # exc = []
        # for ful in self.full:
        #     if not ful in self.body["Pid"]:
        #         exc.append(ful)
        # for ex in exc:
        #     self.full.remove(ex)

        self.save()

