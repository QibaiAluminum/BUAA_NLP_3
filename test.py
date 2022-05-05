import jieba
import random

with open("./merge.txt", "r", encoding="utf-8") as f:
    paras = [eve.strip("\n") for eve in f]
for i in range(len(paras)):
    paras[i]=jieba.lcut(paras[i])
corpus=[]
for i in range(200):
    ran=int(random.random()*len(paras))
    cori=[]
    while(len(cori)<500):
        cori=cori+paras[ran]
        ran=ran+1