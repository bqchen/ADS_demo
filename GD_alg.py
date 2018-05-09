#!/usr/sbin/python
#-*- encoding:utf-8 -*-
#
# http://www.quuxlabs.com/blog/2010/09/matrix-factorization-a-simple-tutorial-and-implementation-in-python/
#

import operator
import numpy as np
import xlrd
import math
with open('ratings_Movies_and_TV.txt') as f:
    prefs_str = ''.join(f.readlines())

# file_path = 'ratings_Movies_and_TV.xlsx'
# data = xlrd.open_workbook(file_path)
# table = data.sheet_by_name('ratings_Movies_and_TV')
# #获取总行数
# nrows = table.nrows
# #获取总列数
# ncols = table.ncols
# for i in range(0,nrows):
#     prefs_str = table.row_values(i)
#     print(prefs_str)

# {'andy': {'霍乱时期的爱情': 1},...}
def read_prefs(prefs_str):
    itemset = set()
    userset = set()
    # rating = set()
    # itemset = []
    # userset = []

    prefs = {}
    for line in prefs_str.split('\n'):
        parts = line.rstrip().split()
        if len(parts) == 3:
            userId, itemId, rating = parts
            itemset.add(itemId)
            userset.add(userId)
            # itemset.append(itemId)
            # userset.append(userId)
            prefs.setdefault(userId,{})
            prefs[userId].update({itemId:rating})
            # prefs[userId].update({itemId:1})

    mat = np.zeros((len(userset), len(itemset)), dtype=int)

    for user in prefs:
        for item in prefs[user]:
            i = list(userset).index(user)
            j = list(itemset).index(item)
            mat[i][j] = prefs[user][item]
    return prefs, mat, list(itemset), list(userset)

def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - alpha * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - alpha * Q[k][j])
        eR = np.dot(P,Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        if e < 0.001:
            break
    return P, Q.T


prefs, mat, items, users = read_prefs(prefs_str)
#
# # print(prefs)
print(mat)
print(items)
print(users)

R = mat
    # [
    #  [5,3,0,1],
    #  [4,0,0,1],
    #  [1,1,0,5],
    #  [1,0,0,4],
    #  [0,1,5,4],
    # ]

R = np.array(R)

N = len(R)
M = len(R[0])
K = 2

P = np.random.rand(N,K)
Q = np.random.rand(M,K)

nP, nQ = matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02)
nR = np.dot(nP, nQ.T)
# print(repr(R))
# print(repr(nP))
# print(repr(nQ))
# print(repr(nR))

print('\n======= 用户矩阵 ========')
for i in range(0, len(nP)):
    print(users[i], nP[i])

print('\n======= 电影矩阵 ========')
for i in range(0, len(nQ)):
    print(items[i], nQ[i])

print('\n======= Matrix Factorization 推荐结果 ========')
f = open("recommend_result.txt", 'w')
f1 = open("recommend_result_sorted.txt", 'w')
tmp_score = [2]
result = []
for i in range(0, len(nP)):
    for j in range(0, len(nQ)):
        score = np.sum(nP[i] * nQ[j])
        score=round(score,3)
        # tmp_score[0] = score
        # if score > 1.1 and R[i][j] == 0:
        if R[i][j] == 0:
            # result = [users[i], items[j], score]
            result.append([users[i],score,items[j]])
            f.write(str(users[i])+' '+ str(items[j])+'  '+str(score)+'\n')
            # print(users[i], items[j], score)

# def cmp(data1,data2):
#     if data1[2] > data2[2]:
#         return -1
#     if data1[2] < data2[2]:
#         return 1
#     return 0


# sorted(result, key=cmp)
result=np.array(result)
for i in range(len(result)):
    result[i][1] = result[i][1].astype('float')
# result=result[np.lexsort(result.T)]
# result=result[np.lexsort(result[:,::-1].T)]
result=result[np.lexsort([result[:,1],result[:,0]]),:]
for i in range(len(result)):
    for j in range(len(result[i])):
        f1.write(str(result[i][j])+'    ')
    f1.write('\n')

# tmp = result[len(result)-1][2]
# count = 0
# for i in result[::-1]:
#     if result[i][0]==result[i-1][0] and tmp <= result[i][2]:
#         if count <= 1:
#             count = count + 1
#             f2.write(str(result[i]))
#     f2.write('\n')





# print(result)






f.close()