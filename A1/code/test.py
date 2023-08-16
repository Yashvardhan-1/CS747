import numpy as np
import math
import time





def tomp(succ, tot):
    return np.random.beta(succ+1, tot-succ+1)

vec_tomp = np.vectorize(tomp)

succ = np.array([10, 11, 140, 0, 5, 31])
tot = np.array([21, 43, 150, 2, 7, 80])

mp = {}
for i in range(100):
    tomp = vec_tomp(succ, tot)
    m = np.argmax(tomp)
    if str(m) in mp.keys():
        mp[str(m)] += 1
    else:
        mp[str(m)] = 1

keys = []
values = []
for k in mp.keys():
    tot[int(k)] += mp[k]
    print(f"{k} {mp[k]}")
    keys += [int(k)]
    values += [mp[k]]

print(keys)
print(values)


# def KL(p, q):
# 	if p == 1:
# 		return p*np.log(p/q)
# 	elif p == 0:
# 		return (1-p)*np.log((1-p)/(1-q))
# 	else:
# 		return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))

# def KL_UCB(e_mean, steps, pulls, c=3):
#     l = e_mean
#     r = 1
#     eps = 1e-9
#     temp = e_mean
#     qold = e_mean
#     q = l + (r-l)/2
#     rhs = (math.log(steps) + c*math.log(math.log(steps)))/pulls

#     while(abs(qold-q)>eps):
#         # print(f"l:{l} and r:{r} and m:{q}")
#         temp = KL(e_mean, q)
#         # print(temp) 
#         if temp > rhs:
#             r = q
#         elif temp == 0:
#             break
#         else:
#             l = q
#         qold = q
#         q = l + (r-l)/2

#     # print(q)
#     return float(q)


# I = np.array([0.8, 0.2, 0.9, 0.3, 0.1, 0.1, 0.1], dtype = float)
# T = np.array([10, 5, 6, 9, 5, 2, 1400])
# steps = np.sum(T)
# print("yash")

# vec_KL = np.vectorize(KL_UCB)

# l = []
# for i in range(I.size):
#     l += [KL_UCB(I[i], steps, T[i])]
#     pass

# l = np.array(l)
# x = vec_KL(I,steps,T)

# for l1, l2 in zip(l,x):
#     print(f"l1:{l1} __ l2:{l2}")