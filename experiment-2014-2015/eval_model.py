"""
Created on Tue Sep 15 17:11:39 2015
@author: Max W. Y. Lam
"""
import sys
sys.path.append("../")
from models import basketball_model
import matplotlib.pyplot as plt


def performance_given_threshold(res, thre):
    corr, incorr, rej = 0, 0, 0
    for prob, zdiff in res:
        if(prob > thre):
            if(zdiff < 0):
                corr += 1
            elif(zdiff > 0):
                incorr += 1
        elif(1-prob > thre):
            if(zdiff > 0):
                corr += 1
            elif(zdiff < 0):
                incorr += 1
        else:
            rej += 1
    return corr, incorr, rej

bas = basketball_model()
bas.load_data()
res = bas.eval_accuracy(2015, True)
n = len(res)
line_x, line_y, line_z = [], [], []
for i in range(501):
    thre = 0.5+i*1./1000
    line_x.append(thre)
    corr, incorr, rej = performance_given_threshold(res, thre)
    acc = corr*100./n
    line_y.append(acc)
    rejp = rej*100./n
    line_z.append(rejp)
fig, ax = plt.subplots(figsize=(10, 6), dpi=80)
ax.plot(line_x, line_y, 'b-', label="Accuracy")
ax.plot(line_x, line_z, 'r--', label="Rejection percentage")
plt.xlim([0.5,1.0])
plt.ylim([0.,100.])
plt.xlabel("Acceptance Threshold of Winning Probability")
plt.ylabel("%")
plt.legend(loc="upper center")
plt.tight_layout()
plt.show()