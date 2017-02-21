################################################################################
#  TLGProb: Two-Layer Gaussian Process Regression Model For
#           Winning Probability Calculation of Two-Team Sports
#  Github: https://github.com/MaxInGaussian/TLGProb
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

try:
    from TLGProb import TLGProb
except:
    print("TLGProb is not installed yet! Trying to call directly from source...")
    from sys import path
    path.append("../")
    from TLGProb import TLGProb
    print("done.")

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

TLGProb_NBA = TLGProb()
TLGProb_NBA.load_data()
res = TLGProb_NBA.eval_accuracy(2015, True)

## Visualize Result of Evaluation
import numpy as np
import matplotlib.pyplot as plt
n = len(res)
thres, corrs, incorrs = [], [], []
line_x, line_y, line_z = [], [], []
for i in range(501):
    thre = 0.5+i*1./1000
    line_x.append(thre)
    corr, incorr, rej = performance_given_threshold(res, thre)
    acc = 100 if corr+incorr == 0 else corr*100./(corr+incorr)
    thres.append(thre)
    corrs.append(corr)
    incorrs.append(incorr)
    line_y.append(acc)
    rejp = rej*100./n
    line_z.append(rejp)
fig, ax = plt.subplots(figsize=(10, 8), dpi=160)
ax.stackplot(thres, [corrs, incorrs], colors=["green", "red"])
ax.set_xlim([0.5, 1])
p1 = Rectangle((0, 0), 1, 1, fc="green")
p2 = Rectangle((0, 0), 1, 1, fc="red")
legend([p1, p2], ['Correct', 'Incorrect'])
ax.plot(line_x, line_y, 'b-', label="Accuracy")
ax.plot(line_x, line_z, 'r--', label="Rejection percentage")
fig, ax = plt.subplots(figsize=(10, 8), dpi=160)
ax.plot(line_x, line_y, 'b-', label="Accuracy")
ax.plot(line_x, line_z, 'r--', label="Rejection percentage")
plt.xlim([0.5,1.0])
plt.ylim([0.,100.])
plt.xlabel("Acceptance Threshold of Winning Probability")
plt.ylabel("%")
plt.legend(loc="upper center")
plt.tight_layout()
plt.show()