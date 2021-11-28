
import subprocess
import numpy as np
from matplotlib import pyplot as plt

# get line numbers of "---- reached final state ----"
command1 = ['cat', 'logs/train_out.log']
command2 = ['grep', '-n', 'reached']
command3 = ['cut', '-d:', '-f1']
p1 = subprocess.Popen(command1, stdout=subprocess.PIPE)
p2 = subprocess.Popen(command2, stdin=p1.stdout, stdout=subprocess.PIPE)
p3 = subprocess.Popen(command3, stdin=p2.stdout, stdout=subprocess.PIPE)
out2 = p3.communicate()[0].decode('utf-8').split('\n')[:-1]
# a2 = np.array(list(map(lambda x: int(x), out2)))
a2 = np.array([int(x_) for x_ in out2 if len(x_)>0 ])

lengths = []
indxs = []
for i in range(19):
    # get line numbers of ind=I
    command1 = ['cat', 'logs/train_out.log']
    command2 = ['grep', '-n', 'ind={},'.format(i)]
    command3 = ['cut', '-d:', '-f1']
    p1 = subprocess.Popen(command1, stdout=subprocess.PIPE)
    p2 = subprocess.Popen(command2, stdin=p1.stdout, stdout=subprocess.PIPE)
    p3 = subprocess.Popen(command3, stdin=p2.stdout, stdout=subprocess.PIPE)
    out = p3.communicate()[0].decode('utf-8').split('\n')[:-1]
    if len(out)<=0:
        print(f'i={i}')
        continue
    else:
        print(f'i={i}')
    # a = np.array(list(map(lambda x: int(x), out)))
    del out[-1]
    a = np.array([int(x_) for x_ in out if len(x_)>0 ])
    indxs.append(a)
    lengths.append([np.min(c[np.where(c>0)]) for c in [a2-a_ for a_ in a]])

# plt.ion()
# plt.figure('0')
l = [len(l_) for l_ in lengths]
# plt.plot(l)
# plt.show()

l_max = np.max(l)
indxs_max = np.max([np.max(iii) for iii in indxs])
fig, axs = plt.subplots(len(lengths),1)
# axs.set_ylim()
for i, (l_, axs_, indxs_) in enumerate(zip(lengths, axs, indxs)):
    axs_.plot(indxs_,l_,marker='.')
    axs_.set_ylabel(f'{len(l_)}')
    axs_.set_ylim([0, 80])
    axs_.set_xlim([0, indxs_max])
plt.show()

# plt.waitforbuttonpress()

