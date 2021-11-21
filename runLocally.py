import subprocess
import time
import os
import signal
from multiprocessing import Pool
import errno
from contextlib import contextmanager
TIME = 2 * 60  # (30 min)
from functools import wraps

import os
thisFile = os.path.abspath(__file__)
thisFileDir = os.path.dirname(thisFile)
proc_list = []
a = []

fid_sim_out = open('logs/sim_out.log', 'a', buffering=1)
a.append(fid_sim_out)
fid_sim_err = open('logs/sim_err.log', 'a', buffering=1)
a.append(fid_sim_err)
fid_train_out = open('logs/train_out.log', 'a', buffering=1)
a.append(fid_train_out)
fid_train_err = open('logs/train_err.log', 'a', buffering=1)
a.append(fid_train_err)
fid_rviz_out = open('logs/rviz_out.log', 'a', buffering=1)
a.append(fid_rviz_out)
fid_rviz_err = open('logs/rviz_err.log', 'a', buffering=1)
a.append(fid_rviz_err)

for a_ in a:
    a_.write('---------------------------------------')


from testEnvVars import my_env_mod
# # my_env_mod = os.environ.copy()
# # import pdb;pdb.set_trace()
my_env_mod['PKG_CONFIG_PATH'] = '/home/roees/tmp_ws/devel/lib/pkgconfig:'+my_env_mod['PKG_CONFIG_PATH']
my_env_mod['LD_LIBRARY_PATH'] = '/home/roees/tmp_ws/devel/lib:'+my_env_mod['LD_LIBRARY_PATH']
command_run_simulation  = '~/tmp_ws/src/cf_simple_sim/utils/simulator.sh gui:=true'
sim_proc = subprocess.Popen([command_run_simulation], shell=True, stdout=fid_sim_out, stderr=fid_sim_err, env=my_env_mod)
proc_list.append(sim_proc)
time.sleep(5)

command_run_train  = thisFileDir+'/launcher.py 31'
print(command_run_train)
train_proc = subprocess.Popen([command_run_train], shell=True, stdout=fid_train_out, stderr=fid_train_err)
proc_list.append(train_proc)

command_launch_rviz = 'rviz -d ~/tmp_ws/src/cf_simple_sim/utils/conf.rviz'
rviz_proc = subprocess.Popen([command_launch_rviz], shell=True, stdout=fid_rviz_out, stderr=fid_rviz_err)
proc_list.append(rviz_proc)


time.sleep(TIME)
for p in proc_list:
    os.killpg(os.getpgid(p.pid), signal.SIGTERM)  # Send the signal to all the process groups

fid_sim_out.close()
fid_sim_err.close()
fid_train_out.close()
fid_train_err.close()
fid_rviz_out.close()
fid_rviz_err.close()