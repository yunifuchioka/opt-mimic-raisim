import os
import numpy as np
import raisimpy as raisim
import time

raisim.World.setLicenseFile(os.path.dirname(
    os.path.abspath(__file__)) + "/../../rsc/activation.raisim")
go1_urdf_file = os.path.dirname(os.path.abspath(
    __file__)) + "/../../rsc/go1/urdf/go1_v2.urdf"

world = raisim.World()
world.setTimeStep(0.001)
server = raisim.RaisimServer(world)
ground = world.addGround()

data = np.loadtxt("go1_motions/raise your rear right leg.csv",delimiter=",")

# data front right --> sim rear left

data_pos = data[:,6:18]
data_vel = data[:,18:]

dataFL = data_pos[:,0:3]
dataFL_vel = data_vel[:,0:3]
dataFR = data_pos[:,3:6]
dataFR_vel = data_vel[:,3:6]
dataRL = data_pos[:,6:9]
dataRL_vel = data_vel[:,6:9]
dataRR = data_pos[:,9:]
dataRR_vel = data_vel[:,9:]

# SIM INFORMATION
# Front Left --> [16:]
# Front Right --> [13:16]
# Rear Left --> [10:13]
# Rear Right --> [7:10]

# Control Information
# Frequency: 50Hz


go1 = world.addArticulatedSystem(go1_urdf_file)
go1.setName("Go1")

go1_nominal_joint_config = np.array([0, -1.5, 0.30, 1.0, 0.0, 0.0, 0.0, 0.0, 0.8, -1.7,
                                        0.00, 0.8, -1.7, 0.0, 0.8, -1.7, 0.00, 0.8, -1.7])

offset_pl = np.array([0.0,0.8,-1.7])

go1.setGeneralizedCoordinate(go1_nominal_joint_config)
go1.setPdGains(200*np.ones([18]), np.ones([18]))
go1.setPdTarget(go1_nominal_joint_config, np.zeros([18]))

server.launchServer(8080)

ep_length = len(data)
run_time = ep_length*20+1

def updateTargets(index):
    p_targets = np.zeros(19)
    d_targets = np.zeros(18)
    p_targets[7:10] = dataRR[index] + offset_pl
    p_targets[10:13] = dataRL[index] + offset_pl
    p_targets[13:16] = dataFR[index] + offset_pl
    p_targets[16:] = dataFL[index] + offset_pl

    d_targets[6:9] = dataRR_vel[index]
    d_targets[9:12] = dataRL_vel[index]
    d_targets[12:15] = dataFR_vel[index]
    d_targets[15:] = dataFL_vel[index]

    go1.setPdTarget(p_targets,d_targets)
    world.integrate()

time.sleep(2)
world.integrate1()
counter = 1

for i in range(1000000000):
    time.sleep(0.001)
    world.integrate()
    if i % 20 == 0:
        updateTargets(counter)
        counter+=1

server.killServer()
