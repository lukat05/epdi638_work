# Imports

from pyDOE import lhs
from statistics import median
import copy as cp
import numpy.matlib as npm
import seaborn as sns
import numpy as np
import random
import matplotlib.pyplot as plt
import io
import imageio.v3 as iio
import matplotlib
from IPython import display
import time
import multiprocessing as mp

# Parameters

nrs = [200., 300., 400., 500., 600., 700., 800.]
sim_pred_means = []
sim_prey_means = []

r_init = 100 # initial rabbit population
mr = 0.03 # magnitude of movement of rabbits
dr = 1.0 # death rate of rabbits when it faces foxes ***
rr = 0.1 # reproduction rate of rabbits

f_init = 30 # initial fox population
mf = 0.05 # magnitude of movement of foxes
df = 0.1 # death rate of foxes when there is no food ***
rf = 0.5 # reproduction rate of foxes ***

cd = 0.02 # radius for collision detection
cdsq = cd ** 2

# Model Functions

class agent:
    pass

def initialize():
    global agents, rdata, fdata
    agents = []
    rdata = []
    fdata = []
    for i in range(r_init + f_init):
        ag = agent()
        ag.type = 'r' if i < r_init else 'f'
        ag.x = random.random()
        ag.y = random.random()
        agents.append(ag)

def observe():
    global agents, rdata, fdata

    plt.subplot(2, 1, 1)
    plt.cla()
    rabbits = [ag for ag in agents if ag.type == 'r']
    if len(rabbits) > 0:
        x = [ag.x for ag in rabbits]
        y = [ag.y for ag in rabbits]
        plt.plot(x, y, 'b.')
    foxes = [ag for ag in agents if ag.type == 'f']
    if len(foxes) > 0:
        x = [ag.x for ag in foxes]
        y = [ag.y for ag in foxes]
        plt.plot(x, y, 'ro')
    plt.axis('image')
    plt.axis([0, 1, 0, 1])

    fig = plt.gcf()
    fig.set_size_inches(6, 8)

    plt.subplot(2, 1, 2)
    plt.cla()
    plt.xlim(0, 1000)
    plt.plot(rdata, label = 'prey')
    plt.plot(fdata, label = 'predator')
    plt.legend()
    # plt.show()

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image

def update():
    global agents, rdata, fdata
    t = 0.
    while t < 1. and len(agents) > 0:
        t += 1. / len(agents)
        update_one_agent()

    rdata.append(sum([1 for x in agents if x.type == 'r']))
    fdata.append(sum([1 for x in agents if x.type == 'f']))

def update_one_agent():
    global agents
    if agents == []:
        return

    ag = random.choice(agents)

    # simulating random movement
    m = mr if ag.type == 'r' else mf
    ag.x += random.uniform(-m, m)
    ag.y += random.uniform(-m, m)
    ag.x = 1 if ag.x > 1 else 0 if ag.x < 0 else ag.x
    ag.y = 1 if ag.y > 1 else 0 if ag.y < 0 else ag.y

    # detecting collision and simulating death or birth
    neighbors = [nb for nb in agents if nb.type != ag.type
                 and (ag.x - nb.x)**2 + (ag.y - nb.y)**2 < cdsq]

    if ag.type == 'r':
        if len(neighbors) > 0: # if there are foxes nearby
            if random.random() < dr:
                agents.remove(ag)
                return
        if random.random() < rr*(1-sum([1 for x in agents if x.type == 'r'])/nr):
            agents.append(cp.copy(ag))
    else:
        if len(neighbors) == 0: # if there are no rabbits nearby
            if random.random() < df:
                agents.remove(ag)
                return
        else: # if there are rabbits nearby
            if random.random() < rf:
                agents.append(cp.copy(ag))

def run_single_simulation(nr_val):
    global nr
    nr = nr_val

    initialize()
    for _ in range(400):
        update()

    return rdata[-1], fdata[-1]

if __name__ == '__main__':

    with mp.Pool(processes=4) as pool:
        for _nr in nrs:
            results = pool.map(run_single_simulation, [_nr] * 5)
            prey_vals = [res[0] for res in results]
            pred_vals = [res[1] for res in results]

            sim_prey_means.append(np.mean(prey_vals))
            sim_pred_means.append(np.mean(pred_vals))

    beta_pred, a_pred = np.polyfit(nrs, sim_pred_means, 1)
    beta_prey, a_prey = np.polyfit(nrs, sim_prey_means, 1)

    plt.scatter(nrs, sim_pred_means, color = 'orange', label='Predator Averages')
    plt.scatter(nrs, sim_prey_means, color = 'blue', label='Prey Averages')
    plt.plot(nrs, beta_pred * np.array(nrs) + a_pred, color = 'orange', label='Predator LSRL')
    plt.plot(nrs, beta_prey * np.array(nrs) + a_prey, color = 'blue', label='Prey LSRL')
    plt.xlabel('NR')
    plt.ylabel('Mean Final Agents')
    plt.legend()
    plt.savefig('images/nr_LSRL.png', dpi=300)

