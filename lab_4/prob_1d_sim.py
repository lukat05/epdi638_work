from pyDOE import lhs
import numpy.matlib as npm
from statistics import median
import copy as cp
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


nr = 500. # carrying capacity of rabbits ***

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

nsamples = 100
reruns = 2
nparams = 4

# Set up parameter array
params = npm.repmat(lhs(nparams, samples = nsamples),reruns,1)
# Each row is a new parameter set

results = []
for i in range(0, nsamples * reruns):
    print(f"Starting Sample {i + 1} of {nsamples * reruns}... ({(i / (nsamples * reruns)) * 100:.1f}%)")
    nr = params[i, 0] * (700 - 200) + 200  # carrying capacity for rabbits (200 to 700)
    dr = params[i, 1] * 0.5 + 0.5  # death probability for rabbits when encountering fox (0.5 to 1)
    df = params[i, 2] * 0.25  # death probability for foxes when no food available (0 to 0.25)
    rf = params[i, 3] * 0.5 + 0.25  # reproduction probability for foxes when food available (0.25 to 0.75)

    # and the rest of your code goes here!

    initialize()
    for t in range(400):
        update()
        if t % 100 == 0:
            print(f'{t / 4} % done...')
            print(f'{t / 4} % done...')
    results.append((rdata[-1], fdata[-1]))

final_prey = [res[0] for res in results]
final_pred = [res[1] for res in results]

nr_samples = params[:, 0] * (800 - 200) + 200
dr_samples = params[:, 1]
df_samples = params[:, 2]
rf_samples = params[:, 3]

param_names = ['nr (Carrying Capacity)', 'dr (Prey Death Rate)',
               'df (Predator Death Rate)', 'rf (Predator Repro Rate)']
param_samples = [nr_samples, dr_samples, df_samples, rf_samples] # From your LHS logic

# Outputs from Part D
outputs = [final_prey, final_pred]
output_names = ['Final Prey Population', 'Final Predator Population']

fig, axes = plt.subplots(4, 2, figsize=(12, 16))
fig.tight_layout(pad=5.0)

for i in range(4): # Loop through each parameter
    for j in range(2): # Loop through each output (Prey vs Pred)
        axes[i, j].scatter(param_samples[i], outputs[j], alpha=0.6, color='teal')
        axes[i, j].set_xlabel(param_names[i])
        axes[i, j].set_ylabel(output_names[j])
        axes[i, j].set_title(f'{param_names[i]} vs {output_names[j]}')

plt.savefig('images/scat_collage.png', dpi = 300)

# Part D: Histograms of final populations
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(final_prey, bins=15, color='blue', edgecolor='black', alpha=0.7)
plt.title('Distribution of Final Prey Populations')
plt.xlabel('Population Size')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(final_pred, bins=15, color='orange', edgecolor='black', alpha=0.7)
plt.title('Distribution of Final Predator Populations')
plt.xlabel('Population Size')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('images/final_hist.png', dpi = 300)

# Part D: 2D Scatterplot (Predator vs Prey)
plt.figure(figsize=(8, 6))
plt.scatter(final_prey, final_pred, alpha=0.5, c='purple')
plt.title('Final Predator vs. Final Prey Populations')
plt.xlabel('Final Prey')
plt.ylabel('Final Predator')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('images/final_pred_pray_final_scat.png', dpi=300)

plt.figure(figsize=(8, 6))
plt.tricontourf(params[:, 1], params[:, 3], final_pred, cmap='viridis')
plt.colorbar(label='Final Pred Population')
plt.xlabel('dr (Prey Death Rate)')
plt.ylabel('rf (Pred Reproduction Rate)')
plt.title('Heatmap: Pred Pop as a function of dr and rf')
plt.savefig('images/heatmap_fox_pop.png', dpi=300)