import matplotlib.pyplot as plt
from matplotlib import rc, rcParams

plt.rc('xtick', labelsize=20) # tick labels
plt.rc('xtick.major', size=8)
plt.rc('xtick.minor', size=4)
plt.rc('ytick', labelsize=20) # tick labels
plt.rc('ytick.major', size=8)
plt.rc('ytick.minor', size=4)

#xtick.major.width    : 0.8    ## major tick width in points
#xtick.minor.width    : 0.6

plt.rcParams['axes.linewidth']  = 2
plt.rcParams['axes.labelsize']  = 24 # axes labels
plt.rcParams['axes.titlesize']  = 22
plt.rcParams['xtick.top']       = True
plt.rcParams['ytick.right']     = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

plt.rcParams['xtick.major.width'] = 2
plt.rcParams['xtick.minor.width'] = 2
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['ytick.minor.width'] = 2

plt.rcParams['legend.handlelength']  = 1.5
plt.rcParams['legend.fontsize']      = 18
plt.rcParams['legend.handletextpad'] = 0.5
plt.rcParams['legend.borderaxespad'] = 1.
plt.rcParams['legend.borderpad']     = 0.5
plt.rcParams['legend.edgecolor']     = 'black'
plt.rcParams['legend.numpoints']     = 1
plt.rcParams['legend.labelspacing']  = 0.3
plt.rcParams['legend.frameon']       = True
plt.rcParams['legend.framealpha']    = 0.4
plt.rcParams['legend.handlelength'] = 1.125
plt.rcParams['legend.handleheight'] = 0.8
plt.rcParams['text.latex.preamble'] = r'\usepackage{newtxmath}'
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif']  = 'Times'

# rc('text', usetex=True)

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Helvetica"]})
# for Palatino and other serif fonts use:
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
})
