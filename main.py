import os
import pandas as pd
from utils.analyze import NetworkAnalysis
import matplotlib.pyplot as plt
 
path = os.getcwd()
df = pd.read_csv(path+'\data\cleaned_data.csv')
df = df.drop(df.columns[0],axis=1)

network_analysis_obj = NetworkAnalysis(df)
monopartite_full = network_analysis_obj.gen_network(network_type='monopartite_full')
monopartite_gcc = network_analysis_obj.gen_network(network_type='monopartite_gcc')

fig = plt.figure(figsize=(50, 25), constrained_layout=True)
ax1 = fig.add_subplot(121)
network_analysis_obj.plot_network(monopartite_full, network_type='monopartite_full')
ax1.set_title('Monopartite Full', fontdict={'fontsize': 75})

ax2 = fig.add_subplot(122)
network_analysis_obj.plot_network(monopartite_gcc, network_type='monopartite_gcc')
ax2.set_title('Monopartite GCC', fontdict={'fontsize': 75})

plt.show()

network_analysis_obj = NetworkAnalysis(df)
bipartite_full = network_analysis_obj.gen_network(network_type='bipartite_full')
bipartite_gcc = network_analysis_obj.gen_network(network_type='bipartite_gcc')

fig = plt.figure(figsize=(50, 25), constrained_layout=True)
ax1 = fig.add_subplot(121)
network_analysis_obj.plot_network(bipartite_full, network_type='bipartite_full')
ax1.set_title('Bipartite Full', fontdict={'fontsize': 75})

ax2 = fig.add_subplot(122)
network_analysis_obj.plot_network(bipartite_gcc, network_type='bipartite_gcc')
ax2.set_title('Bipartite GCC', fontdict={'fontsize': 75})

plt.show()