# %% import packages
import itertools
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
from networkx.algorithms.community import girvan_newman, modularity
from pprint import pprint as pp

# %% load in the dataset
path = os.getcwd()
df = pd.read_csv(path+'/cleaned_data_1000_nodes.csv')
df = df.drop(df.columns[0], axis=1)
df.head(5)

# %% process dataset
userSpace = df[(df.from_address_type == 'EOA') & (df.to_address_type == 'EOA')]
contractSpace = df[(df.from_address_type == 'Contract')
                   & (df.to_address_type == 'Contract')]
bipartiteSpace = df[((df.from_address_type == 'EOA') & (df.to_address_type == 'Contract')) | (
    (df.from_address_type == 'Contract') & (df.to_address_type == 'EOA'))]


userSpace = userSpace[['from_address', 'to_address', 'value']]
userSpace.head()

# %% creating a graph
G = nx.from_pandas_edgelist(userSpace,                # the df containing the data
                            source='from_address',        # first element of the dyad
                            target='to_address',        # second element of the dyad
                            edge_attr='value')

pp(nx.info(G))

# %% drawing the network
pos = nx.spring_layout(G)

# %% girvan-newman method
solutions = girvan_newman(G)

tuple(sorted(c) for c in next(solutions))

k = 20

modularity_scores = dict()

# iterate over solutions
for community in itertools.islice(solutions, k):
    solution = list(sorted(c) for c in community)
    score = modularity(G, solution)
    modularity_scores[len(solution)] = score
    
# %% plot modularity dataset
fig = plt.figure()
pos = list(modularity_scores.keys())
values = list(modularity_scores.values())
ax = fig.add_subplot(1, 1, 1)
ax.stem(pos, values)
ax.set_xticks(pos)
ax.set_xlabel(r'Number of communities detected')
ax.set_ylabel(r'Modularity score')
plt.show()

# %% creating a function for graphing
pos = nx.spring_layout(G)

def draw(G, pos, measures, measure_name):

    nodes = nx.draw_networkx_nodes(G, pos, node_size=0, cmap=plt.cm.plasma,
                                   node_color=list(measures.values()),
                                   nodelist=measures.keys())
    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1, base=10))
    # labels = nx.draw_networkx_labels(G, pos)
    edges = nx.draw_networkx_edges(G, pos)
    
    plt.title(measure_name)
    plt.colorbar(nodes)
    plt.axis('off')
    plt.show()

# %% betweenness centrality
bet_centrality = nx.betweenness_centrality(G)

draw(G, pos, nx.betweenness_centrality(G), 'Betweenness Centrality')


# %% degree centrality
node_dc = nx.degree_centrality(G)

draw(G, pos, nx.degree_centrality(G), 'Degree Centrality')

# %% closesness centrality
close_centrality = nx.closeness_centrality(G)

draw(G, pos, nx.closeness_centrality(G), 'Closeness Centrality')

# %%
