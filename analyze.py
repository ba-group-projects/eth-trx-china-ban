import networkx as nx
import pandas as pd
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt


class NetworkAnalysis:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def gen_network(self) -> nx.graph.Graph:
        """
        transform the pandas.DataFrame into nx.graph.Graph objects
        :return:
            nx.graph.Graph
        """
        G = nx.from_pandas_edgelist(self.data,                # the df containing the data
                            source='from_address',        # first element of the dyad
                            target='to_address',        # second element of the dyad
                            edge_attr='value')
        return G

    def plot_network(self, network: nx.graph.Graph):
        plt.figure(figsize=(50,50))
        pos=nx.spring_layout(network, k=0.15)
        nx.draw_networkx(network,pos,arrows=True,node_size=50, node_color='blue', with_labels=False)
        plt.show()

    def cal_degree_of_nodes(self, network: nx.graph.Graph) -> dict:
        
        # self.network = network
        # self.degreeNode = degreeNode      # ask Benedict

        self.degreeNode = dict(network.degree())
        return self.degreeNode


    def cal_weight_of_nodes(self, network: nx.graph.Graph) -> dict:
        """
        Input the network class and output the weight between each nodes.
            {"from_address_1":
                [{"to_address_1":12},"to_address_2":13}],
             "from_address_2":
                [{"to_address_1":12},"to_address_2":13}],
            }
        :param network:
        :return: the weight between each nodes
        """
        pass

    def draw_degree_distribution(self, network: nx.graph) -> None:
        
        ## Creating parameters for degre distribution
        n = len(network.nodes)

        # get nodal degree 'k' data as list
        k_g = sorted([d for n, d in self.degreeNode], reverse=True)  #add degreeNode here

        # get 'p_k'
        p_k = np.unique(k_g, return_counts=True)
   
        ## Creating parameters for poisson distribution
        # Creating parameters for poisson (lam = average degree of graph)
        average = sum(k_g)/n
        poisson_dist = np.random.poisson(lam=average, size=len(network.nodes()))
        k_poisson = Counter(poisson_dist)
        n_poisson = len(poisson_dist)

        x_poisson = list(k_poisson.keys())
        x_poisson = pd.DataFrame(x_poisson)
        y_poisson = list(k_poisson.values())
        y_poisson = pd.DataFrame(y_poisson)


        ## Create figure
        fig = plt.figure(figsize=(20, 12))

        # Add plot 1 (degree distribution)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.scatter(x_poisson, y_poisson/n_poisson, color='red')
        ax1.set_ylabel("count")
        ax1.set_xlabel("($k$)")

        # Add plot 2 (Poisson distribution)
        ax2 = fig.add_subplot(1, 1, 1)
        ax2.scatter(p_k[0], p_k[1]/n, marker='o', color='black', alpha=0.7)
        ax2.set_ylabel("$Pr(k = k_{i})$")
        ax2.set_xlabel("Degree $k$")
        ax2.set_yscale('log')
        ax2.set_xscale('log')

        ax2.set_title("Degree Distribution")

        plt.show()

    def cal_betweenness_centrality(self, network: nx.graph) -> dict:
        """
        Use betweenness_centrality method in networkx to calculate betweenness
        :param network:
        :return: dict
        """
        pass

    def detect_core_periphery(self, network: nx.graph):
        """
        There is a package for detecting core-periphery structure in networks on github, we could directly use this one:
            link:https://github.com/skojaku/core-periphery-detection
        Let's figure out how to use it in our model.
        :param network: 
        :return:
        """
        pass
