import networkx as nx
import pandas as pd
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
        plt.figure(figsize=(20, 20))
        pos=nx.spring_layout(network, k=0.15)
        nx.draw_networkx(network,pos,arrows=True,node_size=25, node_color='blue', with_labels=False)
        plt.show()

    def cal_degree_of_nodes(self, network: nx.graph.Graph) -> dict:
        """
        Input the network class and output the degree of each nodes.
            eg. {"0x6c96ff26ee153996616b3ab8e6a21c3a8da061f1":12,"0x2faf487a4414fe77e2327f0bf4ae2a264a776ad2":31,....}
        :param network:
        :return: the weight between each nodes
        """
        pass

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
        """
        According the degree of nodes we get from cal_degree_of_nodes, draw the distribution with plt
        :param network: nx.graph
        :return: None
        """
        pass

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
