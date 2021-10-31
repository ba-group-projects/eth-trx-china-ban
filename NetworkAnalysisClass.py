# %%

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

# %%

my_data = ("cleaned_data.csv")

df = pd.read_csv(my_data, header=0)

df.head()


# %%


class NetworkAnalysis:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def gen_network(self) -> nx.graph.Graph:
        """
        transform the pandas.DataFrame into nx.graph.Graph objects
        :return:
            nx.graph.Graph
        """ 
        G = nx.from_pandas_edgelist(self.data,          # the df containing the data
                            source='from_address',      # first element of the dyad
                            target='to_address',        # second element of the dyad
                            edge_attr='value')          # weight
        return G


    def cal_degree_of_nodes(self, network: nx.graph.Graph) -> dict:
        self.network = network
        
        G = network
        nodeDegree = G.degree
        return nodeDegree


    def draw_degree_distribution(self, network: nx.graph) -> None:
        """
        0According the degree of nodes we get from cal_degree_of_nodes, draw the distribution with plt
        :param network: nx.graph
        :return: None
        """

        self #to be filled later)

        # Obtaining the degree sequence
        k = sorted([d for n, d in cal_degree_of_nodes(self)], reverse=True)
        # Count of nodes with degree 'k = k_i'
        p_k = np.unique(k, return_counts=True)

        # Creating the figure
        fig = plt.figure(figsize=(9, 6))

        # Creating the plot
        ax = fig.add_subplot(1, 1, 1)

        # Plotting the graph
        plt.bar(p_k[0], p_k[1], width=0.9, color="b")

        # Labelling the graph
        plt.title("Degree Histogram")
        plt.ylabel("Count")
        plt.xlabel("Degree")

        plt.show()

        pass

# %%
if __name__ == '__main__':
    
    clean_data = NetworkAnalysis(df)
    G = clean_data.gen_network()
    dict = G.cal_degree_of_nodes()


    # nx.draw(G)      #to test and see if it works, which it does!

    # %%

# %%
