# %%

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

# %%

my_data = ("dataCleaned/cleaned_data.csv")

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
        
        g = network
        
        self.degreeNode = g.degree()
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
        """
        0According the degree of nodes we get from cal_degree_of_nodes, draw the distribution with plt
        :param network: nx.graph
        :return: None
        """

       


        # This is for the histogram 
        # note that the variable DegreeNode is the variable generated from cal_degree_of_nodes

        # Obtaining the degree sequence
        k = sorted([d for n, d in degreeNode], reverse=True)
        # Count of nodes with degree 'k = k_i'
        p_k = np.unique(k, return_counts=True)

        #%% degree distribution plot (Log Scale)

        # create figure
        fig = plt.figure(figsize=(9, 6))

        # create plot
        ax = fig.add_subplot(1, 1, 1)

        # plot data
        plt.bar(p_k[0], p_k[1], width=0.9, color="b")

        # transform the scale of axes
        ax.set_xscale('log')
        ax.set_yscale('log')


        # aesthetics
        plt.title("Degree Histogram")
        plt.ylabel("Count")
        plt.xlabel("Degree")

        plt.show()  



        # cumulative plot
        # note that degreeNodes is the dict generated from cal_degree_of_nodes

        # k = list(self.degreeNodes.values())

        # ds = collections.Counter(k)
        # fig = plt.figure(figsize=(6, 4))
        # ax = fig.add_subplot(1, 1, 1)
        # ax.scatter(ds.keys(), ds.values(), color='k')
        # ax.set_yscale('log')
        # ax.set_xscale('log')
        # ax.set_xlabel('Degree')
        # ax.set_ylabel('Counts of nodes')
        # plt.show()


        

# %%
if __name__ == '__main__':
    
    clean_data = NetworkAnalysis(df)
    G = clean_data.gen_network()
    dict = clean_data.cal_degree_of_nodes(G)
    clean_data.draw_degree_distribution(G)


    # nx.draw(G)      #to test and see if it works, which it does!

    # %%

# %%
