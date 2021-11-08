import networkx as nx
import pandas as pd
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import powerlaw


class NetworkAnalysis:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def gen_network(self) -> nx.graph.Graph:
        """
        transform the pandas.DataFrame into nx.graph.Graph objects
        :return:
            nx.graph.Graph
        """
        pass

    def cal_degree_of_nodes(self, network: nx.graph.Graph) -> dict:
        
        self.degreeNode = network.degree()
        return dict(self.degreeNode)


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
        # can't make it into a dictionary

        self.nodesWeight = network.edges(data = 'value')
        return self.nodesWeight


    def draw_degree_distribution(self, network: nx.graph) -> None:
        """
        
        function returns the following: 
        1) Degree Distribution (Power Law vs Poisson)
        2) Degree Histogram
        3) Rank Plot
        4) Degree Distribution (with Power Law function)
        
        """
        ### 1 - Degree Distribution (Power Law vs Poisson)
        ## Creating parameters for degre distribution
        n = len(network.nodes)

        # get nodal degree 'k' data as list
        k_g = sorted([d for e, d in self.degreeNode], reverse=True) 

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
        fig1 = plt.figure(figsize=(20, 12))

        # Add plot 1 (degree distribution)
        ax1 = fig1.add_subplot(1, 1, 1)
        ax1.scatter(x_poisson, y_poisson/n_poisson, color='red')
        ax1.set_ylabel("count")
        ax1.set_xlabel("($k$)")

        # Add plot 2 (Poisson distribution)
        ax2 = fig1.add_subplot(1, 1, 1)
        ax2.scatter(p_k[0], p_k[1]/n, marker='o', color='black', alpha=0.7)
        ax2.set_ylabel("$Pr(k = k_{i})$")
        ax2.set_xlabel("Degree $k$")
        ax2.set_yscale('log')
        ax2.set_xscale('log')

        ax2.set_title("Degree Distribution")

        # plt.show()

        ### 2 Degree distribution histogram
        # create figure
        fig2 = plt.figure(figsize=(20, 12))

        # create plot
        ax = fig2.add_subplot(1, 1, 1)

        # plot data
        plt.bar(p_k[0], p_k[1], width=0.9, color="b")

        # transform the scale of axes
        ax.set_xscale('log')
        ax.set_yscale('log')

        # aesthetics
        plt.title("Degree Histogram")
        plt.ylabel("Count")
        plt.xlabel("Degree")

        # plt.show()  


        ### 3 Rank Plot

        # initialize a new figure and plot the data contestually
        
        fig3 = plt.figure(figsize = (20, 12))
         # create plot
        ax = fig3.add_subplot(1, 1, 1)

        # plot data
        plt.loglog(k_g, "b-", marker="o")

        # axes properties
        plt.title("Degree rank plot")
        plt.ylabel("degree")
        plt.xlabel("rank")

        plt.show()


        ### 4 Degree Distribution with Power Law Function

        # color scheme
        plt.style.use('fivethirtyeight')

        # matplotlib optioons
        # ticks
        import pylab
        pylab.rcParams['xtick.major.pad']='8'
        pylab.rcParams['ytick.major.pad']='8'
        #pylab.rcParams['font.sans-serif']='Arial'
        # fotns
        from matplotlib import rc
        rc('font', family='sans-serif')
        rc('font', size=10.0)
        rc('text', usetex=False)
        from matplotlib.font_manager import FontProperties
        panel_label_font = FontProperties().copy()
        panel_label_font.set_weight("bold")
        panel_label_font.set_size(12.0)
        panel_label_font.set_family("sans-serif")

        # data series to plot
        powerlaw_x = [d for n, d in self.degreeNode()]
        
        # fit powerlaw
        fit = powerlaw.Fit(powerlaw_x)
        alpha = fit.power_law.alpha
        sigma = fit.power_law.sigma
        cmp = fit.distribution_compare('power_law', 'exponential')
        ## Plot degree distribution
        # create figure
        fig = plt.figure(figsize=(20, 12))

        # add plot
        ax = fig.add_subplot(1, 1, 1)

        # plot data
        powerlaw.plot_pdf(powerlaw_x, color='orange', ax=ax)
        powerlaw.plot_pdf(powerlaw_x, linear_bins=True, color='black', ax=ax)

        # title
        ax.set_title("Degree distribution for a simulated\nScale Free Network")

        # labels
        ax.set_ylabel(r"$p_{k}$")
        ax.set_xlabel(r"$k$")

        # show plot
        plt.show()
        ## Identify the scaling range
        ## create figure
        fig = plt.figure(figsize=(9, 4))

        # add plots
        ax0 = fig.add_subplot(1, 2, 1)
        ax1 = fig.add_subplot(1, 2, 2)

        # plot data

        # panel A -- we assume the distribution has no upper bound
        # --------------------------------------------------------
        fit = powerlaw.Fit(powerlaw_x, discrete=True, xmax=None, ax=ax0)
        fit.plot_ccdf(color='grey', label=r"Empirical, no $k_{max}$", ax=ax0)
        fit.power_law.plot_ccdf(color='black', linestyle='--', label=r"Fit, no $k_{max}$", ax=ax0)

        # axes
        ax0.set_ylim(10**-4, 2)
        ax0.set_xlim(np.min(powerlaw_x), 10**4)

        # labels
        ax0.set_ylabel(u"$p(k ≥ k_{i})$")
        ax0.set_xlabel(r"$k$")
        handles, labels = ax0.get_legend_handles_labels()

        # textbox
        ax0.text(10**2, 10**-0.5, r'A (no $k_{max}$)', verticalalignment='center', fontsize=18)

        # legend
        leg = ax0.legend(handles, labels, loc=3)
        leg.draw_frame(False)

        # panel B -- we assume the distribution has an upper bound = N (the size of the network)
        # --------------------------------------------------------------------------------------
        fit = powerlaw.Fit(powerlaw_x, discrete=True, xmax=n, sharey=True)
        fit.plot_ccdf(color='grey', label=r"Empirical, $k_{max}=%s$" % n, ax=ax1)
        fit.power_law.plot_ccdf(color='orange', linestyle='--', label=r"Fit, $k_{max}=%s$" % n, ax=ax1)

        # axes
        ax1.set_ylim(10**-4, 2)
        ax1.set_xlim(np.min(powerlaw_x), 10**4)

        # labels
        # ax1.set_ylabel(u"$p(k ≥ k_{i})$")
        ax1.set_xlabel(r"$k$")
        handles, labels = ax1.get_legend_handles_labels()

        # textbox
        ax1.text(10**2, 10**-0.5, r'B ($k_{max} = %s$)' % n, verticalalignment='center', fontsize=18)

        # legend
        leg = ax1.legend(handles, labels, loc=3)
        leg.draw_frame(False)

        # show plot
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
