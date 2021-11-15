import os
import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms import bipartite as bp
import matplotlib.pyplot as plt
from collections import Counter
from prettytable import PrettyTable
from operator import itemgetter

class NetworkAnalysis:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def gen_network(self, network_type) -> nx.graph.Graph:
        """
        transform the pandas.DataFrame into nx.graph.Graph objects
        :return:
            nx.graph.Graph
        """
        df = self.data

        # select top 5k transactions for pre ban
        df = df.sort_values(by=['value'], ascending=False).head(5000)
        
        if network_type == 'monopartite':
            mono_df = df[(df.from_address_type == 'EOA') & (df.to_address_type == 'EOA')].reset_index(drop=True)

            mono_network = nx.DiGraph()
            mono_network.add_nodes_from(np.unique(np.array(mono_df['from_address'].append(mono_df['to_address']))))
            mono_network.add_weighted_edges_from(list(zip(mono_df['from_address'],mono_df['to_address'],mono_df['value'])))

            final_network = mono_network

        elif network_type == 'bipartite':
            bi_df = df[((df.from_address_type == 'EOA') & (df.to_address_type == 'Contract')) | ((df.from_address_type == 'Contract') & (df.to_address_type == 'EOA'))].reset_index(drop=True)

            bi_users = np.unique(np.array(bi_df['from_address']))
            bi_contracts = np.unique(np.array(bi_df['to_address']))
            bi_edges = list(zip(bi_df['from_address'], bi_df['to_address'], bi_df['value']))

            bi_network = nx.DiGraph()
            bi_network.add_nodes_from(bi_users, bipartite=0)
            bi_network.add_nodes_from(bi_contracts, bipartite=1)
            bi_network.add_weighted_edges_from(bi_edges)

            final_network = bi_network

        return final_network

    def plot_network(self, network: nx.graph.Graph, network_type):
        # aesthetic variables
        seed = 1; user_color = 'blue'; contract_color = 'red'
        node_size = 200; contract_node_shape = 's'; alpha = 0.25
        arrowsize=30; edge_width=1.5

        if network_type == 'monopartite':
            mono_pos=nx.spring_layout(network, seed = seed)
            nx.draw_networkx_nodes(network, pos=mono_pos, nodelist=list(network.nodes()),node_color=user_color, 
                                   node_size=node_size, alpha=alpha)
            nx.draw_networkx_edges(network, pos=mono_pos, width=edge_width, alpha=alpha, arrows=True, arrowsize=arrowsize)                                

        elif network_type == 'bipartite':
            users=[]; contracts=[]
            for node in network.nodes(data=True):
                address = node[0]
                attribute = node[1]['bipartite']
                if attribute == 0:
                    users.append(address)
                elif attribute == 1:
                    contracts.append(address)

            bi_pos=nx.spring_layout(network, seed = seed)
            nx.draw_networkx_nodes(network, pos=bi_pos, nodelist=list(users),node_color=user_color, 
                                   node_size=node_size, alpha=alpha)
            nx.draw_networkx_nodes(network, pos=bi_pos, nodelist=list(contracts),node_color=contract_color,
                                   node_size=node_size, alpha=alpha)
            nx.draw_networkx_edges(network, pos=bi_pos, width=edge_width, alpha=alpha, arrows=True, arrowsize=arrowsize)

    def plot_network_comparison(obj_list, network_list, figsize=(50,50), title_fontsize=75):
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        ax1 = fig.add_subplot(221)
        ax1.set_axis_off()
        ax1_1 = fig.add_subplot(221)
        ax1_1.set_title('Monopartite Before Ban', fontdict={'fontsize': title_fontsize})
        obj_list[0].plot_network(network_list[0], network_type='monopartite')

        ax2 = fig.add_subplot(222)
        ax2.set_axis_off()
        ax2_1 = fig.add_subplot(222)
        ax2_1.set_title('Monopartite After Ban', fontdict={'fontsize': title_fontsize})
        obj_list[1].plot_network(network_list[1], network_type='monopartite')

        ax3 = fig.add_subplot(223)
        ax3.set_axis_off()
        ax3_1 = fig.add_subplot(223)
        ax3_1.set_title('Bipartite Before Ban', fontdict={'fontsize': title_fontsize})
        print(f'Network before ban is bipartite: {nx.is_bipartite(network_list[2])}')
        obj_list[0].plot_network(network_list[2], network_type='bipartite')

        ax4 = fig.add_subplot(224)
        ax4.set_axis_off()
        ax4_1 = fig.add_subplot(224)
        ax4_1.set_title('Bipartite After Ban', fontdict={'fontsize': title_fontsize})
        print(f'Network after ban is bipartite: {nx.is_bipartite(network_list[3])}')
        obj_list[1].plot_network(network_list[3], network_type='bipartite')
        plt.show()

    def cal_degree(network: nx.graph.Graph):
        dv = dict(network.degree())
        k = list(dv.values())
        return k

    def degree_stats(pre_ban_mono_k, post_ban_mono_k, pre_ban_bi_k, post_ban_bi_k):
        print("Overall Degree Summary Statistics:")
        # pre_ban_mono_dv = dict(network1.degree())
        # pre_ban_mono_k = list(pre_ban_mono_dv.values())

        # post_ban_mono_dv = dict(network2.degree())
        # post_ban_mono_k = list(post_ban_mono_dv.values())

        # pre_ban_bi_dv = dict(network3.degree())
        # pre_ban_bi_k = list(pre_ban_bi_dv.values())

        # post_ban_bi_dv = dict(network4.degree())
        # post_ban_bi_k = list(post_ban_bi_dv.values())                      

        degree_table = PrettyTable(["Stats", "Monopartite Pre-Ban", "Monopartite Post-Ban", 
                                    "Bipartite Pre-Ban", "Bipartite Post-Ban"])

        degree_table.add_row(['Mean',round(np.mean(pre_ban_mono_k),2), round(np.mean(post_ban_mono_k),2), 
                            round(np.mean(pre_ban_bi_k),2), round(np.mean(post_ban_bi_k),2)])
        degree_table.add_row(['Min',np.min(pre_ban_mono_k), np.min(post_ban_mono_k), 
                            np.min(pre_ban_bi_k), np.min(post_ban_bi_k)])
        degree_table.add_row(['Max',np.max(pre_ban_mono_k), np.max(post_ban_mono_k), 
                            np.max(pre_ban_bi_k), np.max(post_ban_bi_k)])
        degree_table.add_row(['Std',round(np.std(pre_ban_mono_k),2), round(np.std(post_ban_mono_k),2), 
                            round(np.std(pre_ban_bi_k),2), round(np.std(post_ban_bi_k),2)])
        print(degree_table)

##### FIX AXES RANGES #####
    def plot_degree_dist(network1: nx.graph.Graph, network2: nx.graph.Graph, suptitle_label, figsize=(12,8)):
        alpha = 0.5
        fig = plt.figure(figsize=figsize,constrained_layout=True)
        plt.suptitle(suptitle_label, fontsize=15)
        # Overall (formula is very similar as the degree distribution summary)
        k_overall = sorted([d for n, d in network1.degree()], reverse=True) #Gets all the degree in a list
        p_k_overall = np.unique(k_overall, return_counts=True)

        # create plot
        ax1 = fig.add_subplot(231)
        # plot data
        ax1.bar(p_k_overall[0], p_k_overall[1], width=1, color="k", edgecolor="black",alpha=alpha)
        # transform the scale of axes
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        # aesthetics
        ax1.set_title("Degree Dist. Before Ban")
        ax1.set_ylabel("Count")
        ax1.set_xlabel("Degree")

        ## Indegree Plot
        k_indegree = sorted([d for n, d in network1.in_degree()], reverse=True)
        p_k_indegree = np.unique(k_indegree, return_counts=True)

        # create plot
        ax2 = fig.add_subplot(232)
        # plot data
        ax2.bar(p_k_indegree[0], p_k_indegree[1], width=1, color="r", edgecolor="black",alpha=alpha)
        # transform the scale of axes
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        # aesthetics
        ax2.set_title("In-Degree Dist. Before Ban")
        ax2.set_ylabel("Count")
        ax2.set_xlabel("Degree")

        ## OutDegree Plot
        k_outdegree = sorted([d for n, d in network1.out_degree()], reverse=True)
        p_k_outdegree = np.unique(k_outdegree, return_counts=True)

        # create plot
        ax3 = fig.add_subplot(233)
        # plot data
        ax3.bar(p_k_outdegree[0], p_k_outdegree[1], width=1, color="g", edgecolor="black",alpha=alpha)
        # transform the scale of axes
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        # aesthetics
        ax3.set_title("Out-Degree Dist. Before Ban")
        ax3.set_ylabel("Count")
        ax3.set_xlabel("Degree")

        # Overall (formula is very similar as the degree distribution summary)
        k_overall = sorted([d for n, d in network2.degree()], reverse=True) #Gets all the degree in a list
        p_k_overall = np.unique(k_overall, return_counts=True)
        
        # create plot
        ax4 = fig.add_subplot(234)
        # plot data
        ax4.bar(p_k_overall[0], p_k_overall[1], width=1, color="k", edgecolor="black",alpha=alpha)
        # transform the scale of axes
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        # aesthetics
        ax4.set_title("Degree Dist. After Ban")
        ax4.set_ylabel("Count")
        ax4.set_xlabel("Degree")

        ## Indegree Plot
        k_indegree = sorted([d for n, d in network2.in_degree()], reverse=True)
        p_k_indegree = np.unique(k_indegree, return_counts=True)

        # create plot
        ax5 = fig.add_subplot(235)
        # plot data
        ax5.bar(p_k_indegree[0], p_k_indegree[1], width=1, color="r", edgecolor="black",alpha=alpha)
        # transform the scale of axes
        ax5.set_xscale('log')
        ax5.set_yscale('log')
        # aesthetics
        ax5.set_title("In-Degree Dist. After Ban")
        ax5.set_ylabel("Count")
        ax5.set_xlabel("Degree")

        ## OutDegree Plot
        k_outdegree = sorted([d for n, d in network2.out_degree()], reverse=True)
        p_k_outdegree = np.unique(k_outdegree, return_counts=True)

        # create plot
        ax6 = fig.add_subplot(236)
        # plot data
        ax6.bar(p_k_outdegree[0], p_k_outdegree[1], width=1, color="g", edgecolor="black",alpha=alpha)
        # transform the scale of axes
        ax6.set_xscale('log')
        ax6.set_yscale('log')
        # aesthetics
        ax6.set_title("Out-Degree Dist. After Ban")
        ax6.set_ylabel("Count")
        ax6.set_xlabel("Degree")

        plt.show()

    def plot_power_law(network1: nx.graph.Graph,network2: nx.graph.Graph,network3: nx.graph.Graph,network4: nx.graph.Graph,figsize=(12, 8)):
        fig = plt.figure(figsize=(12, 8), constrained_layout=True)

        # Create n value for finding the k below
        n = len(network1.nodes)
        # get nodal degree 'k' data as list
        k_g = sorted([d for n, d in network1.degree()], reverse=True)
        # get 'p_k'
        p_k = np.unique(k_g, return_counts=True)
        
        ax1 = fig.add_subplot(221)
        # --+ plot data
        ax1.scatter(p_k[0], p_k[1]/n, marker='o', color='black', alpha=0.7)
        # --+ title
        ax1.set_title("Monopartite Before Ban")
        ax1.set_ylabel("$Pr(k = k_{i})$")
        ax1.set_xlabel("Degree $k$")
        ax1.set_yscale('log')
        ax1.set_xscale('log')

        n = len(network2.nodes)
        # get nodal degree 'k' data as list
        k_g = sorted([d for n, d in network2.degree()], reverse=True)
        # get 'p_k'
        p_k = np.unique(k_g, return_counts=True)

        ax2 = fig.add_subplot(222)
        # --+ plot data
        ax2.scatter(p_k[0], p_k[1]/n, marker='o', color='black', alpha=0.7)
        # --+ title
        ax2.set_title("Monopartite After Ban")
        ax2.set_ylabel("$Pr(k = k_{i})$")
        ax2.set_xlabel("Degree $k$")
        ax2.set_yscale('log')
        ax2.set_xscale('log')

        n = len(network3.nodes)
        # get nodal degree 'k' data as list
        k_g = sorted([d for n, d in network3.degree()], reverse=True)
        # get 'p_k'
        p_k = np.unique(k_g, return_counts=True)

        ax3 = fig.add_subplot(223)
        # --+ plot data
        ax3.scatter(p_k[0], p_k[1]/n, marker='o', color='black', alpha=0.7)
        # --+ title
        ax3.set_title("Bipartite Before Ban")
        ax3.set_ylabel("$Pr(k = k_{i})$")
        ax3.set_xlabel("Degree $k$")
        ax3.set_yscale('log')
        ax3.set_xscale('log')

        n = len(network4.nodes)
        # get nodal degree 'k' data as list
        k_g = sorted([d for n, d in network4.degree()], reverse=True)
        # get 'p_k'
        p_k = np.unique(k_g, return_counts=True)

        ax4 = fig.add_subplot(224)
        # --+ plot data
        ax4.scatter(p_k[0], p_k[1]/n, marker='o', color='black', alpha=0.7)
        # --+ title
        ax4.set_title("Bipartite After Ban")
        ax4.set_ylabel("$Pr(k = k_{i})$")
        ax4.set_xlabel("Degree $k$")
        ax4.set_yscale('log')
        ax4.set_xscale('log')
        plt.show()

    def cal_betweenness_cen(network: nx.graph.Graph):
        # Calculating the Betweenness Centrality for a network
        bet_centrality = nx.betweenness_centrality(network)  # should take a few minutes
        nx.set_node_attributes(network, bet_centrality, 'betweenness centrality') 

        # Creating the sorted_betweenness value
        sorted_betweenness = sorted(bet_centrality.items(), key=itemgetter(1), reverse=True)

        # Finding the summary stats on between centrality
        bet_list = list(bet_centrality.values())
        return  bet_list

    def cal_eig_cen(network: nx.graph.Graph):
        # Calculating the Eigenvector CEntrality for a network
        ev_centrality = nx.eigenvector_centrality(network, max_iter=200)
        nx.set_node_attributes(network, ev_centrality, 'eigenvector centrality')

        # Creating the sorted_ev_centrality value
        sorted_ev_centrality = sorted(
            ev_centrality.items(), key=itemgetter(1), reverse=True)

        # Finding the summary stats on eigenvector centrality
        eg_list = list(ev_centrality.values())
        return eg_list

    def cal_deg_cen(network: nx.graph.Graph):
        deg_centrality = nx.degree_centrality(network)
        nx.set_node_attributes(network, deg_centrality, 'degree centrality')
        sorted_deg_centrality = sorted(
            deg_centrality.items(), key=itemgetter(1), reverse=True)

        deg_list = list(deg_centrality.values())
        return deg_list

    def cal_close_cen(network: nx.graph.Graph):
        close_centrality = nx.closeness_centrality(network)
        nx.set_node_attributes(network, close_centrality, 'closeness centrality')

        sorted_close_centrality = sorted(
            close_centrality.items(), key=itemgetter(1), reverse=True)

        close_list = list(close_centrality.values())
        return close_list

    def centrality_stats(bc_1,bc_2,ec_1,ec_2,dc_1,dc_2,cc_1,cc_2,title):
        print(title+':')

        centrality_table_1 = PrettyTable(["Stats", "Betweeness C. Pre-Ban", "Betweeness C. Post-Ban",
                                        "Eigenvector C. Pre-Ban", "Eigenvector C. Post-Ban"])
                                    
        centrality_table_1.add_row(['Mean',round(np.mean(bc_1),5), round(np.mean(bc_2),5), 
                            round(np.mean(ec_1),5), round(np.mean(ec_2),5)]) 
        centrality_table_1.add_row(['Min',round(np.min(bc_1),5), round(np.min(bc_2),5), 
                            round(np.min(ec_1),5), round(np.min(ec_2),5)]) 
        centrality_table_1.add_row(['Max',round(np.max(bc_1),5), round(np.max(bc_2),5), 
                            round(np.max(ec_1),5), round(np.max(ec_2),5)]) 
        centrality_table_1.add_row(['Std',round(np.std(bc_1),5), round(np.std(bc_2),5), 
                            round(np.std(ec_1),5), round(np.std(ec_2),5)])
        print(centrality_table_1)

        centrality_table_2 = PrettyTable(["Stats","Degree C. Pre-Ban", "Degree C. Post-Ban",
                                        "Closeness C. Pre-Ban", "Closeness C. Post-Ban",])

        centrality_table_2.add_row(['Mean',round(np.mean(dc_1),5), round(np.mean(dc_2),5), 
                            round(np.mean(cc_1),5), round(np.mean(cc_2),5)])
        centrality_table_2.add_row(['Min',round(np.min(dc_1),5), round(np.min(dc_2),5), 
                            round(np.min(cc_1),5), round(np.min(cc_2),5)])
        centrality_table_2.add_row(['Max',round(np.max(dc_1),5), round(np.max(dc_2),5), 
                            round(np.max(cc_1),5), round(np.max(cc_2),5)])
        centrality_table_2.add_row(['Std',round(np.std(dc_1),5), round(np.std(dc_2),5), 
                            round(np.std(cc_1),5), round(np.std(cc_2),5)])
        print(centrality_table_2)                            

    def plot_modularity_scores(modularity_scores, figsize):
        fig = plt.figure(figsize=figsize)

        xlabel = 'Number of communities detected'
        ylabel = 'Modularity score'

        pre_ban_mono = modularity_scores[modularity_scores['network'] == 'pre_ban_mono']    
        pos = pre_ban_mono['num_communities']
        values = pre_ban_mono['modularity_score']
        ax1 = fig.add_subplot(221)
        ax1.stem(pos, values)
        ax1.set_xticks(pos)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.set_title('Modularity score for Monopartite before ban')

        post_ban_mono = modularity_scores[modularity_scores['network'] == 'pre_ban_mono']    
        pos = post_ban_mono['num_communities']
        values = post_ban_mono['modularity_score']
        ax2 = fig.add_subplot(222)
        ax2.stem(pos, values)
        ax2.set_xticks(pos)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)
        ax2.set_title('Modularity score for Monopartite after ban')

        pre_ban_bi = modularity_scores[modularity_scores['network'] == 'pre_ban_mono']
        pos = pre_ban_bi['num_communities']
        values = pre_ban_bi['modularity_score']
        ax3 = fig.add_subplot(223)
        ax3.stem(pos, values)
        ax3.set_xticks(pos)
        ax3.set_xlabel(xlabel)
        ax3.set_ylabel(ylabel)
        ax3.set_title('Modularity score for Bipartite before ban')

        post_ban_bi = modularity_scores[modularity_scores['network'] == 'pre_ban_mono']
        pos = post_ban_bi['num_communities']
        values = post_ban_bi['modularity_score']
        ax4 = fig.add_subplot(224)
        ax4.stem(pos, values)
        ax4.set_xticks(pos)
        ax4.set_xlabel(xlabel)
        ax4.set_ylabel(ylabel)
        ax4.set_title('Modularity score for Bipartite after ban')
        
        plt.show()

######################################################################

    def old_gen_network(self, network_type) -> nx.graph.Graph:
        """
        transform the pandas.DataFrame into nx.graph.Graph objects
        :return:
            nx.graph.Graph
        """
        df = self.data
        userSpace = df[(df.from_address_type == 'EOA') & (df.to_address_type == 'EOA')]
        userSpace = userSpace[['from_address','to_address','value']]
        bipartiteSpace = df[((df.from_address_type == 'EOA') & (df.to_address_type == 'Contract')) | ((df.from_address_type == 'Contract') & (df.to_address_type == 'EOA'))]
        bipartiteSpace = bipartiteSpace[['from_address','to_address','value']]

        if network_type == 'monopartite_full':
            final_network = nx.from_pandas_edgelist(userSpace,source='from_address',target='to_address',edge_attr='value')

        elif network_type == 'monopartite_gcc':
            base_network = nx.from_pandas_edgelist(userSpace,source='from_address',target='to_address',edge_attr='value')
            
            gcc = list(max(nx.connected_components(base_network), key=lambda x: len(x)))
            gcc_df = userSpace[userSpace['from_address'].isin(gcc) | userSpace['to_address'].isin(gcc)]

            final_network = nx.from_pandas_edgelist(gcc_df,source='from_address',target='to_address',edge_attr='value')

        elif network_type == 'bipartite_full':
            final_network = nx.Graph()
            final_network.add_nodes_from(np.unique(np.array(bipartiteSpace['from_address'])), bipartite=0)
            final_network.add_nodes_from(np.unique(np.array(bipartiteSpace['to_address'])), bipartite=1)
            final_network.add_edges_from(list(zip(bipartiteSpace['from_address'], bipartiteSpace['to_address'])))

        elif network_type == 'bipartite_gcc':
            base_network = nx.Graph()
            base_network.add_nodes_from(np.unique(np.array(bipartiteSpace['from_address'])), bipartite=0)
            base_network.add_nodes_from(np.unique(np.array(bipartiteSpace['to_address'])), bipartite=1)
            base_network.add_edges_from(list(zip(bipartiteSpace['from_address'], bipartiteSpace['to_address'])))

            gcc = list(max(nx.connected_components(base_network), key=lambda x: len(x)))
            gcc_df = bipartiteSpace[bipartiteSpace['from_address'].isin(gcc) | bipartiteSpace['to_address'].isin(gcc)]
            gcc_users = np.unique(np.array(gcc_df['from_address']))
            gcc_contracts = np.unique(np.array(gcc_df['to_address']))
            gcc_edges = list(zip(gcc_df['from_address'], gcc_df['to_address']))

            final_network = nx.Graph()
            final_network.add_nodes_from(gcc_users, bipartite=0)
            final_network.add_nodes_from(gcc_contracts, bipartite=1)
            final_network.add_edges_from(gcc_edges)

        return final_network    

    def old_plot_network(self, network: nx.graph.Graph, network_type):
        seed = 100; user_color = 'blue'; contract_color = 'red'
        node_size = 200; contract_node_shape = 's'; alpha = 0.5

        if network_type == 'monopartite_full':
            # print(f'{network_type} network is connected: {nx.is_connected(network)}')
            
            pos=nx.spring_layout(network, seed = seed)
            nx.draw(network,pos,arrows=True,node_size=node_size,
                                node_color=user_color, alpha=alpha)

        elif network_type == 'monopartite_gcc':
            # print(f'{network_type} network is connected: {nx.is_connected(network)}')

            pos=nx.spring_layout(network, seed = seed)
            nx.draw(network,pos,arrows=True,node_size=node_size,
                                node_color=user_color, alpha=alpha)     

        elif network_type == 'bipartite_full':
            # print(f'{network_type} network is connected: {nx.is_connected(network)}')
            print(f'{network_type} network is bipartite: {nx.is_bipartite(network)}')
            
            users=[]; contracts=[]
            for node in network.nodes(data=True):
                address = node[0]
                attribute = node[1]['bipartite']
                if attribute == 0:
                    users.append(address)
                else:
                    contracts.append(address)

            pos = nx.spring_layout(network, seed = seed)   
            # draw the network - user nodes
            nx.draw(network,pos,arrows=True,nodelist=list(users),
                                node_color=user_color,alpha=alpha,node_size=node_size)
            # draw the network - contract nodes
            nx.draw(network,pos,arrows=True,nodelist=list(contracts),
                                node_color=contract_color,alpha=alpha,node_shape=contract_node_shape,node_size=node_size)

        elif network_type == 'bipartite_gcc':
            # print(f'{network_type} network is connected: {nx.is_connected(network)}')
            print(f'{network_type} network is bipartite: {nx.is_bipartite(network)}')
            
            # gcc_users = bp.sets(network)[0]
            # gcc_contracts = bp.sets(network)[1]

            gcc_users=[]; gcc_contracts=[]
            for node in network.nodes(data=True):
                address = node[0]
                attribute = node[1]['bipartite']
                if attribute == 0:
                    gcc_users.append(address)
                else:
                    gcc_contracts.append(address)
            
            pos = nx.spring_layout(network, seed = seed)
            # draw the network - user nodes
            nx.draw(network,pos,arrows=True,nodelist=list(gcc_users),
                    node_color=user_color,alpha=alpha,node_size=node_size)
            # draw the network - contract nodes
            nx.draw(network,pos,arrows=True,nodelist=list(gcc_contracts),
                    node_color=contract_color,alpha=alpha,node_shape=contract_node_shape,node_size=node_size)

    def old_plot_degree_dist(network1: nx.graph,network2: nx.graph) -> None:
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

    def old_cal_weight_of_nodes(self, network: nx.graph.Graph) -> dict:
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

    def old_cal_betweenness_centrality(self, network: nx.graph) -> dict:
        """
        Use betweenness_centrality method in networkx to calculate betweenness
        :param network:
        :return: dict
        """
        pass

######################################################################

# if __name__ == '__main__':
#     path = os.getcwd()
#     df = pd.read_csv(path+'\data\cleaned_data.csv')
#     # df = pd.read_csv(path+'\\data\\data_example.csv')
#     df = df.drop(df.columns[0],axis=1)

#     network_analysis_obj = NetworkAnalysis(df)
#     monopartite_full = network_analysis_obj.gen_network(network_type='monopartite_full')
#     monopartite_gcc = network_analysis_obj.gen_network(network_type='monopartite_gcc')

#     fig = plt.figure(figsize=(50, 25), constrained_layout=True)
#     ax1 = fig.add_subplot(121)
#     network_analysis_obj.plot_network(monopartite_full, network_type='monopartite_full')
#     ax1.set_title('Monopartite Full', fontdict={'fontsize': 75})

#     ax2 = fig.add_subplot(122)
#     network_analysis_obj.plot_network(monopartite_gcc, network_type='monopartite_gcc')
#     ax2.set_title('Monopartite GCC', fontdict={'fontsize': 75})

#     plt.show()

#     network_analysis_obj = NetworkAnalysis(df)
#     bipartite_full = network_analysis_obj.gen_network(network_type='bipartite_full')
#     bipartite_gcc = network_analysis_obj.gen_network(network_type='bipartite_gcc')

#     fig = plt.figure(figsize=(50, 25), constrained_layout=True)
#     ax1 = fig.add_subplot(121)
#     network_analysis_obj.plot_network(bipartite_full, network_type='bipartite_full')
#     ax1.set_title('Bipartite Full', fontdict={'fontsize': 75})

#     ax2 = fig.add_subplot(122)
#     network_analysis_obj.plot_network(bipartite_gcc, network_type='bipartite_gcc')
#     ax2.set_title('Bipartite GCC', fontdict={'fontsize': 75})

#     plt.show()