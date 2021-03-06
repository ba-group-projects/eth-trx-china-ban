import os
import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms import bipartite as bp
import matplotlib.pyplot as plt
from collections import Counter
from prettytable import PrettyTable
from operator import itemgetter
import seaborn as sns

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

    def plot_network_comparison(obj_list, network_list, network_type, figsize=(50,25), title_fontsize=75):
        
        if network_type == 'monopartite':
            fig = plt.figure(figsize=figsize, constrained_layout=True)
            ax1 = fig.add_subplot(121)
            ax1.set_axis_off()
            ax1_1 = fig.add_subplot(121)
            ax1_1.set_title('Monopartite Before Ban', fontdict={'fontsize': title_fontsize})
            obj_list[0].plot_network(network_list[0], network_type='monopartite')

            ax2 = fig.add_subplot(122)
            ax2.set_axis_off()
            ax2_1 = fig.add_subplot(122)
            ax2_1.set_title('Monopartite After Ban', fontdict={'fontsize': title_fontsize})
            obj_list[1].plot_network(network_list[1], network_type='monopartite')

            fig.savefig('./figure/monopartite_full_network_comparison.jpg')
            plt.show()
        
        elif network_type == 'bipartite':
            fig = plt.figure(figsize=figsize, constrained_layout=True)
            ax1 = fig.add_subplot(121)
            ax1.set_axis_off()
            ax1_1 = fig.add_subplot(121)
            ax1_1.set_title('Bipartite Before Ban', fontdict={'fontsize': title_fontsize})
            print(f'Network before ban is bipartite: {nx.is_bipartite(network_list[0])}')
            obj_list[0].plot_network(network_list[0], network_type='bipartite')

            ax2 = fig.add_subplot(122)
            ax2.set_axis_off()
            ax2_1 = fig.add_subplot(122)
            ax2_1.set_title('Bipartite After Ban', fontdict={'fontsize': title_fontsize})
            print(f'Network after ban is bipartite: {nx.is_bipartite(network_list[1])}')
            obj_list[1].plot_network(network_list[1], network_type='bipartite')

            fig.savefig('./figure/bipartite_full_network_comparison.jpg')
            plt.show()

    def gen_subset_network(self, id_node, network: nx.graph.Graph, network_type):
        df = self.data
        # select top 5k transactions for pre ban
        df = df.sort_values(by=['value'], ascending=False).head(5000)

        id_node_dict = dict(zip(id_node['node'], id_node['id']))
        
        if network_type == 'monopartite':
            # get top 5 nodes based on degree
            mono_deg = pd.DataFrame(network.degree(),columns=['node','degree']).sort_values(by='degree',ascending=False).head(5).reset_index(drop=True)
            mono_top = mono_deg['node']
            mono_top_dict = {key: value for key, value in id_node_dict.items() if key in list(mono_top)}

            # create subset df
            mono_df = df[(df.from_address_type == 'EOA') &
                                        (df.to_address_type == 'EOA')].reset_index(drop=True)
            mono_subset_df = mono_df[mono_df['from_address'].isin(mono_top) | 
                                                    mono_df['to_address'].isin(mono_top)].reset_index(drop=True)

            # generate network
            mono_subset = nx.DiGraph()
            mono_subset.add_nodes_from(np.unique(np.array(mono_subset_df['from_address'].append(mono_subset_df['to_address']))))
            mono_subset.add_weighted_edges_from(list(zip(mono_subset_df['from_address'],mono_subset_df['to_address'],mono_subset_df['value'])))

            final_network = mono_subset

        elif network_type == 'bipartite':
            # get top 5 nodes based on degree
            bi_deg = pd.DataFrame(network.degree(),columns=['node','degree']).sort_values(by='degree',ascending=False).head(5).reset_index(drop=True)
            bi_top = bi_deg['node']
            bi_top_dict = {key: value for key, value in id_node_dict.items() if key in list(bi_top)}

            # create subset df
            bi_df = df[((df.from_address_type == 'EOA') & (df.to_address_type == 'Contract')) | ((df.from_address_type == 'Contract') & (df.to_address_type == 'EOA'))].reset_index(drop=True)
            bi_subset_df = bi_df[bi_df['from_address'].isin(bi_top) | 
                                                    bi_df['to_address'].isin(bi_top)].reset_index(drop=True)

            # generate network
            bi_subset_users = np.unique(np.array(bi_subset_df['from_address']))
            bi_subset_contracts = np.unique(np.array(bi_subset_df['to_address']))
            bi_subset_edges = list(zip(bi_subset_df['from_address'], bi_subset_df['to_address'], bi_subset_df['value']))

            bi_subset = nx.DiGraph()
            bi_subset.add_nodes_from(bi_subset_users, bipartite=0)
            bi_subset.add_nodes_from(bi_subset_contracts, bipartite=1)
            bi_subset.add_weighted_edges_from(bi_subset_edges)

            final_network = bi_subset

        return final_network            

    def plot_subset_network_comparison(id_node,network1,network2,network_type):
        seed = 15; user_color = 'blue'; contract_color = 'red'
        node_size = 500; contract_node_shape = 's'; alpha = 0.25
        arrowsize=30; edge_width=1.5

        id_node_dict = dict(zip(id_node['node'], id_node['id']))

        if network_type == 'monopartite':
            fig = plt.figure(figsize=(50,25), constrained_layout=True)
            ax1 = fig.add_subplot(121)
            ax1.set_axis_off()
            ax1_1 = fig.add_subplot(121)
            ax1_1.set_title('Monopartite Subset Before Ban', fontdict={'fontsize': 75})
            network = network1

            # get top 5 nodes based on degree
            mono_deg = pd.DataFrame(network.degree(),columns=['node','degree']).sort_values(by='degree',ascending=False).head(5).reset_index(drop=True)
            mono_top = mono_deg['node']
            mono_top_dict = {key: value for key, value in id_node_dict.items() if key in list(mono_top)}

            pre_ban_mono_subset_pos=nx.spring_layout(network, seed = seed)
            nx.draw_networkx_nodes(network, pos=pre_ban_mono_subset_pos, nodelist=list(network.nodes()),node_color=user_color, 
                                node_size=node_size, alpha=alpha)
            nx.draw_networkx_edges(network, pos=pre_ban_mono_subset_pos, width=edge_width, alpha=alpha, arrows=True, arrowsize=arrowsize)
            nx.draw_networkx_labels(network, pos=pre_ban_mono_subset_pos, labels=mono_top_dict, 
                                    font_color='red', font_size=50, verticalalignment='top',clip_on=False)

            ax2 = fig.add_subplot(122)
            ax2.set_axis_off()
            ax2_1 = fig.add_subplot(122)
            ax2_1.set_title('Monopartite Subset After Ban', fontdict={'fontsize': 75})
            network = network2

            # get top 5 nodes based on degree
            mono_deg = pd.DataFrame(network.degree(),columns=['node','degree']).sort_values(by='degree',ascending=False).head(5).reset_index(drop=True)
            mono_top = mono_deg['node']
            mono_top_dict = {key: value for key, value in id_node_dict.items() if key in list(mono_top)}

            post_ban_mono_subset_pos=nx.spring_layout(network, seed = seed)
            nx.draw_networkx_nodes(network, pos=post_ban_mono_subset_pos, nodelist=list(network.nodes()),node_color=user_color, 
                                node_size=node_size, alpha=alpha)
            nx.draw_networkx_edges(network, pos=post_ban_mono_subset_pos, width=edge_width, alpha=alpha, arrows=True, arrowsize=arrowsize)
            nx.draw_networkx_labels(network, pos=post_ban_mono_subset_pos, labels=mono_top_dict, 
                                    font_color='red', font_size=50, verticalalignment='top',clip_on=True)

            fig.savefig('./figure/monopartite_subset_network_comparison.jpg')
            plt.show()
        
        elif network_type == 'bipartite':
            fig = plt.figure(figsize=(50,25), constrained_layout=True)
            ax1 = fig.add_subplot(121)
            ax1.set_axis_off()
            ax1_1 = fig.add_subplot(121)
            ax1_1.set_title('Bipartite Subset Before Ban', fontdict={'fontsize': 75})
            network = network1

            bi_deg = pd.DataFrame(network.degree(),columns=['node','degree']).sort_values(by='degree',ascending=False).head(5).reset_index(drop=True)
            bi_top = bi_deg['node']
            bi_top_dict = {key: value for key, value in id_node_dict.items() if key in list(bi_top)}        

            users=[]; contracts=[]
            for node in network.nodes(data=True):
                address = node[0]
                attribute = node[1]['bipartite']
                if attribute == 0:
                    users.append(address)
                elif attribute == 1:
                    contracts.append(address)

            pre_ban_bi_subset_pos=nx.spring_layout(network, seed = seed)
            nx.draw_networkx_nodes(network, pos=pre_ban_bi_subset_pos, nodelist=list(users),node_color=user_color, 
                                    node_size=node_size, alpha=alpha)
            nx.draw_networkx_nodes(network, pos=pre_ban_bi_subset_pos, nodelist=list(contracts),node_color=contract_color,
                                    node_size=node_size)
            nx.draw_networkx_edges(network, pos=pre_ban_bi_subset_pos, width=edge_width, alpha=alpha, arrows=True, arrowsize=arrowsize)
            nx.draw_networkx_labels(network, pos=pre_ban_bi_subset_pos, labels=bi_top_dict, 
                                    font_color='red', font_size=50, verticalalignment='bottom',clip_on=True)

            ax2 = fig.add_subplot(122)
            ax2.set_axis_off()
            ax2_1 = fig.add_subplot(122)
            ax2_1.set_title('Bipartite Subset After Ban', fontdict={'fontsize': 75})
            network = network2

            bi_deg = pd.DataFrame(network.degree(),columns=['node','degree']).sort_values(by='degree',ascending=False).head(5).reset_index(drop=True)
            bi_top = bi_deg['node']
            bi_top_dict = {key: value for key, value in id_node_dict.items() if key in list(bi_top)}          

            users=[]; contracts=[]
            for node in network.nodes(data=True):
                address = node[0]
                attribute = node[1]['bipartite']
                if attribute == 0:
                    users.append(address)
                elif attribute == 1:
                    contracts.append(address)

            post_ban_bi_subset_pos=nx.spring_layout(network, seed = seed)
            nx.draw_networkx_nodes(network, pos=post_ban_bi_subset_pos, nodelist=list(users),node_color=user_color, 
                                    node_size=node_size, alpha=alpha)
            nx.draw_networkx_nodes(network, pos=post_ban_bi_subset_pos, nodelist=list(contracts),node_color=contract_color,
                                    node_size=node_size)
            nx.draw_networkx_edges(network, pos=post_ban_bi_subset_pos, width=edge_width, alpha=alpha, arrows=True, arrowsize=arrowsize)
            nx.draw_networkx_labels(network, pos=post_ban_bi_subset_pos, labels=bi_top_dict, 
                                    font_color='red', font_size=50, verticalalignment='bottom',clip_on=True)

            fig.savefig('./figure/bipartite_subset_network_comparison.jpg')
            plt.show()

    def cal_degree(network: nx.graph.Graph):
        dv = dict(network.degree())
        k = list(dv.values())
        return k

    def degree_stats(pre_ban_mono_k, post_ban_mono_k, pre_ban_bi_k, post_ban_bi_k):
        print("Overall Degree Summary Statistics:")
                    
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
        ax1.set_xlim(0,150)
        ax1.set_ylim(0,1200)

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
        ax2.set_xlim(0,150)
        ax2.set_ylim(0,1200)

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
        ax3.set_xlim(0,150)
        ax3.set_ylim(0,1200)

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
        ax4.set_xlim(0,150)
        ax4.set_ylim(0,1200)

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
        ax5.set_xlim(0,150)
        ax5.set_ylim(0,1200)

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
        ax6.set_xlim(0,150)
        ax6.set_ylim(0,1200)      
        plt.show()
        fig.savefig(f"./figure/{suptitle_label.replace(' ','_')}.jpg")

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
        fig.savefig("./figure/plot_power_law.png")

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

    def plot_centrality(network: nx.graph.Graph):
        bet_centrality = nx.betweenness_centrality(network)
        ev_centrality = nx.eigenvector_centrality(network, max_iter=200)
        deg_centrality = nx.degree_centrality(network)
        close_centrality = nx.closeness_centrality(network)
        cen_df = pd.DataFrame({'degree': deg_centrality, 'ev_centrality': ev_centrality,
                        'betweenness_centrality': bet_centrality, 'close_centrality': close_centrality}
                        )
        # --+ correlation matrix
        cen_df.corr()
        # --+ scatter plot matrix
        sns.pairplot(cen_df)
        plt.savefig('./figure/centrality_plot.png')    

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
        fig.savefig('./figure/modularity_scores.png')
