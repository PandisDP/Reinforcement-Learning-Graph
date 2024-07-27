from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from sklearn import metrics
import pandas as pd

class ProcessDeep:
    def __init__(self):
        pass

    def __reward_distribution(self,edges,method):
        if method == 'mean':
            return np.mean([edge[2]['reward'] for edge in edges])
        elif method == 'max':
            return max([edge[2]['reward'] for edge in edges])
        elif method == 'min':
            return min([edge[2]['reward'] for edge in edges])
        elif method == 'sum':
            return sum([edge[2]['reward'] for edge in edges])
        else:
            raise ValueError('Método no válido. Métodos válidos: mean, max, min, sum, path_reward_weighted')

    def compute_pagerank_with_weights(self,graph, alpha=0.85):
        self.transform_rewards_to_weights(graph)
        pagerank_scores = nx.pagerank(graph, alpha=alpha, weight='weight')
        df = pd.DataFrame(list(pagerank_scores.items()), columns=['Node', 'PageRank'])
        df.to_csv('pagerank_nodes', index=False)
        return pagerank_scores

    def transform_rewards_to_weights(self,graph):
        # Transform the negative rewards into weights
        min_reward = min(data['reward'] for _, _, data in graph.edges(data=True))
        # if the minimum reward is negative, we adjust all rewards to be non-negative
        if min_reward < 0:
            adjustment = -min_reward
            for u, v, data in graph.edges(data=True):
                data['weight'] = max(0, data['reward'] + adjustment)
        else:
            for u, v, data in graph.edges(data=True):
                data['weight'] = max(0, data['reward'])
        # Normaalice the weights
        for node in graph.nodes:
            total_weight = sum(data['weight'] for _, _, data in graph.in_edges(node, data=True))
            if total_weight > 0:
                for _, target, data in graph.in_edges(node, data=True):
                    data['weight'] /= total_weight
    def __node_rewards_back(self,graph):
        leaf_nodes = {node for node in graph.nodes() if graph.out_degree(node) == 0}
        node_values = {}  
        count_values={} 
        for node in leaf_nodes:
            visited = set()  # Set to store visited nodes
            stack = [(node,0,0)] 
            while stack:
                node, current_value,count_value = stack.pop()
                if node not in visited:
                    visited.add(node)
                    node_values[node] = current_value  
                    count_values[node]= count_value
                    for parent in graph.predecessors(node):
                        if parent not in visited:
                            edge_value = graph[parent][node].get('reward', 0)
                            new_value = current_value + edge_value
                            new_count= count_value+1
                            stack.append((parent, new_value,new_count))   
        node_divisions = {node: node_values[node] / count_values[node] for node in node_values if count_values[node] != 0}                                    
        return node_divisions

    def determine_optimal_clusters(self,q_graph_obj,
                                qtable_filename_base='q_table.joblib'
                                ,n_clusters=10,method='mean'):
        q_graph_learn= q_graph_obj.load_q_table(qtable_filename_base)
        node_rewards = {}
        if method != 'page_rank':
            for node in q_graph_learn.G.nodes():
                edges = q_graph_learn.G.edges(node, data=True)
                if edges:
                    _reward = self.__reward_distribution(edges,method)
                    node_rewards[node] = _reward
                else:
                    node_rewards[node] = 0  # Si el nodo no tiene aristas, asignamos una recompensa de 0
        else:
            node_rewards = self.compute_pagerank_with_weights(q_graph_learn.G, alpha=0.85)           
        rewards = np.array(list(node_rewards.values())).reshape(-1, 1)
        rewards_normalized = (rewards - rewards.min()) / (rewards.max() - rewards.min())
        # Method of the Elbow
        inertia = []
        silhouette_scores = []
        for k in range(2, n_clusters):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto', max_iter=1000)
            kmeans.fit(rewards_normalized)
            inertia.append(kmeans.inertia_)
            clusters = kmeans.fit_predict(rewards_normalized)
            silhouette_scores.append(silhouette_score(rewards_normalized, clusters))
            labels = kmeans.labels_
            metric_sc=metrics.silhouette_score(rewards_normalized, labels, metric='euclidean')
            print('Silhouette Score:',metric_sc)

        plt.figure(figsize=(10, 7))
        plt.plot(range(2,n_clusters), inertia, 'bx-')
        plt.xlabel('Número de clusters')
        plt.ylabel('Inercia')
        plt.title('Método del Codo para encontrar el número óptimo de clusters')
        plt.show()
        # Método de la Silueta
        plt.figure(figsize=(10, 7))
        plt.plot(range(2, n_clusters), silhouette_scores, 'bx-')
        plt.xlabel('Número de clusters')
        plt.ylabel('Puntuación de Silueta')
        plt.title('Método de la Silueta para encontrar el número óptimo de clusters')
        plt.show()

    def clustering_analysis(self,graph_obj,qtable_filename_base='q_table.joblib',
                            n_clusters=3,method='mean'):
        # 1. We calculate the average reward for each node
        q_graph_learn= graph_obj.load_q_table(qtable_filename_base)
        node_rewards = {}
        if method != 'page_rank':
            for node in q_graph_learn.G.nodes():
                edges = q_graph_learn.G.edges(node, data=True)
                if edges:
                    _reward = self.__reward_distribution(edges,method)
                    node_rewards[node] = _reward
                else:
                    node_rewards[node] = 0  # If the node has no edges, we assign a reward of 0
        else:
            node_rewards = self.compute_pagerank_with_weights(q_graph_learn.G, alpha=0.85)          
        # 2. We normalize the rewards
        rewards = np.array(list(node_rewards.values())).reshape(-1, 1)
        rewards_normalized = (rewards - rewards.min()) / (rewards.max() - rewards.min())
        # 3. We apply KMeans to cluster the nodes
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto', max_iter=1000)  
        clusters = kmeans.fit_predict(rewards_normalized)
        # 4. Visualize the clusters
        pos = nx.spring_layout(q_graph_learn.G)  # Positions for all nodes
        plt.figure(figsize=(10, 7))
        for i, cluster in enumerate(set(clusters)):
            nodes_in_cluster = [node for node, cluster_id in
                                zip(q_graph_learn.G.nodes(), clusters) if cluster_id == cluster]
            #node_sizes = [node_rewards[node] * 1000 for node in nodes_in_cluster] # Node size is proportional to the reward
            nx.draw_networkx_nodes(q_graph_learn.G, pos, nodelist=nodes_in_cluster,
                                    node_color=plt.cm.tab10(i), node_size=300)
        nx.draw_networkx_edges(q_graph_learn.G, pos)
        plt.savefig('Deep_Analysis/Clusters.png', format='png', dpi=300)     
        
    def graphics_network(self,graph_obj,qtable_filename_base='q_table.joblib'):
        if qtable_filename_base != '':
            #self.empty_path('Deep_Analysis')
            q_graph_learn= graph_obj.load_q_table(qtable_filename_base)
            max_reward = max((data['reward'] for u, v, data in q_graph_learn.G.edges(data=True)))
            min_reward = min((data['reward'] for u, v, data in q_graph_learn.G.edges(data=True)))
            # Normalizar los rewards en las aristas
            for u, v, data in q_graph_learn.G.edges(data=True):
                normalized_reward = (data['reward'] - min_reward) / (max_reward - min_reward) if max_reward != min_reward else 0
                q_graph_learn.G[u][v]['normalized_reward'] = normalized_reward
            pos = nx.spring_layout(q_graph_learn.G, weight='normalized_reward', scale=1, iterations=100)    
            #pos = nx.spring_layout(q_graph_learn.G, weight='reward',scale=1,iterations=100)
            plt.figure(figsize=(12, 8)) 
            nx.draw(q_graph_learn.G,pos,with_labels=False, node_color='skyblue', 
                    node_size=50, edge_color='k', linewidths=0.05, font_size=5)
            plt.title('Visualización del Grafo de Aprendizaje Q')
            plt.savefig('Deep_Analysis/Learning_graph_Q.png', format='png', dpi=300)
            print('Number of States:',q_graph_learn.G.number_of_nodes())
            print('Number of Relationsip betewen States:',q_graph_learn.G.number_of_edges())         