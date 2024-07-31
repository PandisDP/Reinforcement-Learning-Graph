import networkx as nx
import copy
import numpy as np

class Q_Graph:
    def __init__(self):
        '''
        Create a graph
        env: enviroment where the graph is created
        '''
        self.G = nx.DiGraph()

    def create_state(self,state_id,state_id_position,lst_action,lst_next_state_id, lst_reward,lst_position):
        '''
        Create a state in the graph
        state_id: id of the state
        state_id_position: position of the state
        lst_action: list of actions
        lst_next_state_id: list of next states
        lst_reward: list of rewards
        lst_position: list of positions of the next states
        '''
        if state_id not in self.G:
            node_attr = {'state': state_id, 
                        'position': state_id_position}
            self.G.add_node(state_id, **node_attr)

        for i ,e in enumerate (lst_next_state_id):
            if e not in self.G:
                node_attr = {'state': e, 
                            'position': lst_position[i]}
                self.G.add_node(e, **node_attr)
            if not self.G.has_edge(state_id, e):    
                self.G.add_edge(state_id, e, action=lst_action[i], reward=lst_reward[i])

    def update_state(self,state_id,action,next_state_id,reward):
        '''
        Update the reward of the state
        state_id: id of the state
        action: action
        next_state_id: id of the next state
        reward: reward
        '''
        if state_id not in self.G:
            self.G.add_node(state_id, state=state_id)
        if next_state_id not in self.G:
            self.G.add_node(next_state_id, state=next_state_id)
        self.G.add_edge(state_id, next_state_id, action=action, reward=reward)
    
    def get_value_state_action(self,state_id,action):
        '''
        Get the reward of the state
        state_id: id of the state
        action: action
        return: reward
        
        '''
        if state_id in self.G:
            aristas = self.G.out_edges(state_id, data=True)
            if len(aristas) == 0:
                return 0
            for from_node, to_node, data in aristas:
                if data['action'] == action:
                    return data['reward']
            return 0    
        else:
            return 0

    def get_best_action(self, state_id):
        '''
        Get the best action of the state
        state_id: id of the state
        return: action, next state, reward  
        '''
        aristas = self.G.out_edges(state_id, data=True)
        max_reward = -999999999999
        best_action = None
        best_next_state = None
        for from_node, to_node, data in aristas:
            state_dest = self.G.nodes[to_node]['state']
            if data['reward'] > max_reward:
                max_reward = data['reward']
                best_action = data['action']
                best_next_state = state_dest
        return best_action, best_next_state, max_reward
    

    def find_state_bool(self, nodo_id):
        '''
        Check if the state exist
        nodo_id: id of the state
        return: True if the state exist, False otherwise'''
        return nodo_id in self.G
    
    def find_state_to_next_state(self,nodo_id):
        '''
        Check if the state has next states
        nodo_id: id of the state    
        return: True if the state has next states, False otherwise
        '''
        if nodo_id in self.G:
            aristas = self.G.out_edges(nodo_id, data=True)
            if len(aristas) == 0:
                return False
            else:
                return True
        else:
            return False
        
    def generate_datasets_paths(self,path_end):
        '''
        Generate the list of positions and rewards of the path
        path_end: list of states
        return: list of positions and rewards
        '''
        lst_position=[]
        lst_reward=[]
        #env= copy.deepcopy(self.enviroment)
        for idx in range(len(path_end)-1):
            node_i= self.G.nodes[path_end[idx]]
            edge_data = self.G.get_edge_data(path_end[idx], path_end[idx+1])
            action_i = edge_data.get('action', 0.0)
            #env.position= node_i['position']
            #reward,*_= env.make_action(action_i)
            #lst_reward.append(reward)
            lst_position.append(node_i['position'])
        lst_position.append(self.G.nodes[path_end[-1]]['position'])    
        return lst_position,lst_reward
    
    def find_best_path_ford_bellman(self,init_state,middle_state,final_state):
        '''
        Find the best path from init_state to final_state passing through middle_state
        init_state: id of the initial state
        middle_state: id of the middle state
        final_state: id of the final state
        return: list of states and list of rewards
        '''
        if init_state in self.G and final_state in self.G and middle_state in self.G:
            inverted_graph = self.G.copy()
            for u,v,data in inverted_graph.edges(data=True):
                data['weight'] = -data['reward'] 
            path_to_pickup = nx.bellman_ford_path(inverted_graph, init_state, middle_state, weight='weight')
            path_to_dropoff = nx.bellman_ford_path(inverted_graph, middle_state, final_state, weight='weight')
            path_end= path_to_pickup + path_to_dropoff[1:]
            lst_position,lst_reward=self.generate_datasets_paths(path_end)   
            return lst_position,lst_reward
        else:
            return [],[]

    def find_best_path(self,init_state,middle_state,final_state):
        '''
        Find the best path from init_state to final_state passing through middle_state
        init_state: id of the initial state
        middle_state: id of the middle state
        final_state: id of the final state
        return: list of states and list of rewards
        '''
        if init_state in self.G and final_state in self.G and middle_state in self.G:
            inverted_graph = self.G.copy()
            # Encuentra el mínimo 'reward' negativo en el grafo invertido
            for u,v,data in inverted_graph.edges(data=True):
                data['weight'] = 1/np.exp(data['reward'])
            path_to_pickup = nx.shortest_path(inverted_graph, init_state, middle_state, weight='weight')
            path_to_dropoff = nx.shortest_path(inverted_graph, middle_state, final_state, weight='weight')
            path_end= path_to_pickup + path_to_dropoff[1:]
            lst_position,lst_reward=self.generate_datasets_paths(path_end)   
            return lst_position,lst_reward
        else:
            return [],[]
        
    def all_paths_with_intermediate(self, init_state, middle_state, final_state):
        '''
        Find all paths between two states with an intermediate state
        init_state: id of the initial state
        middle_state: id of the intermediate state
        final_state: id of the final state
        return: list of lists of states
        '''
        if init_state in self.G and final_state in self.G and middle_state in self.G:
            all_paths = []
            paths_to_middle = list(nx.all_simple_paths(self.G, init_state, middle_state))
            paths_from_middle = list(nx.all_simple_paths(self.G, middle_state, final_state))
            print('Paths to middle state found', len(paths_to_middle))
            for path_to_middle in paths_to_middle:
                for path_from_middle in paths_from_middle:
                    # Combine the paths, avoiding duplicate middle_state
                    combined_path = path_to_middle + path_from_middle[1:]
                    all_paths.append(combined_path)
            print('All paths with intermediate state found', len(all_paths))        
            set_path=[]
            for path in all_paths:
                lst_position=[]
                lst_reward=[]
                for idx in range(len(path)-1):
                    node_i= self.G.nodes[path[idx]]
                    node_j= self.G.nodes[all_paths[idx+1]]
                    edge_data = self.G.get_edge_data(path[idx], path[idx+1])
                    reward = edge_data.get('reward', 0.0)
                    lst_reward.append(reward)
                    lst_position.append(node_i['position'])
                lst_position.append(self.G.nodes[all_paths[-1]]['position'])   
                set_path.append([len(lst_position),sum(lst_reward),lst_position])        
            return set_path
        else:
            return []         

    def find_state(self, nodo_id):
        if nodo_id in self.G:
            state = self.G.nodes[nodo_id]['state']
            aristas = self.G.out_edges(nodo_id, data=True)
            print(f"From state: {state} (Node {nodo_id}):")
            for from_node, to_node, data in aristas:
                state_dest = self.G.nodes[to_node]['state']
                print(f" Action: {data['action']} -> Next State: {state_dest} (Nodo {to_node}) with reward: {data['reward']}")
        else:
            print("Node dont exist.")


