import networkx as nx
import random

class Q_Graph:
    def __init__(self):
        self.G = nx.DiGraph()

    def create_state(self,state_id,lst_action,lst_next_state_id, lst_reward):
        # Add nodes if not exist
        if state_id not in self.G:
            self.G.add_node(state_id, state=state_id)

        for i ,e in enumerate (lst_next_state_id):
            if e not in self.G:
                self.G.add_node(e, state=e)
            if not self.G.has_edge(state_id, e):    
                self.G.add_edge(state_id, e, action=lst_action[i], reward=lst_reward[i])

    def update_state(self,state_id,action,next_state_id,reward):
        if state_id not in self.G:
            self.G.add_node(state_id, state=state_id)
        if next_state_id not in self.G:
            self.G.add_node(next_state_id, state=next_state_id)
        self.G.add_edge(state_id, next_state_id, action=action, reward=reward)
    
    def get_value_state_action(self,state_id,action):
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
        return nodo_id in self.G
    
    def find_state_to_next_state(self,nodo_id):
        if nodo_id in self.G:
            aristas = self.G.out_edges(nodo_id, data=True)
            if len(aristas) == 0:
                return False
            else:
                return True
        else:
            return False

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


