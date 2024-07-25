import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from joblib import dump, load
import os
import shutil
from Q_Graphs import Q_Graph

class QLearning:
    def __init__(self,field,path_predicts='Episodes'):
        #Field variables 
        self.field= field
        self.number_of_states= field.get_number_of_states()
        self.number_of_actions= field.number_of_actions
        #self.q_table= np.zeros((self.number_of_states,self.number_of_actions))
        self.q_graph= Q_Graph()
        #Training variables
        self.episodes=0 #number of episodes
        self.path_predicts=path_predicts #path to save the results of the predictions
        self.total_reward=0
        self.total_reward_training=[]
        self.steps_training_average=0
        #Hyperparameters
        self.epsilon=0.1
        self.min_epsilon=0.01
        self.decay_epsilon=0.01
        self.alpha=0.1
        self.gamma=0.6

    def training(self,n_iterations=1000,max_epsilon=0.1,alpha=0.1,gamma=0.6,min_epsilon=0.01,decay_epsilon=0.001,save_learning=True):
        self.episodes= n_iterations
        self.epsilon=max_epsilon
        self.min_epsilon=min_epsilon
        self.decay_epsilon=decay_epsilon
        self.alpha=alpha
        self.gamma=gamma
        self.steps_training_average=0
        self.total_reward_training=[]
        steps_total=0
        for iter_n in range(n_iterations):
            var,steps,done=self.learning_process(copy.deepcopy(self.field),self.epsilon,self.alpha,self.gamma,self.min_epsilon,self.decay_epsilon)
            self.total_reward_training.append(self.total_reward)
            steps_total+=steps     
        self.steps_training_average=steps_total/(iter_n+1)
        if save_learning:
            self.save_q_table(self.q_graph)

    def predict(self,qtable_filename_base='q_table.joblib',hyperparams_file='params',re_training=False ,re_training_epi=1000, print_episode=False):
        if qtable_filename_base != '':
            q_table_base= self.load_q_table(qtable_filename_base)
            hyperparams= self.load_q_table(hyperparams_file)
            self.q_graph=q_table_base
            self.epsilon=hyperparams['epsilon']
            self.alpha=hyperparams['alpha']
            self.gamma=hyperparams['gamma']
            if re_training==False:   
                var,steps,done=self.learning_process(self.field,self.epsilon,self.alpha,self.gamma,self.min_epsilon,self.decay_epsilon,print_episode)
                states_path= self.field.allposicions    
                return steps,self.total_reward,states_path 
            else:
                self.training(re_training_epi,self.epsilon,self.alpha,self.gamma,self.min_epsilon,self.decay_epsilon,True)
                var,steps,done=self.learning_process(self.field,self.epsilon,self.alpha,self.gamma,self.min_epsilon,self.decay_epsilon,print_episode)
                states_path= self.field.allposicions
                return steps,self.total_reward,states_path
        else:
            return 0,0,[]   
    def find_memory_status(self,qtable):
        (pos_x,pos_y)= self.field.item_dropoff
        tmp_state= self.field.get_state_xy(pos_x,pos_y)
        #print('Memory status: ',qtable[tmp_state])

    def convergence(self, lst_qtable, threshold=0.01,steps_evaluations=5):
        if len(lst_qtable) < steps_evaluations:
            return False
        avg_diff = np.sum(lst_qtable[-1] - sum(lst_qtable[-steps_evaluations:-1]) /(steps_evaluations-1))
        if abs(avg_diff) <= threshold:
            return True
        return False
    
    def learning_process(self,field,epsilon=0.1,alpha=0.1,gamma=0.6,min_epsilon=0.01,decay_epsilon=0.999,print_episode=False):
        done= False
        steps=0
        self.total_reward=0
        if print_episode:
            field.graphics(field.allposicions,f"episode {steps}") 
        while not done:
            state= field.get_state() #get the id of the state of game
            lst_states,lst_actions,lst_rewards= field.get_possibles_states()
            lst_rewards=[0,0,0,0,0,0]
            self.q_graph.create_state(state,lst_actions,lst_states,lst_rewards)

            if random.uniform(0,1)<epsilon:
                action= random.randint(0,field.number_of_actions-1)
            else:
                action,_,_= self.q_graph.get_best_action(state) 
            reward,done=field.make_action(action) # get the reward of the action
            self.total_reward+=reward 
            next_state= field.get_state() #get the id of the next state
            if self.q_graph.find_state_to_next_state(next_state):
                _,_,best_next_reward= self.q_graph.get_best_action(next_state)
            else:
                lst_states,lst_actions,_= field.get_possibles_states()
                self.q_graph.create_state(next_state,lst_actions,lst_states,lst_rewards)
                _,_,best_next_reward= self.q_graph.get_best_action(next_state) 
            #next_state_max= np.max(self.q_table[next_state])
            current_q= self.q_graph.get_value_state_action(state,action)
            update_reward= (1-alpha)*current_q+alpha*(reward+gamma*best_next_reward-current_q)
            self.q_graph.update_state(state,action,next_state,update_reward)
            #self.q_table[state,action]= (1-alpha)*self.q_table[state,action]+alpha*(reward+gamma*best_next_reward- self.q_table[state,action])
            steps= steps+1
            epsilon= min_epsilon+(epsilon-min_epsilon)*np.exp(-decay_epsilon*steps)
            #print(f'Steps : {steps}' , 'Action: ',action , 'Reward: ',reward, 'Total Reward: ',self.total_reward) 
            if print_episode:
                field.graphics(field.allposicions,f"episode {steps}")
                self.total_reward_training.append(self.total_reward)
                self.graphics_reward_training(f"rewars {steps}")   
        return field,steps,done 
    
    def hyperparameters_training(self,iterations,epsilon_values,alpha_values,gamma_values,min_epsilon,decay_epsilon):
        best_reward = float('inf')
        best_hiperparamters = {}
        iter_=0
        for epsilon in epsilon_values:
            for alpha in alpha_values:
                for gamma in gamma_values: 
                    iter_+=1
                    # Entrena tu modelo y evalúa el rendimiento
                    self.training(iterations, epsilon, alpha, gamma,min_epsilon,decay_epsilon,False)
                    # Actualiza los mejores hiperparámetros si es necesario
                    if self.steps_training_average < best_reward:
                        best_reward = self.steps_training_average
                        best_hiperparamters = {'gamma': gamma, 'epsilon': epsilon, 'alpha': alpha} 
                    self.reset_qtable()     
                    print(f'running {iter_}',' acurracy: ', best_reward)        
        print("Best hiperparameters:", best_hiperparamters)
        self.save_q_table(best_hiperparamters,'best_hiperparameters.joblib')
        return best_hiperparamters
    
    def reset_qtable(self):
        self.G = Q_Graph()
        #self.q_table = np.zeros((self.number_of_states, self.number_of_actions))
    
    def save_q_table(self,q_table, filename='q_table.joblib'):
        dump(q_table, filename)
        print(f'Saved to {filename}')

    def load_q_table(self,filename='q_table.joblib'):
        q_table = load(filename)
        return q_table
    
    def empty_path(self,path):
        for nombre in os.listdir(path):
            ruta_completa = os.path.join(path, nombre)
            try:
                if os.path.isfile(ruta_completa) or os.path.islink(ruta_completa):
                    os.remove(ruta_completa)
                elif os.path.isdir(ruta_completa):
                    shutil.rmtree(ruta_completa)
            except Exception as e:
                print(f'Error {ruta_completa}. reason: {e}')

    def graphics_reward_training(self,name_fig):
        plt.plot(self.total_reward_training)
        plt.ylabel('Values')
        plt.xlabel('Iterations')
        plt.title('Rewards')
        name_fig_path = self.path_predicts + '/' +name_fig
        plt.savefig(name_fig_path)
        plt.close()
