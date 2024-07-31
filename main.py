import random
import numpy as np
import pandas as pd
from Games import Field
from QLearning import QLearning
from Q_Graphs import Q_Graph
from DeepAnalysis import ProcessDeep
import seaborn as sns
import matplotlib.pyplot as plt


def random_solutions():
    size=10
    item_pickup=(0,0)
    item_dropoff=(9,9)
    start_position=(9,0)
    field = Field(size,item_pickup,item_dropoff,start_position)
    done= False
    steps=0
    while not done:
        action= random.randint(0,5)
        reward,done=field.make_action(action) 
        steps= steps+1
    return steps    

def qlearning_training(training_iter,size, item_pickup, item_dropoff, start_position, zones_block):
    field = Field(size,item_pickup,item_dropoff,start_position,zones_block,'Episodes')
    field.empty_predict_data()
    qlearning= QLearning(field,'Training')
    print('Hyperparameters training')
    epsilon_valores = [0.1] #Exploration rate
    min_epsilon = 0.01
    decay_epsilon = 0.01
    alpha_valores = [0.01,0.1] #Learning rate
    gamma_valores = [0.1,0.3,0.5,0.8] #Discount factor
    hyperparams=qlearning.hyperparameters_training(training_iter,epsilon_valores,alpha_valores,gamma_valores,min_epsilon,decay_epsilon)
    print('Training Process')
    qlearning.training(training_iter,hyperparams['epsilon'],hyperparams['alpha'],hyperparams['gamma'],min_epsilon,decay_epsilon,True)
    qlearning.graphics_reward_training('Reward_Training')
    print('Training done')
    
def qlearning_predict(size, item_pickup, item_dropoff, start_position, zones_block ,print_episode=False,re_training_bool=False,re_training_epi=1000):
    field = Field(size,item_pickup,item_dropoff,start_position,zones_block,'Episodes')
    field.empty_predict_data()
    qlearning= QLearning(field,'Episodes')
    steps,reward,status_all= qlearning.predict('q_table.joblib','best_hiperparameters.joblib',re_training_bool,re_training_epi,print_episode)
    #field.graphics(status_all,'Path_Complete')
    return steps,reward,status_all

def qlearning_predict_update(size, item_pickup, item_dropoff, start_position, zones_block ,print_episode=False,re_training_epi=1000):
    field = Field(size,item_pickup,item_dropoff,start_position,zones_block,'Episodes')
    field.empty_predict_data()
    qlearning= QLearning(field,'Episodes')
    steps,reward,status_all= qlearning.dynamic_predict('q_table.joblib','best_hiperparameters.joblib',re_training_epi,print_episode)
    #field.graphics(status_all,'Path_Complete')
    return steps,reward,status_all

def Analysis_Prediction(iterations,size, item_pickup, item_dropoff, start_position, zones_block ,print_episode=False):
    list_solutions=[]
    for i in range(iterations):
        #print('Iteration: ',i)
        steps,reward,status_all= qlearning_predict(size, item_pickup, item_dropoff, start_position, zones_block ,print_episode,False,10000)
        list_solutions.append([steps,reward,status_all])
    df = pd.DataFrame(list_solutions, columns=['steps', 'reward', 'Path']) 
    df['Path'] = df['Path'].apply(lambda x: str(x))
    df['Count'] = df.groupby('Path')['Path'].transform('count')
    df = df.drop_duplicates(subset=['Path']).reset_index(drop=True)
    df= df.sort_values(by=['reward'], ascending=False).reset_index(drop=True)
    total_count = df['Count'].sum()
    df['Percentage'] = (df['Count'] / total_count) * 100
    n_rewars_positive= df[df['reward']>=0]['Count'].sum()
    print('Percentage of positive rewards: ',(n_rewars_positive/total_count)*100)
    df.to_csv('path_to_your_file.csv', index=True)
    print(df)
    plt.figure(figsize=(10, 6))
    min_size=10
    max_size=1000
    sizes = min_size + (max_size - min_size) * (df['Percentage'] / df['Percentage'].max())
    plt.scatter(df['steps'], df['reward'], s=sizes, alpha=0.5, c='blue', edgecolors='w', linewidth=0.5)
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.title('Scatter Plot of Steps vs Reward with Percentage Size')
    plt.grid(True)
    plt.savefig('Deep_Analysis/scatter_plot.png')

    # Crear el gráfico de dispersión en 3D
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['steps'], df['reward'], df['Count'], s=sizes, alpha=0.5, c='blue', edgecolors='w', linewidth=0.5)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Reward')
    ax.set_zlabel('Count')
    ax.set_title('3D Scatter Plot of Steps vs Reward vs Count with Percentage Size')
    plt.savefig('Deep_Analysis/scatter_plot_3d.png')


def qtable_analysis(name_file='q_table.joblib'):
    size=10
    start_position=(9,0)
    item_pickup=(1,1)
    item_dropoff=(7,7)
    zones_block=[(4,0),(4,1),(4,2),(4,3),(2,6),(2,7),(2,8),(2,9),(4,8),(5,8),(6,8),(7,6),(8,6),(9,6)]
    field = Field(size,item_pickup,item_dropoff,start_position,zones_block,'Episodes')
    qlearning_predict= QLearning(field,'Episodes')
    deep_analysis= ProcessDeep()
    deep_analysis.graphics_network(qlearning_predict,name_file)
    deep_analysis.clustering_analysis(qlearning_predict,'q_table.joblib',5,'page_rank')
    #deep_analysis.determine_optimal_clusters(qlearning_predict,name_file,20,'page_rank')
    

if __name__ == "__main__":
    training_iter=100000
    size=10
    start_position=(9,0) # (9,0)
    item_pickup=(1,1)# (1,1)
    item_dropoff=(7,7) # (7,7)
    #zones_block=[(4,0),(4,1),(4,2),(4,3),(2,6),(2,7),(2,8),(2,9),(4,8),(5,8),(6,8),(7,6),(8,6),(9,6)]
    zones_block=[(4,0),(4,1),(4,2),(4,3),(2,6),(2,7),(2,8),(2,9),(4,8),(5,8),(6,8),(7,6),(8,6),(9,6)]
    #print(random_solutions())
    #print(qlearning_training(training_iter,size, item_pickup, item_dropoff, start_position, zones_block))
    #print(qlearning_predict(size, item_pickup, item_dropoff, start_position, zones_block,True,True,10))
    #print(qlearning_predict_update(size, item_pickup, item_dropoff, start_position, zones_block,True,100))
    #qtable_analysis('q_table.joblib')
    Analysis_Prediction(10000,size, item_pickup, item_dropoff, start_position, zones_block ,False)
