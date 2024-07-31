import pandas as pd
from Games import Field
from QLearning import QLearning
from DeepAnalysis import ProcessDeep
import matplotlib.pyplot as plt

def qlearning_training(training_iter,size, item_pickup, item_dropoff,
                        start_position, zones_block):
    '''
    Training process of the Q-Learning algorithm--> This function first determine the 
    best hyperparameters and then train the model and save the best hyperparameters,
    after that running the training process with the best hyperparameters and save the q_table   
    into the main folder

    Parameters:
    training_iter: number of iterations
    size: size of the field
    item_pickup: position of the item to pick up
    item_dropoff: position of the item to drop off
    start_position: initial position of the agent
    zones_block: list of positions that are blocked
    '''
    field = Field(size,item_pickup,item_dropoff,start_position,zones_block,'Episodes')
    field.empty_predict_data()
    qlearning= QLearning(field,'Training')
    print('Hyperparameters training')
    epsilon_valores = [0.1] #Exploration rate
    min_epsilon = 0.01
    decay_epsilon = 0.01
    alpha_valores = [0.01,0.1] #Learning rate
    gamma_valores = [0.1,0.3,0.5,0.8] #Discount factor
    hyperparams=qlearning.hyperparameters_training(training_iter,epsilon_valores,
                        alpha_valores,gamma_valores,min_epsilon,decay_epsilon)
    print('Training Process')
    qlearning.training(training_iter,hyperparams['epsilon'],hyperparams['alpha'],
                    hyperparams['gamma'],min_epsilon,decay_epsilon,True)
    qlearning.graphics_reward_training('Reward_Training')
    print('Training done')
    
def qlearning_predict(size, item_pickup, item_dropoff, start_position, 
                    zones_block ,print_episode=False,re_training_bool=False,re_training_epi=1000):
    '''
    Predict process of the Q-Learning algorithm where the model is loaded and the agent and determine
    the best path to pick up and drop off the item.

    Parameters:
    size: size of the field
    item_pickup: position of the item to pick up
    item_dropoff: position of the item to drop off
    start_position: initial position of the agent
    zones_block: list of positions that are blocked
    print_episode: print the episode
    re_training_bool: retrain the model
    re_training_epi: number of episodes to retrain the model'''
    field = Field(size,item_pickup,item_dropoff,start_position,zones_block,'Episodes')
    field.empty_predict_data()
    qlearning= QLearning(field,'Episodes')
    steps,reward,status_all= qlearning.predict('q_table.joblib','best_hiperparameters.joblib',
                                            re_training_bool,re_training_epi,print_episode)
    return steps,reward,status_all

def qlearning_predict_dynamic(size, item_pickup, item_dropoff, start_position, 
                            zones_block ,print_episode=False,re_training_epi=1000):
    '''
    Predict process of the Q-Learning algorithm where the model is loaded and the agent and determine
    the best path to pick up and drop off the item in dynamic mode, for example if the agent determine 
    that the asigned work is not in the memory, the agent will retrain the model and then predict the best path.
    
    Parameters:
    size: size of the field
    item_pickup: position of the item to pick up
    item_dropoff: position of the item to drop off
    start_position: initial position of the agent
    zones_block: list of positions that are blocked
    print_episode: print the episode
    re_training_epi: number of episodes to retrain the model'''
    field = Field(size,item_pickup,item_dropoff,start_position,zones_block,'Episodes')
    field.empty_predict_data()
    qlearning= QLearning(field,'Episodes')
    steps,reward,status_all= qlearning.dynamic_predict('q_table.joblib','best_hiperparameters.joblib'
                            ,re_training_epi,print_episode)
    return steps,reward,status_all

def Analysis_Prediction(iterations,size, item_pickup, item_dropoff, start_position,
                zones_block ,print_episode=False):
    '''
    Analysis of the prediction process of the Q-Learning algorithm where the model is loaded and the agent and determine
    the best path to pick up and drop off the item for a number of iterations in order to analyze the probability of
    the positive rewards , total paths and the percentage of each path for specific number of iterations or samples
    Parameters:
    iterations: number of iterations
    size: size of the field
    item_pickup: position of the item to pick up
    item_dropoff: position of the item to drop off
    start_position: initial position of the agent
    zones_block: list of positions that are blocked
    print_episode: print the episode'''
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
    '''
    Analysis of the q_table of the Q-Learning algorithm is necessary to determine clusters into the q_table
    and others ways to analyze how the agent is learning the best path to pick up and drop off the item.
    Parameters:
    name_file: name of the file that contains the q_table
    '''
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
    print(qlearning_training(training_iter,size, item_pickup, item_dropoff, start_position, zones_block))
    #print(qlearning_predict(size, item_pickup, item_dropoff, start_position, zones_block,True,False,10))
    print(qlearning_predict_dynamic(size, item_pickup, item_dropoff, start_position, zones_block,True,100))
    #qtable_analysis('q_table.joblib')
    #Analysis_Prediction(10000,size, item_pickup, item_dropoff, start_position, zones_block ,False)
