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
    epsilon_valores = [0.1,0.3] #Exploration rate
    min_epsilon = 0.01
    decay_epsilon = 0.01
    alpha_valores = [0.001,0.01,0.1] #Learning rate
    gamma_valores = [0.1,0.3,0.5] #Discount factor
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
    total_count = df['Count'].sum()
    df['Percentage'] = (df['Count'] / total_count) * 100
    #df= df.sort_values(by='Count', ascending=False)
    df.to_csv('path_to_your_file.csv', index=True)
    print(df)

    sns.kdeplot(df['Count'], fill=True)
    plt.title('Density Plot of Count')
    plt.xlabel('Count')
    plt.ylabel('Density')
    plt.show()
    sns.ecdfplot(df['Count'])
    plt.title('CDF of Count')
    plt.xlabel('Count')
    plt.ylabel('CDF')
    plt.show()

    # Asumiendo que df es tu DataFrame y ya está definido como se muestra en el fragmento de código anterior
    plt.figure(figsize=(10, 6))  # Ajusta el tamaño del gráfico según sea necesario
    plt.bar(df.index, df['Percentage'])  # Crea un gráfico de barras
    plt.title('Bar Plot of Count by Class')
    plt.xlabel('Class')  # Asume que el índice representa la clase
    plt.ylabel('Percentage')
    plt.xticks(rotation=45)  # Rota las etiquetas del eje X para mejor legibilidad si es necesario
    plt.show()


def qtable_analysis(name_file='q_table.joblib'):
    size=10
    start_position=(9,0)
    item_pickup=(1,1)
    item_dropoff=(7,7)
    zones_block=[(4,0),(4,1),(4,2),(4,3),(2,6),(2,7),(2,8),(2,9),(4,8),(5,8),(6,8),(7,6),(8,6),(9,6)]
    field = Field(size,item_pickup,item_dropoff,start_position,zones_block,'Episodes')
    qlearning_predict= QLearning(field,'Episodes')
    deep_analysis= ProcessDeep()
    #qlearning_predict.analysys_process_learning('q_table.joblib')
    #deep_analysis.clustering_analysis(qlearning_predict,'q_table.joblib',5,'path_reward_weighted')
    #deep_analysis.determine_optimal_clusters(qlearning_predict,'q_table.joblib',20,'path_reward_weighted')
    

def test():
    q= Q_Graph()
    # Uso de la función de búsqueda

    # Ejemplo de uso de la función con identificadores de nodos explícitos
    q.add_state(1,'A',2,-5)
    q.add_state(2,'B',4,-5)
    q.add_state(2,'C',3,-15)
    q.add_state(3,'A',1,-5)
    print(q.G.nodes.data(True))
    print(q.G.edges.data(True))
    q.find_state(4)
    print(q.get_best_action(4,3))


if __name__ == "__main__":
    training_iter=100000
    size=10
    start_position=(9,0) # (9,0)
    item_pickup=(1,1)# (1,1)
    item_dropoff=(7,7) # (8,8)
    zones_block=[(4,0),(4,1),(4,2),(4,3),(2,6),(2,7),(2,8),(2,9),(4,8),(5,8),(6,8),(7,6),(8,6),(9,6)]
    #print(random_solutions())
    print(qlearning_training(training_iter,size, item_pickup, item_dropoff, start_position, zones_block))
    #print(qlearning_predict(size, item_pickup, item_dropoff, start_position, zones_block,True,False,10000))
    #qtable_analysis('q_table.joblib')
    #Analysis_Prediction(10000,size, item_pickup, item_dropoff, start_position, zones_block ,False)
    #test()