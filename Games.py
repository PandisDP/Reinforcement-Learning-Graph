import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import random
class Field:
    def __init__(self,size,item_pickup,item_dropoff,start_position,zones_blocks=[],path_predicts='Episodes'):
        '''
        size: size of the field
        item_pickup: position of the item to pickup
        item_dropoff: position of the item to dropoff
        start_position: initial position
        zones_blocks: list of tuples with the position of the blocks
        path_predicts: path to save the predictions
        '''
        self.size = size
        self.item_pickup = item_pickup
        self.item_dropoff = item_dropoff
        self.position = start_position
        self.position_start= start_position
        self.block_zones=zones_blocks
        self.item_in_car= False
        self.number_of_actions=6
        self.allposicions = []
        self.path_predicts= path_predicts
        self.save_state()

    def get_path_status(self):
        '''
        Get the status of the path
        return: status of the path
        '''
        status_position_start= self.get_state_xy(self.position_start[0],self.position_start[1],self.item_in_car,self.item_pickup)
        status_item_pickup= self.get_state_xy(self.item_pickup[0],self.item_pickup[1],self.item_in_car,self.item_pickup)
        status_item_dropoff= self.get_state_xy(self.item_dropoff[0],self.item_dropoff[1],self.item_in_car,self.item_pickup)
        return status_position_start,status_item_pickup,status_item_dropoff

    def get_number_of_states(self):
        '''
        Get the number of states
        return: number of states
        '''
        return (self.size**4)*2 
    
    def get_possibles_states(self):
        '''
        Get the possibles states
        return: list of possibles states
        '''
        actions_= []
        states_ = []
        rewards_ = []
        positions_ = []
        for i in range(self.number_of_actions):
            reward,done,position,item_in_car,item_pickup = self.make_action_virtual(i)       
            state_var= self.get_state_xy(position[0],position[1],item_in_car,item_pickup )
            actions_.append(i)
            states_.append(state_var)
            rewards_.append(reward)
            positions_.append(position)
        return states_,actions_,rewards_,positions_
    
    def get_explore_states(self):
        '''
        Get the explore states
        return: list of explore states and rewards 
        '''
        action_var= random.randint(0,5)
        reward,done,position,item_in_car,item_pickup = self.make_action_virtual(action_var)       
        state_var= self.get_state_xy(position[0],position[1],item_in_car,item_pickup )
        return state_var,action_var,reward
    
    def get_state_xy(self,pos_x,pos_y,item_in_car,item_pickup):
        '''
        Get the state of the position
        pos_x: x position
        pos_y: y position
        item_in_car: item in the car
        item_pickup: item to pickup
        return: state
        '''
        state= pos_x*self.size*self.size*self.size*2
        state+= pos_y*self.size*self.size*2
        state+= item_pickup[0]*self.size*2
        state+= item_pickup[1]*2   
        if item_in_car:
            state+=1
        return state

    def get_state(self):
        '''
        Get the state of the position
        return: state
        '''
        state= self.position[0]*self.size*self.size*self.size*2
        state+= self.position[1]*self.size*self.size*2
        state+= self.item_pickup[0]*self.size*2
        state+= self.item_pickup[1]*2   
        if self.item_in_car:
            state+=1
        return state    
    
    def save_state(self):
        '''
        Save the state
        '''
        self.allposicions.append(self.position)

    def graphics(self,puntos,name_fig):
        '''
        Graphics the path
        puntos: list of positions
        name_fig: name of the figure
        '''
        # Crear una cuadrícula de 10x10
        cuadricula = np.zeros((10, 10))
        # Marcar los puntos en la cuadrícula
        for punto in puntos:
            cuadricula[punto] = 1
        # Crear la figura y el eje para el plot
        fig, ax = plt.subplots()
        # Usar 'imshow' para mostrar la cuadrícula como una imagen
        # 'cmap' define el mapa de colores, 'Greys' es bueno para gráficos en blanco y negro
        ax.imshow(cuadricula, cmap='Greys', origin='lower')
        # Ajustar los ticks para que coincidan con las posiciones de la cuadrícula
        ax.set_xticks(np.arange(-.5, 10, 1))
        ax.set_yticks(np.arange(-.5, 10, 1))
        # Dibujar las líneas de la cuadrícula
        ax.grid(color='black', linestyle='-', linewidth=2)
        # Ajustar el límite para evitar cortes
        ax.set_xlim(-0.5, 9.5)
        ax.set_ylim(-0.5, 9.5)
        for punto in self.block_zones:
            ax.scatter(punto[1], punto[0], color='red', marker='X', s=100) 
        for punto in puntos:
            ax.text(punto[1], punto[0], '✔', color='white', ha='center', va='center', fontsize=10)

        lst_start=[self.position_start, self.item_pickup,self.item_dropoff]
        for punto in lst_start:
            ax.scatter(punto[1], punto[0], color='blue',marker='*', s=100)  
        name_fig_path = self.path_predicts + '/' +name_fig
        plt.savefig(name_fig_path)
        plt.close()

    def empty_predict_data(self):
        '''
        Empty the predict data
        '''
        path=self.path_predicts
        for nombre in os.listdir(path):
            ruta_completa = os.path.join(path, nombre)
            try:
                if os.path.isfile(ruta_completa) or os.path.islink(ruta_completa):
                    os.remove(ruta_completa)
                elif os.path.isdir(ruta_completa):
                    shutil.rmtree(ruta_completa)
            except Exception as e:
                print(f'Error {ruta_completa}. reason: {e}')

    def block_zones_evaluation(self,position):
        '''
        Evaluate if the position is in a block zone
        position: position to evaluate
        return: True if the position is in a block zone, False otherwise'''
        if position in self.block_zones:
            return True
        return False

    def make_action(self,action):
        '''
        Make an action
        action: action to make where in somes cases the position is updated
        return: reward and done
        '''
        (x,y) = self.position
        if action ==0: #down    
            if y==self.size-1:
                return -10,False #reward,done
            else:
                self.position = (x,y+1)
                self.save_state()
                if self.block_zones_evaluation(self.position):
                    return -100,False
                return -1,False
        elif action ==1: #up
            if y==0:
                return -10,False
            else:
                self.position = (x,y-1)
                self.save_state()
                if self.block_zones_evaluation(self.position):
                    return -100,False
                return -1,False
        elif action ==2: #left
            if x==0:
                return -10,False
            else:
                self.position = (x-1,y)
                self.save_state()
                if self.block_zones_evaluation(self.position):
                    return -100,False
                return -1,False
        elif action ==3: #right
            if x==self.size-1:
                return -10,False
            else:
                self.position = (x+1,y)
                self.save_state()
                if self.block_zones_evaluation(self.position):
                    return -100,False
                return -1,False
        elif action ==4: #pickup
            if self.item_in_car:
                return -10,False
            elif self.item_pickup != (x,y):
                return -10,False
            else:
                self.item_in_car = True
                return 20,False
        elif action ==5: #dropoff
            if not self.item_in_car:
                return -10,False
            elif self.item_dropoff != (x,y):
                return -10,False
            else:
                self.item_in_car = False
                return 20,True 
            
    def make_action_virtual(self,action):
            '''
            Make an action in virtual mode that means that the position is not updated
            action: action to make
            return: reward, done, position, item_in_car, item_pickup'''
            (x,y) = self.position
            virtual_position = (x,y)
            item_in_car=self.item_in_car
            item_pickup=self.item_pickup
            item_dropoff=self.item_dropoff
            if action ==0: #down    
                if y==self.size-1:
                    return -10,False,virtual_position,item_in_car,item_pickup 
                else:
                    virtual_position = (x,y+1)
                    if self.block_zones_evaluation(virtual_position):
                        return -100,False, virtual_position,item_in_car,item_pickup 
                    return -1,False, virtual_position,item_in_car,item_pickup 
            elif action ==1: #up
                if y==0:
                    return -10,False,virtual_position,item_in_car,item_pickup 
                else:
                    virtual_position= (x,y-1)
                    if self.block_zones_evaluation(virtual_position):
                        return -100,False, virtual_position,item_in_car,item_pickup 
                    return -1,False, virtual_position,item_in_car,item_pickup 
            elif action ==2: #left
                if x==0:
                    return -10,False,virtual_position,item_in_car,item_pickup 
                else:
                    virtual_position = (x-1,y)
                    if self.block_zones_evaluation(virtual_position):
                        return -100,False, virtual_position,item_in_car,item_pickup 
                    return -1,False,virtual_position,item_in_car,item_pickup 
            elif action ==3: #right
                if x==self.size-1:
                    return -10,False,virtual_position,item_in_car,item_pickup 
                else:
                    virtual_position = (x+1,y)
                    if self.block_zones_evaluation(virtual_position):
                        return -100,False, virtual_position,item_in_car,item_pickup 
                    return -1,False,virtual_position,item_in_car,item_pickup 
            elif action ==4: #pickup
                if item_in_car:
                    return -10,False,virtual_position,item_in_car,item_pickup 
                elif item_pickup != (x,y):
                    return -10,False,virtual_position,item_in_car,item_pickup 
                else:
                    item_in_car = True
                    return 20,False,virtual_position,item_in_car,item_pickup 
            elif action ==5: #dropoff
                if not item_in_car:
                    return -10,False,virtual_position,item_in_car,item_pickup 
                elif item_dropoff != (x,y):
                    return -10,False,virtual_position,item_in_car,item_pickup 
                else:
                    item_in_car = False
                    return 20,True ,virtual_position,item_in_car,item_pickup 

