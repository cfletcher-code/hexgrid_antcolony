# -*- coding: utf-8 -*-


import numpy as np
import cv2
import pandas as pd
from enum import Enum ,IntEnum
import random 

class AgentState(Enum):
    FORAGING = 0
    RETURNING = 1
    
class SpaceType(Enum):
    EMPTY = 0
    WALL = 1
    NEST = 2
    
class GridDirection(IntEnum):
    UPLEFT  = 0
    UP = 1
    UPRIGHT = 2
    DOWNRIGHT  = 3
    DOWN = 4
    DOWNLEFT = 5

class Nest():
    
    def __init__(self,position,grid_spaces):
        self.centre_position = position
        self.food = 0
        self.grid_spaces = grid_spaces
        
    def add_food(self,food):
        self.food+=food
    
class AgentManager():

    def __init__(self,grid):
        self.agents = []
        self.grid = grid
        self.nests = []
    
    def add_agent_at_position(self,position):
        if(self.grid.check_agent_position_valid(position)):
            if(not self.is_agent_at_position(position)):
                agent = Agent(len(self.agents),position)
                self.agents.append(agent)
                return True
            else:
                print("Agent already at position " + str(position))
        else:
            print("Position not valid for agent placement")
        return False
    def is_agent_at_position(self,position):
        for agent in self.agents:
            if(agent.position == position):
                return True
        return False
    
    def is_position_free_for_agent(self,position):
        if(self.grid.check_agent_position_valid(position)
            and not self.is_agent_at_position(position)):
            return True
        return False

    def create_nest_around_point(self,position,radius):
        grid_spaces = self.grid.get_circle_positions_around_point(position, radius)
        nest = Nest(position,grid_spaces)
        self.nests.append(nest)
        for grid_space in grid_spaces:
            self.grid.grid[grid_space[0],grid_space[1]].type = SpaceType.NEST
            self.grid.grid[grid_space[0],grid_space[1]].nest = nest
            self.grid.grid[grid_space[0],grid_space[1]].nest_value = 1
            
    def spawn_agents_around_point(self,num_agents,position,radius=None,desired_density=0.4,strict_radius=False):
        if(radius==None):
            desired_spaces = int(num_agents/desired_density)
            proposed_radius= 0 
            while(True):
                if(3*proposed_radius*(proposed_radius+1)+1 > desired_spaces):
                    radius=proposed_radius
                    break
                proposed_radius+=1
                
        potential_spawn_points = []
        while(True):
            potential_spawn_points = [agent_position for agent_position in 
                        self.grid.get_circle_positions_around_point(position, radius)
                        if self.is_position_free_for_agent(agent_position)]            
            if(len(potential_spawn_points)>=num_agents):
                break
            else:
                if(strict_radius):
                    print("Unable to spawn " + str(num_agents-len(potential_spawn_points)) + " agents...")
                    num_agents = len(potential_spawn_points)
                    break
                radius+=1
        
        potential_spawn_point_indices = [i for i in range(len(potential_spawn_points))]
        spawn_point_indices = np.random.choice(potential_spawn_point_indices,num_agents,replace=False)
        
        for spawn_point_index in spawn_point_indices:
            #print("spawning at " + str(potential_spawn_points[spawn_point_index]))
            self.add_agent_at_position(potential_spawn_points[spawn_point_index])
            
    
        
        
    def iterate_system(self):
        for agent in self.agents:
            self.agent_behaviour(agent)
        self.grid.iterate_grid()
            
    def agent_behaviour(self,agent):
        
        current_grid_space = self.grid.grid[agent.position[0],agent.position[1]]
        
        if(agent.state == AgentState.FORAGING):
            if(current_grid_space.food>0):
                food_taken = min(agent.max_capacity-agent.food,current_grid_space.food)
                agent.food += food_taken
                current_grid_space.take_food(food_taken)
                if(agent.max_capacity==agent.food):
                    agent.direction = GridDirection((int(agent.direction)+3)%6)
                    agent.state = AgentState.RETURNING
                    
        elif(agent.state == AgentState.RETURNING):
            if(current_grid_space.nest_value==1):
                nest = current_grid_space.nest
                nest.add_food(agent.food)
                agent.food = 0
                agent.state = AgentState.FORAGING
      
            
        forward_dir_int = int(agent.direction)
        #print(forward_dir_int)
        direction_ints_raw = [r for r in range(forward_dir_int-3,forward_dir_int+3)]

        agent_params = agent.agent_parameters
        direction_modulo_raw = np.array([r%6 for r in direction_ints_raw])
        direction_modulo = []
        direction_ints = []
        for i in range(len(direction_modulo_raw)):
            if(self.grid.check_agent_position_valid(self.grid.get_position_in_direction(agent.position, GridDirection(direction_modulo_raw[i])))):
                direction_modulo.append(direction_modulo_raw[i])
                direction_ints.append(direction_ints_raw[i])
        #print(direction_modulo)
        
        fov_variance = agent.agent_parameters["fov_var"]
        fov_weights = np.array([normal_values_given_sd(dir_int-forward_dir_int,fov_variance) for dir_int in direction_ints])
        
        grid_positions = [self.grid.raycast_in_direction(agent.position, direction, 
                                                         agent_params["perception_range"]) for direction in direction_modulo]
      
        grid_pos_objs = [[self.grid.grid[x,y] for x,y in pos] for pos in grid_positions]
        
        grid_pos_stats = [[np.array([pos.food,pos.positive_pher,pos.negative_pher,pos.forage_pher,pos.nest_value]) for pos in pos_list] for pos_list in grid_pos_objs]
        grid_pos_falloff = [np.array([exp_falloff(i,agent_params["perception_falloff"]) for i in range(len(pos_list))]) for pos_list in grid_pos_objs]
        grid_pos_falloff_norm = [falloff/np.sum(falloff) for falloff in grid_pos_falloff]
        
        #print(grid_pos_falloff_norm)
        grid_pos_weights = [np.sum([stats[i]*falloff[i] for i in range(len(stats))],axis=0) for stats,falloff in zip(grid_pos_stats,grid_pos_falloff_norm)]
        
        is_position_free = [self.is_position_free_for_agent(pos[0]) for pos in grid_positions]
        
        attractor_stats = agent_params["attractor_weights"]
        if(agent.state == AgentState.FORAGING):
            attractor_weights = np.array([attractor_stats["food"],attractor_stats["pos_pher"],attractor_stats["neg_pher"],0,0])
        elif(agent.state == AgentState.RETURNING): 
            attractor_weights = np.array([0,0,attractor_stats["neg_pher"],attractor_stats["for_pher"],attractor_stats["nest"]])
  
        old_position = agent.position
        
        if(not any(is_position_free)):
            fov_weights = fov_weights/np.sum(fov_weights)
            if(len(direction_modulo)!=0):
                chosen_direction = np.random.choice(direction_modulo,p=fov_weights)
            else:
                chosen_direction = agent.direction
        else:          
            computed_direction_weights = np.array([np.sum(attractor_weights*grid_pos_weight) for grid_pos_weight in grid_pos_weights])
            #print(fov_weights,computed_direction_weights,is_position_free)
            #TODO test softmax instead
            final_weights = computed_direction_weights*is_position_free
            #final_weights = fov_weights*sigmoid_activation(computed_direction_weights,agent_params["activation_falloff"])*is_position_free
            
            final_weights = softmax_vector(final_weights)
            final_weights = fov_weights*final_weights
            final_weights = final_weights/np.sum(final_weights)
            
            #print(direction_modulo,final_weights)
            
            chosen_direction = np.random.choice(direction_modulo,p=final_weights)
            agent.position = self.grid.get_position_in_direction(agent.position, chosen_direction)
        agent.direction = GridDirection(chosen_direction)
        
        if(np.sum(is_position_free)>2):
            if(agent.state == AgentState.FORAGING):
                current_grid_space.forage_pher = min(1,0.4+current_grid_space.forage_pher)
            elif(agent.state == AgentState.RETURNING):
                current_grid_space.positive_pher = min(1,0.6+current_grid_space.positive_pher)
        else:
            current_grid_space.negative_pher = min(1,0.5+current_grid_space.negative_pher)

        #print(old_position, GridDirection(chosen_direction),agent.position)
        
        
        
def exp_falloff(x,falloff):
    return np.exp(-x*falloff)
 
def softmax_vector(x):
    return np.array([np.exp(xi)/np.sum(np.exp(x)) for xi in x])  

   
def sigmoid_activation(x,falloff):
    return 1/(1+np.exp(-falloff*x)) 
   
def normal_values_given_sd(x,var):
    return np.exp(-x**2/(2*var))/(np.sqrt(np.pi*2*var) )   

class Agent():
    
    def __init__(self,agent_id,position):
        self.agent_id = agent_id
        self.is_alive = True
        self.position = position
        self.max_capacity = 0.5
        self.food = 0
        self.state = AgentState.FORAGING
        self.direction = random.choice(list(GridDirection))
        self.agent_parameters = {"fov_var":0.6,"perception_range":6,"perception_falloff":0.15,
                                 "activation_falloff":0.5,"attractor_weights":{"food":20,"nest":20,"pos_pher":12,"neg_pher":-10,"for_pher":12}}
    
        
from perlin_noise import PerlinNoise   
    
class Grid():
    
    def __init__(self,grid_size,perlin_octaves,wall_threshold):
        self.grid_size = grid_size
        self.agents = []
        self.perlin_noise = PerlinNoise(octaves=perlin_octaves)
        self.grid=np.array([[GridSpace(x,y) for y in range(grid_size[1])] for x in range(grid_size[0])])
        self.update_walls_for_threshold(wall_threshold)
        
        self.directions = directions={1:{GridDirection.UPLEFT:[-1,0],GridDirection.UP:[-2,0],
         GridDirection.UPRIGHT:[-1,1],GridDirection.DOWNRIGHT:[1,1],
         GridDirection.DOWN:[2,0],GridDirection.DOWNLEFT:[1,0]},
         0:{GridDirection.UPLEFT:[-1,-1],GridDirection.UP:[-2,0],
          GridDirection.UPRIGHT:[-1,0],GridDirection.DOWNRIGHT:[1,0],
          GridDirection.DOWN:[2,0],GridDirection.DOWNLEFT:[1,-1]}}
    
    def update_walls_for_threshold(self,wall_threshold):
        for space in self.grid.flatten():
            perlin_level = self.perlin_noise([space.x/self.grid_size[0],space.y/self.grid_size[1]])
            if(perlin_level>wall_threshold):
                space.type = SpaceType.WALL
    
    def find_random_valid_circle(self,radius,max_steps=20):
        step = 0
        while(step<max_steps):
            random_space = self.pick_random_freespace()
            if(self.check_circle_is_valid([random_space.x,random_space.y],radius)):
                return random_space
            step+=1
        print("Unable to find valid circle")
        return None
        
    def check_circle_is_valid(self,position,radius):
        grid_spaces = self.get_circle_positions_around_point(position,radius)
        if (any([not self.check_agent_position_valid(pos) for pos in grid_spaces])):
            return False
        return True
    
    def add_food_cluster(self,position,radius,food_density):
        grid_positions = self.get_circle_positions_around_point(position,radius)
        for pos in grid_positions:
            random_num = random.random()
            if(random_num<food_density):
                self.grid[pos[0],pos[1]].add_food(1)
        
    def pick_random_freespace(self):
        flattened_grid = self.grid.flatten()
        while(True):       
            random_space = np.random.choice(flattened_grid)
            if(random_space.type!=SpaceType.WALL):
                break
        return random_space
        
    def check_agent_position_valid(self,position):
        if(self.check_position_valid(position) and 
           not (self.grid[position[0],position[1]].type == SpaceType.WALL)):
            return True
        return False
        
    def get_circle_positions_around_point(self,position,radius):
        positions_list = [position]
        upleft_pos=up_pos=upright_pos=downright_pos=down_pos= downleft_pos = position
        for curr_radius in range(radius):
            #UPLEFT
            
            upleft_pos = self.get_position_in_direction(upleft_pos,GridDirection.UPLEFT)
            positions_list += self.raycast_in_direction(upleft_pos,GridDirection.UPRIGHT,curr_radius,False)
            positions_list += self.raycast_in_direction(upleft_pos,GridDirection.DOWN,curr_radius,False)
            
            #UPRIGHT
            upright_pos = self.get_position_in_direction(upright_pos, GridDirection.UPRIGHT)
            positions_list += self.raycast_in_direction(upright_pos,GridDirection.UPLEFT,curr_radius,False)
            positions_list += self.raycast_in_direction(upright_pos,GridDirection.DOWN,curr_radius,False)
            #UP
            up_pos = self.get_position_in_direction(up_pos,GridDirection.UP)
            
            #DOWNRIGHT
            downright_pos = self.get_position_in_direction(downright_pos, GridDirection.DOWNRIGHT)
            positions_list += self.raycast_in_direction(downright_pos,GridDirection.DOWNLEFT,curr_radius,False)
            
            #DOWN
            down_pos = self.get_position_in_direction(down_pos, GridDirection.DOWN)
            
            #DOWNLEFT
            downleft_pos = self.get_position_in_direction(downleft_pos, GridDirection.DOWNLEFT)
            positions_list += self.raycast_in_direction(downleft_pos,GridDirection.DOWNRIGHT,curr_radius,False)
            
            positions_list += [upleft_pos,up_pos,upright_pos,downright_pos,down_pos,downleft_pos]
            
        return positions_list
    
    def get_position_type(self,position):
        return self.grid[position[0],position[1]].type
    
    def get_position_neighbors(self,position):
        x = position[0]
        y = position[1]
        if(y%2==0):
            neighbors = [[x-1,y],[x-2,y],[x-1,y+1],
                    [x+1,y-1],[x+2,y],[x+1,y]]
        else:
            neighbors = [[x-1,y-1],[x-2,y],[x-1,y],
                    [x+1,y],[x+2,y],[x+1,y-1]] 
        
        final_neighbors = []
        for neighbor in neighbors:
            neighbor_x = neighbor[0]
            neighbor_y = neighbor[1]
            if ((neighbor_x >= 0 and neighbor_x < self.grid_size[0]) and
                (neighbor_y >= 0 and neighbor_y < self.grid_size[1])):
                final_neighbors.append(neighbor)
        return final_neighbors
     
    def get_position_in_direction(self,position,direction):
        x=position[0]
        y=position[1]
        delta = self.directions[x%2][direction]
        return [x+delta[0],y+delta[1]]
          
    def raycast_in_direction(self,position,direction,ray_length,validity_required=True):
        ray_positions = []
        last_position = position
        for i in range(ray_length):
            next_position = self.get_position_in_direction(last_position,direction)
            if(validity_required):
                if(self.check_position_valid(next_position) and 
                   self.get_position_type(next_position)!=SpaceType.WALL):
                    ray_positions.append(next_position)
                    last_position = next_position
                else:
                    break
            else:
                ray_positions.append(next_position)
                last_position = next_position
        return ray_positions
       
    def check_position_valid(self,position):
        x=position[0]
        y=position[1]
        if(x<0 or x>=self.grid_size[0] or y<0 or y>=self.grid_size[1]):
            return False
        else:
            return True
        
    def get_position_objects(self,position_list):
        return [self.grid[x,y] for x,y in position_list]
    
    def iterate_grid(self):
        for space in self.grid.flatten():
            space.iterate_space()
    
class GridSpace():
     
    
    def __init__(self,x,y) :
        self.x = x
        self.y = y
        self.type = SpaceType.EMPTY
        self.nest = None
        self.nest_value = 0
        self.food = 0
        self.positive_pher = 0
        self.negative_pher = 0
        self.forage_pher = 0
        self.pos_pher_decay_rate = 0.05
        self.forage_pher_decay_rate = 0.2
        self.neg_pher_decay_rate = 0.3
        
    def add_food(self,food_to_add):
        self.food+=food_to_add
        
    def take_food(self,food_to_take):
        food_taken = min(food_to_take,self.food)
        self.food = max(0,self.food-food_to_take)
        return food_taken
    
    def iterate_space(self):
        self.positive_pher*=(1-self.pos_pher_decay_rate)
        self.forage_pher*=(1-self.forage_pher_decay_rate)
        self.negative_pher*=(1-self.neg_pher_decay_rate)
        
        

    
def main():
    grid_size = [100,100]
    perlin_octaves = 10
    wall_threshold = 0.5
    grid = Grid(grid_size,perlin_octaves,wall_threshold)
    agents = AgentManager(grid)
    return agents


# if __name__ == "__main__":
#     main()