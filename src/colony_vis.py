#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 13:08:15 2022

@author: cfletcher1
"""

from tkinter import Tk, Canvas, Frame, BOTH
from ant_colony import AgentManager,Grid, SpaceType
from math import sin,cos,pi
import time

class HexGrid(Frame):
    
    def __init__(self,agents):
        super().__init__()
        self.initUI(agents)
        
    def initUI(self,agents):
        self.master.title("Ant Colony Hex Grid")
        self.pack(fill=BOTH,expand=1)
        self.canvas = Canvas(self)
        self.agents = agents
        self.timer = 0
        
        r = 20
        start_position = [sin(pi/6)*r,0]
        self.grid_spaces = agents.grid.grid.flatten()
        xy_positions = [[space.x,space.y] for space in self.grid_spaces]
        uv_positions = [calculate_cartesian_grid_position(x,y,r) for x,y in xy_positions]
        uv_positions = [[start_position[0]+u,start_position[1]+v] for u,v in uv_positions]
        hexpoints_positions = [calculate_hexpoints_from_tl(u,v,r) for u,v in uv_positions]
       
        
        self.canvas_hexes = [self.canvas.create_polygon(points,
                                              outline='#000000',fill='#ed5876',width=1) for points in hexpoints_positions]
        #[self.canvas.create_text(uv[0]+r/2,uv[1]+r*cos(pi/6),text=str(xy),font=('Helvetica 5 bold')) for uv,xy in zip(uv_positions,xy_positions)]
        self.grid_to_hex = dict(zip(self.grid_spaces,self.canvas_hexes))
        #canvas.pack(fill=BOTH,expand=1)
        #print(self.grid_spaces[0].position,self.grid_to_hex[self.grid_spaces[0]])
        
        nest_centers = [nest.centre_position for nest in agents.nests]
        nest_text_positions = [calculate_cartesian_grid_position(x, y, r) for x,y in nest_centers]
        nest_text_centers = [[u+r/2,v+r*cos(pi/6)] for u,v in nest_text_positions]
        nest_text = [self.canvas.create_text(u,v,text="0",font=('Helvetica 20 bold')) for u,v in nest_text_centers]
        self.nest_to_text = dict(zip(agents.nests,nest_text))
        
        self.iterate_system()
        
    
            
        self.canvas.pack(fill=BOTH,expand=1)
        
    def iterate_system(self):
        if(self.timer<10000):
            self.agents.iterate_system()
                
            for polygon,space in zip(self.canvas_hexes,self.grid_spaces):
                
                if(space.type == SpaceType.WALL):
                    fill_colour = '#000000'
                elif(space.type == SpaceType.NEST):
                    fill_colour = '#61d44a'
                else:
                    fill_colour = '#FFFFFF'
                    food_max_colour = '#fcba03'
                    pos_pher_max_colour = '#4287f5'
                    for_pher_max_colour = '#f542f5'
                    neg_pher_max_colour = '#ff0d2d'
                    if(space.food>0):
                        fill_colour = lerp_hex(fill_colour,food_max_colour,min(1,space.food))
                    else:
                        if(space.positive_pher>0):
                            fill_colour = lerp_hex(fill_colour,pos_pher_max_colour,space.positive_pher)
                        if(space.forage_pher>0):
                            fill_colour = lerp_hex(fill_colour,for_pher_max_colour,space.forage_pher)
                        if(space.negative_pher>0):
                            fill_colour = lerp_hex(fill_colour,neg_pher_max_colour,space.negative_pher)
                        
         
                self.canvas.itemconfig(polygon,fill=fill_colour) 
            
            for agent in self.agents.agents:
                position = agent.position
                self.canvas.itemconfig(self.grid_to_hex[self.agents.grid.grid[position[0],position[1]]],fill='#a2a34b')
            self.after(1,self.iterate_system)
            self.timer+=1
            
            for nest in self.agents.nests:
                self.canvas.itemconfig(self.nest_to_text[nest],text=str(nest.food))
        
        
def calculate_hexpoints_from_tl(u,v,r):
    u_list = [0,1,1+sin(pi/6),1,0,-sin(pi/6)]
    v_list = [0,0,cos(pi/6),2*cos(pi/6),2*cos(pi/6),cos(pi/6)]
    points_list = []
    for i in range(len(u_list)):
        points_list.append(u+r*u_list[i])
        points_list.append(v+r*v_list[i])
    return points_list
    
    
def calculate_cartesian_grid_position(x,y,r):  
    if(x%2==0):
        u=y*2*(sin(pi/6)+1)
        v=x*cos(pi/6)
    else:
        u=(1+y*2)*(sin(pi/6)+1)
        v=x*cos(pi/6)
    return r*u,r*v
  
def lerp_hex(a,b,t):
    a = hex_to_rgb(a)
    b= hex_to_rgb(b)
    x = int(t*(b[0]-a[0]) + a[0])
    y = int(t*(b[1]-a[1]) + a[1])
    z = int(t*(b[2]-a[2]) + a[2])
    return rgb_to_hex((x,y,z))
        
def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2],16) for i in (0,2,4))

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb

def main():
    grid_size = [82,39]
    #82,39,r=20
    
    perlin_octaves = 3
    wall_threshold = 0.05
    grid = Grid(grid_size,perlin_octaves,wall_threshold)
    agents = AgentManager(grid)
    
    num_nests = 5
    nest_rad = 4
    nest_agent_num = 5
    num_food = 7
    food_rad = 3
    for i in range(num_nests):
        spawn_point = agents.grid.find_random_valid_circle(nest_rad)
        if(spawn_point!=None):
            pos = [spawn_point.x,spawn_point.y]
            agents.spawn_agents_around_point(nest_agent_num, pos,nest_rad)
            agents.create_nest_around_point(pos,nest_rad)
        
    for i in range(num_food):
        spawn_point = agents.grid.find_random_valid_circle(food_rad)
        if(spawn_point!=None):
            pos = [spawn_point.x,spawn_point.y]
            agents.grid.add_food_cluster(pos,food_rad,0.5)
            
    root=Tk()
    hexGrid = HexGrid(agents)
    root.mainloop()
    hexGrid.iterate_system()
    

        
        

if __name__ == "__main__":
   main()
    
    
