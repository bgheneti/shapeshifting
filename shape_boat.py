import numpy as np

from convex_hulls_graph import *
from shape import *
from multiboat_trajectory_optimization.boat_models import *
from multiboat_trajectory_optimization.trajectory_planner import *
from shapely.geometry import Point
import numpy as np

class ShapeBoat(TwoInputBoat, object):
    
    linear=True
    
    def __init__(self, boat_shape, obstacle_shape, margin=0):
        self.shape = boat_shape
        self.obstacle_shape = obstacle_shape
        self.hulls = self.obstacle_shape.c_space(boat_shape,margin=margin).partition_around(self.shape, buffered=False)
        self.g = HullGraph(self.hulls)
        self.hull_path = None
    
    def toProblemStates(self, boats_S):
        assert len(boats_S)==1
        return super(ShapeBoat, self).toProblemStates(boats_S)
    
    def toGlobalStates(self, boats_S, state_initial=None):
        assert len(boats_S)==1
        return super(ShapeBoat, self).toGlobalStates(boats_S, state_initial)

    def set_end_points(self, x0, xN):
        self.hull_path = self.g.point_path(Point(x0[0,:2]), Point(xN[0,:2])) 

    def hull_constraint(self, h_bool, x, eq, M=100., mp=None):
        val1 = M*(h_bool-1)+eq['b']
        val2 = eq['A'].dot((x).T)
        return mp.add_leq_constraints(val1, val2, linear=True) if mp is not None else val1<=val2
        
    def add_collision_constraints(self, mp, boats_S, M=20.):
        assert len(boats_S)==1
            
        T = len(boats_S[0])
        H = len(self.hull_path) 
        self.in_hull = mp.NewBinaryVariables(T, H, "c")
        
        for t in range(T):
            mp.AddLinearConstraint(np.sum(self.in_hull[t])==1)
            boat_s = boats_S[0][t]
            for i,h in enumerate(self.hull_path):
                
                self.hull_constraint(self.in_hull[t][i],                                       \
                                     boats_S[0,t,:2],                                          \
                                     self.g.vertex_properties["polygon_eq"][self.g.vertex(h)], \
                                     mp=mp                                                     \
                                    )
                if t<T-1:
                    if i>1:
                        mp.AddLinearConstraint(2*(1-self.in_hull[t][i])>=self.in_hull[t+1][i-1])
                    if i<H-1:
                        mp.AddLinearConstraint((self.in_hull[t][i]-1)+1<=self.in_hull[t+1][i]+self.in_hull[t+1][i+1])
                    if i==H-1:
                        mp.AddLinearConstraint((self.in_hull[t][i]-1)+1<=self.in_hull[t+1][i])                        
                        
        mp.AddLinearConstraint(self.in_hull[0][0]==1)
        mp.AddLinearConstraint(self.in_hull[-1][-1]==1)        
                
    @classmethod
    def boat_dynamics(cls, s, u):

        derivs = np.zeros_like(s)
        
        derivs[0] = s[2];
        derivs[1] = s[3];
        derivs[2] = u[0]-cls.d11 / cls.m11 * s[2] + u[0] / cls.m11;
        derivs[3] = u[1]-cls.d22 / cls.m22 * s[3] + u[1] / cls.m22;
        return derivs

class ShapeBoat_spline(TwoInputBoat, object):
    
    linear=True
    
    def __init__(self, boat_shape, obstacle_shape, margin=0):
        self.shape = boat_shape
        self.obstacle_shape = obstacle_shape
        self.hulls = self.obstacle_shape.c_space(boat_shape,margin=margin).partition_around(self.shape, buffered=False)
        self.g = HullGraph(self.hulls)
        self.hull_path = None
    
    def toProblemStates(self, boats_S):
        assert len(boats_S)==1
        return super(ShapeBoat_spline, self).toProblemStates(boats_S)
    
    def toGlobalStates(self, boats_S, state_initial=None):
        assert len(boats_S)==1
        return super(ShapeBoat_spline, self).toGlobalStates(boats_S, state_initial)

    def set_end_points(self, x0, xN):
        self.hull_path = self.g.point_path(Point(x0[0,:2]), Point(xN[0,:2])) 

    def hull_constraint(self, h_bool, x, eq, M=100., mp=None):
        val1 = M*(h_bool-1)+eq['b']
        val2 = eq['A'].dot((x).T)
#         mp.AddLinearConstraint(x[0]>=-5)
#         mp.AddLinearConstraint(x[1]>=-5)
#         mp.AddLinearConstraint(x[0]<=5)
#         mp.AddLinearConstraint(x[1]<=5)

        return mp.add_leq_constraints(val1, val2, linear=True) if mp is not None else val1<=val2
        
    def edge_constraint(self, e_bool, a, x1, e1, e2, M=100., mp=None):
        const = M*(e_bool-np.ones(2))
        x_dif = a*e1 + (1-a)*e2 - x1
                
        val1 = np.concatenate((const, x_dif))
        val2 = np.concatenate((x_dif, -const))
        return mp.add_leq_constraints(val1, val2, linear=True) if mp is not None else val1==val2

    def edge_line_constraint(self, e_bool, b, x0, x1, x2, M=100., mp=None):
        const = M*(e_bool-np.ones(2))
        x_dif = x0+b*(x1-x0)-x2
                
        val1 = np.concatenate((const, x_dif))
        val2 = np.concatenate((x_dif, -const))
        return True
        return mp.add_leq_constraints(val1, val2, linear=True) if mp is not None else val1==val2    
    
    def add_collision_constraints(self, mp, boats_S, M=20.):
        assert len(boats_S)==1
            
        T = len(boats_S[0])
        H = len(self.hull_path) 
        self.in_hull = mp.NewBinaryVariables(T, H, "hull")
        self.on_edge = mp.NewBinaryVariables(T, H-1, "edge")
        
        self.a = mp.NewContinuousVariables(H-1, "alpha")
        self.b = mp.NewContinuousVariables(H-1, "beta")
        mp.AddConstraint(0. <= self.a[0])
        mp.add_leq_constraints(self.a, np.ones(H-1), linear=True)
        mp.add_leq_constraints(-self.a, np.zeros(H-1), linear=False)        
        
        for t in range(T):
            mp.AddLinearConstraint(np.sum(self.in_hull[t])+np.sum(self.on_edge[t])==1)
            boat_s = boats_S[0][t]
            for i,h in enumerate(self.hull_path):
                
                self.hull_constraint(self.in_hull[t][i],                                       \
                                     boats_S[0,t,:2],                                          \
                                     self.g.vertex_properties["polygon_eq"][self.g.vertex(h)], \
                                     mp=mp                                                     \
                                    )
                if t<T-1:
#                     if i>1:
#                         mp.AddLinearConstraint(2*(1-self.in_hull[t][i])>=self.in_hull[t+1][i-1])
#                     if i<H-1:
#                         mp.AddLinearConstraint((self.in_hull[t][i]-1)+1<=self.in_hull[t+1][i]+self.in_hull[t+1][i+1])
                    if i==H-1:
                        mp.AddLinearConstraint(self.in_hull[t][i]<=self.in_hull[t+1][i])
                    if i<H-1:
                        mp.AddLinearConstraint(self.in_hull[t][i]<=self.in_hull[t+1][i]+self.on_edge[t+1][i])
                if t<T-2 and i<H-1:
                    mp.AddLinearConstraint(self.in_hull[t+2][i+1]+self.in_hull[t][i]<=1+self.on_edge[t+1][i])
                                   
        for t in range(1,T-1):                      
            for i,(h1,h2) in enumerate(zip(self.hull_path[:-1],self.hull_path[1:])):
                self.edge_constraint(self.on_edge[t][i],                                       \
                                     self.a[i],                                                \
                                     boats_S[0,t,:2],                                          \
                                     *self.g.edge_properties["points"][self.g.edge(h1,h2)],    \
                                     mp=mp                                                     \
                                    )
                self.edge_line_constraint(self.on_edge[t][i],                                       \
                                          self.b[i],                                                \
                                          boats_S[0,t-1,:2],                                        \
                                          boats_S[0,t,:2],                                          \
                                          boats_S[0,t+1,:2],                                        \
                                          mp=mp                                                     \
                                         )                                   
                                   
        mp.AddLinearConstraint(self.in_hull[0][0]==1)
        mp.AddLinearConstraint(self.in_hull[-1][-1]==1)    
                
    @classmethod
    def boat_dynamics(cls, s, u):

        derivs = np.zeros_like(s)
        
        derivs[0] = s[2];
        derivs[1] = s[3];
        derivs[2] = u[0]-cls.d11 / cls.m11 * s[2] + u[0] / cls.m11;
        derivs[3] = u[1]-cls.d22 / cls.m22 * s[3] + u[1] / cls.m22;
        return derivs
    
    
    
def check_vertex_constraints(boat):
    ##Check Contains Functions works properly
    X = (np.random.rand(100,2)-0.5)*8
    for x in X:
        for v in boat.g.vertices():
            assert(
                np.all(boat.hull_constraint(1,x,boat.g.vertex_properties["polygon_eq"][v])) == \
                boat.g.vertex_properties['polygon'][v].contains(Point(x))
                  )
          
def check_state_in_hull(path_hull_i, state_i):
    hull_i = boat.hull_path[path_hull_i]
    print boat.g.vertex_properties['polygon'][boat.g.vertex(hull_i)].contains(Point(boats_S[0,state_i][:2]))

        