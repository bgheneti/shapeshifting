import numpy as np

from convex_hulls_graph import *
from shape import *
from multiboat_trajectory_optimization.boat_models import *
from multiboat_trajectory_optimization.trajectory_planner import *
from shapely.geometry import Point
from timeit import time
from pydrake.all import Variable
import numpy as np
import math
from collections import Iterable


class ShapeBoat(ThreeInputBoat, object):
    linear = True
    num_inputs = 4
    S_transition_inds = range(ThreeInputBoat.num_states)
    
    def __init__(self, boat_shape, obstacle_shape, search=True):
        self.shape = boat_shape
        self.obstacle_shape = obstacle_shape
        self.hulls     = None
        self.msums     = None
        self.g         = None
        self.path      = None
        self.hull_path = None
        self.search    = search
    
    def toProblemStates(self, S):
        assert len(S)==1
        return super(ShapeBoat, self).toProblemStates(S)
    
    def toGlobalStates(self, S, state_initial=None):
        assert len(S)==1
        return super(ShapeBoat, self).toGlobalStates(S, state_initial)
    
    def toBoatPlotStates(self, S, shape=None):
        if shape is None:
            shape = self.shape
                        
        assert S.shape[0]==1
        
        theta = S[0,:,2]/180*np.pi
        
        sin = np.sin(theta)
        cos = np.cos(theta)
        zeros = np.zeros(len(theta))
        
        centroids = [(block.x,block.y) for block in shape.blocks]
        
        S_subblocks = np.vstack([np.dstack([cos*x-sin*y, sin*x+cos*y, theta, zeros, zeros, zeros]) for x,y in centroids])
        
        S_subblocks[:,:,:2] += S[:,:,:2]
                
        return S_subblocks
    
    def plot_configurations(self, S, stride=5):
        return super(ShapeBoat, self).plot_configurations(self.toBoatPlotStates(S), stride=stride)
        
    def plot_animation(self, S, input_trajectories=None, show_regions=False):
        return super(ShapeBoat, self).plot_animation(self.toBoatPlotStates(S), input_trajectories=input_trajectories, show_regions=show_regions)
        
    def set_end_points(self, x0, xN, all_hulls=False):
        print "SETTING ENDPOINTS"
        start = time.time()
        self.msums, self.hulls = self.obstacle_shape.c_space_rotate(self.shape, x0[0], xN[0])
        print "Calculated C-Space: %f seconds" % (time.time() - start)
        start = time.time()
        self.g = HullGraph(self.hulls)    
        
        self.plot_hulls(all_hulls=True)
        
        print "Set up graph: %f seconds" % (time.time() - start)

        if self.search:
            self.path  = self.g.point_path(Point(*x0[0,:2]), Point(*xN[0,:2]), x0[0, 2], xN[0,2])
        else:
            self.path = range(self.g.num_vertices())

        self.hull_path = [{ "polygon":    self.g.vertex_properties["polygon"][i],    \
                            "polygon_eq": self.g.vertex_properties["polygon_eq"][i], \
                            "min_angle":  self.g.vertex_properties["min_angle"][i],  \
                            "max_angle":  self.g.vertex_properties["max_angle"][i]} for i in self.path]
        self.plot_hulls()
        
    def plot_hulls(self, S=None, S_knots=None, in_hull=None, all_hulls=False, text=True, both=False):
        if all_hulls:
            chosen_hulls = [x for hulls in self.hulls.values() for x in hulls]
        else: 
            assert self.hull_path is not None
            if in_hull is not None:
                assert in_hull.shape[1] == len(self.hull_path)
            chosen_i = range(len(self.hull_path)) if in_hull is None else np.where(np.sum(in_hull, axis=0)>0)[0] 
            chosen_hulls = [self.hull_path[i]['polygon'] for i in chosen_i]
  
        plt.figure(figsize=(10,10))
        plot_hulls([x for hulls in self.hulls.values() for x in hulls] if both else [], None if S_knots is None else (S_knots[0,:,0], S_knots[0,:,1]), figure=False, color='red', hull_color='black', text=text)
        plot_hulls(chosen_hulls, None if S is None else (S[0,:,0], S[0,:,1]), text=text, figure=False)

        plt.tick_params(labelsize='15')
        plt.xlabel('x (m)', fontsize='15')
        plt.ylabel('y (m)', fontsize='15')
        plt.show()
        
    @staticmethod        
    def hull_constraint(h_bool, x, eq, M=20., mp=None):
        val1 = M*(h_bool-1)+eq['b']
        val2 = eq['A'].dot((x[:2]).T)
        return mp.add_leq_constraints(val1-val2,np.zeros(val1.shape), linear=True) if mp is not None else val1<=val2
    
    @staticmethod
    def angle_constraint(h_bool, x, min_angle, max_angle, M=360, mp=None):
        val1 = M*(h_bool-1)+min_angle
        val2 = x[2]
        val3 = M*(1-h_bool)+max_angle
        return mp.add_leq_constraints([val1-val2, -val3+val2],[0, 0], linear=True) if mp is not None else val1<=val2<=val3
    
    @staticmethod
    def angular_velocity_constraint(h_bool, x, min_angle, max_angle, M=360, mp=None):
        val1 = M*(h_bool-1)+min_angle-max_angle
        val2 = x[5]
        val3 = M*(1-h_bool)+max_angle-min_angle
        return mp.add_leq_constraints([val1-val2, -val3+val2],[0, 0], linear=True) if mp is not None else val1<=val2<=val3

    def add_integer_constraints(self, in_hull, mp):
        N, H = in_hull.shape       
        for t in range(N-1):
            for i in range(H):  
                mp.AddLinearConstraint(in_hull[t][i]<=1-np.sum(in_hull[t+1][i-1:i]))            
                mp.AddLinearConstraint(in_hull[t][i]<=np.sum(in_hull[t+1][i:i+2])) 

    def add_position_collision_constraints(self, S, in_hull, opt_hull, mp, M=20.):
        for t in range(S.shape[1]-1):
            for i,hull in enumerate(self.hull_path):
                for dt in range(2):
                    if opt_hull or in_hull[t][i]:
                        self.hull_constraint(
                                             in_hull[t][i],      \
                                             S[0,t+dt],          \
                                             hull["polygon_eq"], \
                                             mp=mp               \
                                            )

    def add_angle_collision_constraints(self, S, in_hull, mp, M=20., opt_hull=True):
        for t in range(S.shape[0]-1):
            for i,hull in enumerate(self.hull_path):  
                for dt in range(2):
                    if opt_hull or in_hull[t][i]:
                        cls.angle_constraint(
                                             in_hull[t][i],      \
                                             S[0,t+dt],          \
                                             hull["min_angle"],  \
                                             hull["max_angle"],  \
                                             mp=mp               \
                                            )       

                        cls.angular_velocity_constraint(                                         
                                             in_hull[t][i],      \
                                             S[0,t+dt],          \
                                             hull["min_angle"],  \
                                             hull["max_angle"],  \
                                             mp=mp               \
                                            )
                                       
    @staticmethod
    def boat_dynamics(s, u, am):
        derivs = np.zeros_like(s)
        derivs[0] = s[3]
        derivs[1] = s[4]
        derivs[2] = s[5] + .375*u[2] + .125*u[3] + 360*(am[0]-am[1])
        derivs[3] = u[0]
        derivs[4] = u[1]
        derivs[5] = .5*u[2] + .5*u[3]
        return derivs

    @classmethod
    def add_transition_constraints(cls, S, U, angle_mod, mp, S_fix_inds=None):
        if S_fix_inds is None:
            S_fix_inds = []
            
        inds = [x for x in cls.S_transition_inds if x not in S_fix_inds]
        for k in range(1,S.shape[1]):        
            u0 = U[0,k-1] #old input
            s0 = S[0,k-1] #old state
            s  = S[0,k]   #new state
            am = angle_mod[k-1]

            #State transition constraint
            mp.add_equal_constraints(s[inds], s0[inds] + cls.boat_dynamics(s0, u0, am)[inds],linear=cls.linear)
                                       
    @staticmethod
    def add_input_position_cost(S, U, mp):
        return mp.AddQuadraticCost(100*np.sum(U[0,:,:2]**2))

    @staticmethod
    def add_input_angle_cost(S, U, mp):
        return mp.AddQuadraticCost(np.sum((U[0,:,2:]/10)**2))


class ShapeBoat_spline(ShapeBoat, object):
    
    M = 0.5 * np.array([[1, 1, 0],[-2, 2, 0],[1, -2, 1]])
    p_cost_matrix = M.T.dot(np.array([[0, 0, 2]]).T.dot(np.array([[0, 0, 2]])).dot(M))
    max_U = np.array([0.01, 0.01, 10])
    U_rate = 5
    
    S_transition_inds = [2,5]
        
    @classmethod
    def B(cls, u_t):
        return np.array([1, u_t, u_t**2]).dot(cls.M)
    
    @classmethod    
    def dB_dt(cls, u_t, dT):
        return np.array([0, 1, 2*u_t]).dot(cls.M)/dT
    
    @classmethod    
    def d2B_dt2(cls, u_t, dT):
        return np.array([0, 0, 2]).dot(cls.M)/dT**2
    
    @classmethod
    def knots_to_trajectory(cls, S, U, dT_target=1, order=3, dT_round=True, U_rate=None):
        print "trajs", S.shape, U.shape
        S_sample = np.zeros((S.shape[0],S.shape[1]+(order-1)+(order-2),S.shape[2]))
        U_sample = np.zeros((U.shape[0],U.shape[1]+2,U.shape[2]))
        S_sample[:,order-1:-order] = S[:,1:-1]
        S_sample[:,:order-1] = S[:,0,:]
        S_sample[:,-order:] = S[:,-1,:]
        U_sample[:,:-2] = U
        
        if U_rate is None:
            U_rate = cls.U_rate

        dN = dT_target*U_rate
        if not dT_round:
            assert dN.is_integer() 
        
        dN = int(math.ceil(dN))
        dT = float(dN)/U_rate
        
        print "Time Scaling target: %f, result: %f" % (dT_target, dT)
        
        shape = S_sample.shape   
        N = dN*(shape[1]-3)+1
        S_new = np.zeros((1,N,shape[2]))
        U_new = np.zeros((1,N,3))

        for x in range(N): 
            knot_ind = int(x/dN)
            knot_fraction = x/float(dN)-knot_ind
            p = S_sample[0,knot_ind:knot_ind+3,:2]

            S_new[0,x,:2]  = cls.B(knot_fraction).dot(p)
            S_new[0,x,3:5] = cls.dB_dt(knot_fraction, dT).dot(p)
            U_new[0,x,:2]  = cls.d2B_dt2(knot_fraction, dT).dot(p) 
            U_new[0,x,2]   = U_sample[0,knot_ind,2]/dT**2 if knot_fraction<0.5 else U_sample[0,knot_ind,3]/dT**2
            
            if x>0:
                S_new[0,x,5] = S_new[0,x-1,5] + U_new[0,x-1,2]/U_rate
                S_new[0,x,2] = S_new[0,x-1,2] + S_new[0,x-1,5]/U_rate + 0.5*U_new[0,x-1,2]/U_rate**2        
            
        return S_new, U_new
    
    

    @classmethod
    def max_trajectory_U(cls, S, U):
        S_new, U_new = cls.knots_to_trajectory(S, U, 1)
        return np.max(U_new[0], axis=0)

    @classmethod
    def knots_to_feasible_trajectory(cls, S, U):        
        target_dT = max(cls.max_trajectory_U(S, U)/cls.max_U)**0.5        
        return  cls.knots_to_trajectory(S, U, target_dT)
    
    @staticmethod        
    def hull_edge_constraint(h_bool0, h_bool1, a, x0, x2, eq, M=20., mp=None, linear=True):
        x_avg = a*x0 + (1-a)*x2
        val1 = M*(h_bool0-h_bool1-1)+eq['b']
        val2 = eq['A'].dot((x_avg[:2]).T)
        return mp.add_leq_constraints(val1-val2,np.zeros(val1.shape), linear=linear) if mp is not None else val1<=val2

    @staticmethod        
    def hull_edge_constraint2(h_bool0, h_bool1, a, x0, x1, x2, eq, M=20., mp=None, linear=True):
        x_avg = a*(x0+x1)/2+(1-a)*(x1+x2)/2
        val1 = M*(h_bool0-h_bool1-1)+eq['b']
        val2 = eq['A'].dot((x_avg[:2]).T)
        return mp.add_leq_constraints(val1-val2,np.zeros(val1.shape), linear=linear) if mp is not None else val1<=val2    
    
    def add_position_collision_constraints(self, S, in_hull, opt_hull, mp, M=20.):
        super(ShapeBoat_spline, self).add_position_collision_constraints(S, in_hull, opt_hull, mp, M)
        N, H = in_hull.shape

        if opt_hull:
            a = 0.5*np.ones(N-1)
        else:
            a = mp.NewContinuousVariables(N-1)
            mp.add_leq_constraints(a, np.ones(N-1))
            mp.add_leq_constraints(-a, np.zeros(N-1))
            
        for i,hull in enumerate(self.hull_path):
                
            for t in range(1,N):
                t_e = t-1
                
                if opt_hull or (in_hull[t][i]+in_hull[t-1][i]==1):
                    self.hull_edge_constraint2(in_hull[t][i],      \
                                              in_hull[t-1][i],    \
                                              a[t_e],             \
                                              S[0,t-1,:2],        \
                                              S[0,t,:2],        \
                                              S[0,t+1,:2],        \
                                              hull["polygon_eq"], \
                                              mp=mp,              \
                                              linear=opt_hull     \
                                             )

                    self.hull_edge_constraint2(in_hull[t-1][i],    \
                                              in_hull[t][i],      \
                                              a[t_e],             \
                                              S[0,t-1,:2],        \
                                              S[0,t,:2],        \
                                              S[0,t+1,:2],        \
                                              hull["polygon_eq"], \
                                              mp=mp,              \
                                              linear=opt_hull     \
                                             )

    @classmethod    
    def add_input_position_cost(cls, S, U, mp):
        T = S.shape[1]
        for k in range(T):
            P = np.zeros((3,2),S.dtype)
            if k==0:
                P[0] = P[1] = S[0,0,:2]
                P[2] = S[0,k+1,:2]
            elif k==T-1:
                P[0] = S[0,k-1,:2]
                P[1] = P[2] = S[0,T-1,:2]
            else:
                P=S[0,k-1:k+2,:2]
            mp.AddQuadraticCost(100*np.sum(np.multiply(cls.p_cost_matrix.dot(P),P)))

    @staticmethod              
    def boat_dynamics(s, u, am):
        derivs = np.zeros_like(s)
        derivs[2] = s[5] + .375*u[2] + .125*u[3] + 360*(am[0]-am[1])
        derivs[5] = .5*u[2] + .5*u[3]
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
    print boat.g.vertex_properties['polygon'][boat.g.vertex(hull_i)].contains(Point(S[0,state_i][:2]))     