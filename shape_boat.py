import numpy as np

from convex_hulls_graph import *
from shape import *
from multiboat_trajectory_optimization.boat_models import *
from multiboat_trajectory_optimization.trajectory_planner import *
from shapely.geometry import Point
import numpy as np

class ShapeBoat(ThreeInputBoat, object):
    linear=True
    num_inputs=4
    
    def __init__(self, boat_shape, obstacle_shape, margin=0):
        self.shape = boat_shape
        self.obstacle_shape = obstacle_shape
        self.hulls     = None
        self.msums     = None
        self.g         = None
        self.path      = None
        self.hull_path = None
    
    def toProblemStates(self, S):
        assert len(S)==1
        return super(ShapeBoat, self).toProblemStates(S)
    
    def toGlobalStates(self, S, state_initial=None):
        assert len(S)==1
        return super(ShapeBoat, self).toGlobalStates(S, state_initial)

    def set_end_points(self, x0, xN):
        self.msums, self.hulls = self.obstacle_shape.c_space_rotate(self.shape, x0[0], xN[0])
        self.g = HullGraph(self.hulls)        
        
        #self.g.draw_graph()
        
        plot_hulls(self.hulls[(0,0)])
        
        self.path  = self.g.point_path(Point(*x0[0,:2]), Point(*xN[0,:2]), x0[0, 2], xN[0,2])
        
        self.hull_path = [{ "polygon":    self.g.vertex_properties["polygon"][i],    \
                            "polygon_eq": self.g.vertex_properties["polygon_eq"][i], \
                            "min_angle":  self.g.vertex_properties["min_angle"][i],  \
                            "max_angle":  self.g.vertex_properties["max_angle"][i]} for i in self.path]

    @staticmethod        
    def hull_constraint(h_bool, x, eq, M=100., mp=None):
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

    @staticmethod
    def add_integer_constraints(in_hull, on_edge, angle_mod, mp):
        N, H = in_hull.shape  

        mp.AddLinearConstraint(in_hull[0][0]==1)
        mp.AddLinearConstraint(in_hull[-1][-1]==1)   
        
        for t in range(N-1):
            for i in range(H):  
                    if i>1:
                        mp.AddLinearConstraint(2*(1-in_hull[t][i])>=in_hull[t+1][i-1])
                    if i<H-1:
                        mp.AddLinearConstraint((in_hull[t][i]-1)+1<=in_hull[t+1][i]+in_hull[t+1][i+1])
                    
            #reduce searchspace if traversing hulls sequentially
            mp.AddLinearConstraint(in_hull[t][H-1]<=in_hull[t+1][H-1])
        for t in range(1,N):            
            for i in range(H):
                mp.AddLinearConstraint(in_hull[t-1][i]<=in_hull[t][i]+on_edge[t-1])

    def add_position_collision_constraints(self, S, in_hull, on_edge, mp, M=20.):
        for t in range(S.shape[1]-1):
            for i,hull in enumerate(self.hull_path):
                for dt in range(2):
                    self.hull_constraint(
                                         in_hull[t][i],      \
                                         S[0,t+dt],          \
                                         hull["polygon_eq"], \
                                         mp=mp               \
                                        )

    def add_angle_collision_constraints(self, S, in_hull, mp, M=20.):
        for t in range(S.shape[0]-1):
            for i,hull in enumerate(self.hull_path):  
                for dt in range(2):
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
    def add_transition_constraints(cls, S, U, angle_mod, mp):
        for k in range(1,S.shape[1]):        
            u0 = U[0,k-1] #old input
            s0 = S[0,k-1] #old state
            s  = S[0,k]   #new state
            am = angle_mod[k-1]

            #State transition constraint
            mp.add_equal_constraints(s, s0 + cls.boat_dynamics(s0, u0, am),linear=cls.linear)
                                       
    @staticmethod
    def add_input_position_cost(S, U, mp):
        return mp.AddQuadraticCost(np.sum(U[0,:,:2]**2))

    @staticmethod
    def add_input_angle_cost(S, U, mp):
        return mp.AddQuadraticCost(np.sum((U[0,:,2:]/100)**2))


class ShapeBoat_spline(ShapeBoat, object):
    
    M = 0.5 * np.array([[1, 1, 0],[-2, 2, 0],[1, -2, 1]])
    p_cost_matrix = M.T.dot(np.array([[0, 0, 2]]).T.dot(np.array([[0, 0, 2]])).dot(M))
        
    @classmethod
    def B(cls, u_t):
        return np.array([1, u_t, u_t**2]).dot(cls.M)
    
    @classmethod    
    def dB_dt(cls, u_t, dN):
        return np.array([0, 1, 2*u_t]).dot(cls.M)/dN
    
    @classmethod    
    def d2B_dt2(cls, u_t, dN):
        return np.array([0, 0, 2]).dot(cls.M)/dN**2
    
    @classmethod
    def knots_to_trajectory(cls, S, U, dN, order=3):
        S_sample = np.zeros((S.shape[0],S.shape[1]+(order-1)+(order-2),S.shape[2]))
        U_sample = np.zeros((U.shape[0],U.shape[1]+2,U.shape[2]))
        S_sample[:,order-1:-order] = S[:,1:-1]
        S_sample[:,:order-1] = S[:,0,:]
        S_sample[:,-order:] = S[:,-1,:]
        U_sample[:,:-2] = U

        shape = S_sample.shape   
        N = dN*(shape[1]-3)+1
        S_new = np.zeros((1,N,shape[2]))
        U_new = np.zeros((1,N,3))

        for x in range(N): 
            knot_ind = int(x/dN)
            knot_fraction = x/float(dN)-knot_ind
            p = S_sample[0,knot_ind:knot_ind+3,:2]

            S_new[0,x,:2]  = cls.B(knot_fraction).dot(p)
            S_new[0,x,3:5] = cls.dB_dt(knot_fraction, dN).dot(p)
            U_new[0,x,:2]  = cls.d2B_dt2(knot_fraction, dN).dot(p)           
            U_new[0,x,2]   = U_sample[0,knot_ind,2] if knot_fraction<0.5 else U_sample[0,knot_ind,3]

        return S_new, U_new
    
    @staticmethod                                           
    def edge_line_constraint(e_bool, x0, x1, x2, M=100., mp=None):
        const = M*(e_bool-np.ones(2))
        x_dif = x1+(x1-x0)-x2
                                       
        val1 = np.concatenate((const, x_dif))
        val2 = np.concatenate((x_dif, -const))
        return mp.add_leq_constraints(val1-val2, np.zeros(val1.shape), linear=True) if mp is not None else val1==val2  

    def add_position_collision_constraints(self, S, in_hull, on_edge, mp, M=20.):
        super(ShapeBoat_spline, self).add_position_collision_constraints(S, in_hull, on_edge, mp, M)
                 
        for t in range(1,S.shape[1]-1):
            t_e = t-1
            self.edge_line_constraint(on_edge[t_e],    \
                                      S[0,t-1,:2],     \
                                      S[0,t,:2],       \
                                      S[0,t+1,:2],     \
                                      mp=mp            \
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
            mp.AddQuadraticCost(np.sum(np.multiply(cls.p_cost_matrix.dot(P),P)))

    @staticmethod              
    def boat_dynamics(s, u, am):
        derivs = np.zeros_like(s)
        derivs[2] = s[5] + .375*u[2] + .125*u[3] + 360*(am[0]-am[1])
        derivs[5] = .5*u[2] + .5*u[3]
        return derivs
    
    @classmethod                                                    
    def add_transition_constraints(cls, S, U, angle_mod, mp):
        for k in range(1,S.shape[1]):        
            u0 = U[0,k-1] #old input
            s0 = S[0,k-1] #old state
            s  = S[0,k]   #new state
            am = angle_mod[k-1]

            #State transition constraint
            mp.add_equal_constraints(s[[2,5]], s0[[2,5]] + cls.boat_dynamics(s0, u0, am)[[2,5]],linear=cls.linear)

            
            
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