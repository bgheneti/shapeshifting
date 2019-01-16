from shape_boat import ShapeBoat_spline
from pydrake.all import SolutionResult
from multiboat_trajectory_optimization.trajectory_planner import BoatConfigurationPlanning
import numpy as np
import pickle

def shapeshift_trajectory(boat_shape, obstacle_shape, xy0, xyN, N=15, margin=0.0, boat_type=ShapeBoat_spline, opt_angle=True):
    result2 = None
    S_knots = None
    U_knots = None
    
    x0 = np.zeros((1, boat_type.num_states))
    xN = np.zeros((1, boat_type.num_states))
    x0[0, :3] = xy0
    xN[0, :3] = xyN
    boat = boat_type(boat_shape, obstacle_shape, margin=margin)
    boat.set_end_points(x0, xN)
    
    planner = BoatConfigurationPlanning(boat)
    S, U, in_hull, on_edge, mp, result1, solve_time = planner.compute_spline_trajectory(x0, xN, opt_angle=False, N=N)
    success = result1==0
        
    if opt_angle and success:
        S, U, in_hull, on_edge, mp, result2, solve_time = planner.compute_spline_trajectory(x0, xN, opt_position=False, N=N, in_hull=in_hull, on_edge=on_edge, S_initialization=S, S_fix_inds=[0,1])
        success = result2==0

    if boat_type is ShapeBoat_spline:
        S_knots = S
        U_knots = U
        S, U = boat_type.knots_to_trajectory(S_knots, U_knots, 40) if success else (None,None)   
                   
    return {'boat':          boat, 
            'S_knots':       S_knots,
            'U_knots':       U_knots,
            'S' :            S,
            'U':             U, 
            'in_hull':       in_hull,
            'on_edge':       on_edge,
            'result_spline': result1,
            'result_angle':  result2,
            'success':       success,
            'metrics':       metrics(U) if success else None
            }

def metrics(U):
    return analytics({'U_position_cost': U_position_cost(U), 'U_angle_cost': U_angle_cost(U)})

def analytics(costs):
    analytic_ops = [('_sum',np.sum),('_std',np.std),('_avg',np.average), ('',lambda x: x)]
    return {k+fk:fv(v) for k,v in costs.items() for fk,fv in analytic_ops}
     
def U_position_cost(U):
    return np.sum(U[0,:,:2]**2,axis=1)

def U_angle_cost(U):
    return U[0,:,2]**2

def write_experiment(results, label):
    with open('results/MIQP_'+label+'.pickle', 'wb') as f:
        pickle.dump(results, f)