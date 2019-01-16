from shape_boat import ShapeBoat_spline, ShapeBoat
from pydrake.all import SolutionResult
from multiboat_trajectory_optimization.trajectory_planner import BoatConfigurationPlanning
import numpy as np
import pickle

def shapeshift_trajectory(boat_shape, obstacle_shape, xy0, xyN, N=15, boat_type_init=None, boat_type=ShapeBoat_spline, plot=True):
    result_init = None
    boat_init   = None
    S_init      = None
    U_init      = None
    S_knots     = None
    U_knots     = None
    success     = True
    in_hull     = None
    
    x0 = np.zeros((1, boat_type.num_states))
    xN = np.zeros((1, boat_type.num_states))
    x0[0, :3] = xy0
    xN[0, :3] = xyN
    
    if boat_type_init is not None:
        boat_init = boat_type_init(boat_shape, obstacle_shape)
        planner = BoatConfigurationPlanning(boat_init)
        boat_init.set_end_points(x0, xN)
        
        S_init, U_init, in_hull, mp, result_init, solve_time_init = planner.compute_spline_trajectory(x0, xN, N=N)
        success = result_init==0
    
    if success:
        boat = boat_type(boat_shape, obstacle_shape)
        planner = BoatConfigurationPlanning(boat)
        boat.set_end_points(x0, xN)
        
        S, U, in_hull, mp, result, solve_time = planner.compute_spline_trajectory(x0, xN, N=N, in_hull=in_hull, S_initialization=S_init)
        success = result==0

    if success:
        if boat_type is ShapeBoat_spline:
            S_knots = S
            U_knots = U
            S, U = boat_type.knots_to_trajectory(S_knots, U_knots, 40) if success else (None,None)
        if plot:  
            boat.plot_hulls(S, all_hulls=True, text=False)
            boat.plot_hulls(S, in_hull)
        
    return {'boat_init':     boat_init,
            'boat':          boat,
            'S_knots':       S_knots,
            'U_knots':       U_knots,
            'S_init':        S_init,
            'S' :            S,
            'U':             U, 
            'in_hull':       in_hull,
            'result_init':   result_init,
            'result':        result,
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