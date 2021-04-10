import cvxpy as cvx
import numpy as np
import time
def compute_vel_limit(pos_traj, v0, vel_limit, acc_limit):
    vel_lls = []
    vel_uls = []
    vel_lls.append(np.zeros(v0.shape))
    vel_uls.append(np.zeros(v0.shape))
    # vel_limit = self.vel_limit
    # acc_limit = self.acc_limit
    for i in range(1, len(pos_traj)-1):
        # compare with neighbor values, determine the velocity limit
        vl = np.zeros(v0.shape)
        vu = np.zeros(v0.shape)
        vu[(pos_traj[i] > pos_traj[i-1]) & (pos_traj[i+1] > pos_traj[i])] = vel_limit[1]
        vl[(pos_traj[i] > pos_traj[i-1]) & (pos_traj[i+1] > pos_traj[i])] = 0.
        vu[(pos_traj[i] < pos_traj[i-1]) & (pos_traj[i+1] < pos_traj[i])] = 0.
        vl[(pos_traj[i] < pos_traj[i-1]) & (pos_traj[i+1] < pos_traj[i])] = vel_limit[0]
        vel_lls.append(vl)
        vel_uls.append(vu)
    vel_lls.append(np.zeros(v0.shape))
    vel_uls.append(np.zeros(v0.shape))
    
    vel_half_uls = []
    vel_half_lls = []
    for i in range(1, len(pos_traj)):
        vel_half_uls.append(np.maximum(vel_uls[i], vel_uls[i-1]))
        vel_half_lls.append(np.minimum(vel_lls[i], vel_lls[i-1]))
    vel_half_uls = np.array(vel_half_uls)
    vel_half_lls = np.array(vel_half_lls)
    mask = (vel_half_uls==0.) & (vel_half_lls==0.)
    vel_half_uls[mask] = vel_limit[1]
    vel_half_lls[mask] = vel_limit[0]
    return vel_lls, vel_uls, vel_half_lls, vel_half_uls

def compute_vel_time_qcqp(pos_traj, v0, vel_lls, vel_uls, vel_half_lls, vel_half_uls, vel_limit, acc_limit):
    # identify vel=0 points
    # vel_limit = self.vel_limit
    # acc_limit = self.acc_limit

    start_time = time.time()
    t_0s_val = np.zeros(len(pos_traj)-1) + .15
    t_1s_val = np.zeros(len(pos_traj)-1) + 0.
    t_2s_val = np.zeros(len(pos_traj)-1) + .15

    t_0s = cvx.Parameter(t_0s_val.shape, nonneg=True)
    t_1s = cvx.Parameter(t_1s_val.shape, nonneg=True)
    t_2s = cvx.Parameter(t_2s_val.shape, nonneg=True)

    print('after creating parameter...')
    # for i in range(len(t_0s_val)):
    #     print(t_0s_val[i])
    #     t_0s[i].value = t_0s_val[i]
    #     t_1s[i].value = t_1s_val[i]
    #     t_2s[i].value = t_2s_val[i]
    t_0s.value = t_0s_val
    t_1s.value = t_1s_val
    t_2s.value = t_2s_val
    print(t_0s[0].value)
    # ** construct the optimization problem
    vel_vars = [v0]
    vel_halfs = []
    consts = []  # constraints
    sum_vel_ul = 0.
    sum_vel_ll = 0.
    for i in range(1,len(pos_traj)):
        # if i == len(pos_traj)-1:
        #     vel_vars.append(np.zeros(len(v0)))
        # else:
        vel_vars.append(cvx.Variable(len(v0)))
        vel_halfs.append(cvx.Variable(len(v0)))
        first_seg = (vel_vars[i-1] + vel_halfs[i-1]) / 2 * t_0s[i-1]
        second_seg = vel_halfs[i-1] * t_1s[i-1]
        third_seg = (vel_vars[i] + vel_halfs[i-1]) / 2 * t_2s[i-1]
        consts.append(first_seg + second_seg + third_seg <= (pos_traj[i] - pos_traj[i-1] + 1e-5))
        consts.append(first_seg + second_seg + third_seg >= (pos_traj[i] - pos_traj[i-1] - 1e-5))

        consts.append((vel_halfs[i-1] - vel_vars[i-1]) <= acc_limit[1] * t_0s[i-1])
        consts.append((vel_halfs[i-1] - vel_vars[i-1]) >= acc_limit[0] * t_0s[i-1])
        consts.append((vel_vars[i] - vel_halfs[i-1]) <= acc_limit[1] * t_2s[i-1])
        consts.append((vel_vars[i] - vel_halfs[i-1]) >= acc_limit[0] * t_2s[i-1])
        sum_vel_ul += cvx.maximum(vel_vars[i] - vel_uls[i], np.zeros(v0.shape))
        sum_vel_ll += cvx.maximum(vel_lls[i] - vel_vars[i], np.zeros(v0.shape))
        

        sum_vel_ul += cvx.maximum(vel_halfs[i-1] - vel_half_uls[i-1], np.zeros(v0.shape))
        sum_vel_ll += cvx.maximum(vel_half_lls[i-1] - vel_halfs[i-1], np.zeros(v0.shape))

        #     obj = cvx.Minimize(sum_t)
    prob = cvx.Problem(cvx.Minimize(cvx.sum(sum_vel_ul + sum_vel_ll)), consts)

    # TODO: provide an initial guess for the variable
    for i in range(1, len(pos_traj)):
        #vel_vars[i].value = (vel_lls[i] + vel_uls[i]) / 2
        vel_vars[i].value = np.random.random(vel_uls[i].shape) * (vel_uls[i]-vel_lls[i]) + vel_lls[i]
        #vel_halfs[i-1].value = (vel_half_lls[i-1] + vel_half_uls[i-1]) / 2  
        vel_halfs[i-1].value = np.random.random(vel_half_lls[i-1].shape) * (vel_half_uls[i-1]-vel_half_lls[i-1]) + vel_half_lls[i-1]


    print('problem has been setup.')
    print('initialization takes %f time' % (time.time() - start_time))

    while True:
        # for i in range(len(t_0s_val)):
        #     t_0s[i].value = t_0s_val[i]
        #     t_1s[i].value = t_1s_val[i]
        #     t_2s[i].value = t_2s_val[i]

        t_0s.value = t_0s_val
        t_1s.value = t_1s_val
        t_2s.value = t_2s_val
        #result = prob.solve(solver=cvx.OSQP, warm_start=True)
        try:
            result = prob.solve(solver=cvx.OSQP, warm_start=True, verbose=True)
        except:
            result = None
        print(prob.status)
        if result is None or prob.status == 'infeasible':
            # increase all time
            # t_0s = t_0s * 2.
            # t_1s = t_1s * 2.
            # t_2s = t_2s * 2.
            i = np.argmin(t_0s_val + t_1s_val + t_2s_val)
            t_0s_val[i] = t_0s_val[i] + .5
            #t_1s[i] = t_1s[i] + .5
            t_2s_val[i] = t_2s_val[i] + .5
            
        else:
            # check if optimal value is small enough
            # print('opt value:')
            # print(result)
            max_violate_k = 0
            max_violate = -1.
            if result > np.pi / 180:
                # print('result not valid.')
                # print('ts')
                # print(t_0s_val)
                # print(t_1s_val)
                # print(t_2s_val)

                # check which constraint is violated
                for i in range(1, len(vel_vars)):
                    violate_i = 0.
                    inc = False
                    if i < len(vel_vars)-1:
                        mask = (vel_vars[i].value-vel_uls[i]>0) | (vel_vars[i].value-vel_lls[i]<0)
                        violate_i = np.abs(vel_vars[i].value[mask]).sum()
                    mask = (vel_halfs[i-1].value-vel_half_uls[i-1]>0) | (vel_halfs[i-1].value-vel_half_lls[i-1]<0)
                    violate_i += np.abs(vel_halfs[i-1].value[mask]).sum()
                    if violate_i > max_violate:
                        max_violate = violate_i
                        max_violate_k = i
                    # if i < len(vel_vars)-1 and ((vel_vars[i].value-vel_uls[i]>0) | (vel_vars[i].value-vel_lls[i]<0)).sum()>0:
                    #     # increase the time
                    #     inc = True
                    # if ((vel_halfs[i-1].value-vel_uls[i]>0) | (vel_halfs[i-1].value-vel_lls[i]<0)).sum()>0:
                    #     inc = True
                i = max_violate_k
                # t_0s[i-1] = t_0s[i-1] * 2
                # t_1s[i-1] = t_1s[i-1] * 2
                # t_2s[i-1] = t_2s[i-1] * 2
                # increase the smallest one
                t_0s_val[i-1] = t_0s_val[i-1] + 0.3
                t_1s_val[i-1] = t_1s_val[i-1] + 0.3
                #t_1s[i-1] = t_1s[i-1] + 0.5
                t_2s_val[i-1] = t_2s_val[i-1] + 0.3
                # min_t = np.min([t_0s[i-1], t_1s[i-1], t_2s[i-1]])
                # if t_0s[i-1] == min_t:
                #     t_0s[i-1] = t_0s[i-1] + 1.0
                # elif t_1s[i-1] == min_t:
                #     t_1s[i-1] = t_1s[i-1] + 1.0
                # elif t_2s[i-1] == min_t:
                #     t_2s[i-1] = t_2s[i-1] + 1.0
            else:
                print('result valid.')

                # print('ts')
                # print(t_0s)
                # print(t_1s)
                # print(t_2s)
                break
                        

    # The optimal objective value is returned by `prob.solve()`.
    print("status:", prob.status)
    print("optimal value", prob.value)

    # The optimal value for x is stored in `x.value`.
    for i in range(1, len(vel_vars)):
        # print('vel_uls: ')
        # print(vel_uls[i] * 180 / np.pi)
        # print('vel_lls:')
        # print(vel_lls[i] * 180 / np.pi)

        # print('vel_vars[i]: ')
        # print(vel_vars[i].value * 180 / np.pi)
        pass           
    # print('time: ')
    # print(t_0s_val)
    # print(t_1s_val)
    # print(t_2s_val)
    # given the solution, obtain the new break points and vel, time
    new_pos_traj = []
    vel_traj = []
    t_traj = []
    vel_traj.append(vel_vars[0])
    t_traj.append(0.)
    new_pos_traj.append(pos_traj[0])
    for i in range(len(pos_traj)-1):
        t0 = t_0s_val[i]
        t1 = t_1s_val[i]
        t2 = t_2s_val[i]
        if i == 0:
            v0 = vel_vars[i]
        else:
            v0 = vel_vars[i].value
        v_half = vel_halfs[i].value
        # if i == len(pos_traj)-2:
        #     v1 = vel_vars[i+1]
        # else:
        v1 = vel_vars[i+1].value
        dx = (v0+v_half)/2 * t_0s_val[i]
        new_pos_traj.append(new_pos_traj[-1] + dx)
        t_traj.append(t0+t_traj[-1])
        vel_traj.append(v_half)
        
        dx = v_half * t_1s_val[i]
        new_pos_traj.append(new_pos_traj[-1] + dx)
        t_traj.append(t1+t_traj[-1])
        vel_traj.append(v_half)
        
        dx = (v_half+v1)/2 * t_2s_val[i]
        new_pos_traj.append(new_pos_traj[-1] + dx)
        t_traj.append(t2+t_traj[-1])
        vel_traj.append(v1)
    pos_traj = np.array(new_pos_traj)
    vel_traj = np.array(vel_traj)
    t_traj = np.array(t_traj)
    return pos_traj, vel_traj, t_traj

