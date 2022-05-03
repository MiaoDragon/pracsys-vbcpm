"""
compute the average of time for each problem and difficulty level
"""
import time
import subprocess
import os
import pickle
import numpy as np
def main():
    num_objs = [11, 8, 5]
    difficulties = [3, 2, 1]
    probs = [1,2,3,4,5,6,7,8,9,10]
    total_time = {11: {3: [], 2: [], 1: []}, 8: {3: [], 2: [], 1: []}, 5: {3: [], 2: [], 1: []}}    
    total_time2 = {11: {3: {}, 2: {}, 1: {}}, 8: {3: {}, 2: {}, 1: {}}, 5: {3: {}, 2: {}, 1: {}}}    

    for num_obj in num_objs:
        for difficulty in difficulties:
            for prob in probs:
                trials = 5
                max_trial = 6

                for trial in range(max_trial-trials,max_trial):
                    # print(' number of objects: ', num_obj, ', difficulty level: ', difficulty, 
                    #       ', prob: ', prob, ', trial: ', trial, '...')
                    # if the result file already exists, skip
                    fname = 'prob-%d-%d-%d-trial-%d-result.pkl' % (num_obj, difficulty, prob, trial)
                    if os.path.exists(fname):
                        f = open(fname, 'rb')
                        data = pickle.load(f)
                        running_time = data['running_time']
                        f.close()

                        if num_obj == 11 and difficulty == 3 and prob == 1 and trial == 2:
                            print(data)
                        if num_obj == 11 and difficulty == 3 and prob == 4 and trial == 2:
                            print(data)
                        if num_obj == 11 and difficulty == 1 and prob == 7 and trial == 2:
                            print(data)

                        if data['num_reconstructed_objs'] != num_obj:
                            print('num_reconstructed_objs does not match')
                            print('num_obj = %d, difficulty = %d, prob = %d, trial = %d' % \
                                (num_obj, difficulty, prob, trial))
                            print(data)

                        total_time[num_obj][difficulty].append(running_time)
                        if prob not in total_time2[num_obj][difficulty]:
                            total_time2[num_obj][difficulty][prob] = []
                        total_time2[num_obj][difficulty][prob].append(running_time)
    print('num_objs = 11, difficulty = 3:')
    print(total_time[11][3])
    print(np.mean(total_time[11][3]))
    print(total_time2[11][3])
    print((np.array(total_time[11][3])<1000).sum() / len(total_time[11][3]))
    print('num_objs = 11, difficulty = 2:')
    print(total_time[11][2])
    print(np.mean(total_time[11][2]))
    print((np.array(total_time[11][2])<1000).sum() / len(total_time[11][2]))

    print('num_objs = 11, difficulty = 1:')
    print(total_time[11][1])
    print(np.mean(total_time[11][1]))
    print(total_time2[11][1])
    print((np.array(total_time[11][1])<1000).sum() / len(total_time[11][1]))

    print('num_objs = 8, difficulty = 3:')
    print(total_time[8][3])
    print(np.mean(total_time[8][3]))
    print(np.mean(total_time[8][3]))
    print((np.array(total_time[8][3])<800).sum() / len(total_time[8][3]))

    print('num_objs = 8, difficulty = 2:')
    print(total_time[8][2])
    print(np.mean(total_time[8][2]))
    print((np.array(total_time[8][2])<800).sum() / len(total_time[8][2]))

    print('num_objs = 8, difficulty = 1:')
    print(total_time[8][1])
    print(np.mean(total_time[8][1]))
    print((np.array(total_time[8][1])<800).sum() / len(total_time[8][1]))

    print('num_objs = 5, difficulty = 3:')
    print(total_time[5][3])
    print(np.mean(total_time[5][3]))
    print((np.array(total_time[5][3])<400).sum() / len(total_time[5][3]))

    print('num_objs = 5, difficulty = 2:')
    print(total_time[5][2])
    print(np.mean(total_time[5][2]))
    print((np.array(total_time[5][2])<400).sum() / len(total_time[5][2]))

    print('num_objs = 5, difficulty = 1:')
    print(total_time[5][1])
    print(np.mean(total_time[5][1]))
    print((np.array(total_time[5][1])<400).sum() / len(total_time[5][1]))


if __name__ == "__main__":
    main()