"""
running experiment using subprocess
random algorithm
"""
import time
import subprocess
import os
def main():
    num_objs = [11, 8, 5]
    difficulties = [3, 2, 1]
    probs = [1,2,3,4,5]#,6,7,8,9,10]
    algo_type = 1  # 0: random   1: greedy
    for num_obj in num_objs:
        for difficulty in difficulties:
            for prob in probs:
                trials = 5
                max_trial = 6
                if num_obj == 11:
                    timeout = 1000
                elif num_obj == 8:
                    timeout = 800
                else:
                    timeout = 400

                for trial in range(max_trial-trials,max_trial):
                    print(' number of objects: ', num_obj, ', difficulty level: ', difficulty, 
                          ', prob: ', prob, ', trial: ', trial, '...')
                    # if the result file already exists, skip
                    if algo_type == 0:
                        filename = 'random-prob-%d-%d-%d-trial-%d-result.pkl' % (num_obj, difficulty, prob, trial)
                    else:
                        filename = 'multistep-lookahead-prob-%d-%d-%d-trial-%d-result.pkl' % (num_obj, difficulty, prob, trial)

                    if os.path.exists(filename):
                        print('file ', filename, 'exists. skipping...')
                        continue

                    # run execution scene
                    sp1 = subprocess.Popen(["python", "execution_system.py", "1", 
                                    "prob-%d-%d-%d"%(num_obj, difficulty, prob),
                                    "y"], stdout=subprocess.DEVNULL)
                    time.sleep(5)  # wait for some time for the scene to set up
                    sp2 = subprocess.Popen(["python", "task_planner_random.py", 
                                            "prob-%d-%d-%d"%(num_obj, difficulty, prob),
                                            str(trial), str(algo_type), str(timeout), str(num_obj)])#, stdout=subprocess.DEVNULL)
                    sp2.wait()
                    # after task planning is done, terminate execution scene
                    poll = sp1.poll()
                    if poll is None:
                        # process is still alive. kill it
                        sp1.kill()
                    time.sleep(5)  # before next batch, wait a bit

if __name__ == "__main__":
    main()