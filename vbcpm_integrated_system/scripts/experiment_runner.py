"""
running experiment using subprocess
"""
import time
import subprocess
import os
def main():
    num_objs = [11, 8, 5]
    difficulties = [3, 2, 1]
    probs = [1,2,3,4,   5,6,7,8,9,10]
    for num_obj in num_objs:
        for difficulty in difficulties:
            for prob in probs:
                trials = 5
                max_trial = 6

                for trial in range(max_trial-trials,max_trial):
                    print(' number of objects: ', num_obj, ', difficulty level: ', difficulty, 
                          ', prob: ', prob, ', trial: ', trial, '...')
                    # if the result file already exists, skip
                    if os.path.exists('prob-%d-%d-%d-trial-%d-result.pkl' % (num_obj, difficulty, prob, trial)):
                        print('file prob-%d-%d-%d-trial-%d-result.pkl exists. Skipping...' % (num_obj, difficulty, prob, trial))
                        continue

                    # run execution scene
                    sp1 = subprocess.Popen(["python", "execution_system.py", "1", 
                                    "prob-%d-%d-%d"%(num_obj, difficulty, prob),
                                    "y"], stdout=subprocess.DEVNULL)
                    time.sleep(5)  # wait for some time for the scene to set up
                    sp2 = subprocess.Popen(["python", "task_planner.py", 
                                            "prob-%d-%d-%d"%(num_obj, difficulty, prob),
                                            str(trial)])#, stdout=subprocess.DEVNULL)
                    sp2.wait()
                    # after task planning is done, terminate execution scene
                    poll = sp1.poll()
                    if poll is None:
                        # process is still alive. kill it
                        sp1.kill()
                    time.sleep(5)  # before next batch, wait a bit

if __name__ == "__main__":
    main()