# Efficient Zero Montecarlo Tree search
Efficient zero is a monte carlo tree search based model based method. It utilizes 

1.Self-Supervised Consistency Loss: Forces the predicted next state to be mathematically similar to the real observed next state (using a SimSiam-style loss).

2.End-to-End Value Prefix: Instead of predicting single-step rewards $r_t$, it predicts the cumulative discounted sum (value prefix) using an LSTM, smoothing out short-term noise.

3.Model-Based Off-Policy Correction: It uses the current learned model to re-imagine the immediate future of old data, correcting the value 
targets.


# Integeration requirements
I will be performing extensive set of ablations, hence you are required to perform the following integrations to the codebase. Primarily these concern regarding logging based integrations and bash run scripts.

## Wandb integration
You are to look extensively thorughout this codebase and:
1. integrate wandb and get rid of tensorboard based logging, the only logging mechanism should be wandb.
2. entity will be "stablegradients" by default. but can be changed by user argument.
3. Project name will be  "MonteCarloTreeSearch" but can be changedn by user argument.
4. Each run will be assigned a group.\
    4.1 Different that differ only in the seed will share the same group\
    4.2 The name of the group shall contain the environment\
    4.3 The default hyperparameters shall not be present in the group name,\
    4.4 The name of the parameter and its value would be present in the group name only if it is different from the default values, you can refer to the train.sh and core.py to get the default values.\
    4.5 The user should have the option to append into the group name. 
    4.6 The name of each run is just group name and the seed number mentioned 
    4.7 There are certain things that are curretly being logged and you should definitely log them.
    4.8. Beyond that, there are many metrics that are usefull for an ML scientist such as\
        4.8.1 total loss, weighted loss, value loss consistency loss, replay buffer size, episodes collected, lr, grad norm, policy (nominal i.e. policy head) entropy , target policy entropy(entropy of the policy suggested by MCTS), 
5. There are many subprocesses in the entire algorithm, add wandb logs to log how long each process takes as well. 


