from hyperopt import fmin, tpe, hp, space_eval, Trials
from updated_pr import objective

# define an objective function
# def objective(args):
#     case, val = args
#     if case == 'case 1':
#         return val
#     else:
#         return val ** 2
#LR, MS, BS, WE
# define a search space
target = open('log2.txt','w')
space = hp.choice('a',
    [
        (0.01, 10000, 50, 6, target),
        (0.01, 100000, 50, 8, target),
        (0.01, 1000000, 50, 6, target),
        (0.01, 10000, 50, 4, target),
        (0.01, 100000, 50, 8, target),
        (0.01, 1000000, 50, 4, target),
        (1, 10000, 250, 4, target),
        (.5, 10000, 250, 8, target),
        (1, 10000, 250, 6, target)
    ])
trials = Trials()
# minimize the objective over the space
best = fmin(objective, space, algo=tpe.suggest, max_evals=12, trials=trials)
print(trials.trials)
print(best)
# -> {'a': 1, 'c2': 0.01420615366247227}
print(space_eval(space, best))
# -> ('case 2', 0.01420615366247227}