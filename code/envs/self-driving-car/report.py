import pickle
import matplotlib.pyplot as plt

with open('shielded.pkl', 'rb') as f:
    shielded_score = pickle.load(f)
plt.plot(shielded_score)

with open('not_shielded.pkl', 'rb') as f:
    not_shielded_score = pickle.load(f)
plt.plot(not_shielded_score)



plt.show()