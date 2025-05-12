import numpy as np
from sklearn.linear_model import LogisticRegression


def pu_bagging(tu, kp, iZemb):
    obs_votes = defaultdict(list)
    obs_experiences = set()
    iteration = 0
    while len(obs_experiences) != len(tu):
        ta = kp.random.choice(tu, size=len(tu), replace=True).tolist()
        tnb = list(set(ta) - set(tu))
        obs = list(set(tu) - set(ta))
        predictions = one_hotup(ta, tnb, iZemb)
        for ida, prob in predictions.items():
            obs_votes[ida].append(prob)
        if len(obs_experiences) == len(tu):
            print("All nodes have entered")
        obs_experiences = set()
        i = 0
        for ida, probs in obs_votes.items():
            binary_prediction = [1 if prob > 0.5 else 0 for prob in probs]
            avg_prob = np.mean(binary_prediction)
            if avg_prob > 0.5:
                obs_experiences.add(ida)
        iteration += 1
    return obs_experiences

def one_pu(X, tnb, vocab):
    l = []
    for i in range(len(X)):
        l.append(tnb.transform(X[i]).toarray().tolist())
    idtrain = []
    for i in range(len(vocab)):
        idtrain.append(i)
    X_train = [idtrain.index(i) for i in idtrain]
    y_train = [len(tnb) + i for i in len(tnb)]
    X_test = [idtrain.index(i) for i in vocab]
    y_test = [idtrain.index(i) for i in vocab]
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    return clf.predict(X_test), l

def calculate_initial_velocity(data_length, window_size, total_steps):
    total_displacement = data_length - window_size
    initial_velocity = (total_displacement - 0.5) / total_steps
    return initial_velocity

def generate_accelerated_windows(data, window_size, initial_velocity, total_steps):
    total_displacement = len(data) - window_size
    acceleration = (2 * (total_displacement - initial_velocity * total_steps)) / (total_steps ** 2)
    batches = []
    current_position = 0.0
    
    for t in range(total_steps):
        start_idx = int(initial_velocity * t + 0.5 * acceleration * t ** 2)
        if start_idx + window_size > len(data):
            start_idx = len(data) - window_size
        batches.append(data[start_idx : start_idx + window_size])
        current_position = initial_velocity * (t + 1) + 0.5 * acceleration * (t + 1) ** 2
        if current_position >= total_displacement:
            break
    
    return batches
