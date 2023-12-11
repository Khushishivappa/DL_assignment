import numpy as np
import matplotlib.pyplot as plt


def normalize_windows(win_data):
    
    norm_data = []
    for w in win_data:
        norm_win = [((float(p) / float(w[0])) - 1) for p in w]
        norm_data.append(norm_win)
    return norm_data


def load_data(filename, seq_len, norm_win):
    
    fid = open(filename, 'r').read()
    data = fid.split('\n')
    sequence_length = seq_len + 1
    out = []
    for i in range(len(data) - sequence_length):
        out.append(data[i: i + sequence_length])
    if norm_win:
        out = normalize_windows(out)
    out = np.array(out)
    split_ratio = 0.9
    split = round(split_ratio * out.shape[0])
    train = out[:int(split), :]
    np.random.shuffle(train)
    X_tr = train[:, :-1]
    Y_tr = train[:, -1]
    X_te = out[int(split):, :-1]
    Y_te = out[int(split):, -1]
    X_tr = np.reshape(X_tr, (X_tr.shape[0], X_tr.shape[1], 1))
    X_te = np.reshape(X_te, (X_te.shape[0], X_te.shape[1], 1))
    return [X_tr, Y_tr, X_te, Y_te]


def predict_seq_mul(model, data, win_size, pred_len):

    pred_seq = []
    for i in range(len(data)//pred_len):
        current = data[i * pred_len]
        predicted = []
        for j in range(pred_len):
            predicted.append(model.predict(current[None, :, :])[0, 0])
            current = current[1:]
            current = np.insert(current, [win_size - 1], predicted[-1], axis=0)
        pred_seq.append(predicted)
    return pred_seq


def predict_pt_pt(model, data):

    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size, ))
    return predicted


def plot_mul(Y_hat, Y, pred_len):

    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(Y, label='Y')
    # Print the predictions in its respective series-length
    for i, j in enumerate(Y_hat):
        shift = [None for p in range(i * pred_len)]
        plt.plot(shift + j, label='Y_hat')
        plt.legend()
    plt.show()
