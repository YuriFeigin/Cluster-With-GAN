import numpy as np
import matplotlib.pyplot as plt

def ACC(y_true,y_pred):
    Y_pred = y_pred
    Y = y_true
    from sklearn.utils.linear_assignment_ import linear_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size,ind


def CreateImageFromPlot(data):
    fig = plt.figure(figsize=[3.2,2.4])
    ax = fig.gca()
    ax.plot(data)
    ax.axis('tight')
    # plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    fig.canvas.draw()
    plt.show(block=False)
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    w, h = fig.canvas.get_width_height()
    data = data.reshape((h, w, 3))
    plt.close()
    return data