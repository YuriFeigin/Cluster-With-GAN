import abc


class Common(object):
    def parse_inputs(self, inputs, label_ind=1):
        if type(inputs) is list or tuple and len(inputs) == (label_ind + 1):
            assert self.n_classes, 'number of class = 0, while there is labels in the input'
            x = inputs[:-1]
            labels = inputs[-1]
        else:
            assert self.n_classes is None, 'no labels given but number of class > 0'
            x = inputs
            labels = None
        return x, labels