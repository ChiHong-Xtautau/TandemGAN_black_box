import sys
import torch
import os


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.sys_stream = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.log.write(message)
        self.sys_stream.write(message)

    def flush(self):
        pass


def save_results(results, filename):
    filename = filename + ".py"
    dirname = 'res'
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    f = open(os.path.join(dirname, filename), 'a')
    f.writelines(results + '\n')
    f.close()


def accuracy(net, net_name, test_loader, cuda=False, idx=0):
    device = torch.device("cuda:0" if cuda else "cpu")

    net.eval()

    total = 0.0
    correct = torch.tensor(0.0).to(device)
    for i, data in enumerate(test_loader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            predicted = predicted.to(device)
            correct += (predicted == labels).sum()

    if net_name == 'netB':
        print('Accuracy of netB_%d: %.2f %%' % (idx, 100. * float(correct) / total))
    else:
        print('Accuracy of %s: %.2f %%' % (net_name, 100. * float(correct) / total))

    return 100. * float(correct) / total

