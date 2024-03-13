import torch


def binary_accuracy(pred, target, threshold=0.5):
    assert pred.size() == target.size()
    pred = torch.where(pred < threshold, pred.new_tensor(0), pred.new_tensor(1))
    num_correct = (pred == target).sum()
    num_total = target.new_tensor(target.shape[0])
    return num_correct / num_total


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) / batch_size for k in topk]