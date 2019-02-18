


def misclassified(true, pred):
    misclassifiction = round(1.0 * (true != pred).sum().item()/pred.size()[0] * 100, 3)
    string = 'Misclassification Rate: ' + str(misclassifiction) + '%'
    return string
