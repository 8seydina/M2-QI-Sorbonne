# This code implements a neural network for forward and backward propagation "by hand".
# It allows you to manually initialize parameters, perform forward passes, compute loss and accuracy, 
# and update parameters using gradients calculated during backpropagation.


def init_params(nx, nh, ny):
    """
    nx, nh, ny: integers
    out params: dictionnary
    """
    params = {}

    #####################
    ## Your code here  ##
    #####################
    # fill values for Wh, Wy, bh, by
    #the value of the standard deviation
    std = 0.3

    params["Wh"] = torch.randn((nh, nx)) * std
    params["Wy"] = torch.randn((ny, nh)) * std
    params["bh"] = torch.zeros(nh)
    params["by"] = torch.zeros(ny)

    ####################
    ##      END        #
    ####################
    return params

def forward(params, X):
    """
    params: dictionnary
    X: (n_batch, dimension)
    """
    bsize = X.size(0)
    nh = params['Wh'].size(0)
    ny = params['Wy'].size(0)
    outputs = {}

    #####################
    ## Your code here  ##
    #####################
    # fill values for X, htilde, h, ytilde, yhat

    outputs["X"] = X
    outputs["htilde"] = (torch.mm(outputs["X"], params["Wh"].T) + params["bh"])
    outputs["h"] = (torch.tanh(outputs["htilde"]))
    outputs["ytilde"] = (torch.mm(outputs["h"], params["Wy"].T) + params["by"])
    outputs["yhat"] = (torch.exp(outputs["ytilde"]) / torch.sum(torch.exp(outputs["ytilde"]), dim = 1, keepdim = True))

    ####################
    ##      END        #
    ####################

    return outputs['yhat'], outputs

def loss_accuracy(Yhat, Y):

    #####################
    ## Your code here  ##
    #####################

    L = -torch.mean(torch.sum(Y * torch.log(Yhat), dim=1))

    _, indsYhat = torch.max(Yhat, 1)
    _, indsY = torch.max(Y, 1)

    acc = torch.mean((indsYhat == indsY).float())

    ####################
    ##      END        #
    ####################

    return L, acc

def backward(params, outputs, Y):
    bsize = Y.shape[0]
    grads = {}

    #####################
    ## Your code here  ##
    #####################
    # fill values for Wy, Wh, by, bh
    grads["ytilde"] = outputs['yhat'] - Y

    grads["Wy"] = torch.mm(grads["ytilde"].T, outputs["h"])

    #grads["mm"] = torch.mm(grads["ytilde"], params["Wy"])
    #grads["h"] = 1 - outputs["h"] ** 2

    grads["dhtilde"] = (torch.mm(grads["ytilde"], params["Wy"]) * (1 - outputs["h"] ** 2))

    grads["Wh"] = torch.mm(grads["dhtilde"].T, outputs["X"])

    grads["by"] = torch.sum(grads["ytilde"].T, dim = 1)

    grads["bh"] = torch.sum(grads["dhtilde"].T, dim = 1)

    ####################
    ##      END        #
    ####################
    return grads

def sgd(params, grads, eta):

    #####################
    ## Your code here  ##
    #####################
    # update the params values

    params["Wh"] -= eta * grads["Wh"]
    params["Wy"] -= eta * grads["Wy"]
    params["bh"] -= eta * grads["bh"]
    params["by"] -= eta * grads["by"]

    ####################
    ##      END        #
    ####################
    return params