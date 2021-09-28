import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import argparse
import torch.nn as nn
import matplotlib.pyplot as plt
from models import MonotonicNN, IncConcaveNN, ConcaveNN

def f(x_1, x_2, x_3, mode='monotone'):
    if mode == 'monotone':
        out = .001*(x_1**3 + x_1) + x_2 ** 2 + torch.sin(x_3)
    elif mode == 'inc_concave':
        out = .001*(-torch.exp(-x_1) + x_1) + x_2 ** 2 + torch.sin(x_3)
    elif mode == 'concave':
        out = .001*(-x_1**2 + x_1) + x_2 ** 2 + torch.sin(x_3)
    else:
        raise Exception("Invalid mode!")
    return out

def create_dataset(n_samples, mode='monotone'):
    x = torch.randn(n_samples, 3)
    y = f(x[:, 0], x[:, 1], x[:, 2], mode)
    return x, y

class MLP(nn.Module):
    def __init__(self, in_d, hidden_layers):
        super(MLP, self).__init__()
        self.net = []
        hs = [in_d] + hidden_layers + [1]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()  # pop the last ReLU for the output layer
        self.net = nn.Sequential(*self.net)

    def forward(self, x, h):
        return self.net(torch.cat((x, h), 1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-mode", default="inc_concave", help="mode of experiment: [monotone, inc_concave, concave]")
    parser.add_argument("-nb_train", default=10000, type=int, help="Number of training samples")
    parser.add_argument("-nb_test", default=1000, type=int, help="Number of testing samples")
    parser.add_argument("-nb_epoch", default=200, type=int, help="Number of training epochs")
    parser.add_argument("-load", default=False, action="store_true", help="Load a model ?")
    parser.add_argument("-folder", default="", help="Folder")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if args.mode == 'monotone':
        model = MonotonicNN(3, [100, 100, 100], nb_steps=100, dev=device).to(device)
    elif args.mode == 'inc_concave':
        model = IncConcaveNN(3, [100, 100, 100], nb_steps=100, dev=device).to(device)
    elif args.mode == 'concave':
        model = ConcaveNN(3, [100, 100, 100], nb_steps=100, dev=device).to(device)
    else:
        raise Exception("Invalid mode!")
    
    model_mlp = MLP(3, [200, 200, 200]).to(device)
    optim_constraint = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5)
    optim_mlp = torch.optim.Adam(model_mlp.parameters(), 1e-3, weight_decay=1e-5)

    train_x, train_y = create_dataset(args.nb_train, args.mode)
    test_x, test_y = create_dataset(args.nb_test, args.mode)
    b_size = 100

    # args.nb_epoch = 0 # for debug
    # noise = 0.005 * torch.randn(*train_y.shape)
    for epoch in range(0, args.nb_epoch):
        # Shuffle
        idx = torch.randperm(args.nb_train)
        train_x = train_x[idx].to(device)
        train_y = (train_y[idx]).to(device)
        avg_loss_mon = 0.
        avg_loss_mlp = 0.
        for i in range(0, args.nb_train-b_size, b_size):
            # Monotonic
            x = train_x[i:i + b_size].requires_grad_()
            y = train_y[i:i + b_size].requires_grad_()
            y_pred = model(x[:, [0]], x[:, 1:])[:, 0]
            loss = ((y_pred - y)**2).sum()
            optim_constraint.zero_grad()
            loss.backward()
            optim_constraint.step()
            avg_loss_mon += loss.item()
            # MLP
            y_pred = model_mlp(x[:, [0]], x[:, 1:])[:, 0]
            loss = ((y_pred - y) ** 2).sum()
            optim_mlp.zero_grad()
            loss.backward()
            optim_mlp.step()
            avg_loss_mlp += loss.item()

        print(epoch)
        print("\tMLP: ", avg_loss_mlp/args.nb_train)
        print("\tConstraint: ", avg_loss_mon / args.nb_train)

    # <<TEST>>
    # x = train_x[:, 0]
    # h = torch.zeros(x.shape[0], 2).to(device)
    # y = f(x, h[:, 0], h[:, 1], args.mode).detach().cpu().numpy()
    # x = x.detach().cpu().numpy()
    # plt.scatter(x, y, marker='x', label="data", c='g')

    x = torch.arange(-5, 5, .1).unsqueeze(1).to(device)
    h = torch.zeros(x.shape[0], 2).to(device)
    y = f(x[:, 0], h[:, 0], h[:, 1], args.mode).detach().cpu().numpy()
    y_mon = model(x, h)[:, 0].detach().cpu().numpy()
    y_mlp = model_mlp(x, h)[:, 0].detach().cpu().numpy()
    x = x.detach().cpu().numpy()
    plt.plot(x, y, label="groundtruth")
    plt.plot(x, y_mlp, label="MLP model")    
    plt.plot(x, y_mon, label="Constraint model")
    plt.xlabel('x1 (x2=x3=0)')
    plt.ylabel('y')
    plt.legend()
    # plt.show()
    plt.savefig(f"{args.mode}.pdf", bbox_inches='tight', pad_inches=0.0)
    plt.clf()