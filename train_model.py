import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from mnist import MNIST
from sklearn import manifold
import numpy as np
import os, argparse
from matplotlib import offsetbox
from torch.utils.tensorboard import SummaryWriter
from model_class import AutoEncoder

# Read Data as Numpy
def read_data():
    mndata = MNIST('Data/mnist')
    raw_train_images, raw_train_labels = mndata.load_training()
    raw_test_images, raw_test_labels = mndata.load_testing()

    X_tr = np.array(raw_train_images)
    y_tr = np.array(raw_train_labels)
    X_te = np.array(raw_test_images)
    y_te = np.array(raw_test_labels)

    X_tr_imgs = X_tr.reshape([-1, 1,28,28])
    X_te_imgs = X_te.reshape([-1, 1,28,28])

    print("X_tr: {}".format(X_tr.shape))
    print("y_tr: {}".format(y_tr.shape))
    print("X_te: {}".format(X_te.shape))
    print("y_te: {}".format(y_te.shape))
    print("X_tr_imgs: {}".format(X_tr_imgs.shape))
    print("X_te_imgs: {}".format(X_te_imgs.shape))

    return X_tr_imgs, y_tr, X_te_imgs, y_te

# Load as DataLoader
def load_data():
    X_tr_imgs, y_tr, X_te_imgs, y_te = read_data()

    img_tr_tensor = torch.tensor(X_tr_imgs)/255.0 #converting from 0-255 to 0.0-1.0
    y_tr_tensor   = torch.tensor(y_tr)
    img_te_tensor = torch.tensor(X_te_imgs)/255.0 #converting from 0-255 to 0.0-1.0
    y_te_tensor   = torch.tensor(y_te)

    train_dataset = torch.utils.data.TensorDataset(img_tr_tensor, y_tr_tensor)
    test_dataset  = torch.utils.data.TensorDataset(img_te_tensor, y_te_tensor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, num_workers=0)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=10, num_workers=0)

    return train_loader, test_loader

def train_model(model, args):
    train_loader, test_loader = load_data()

    # Loss, opt, summary writer
    loss_fxn  = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 
    save_dir  = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    writer    = SummaryWriter('{}/exp_1'.format(args.summary))
    te_writer = SummaryWriter('{}/test_1'.format(args.summary))

    # Train Loop
    try:
        load_ckpt_num = 5
        checkpoint = torch.load(save_dir+"/ep_{}.ckpt".format(load_ckpt_num))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        global_epoch = checkpoint['global_epoch']
        print("loaded global ep: {}".format(global_epoch))
    except Exception as e:
        print("Loading checkpoint failed! - {}".format(e))
        print("Training from beginning")
        global_epoch = 0
        
    n_epochs = args.epochs
    print("No. of epochs: {}".format(n_epochs))

    for epoch in range(n_epochs):
        train_loss = 0.0
        model.train()
        for data in train_loader:
            images, _ = data
            outputs = model.forward(images)
            loss = loss_fxn(outputs, images)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss+=(loss.item()*images.size(0))

        # Save and evaluate  
        if(epoch%5==4):
            print("Saving Global epoch: {}".format(global_epoch))
            
            torch.save({
                'global_epoch':global_epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict()
            }, save_dir+"/ep_{}.ckpt".format(global_epoch))
            
            writer.add_scalar('train_loss', train_loss/len(train_loader), global_epoch)
            
            model.eval()
            with torch.no_grad():
                te_loss = 0.0
                for te_data in test_loader:
                    te_imgs, _ = data
                    te_out = model.forward(te_imgs)
                    loss_ = loss_fxn(te_out, te_imgs)
                    te_loss += loss_.item() * te_imgs.size(0)
                te_loss = te_loss/len(test_loader)
                te_writer.add_scalar('test_loss', te_loss, global_epoch)

                print("Test Loss: {}".format(te_loss))
        
        global_epoch+=1 
        
        train_loss = train_loss/len(train_loader)
        print("Epoch: {} \t Training Loss: {}".format(epoch, train_loss))
    
    print("Training Done!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training mnist-Muddle!')
    parser.add_argument('--epochs', default=5, type=int,
                        help='No. of epochs to train from the prev. ckpt')
    parser.add_argument('--save_dir', default="./checkpoints/", type=str,
                        help='Dir for saving and loading ckpts')
    parser.add_argument('--summary', default="runs", type=str,
                        help='Dir for Tensorboard summary')

    args = parser.parse_args()
    model = AutoEncoder()
    train_model(model, args)
 