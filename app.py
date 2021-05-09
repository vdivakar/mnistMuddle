import streamlit as st
import numpy as np
import pandas as pd
import random
from PIL import Image

from model_class import AutoEncoder
from mnist import MNIST
import torch

st.title("MNIST Muddle")
st.write("\
## Generating poorly handwritten digits...\n\
... Proving humans' handwriting is still good!\
")

nums_list = [0,1,2,3,4,5,6,7,8,9]

left_col, right_col = st.beta_columns([1,3])
with left_col:
    selected_number = st.radio('Select Number', nums_list, index=0)
    st.button('Re-generate')

with right_col:
    st.image('Data/tsne-latent.png', "t-SNE plot of latent vectors (train images)" , width=400)
######################################################################
load_dir = "checkpoints/"

@st.cache
def load_model():
    model = AutoEncoder()
    try:
        load_ckpt_num = 68
        checkpoint = torch.load(load_dir+"ep_{}.ckpt".format(load_ckpt_num))
        model.load_state_dict(checkpoint['model_state_dict'])
        global_epoch = checkpoint['global_epoch']
        print("loaded global ep: {}".format(global_epoch))
    except Exception as e:
        print("Loading checkpoint failed! - {}".format(e))
        global_epoch = 0
    return model

@st.cache
def load_data():
    mndata = MNIST('Data/mnist')
    batch = mndata.load_training_in_batches(5000)
    (raw_test_images, raw_test_labels) = next(batch)
    X_te = np.array(raw_test_images)
    y_te = np.array(raw_test_labels)
    X_te_imgs = X_te.reshape([-1, 1,28,28])
    
    return X_te_imgs, y_te

@st.cache
def load_avg_latent_vectors():
    images, labels = load_data()
    model = load_model()
    avg_latent_vecs = np.zeros((10,10))
    with torch.no_grad():
        for num in range(0, 10):
            x = torch.tensor(images[labels==num])/255
            l = model.encoder(x) #get latent vectors
            l_mean = l.mean(axis=0)
            l_mean = l_mean.detach().numpy()
            avg_latent_vecs[num] = l_mean
    return avg_latent_vecs

def display_img_array(tensor, texts):
    # l = list(st.beta_columns(len(tensor)))
    l = list(st.beta_columns([1,1,2,2]))
    for i in range(len(l)):
        if i==3:
            temp = "./latent_output.gif"
        else:
            temp = tensor[i].permute(1,2,0).squeeze().detach().numpy()
        with l[i]:
            st.write(texts[i])
            st.image(temp, clamp=True, use_column_width=True)#False, width=128)

def get_rand_img(images, labels, number):
    rule = labels==number
    imgs = torch.tensor(images[rule])
    rand_num = random.randint(0, len(imgs)-1)
    rand_img = imgs[rand_num:rand_num+1]/255
    return rand_img

model = load_model()
images, labels = load_data()
avg_latent_vecs = load_avg_latent_vectors()

a = get_rand_img(images, labels, selected_number)

with torch.no_grad():
    l1 = model.encoder(a)

    #1. Finding distances from clusters centers in latent domain.
    #2. Selecting a different cluster than itself.
    #3. Picking a random image from that nearest cluster.
    cluster_distances = np.linalg.norm(avg_latent_vecs - l1.detach().numpy(), axis=1)
    nearest_clusters = np.argpartition(cluster_distances, 2) # Top 2 indices with minimum distance
    nearest_cluster_idx = nearest_clusters[1] if nearest_clusters[0]==selected_number else nearest_clusters[0]
    b = get_rand_img(images, labels, nearest_cluster_idx)

    l2 = model.encoder(b)
    l_avg = (l1+l2)/2

    o1 = model.decoder(l1)
    o2 = model.decoder(l2)
    o3 = model.decoder(l_avg)

    def save_gif():
        weights = [0.25, 0.35, 0.45, 0.5, 0.55, 0.65, 0.75]
        frames = []
        for w in weights:
            l_wavg  = (1-w)*l1 + w*l2
            out     = model.decoder(l_wavg).squeeze().detach().numpy()*255
            print(out.shape)
            out_img = Image.fromarray(out).convert('L')
            # out_img.save("{}.png".format(w))
            frames.append(out_img)
        frames2 = frames #to play in bounce format
        for f in reversed(frames):
            frames2.append(f)
        frames2[0].save('./latent_output.gif', format='GIF',
                        append_images=frames[1:],
                        save_all=True,
                        duration=185, loop=0)
    save_gif()

    outputs = torch.cat((o1,o2,o3,o3))
    texts = ["Input: {}".format(selected_number),\
             "Mimic: {}".format(nearest_cluster_idx), \
             "Output Img", \
             "Output GIF"]
    display_img_array(outputs, texts)

print("DONE!")

#####################
link_github = '[GitHub Repo Link](https://github.com/vdivakar/mnistMuddle)'
st.markdown(link_github, unsafe_allow_html=True)

link_blog = '[Blog Post Link for details](https://www.divakar-verma.com/post/mnist-muddle)'
st.markdown(link_blog, unsafe_allow_html=True)