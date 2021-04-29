import streamlit as st
import numpy as np
import pandas as pd

st.title("MNIST Muddle")
st.write("\
### Generating poor handwritten digits just to prove your handwritting is good!\
")
df = [1,2,3,4,5]

option = st.sidebar.selectbox(
    'Which number do you like best?',
     df)

'You selected:', option

left_column, right_column = st.beta_columns(2)
pressed = left_column.button('Press me?')
if pressed:
    right_column.write("Woohoo!")

# expander = st.beta_expander("FAQ")
# expander.write("Here you could put in some really, really long explanations...")


# from PIL import Image
# image = Image.open('sprite.png')
# st.image(image, caption='sprite')

######################################################################
import sys
sys.path.append("/Users/dv/Projects/PyTorch-projects/mnistMuddle/")
from model_class import AutoEncoder
from mnist import MNIST
import torch

load_dir = "/Users/dv/Projects/PyTorch-projects/mnistMuddle/checkpoints/"

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
    mndata = MNIST('../Data/mnist')
    raw_test_images, raw_test_labels = mndata.load_testing()
    X_te = np.array(raw_test_images)
    y_te = np.array(raw_test_labels)
    X_te_imgs = X_te.reshape([-1, 1,28,28])
    
    return X_te_imgs, y_te

def display_img(tensor):
    m = tensor.shape[0]
    for i in range(m):
        temp = tensor[i].permute(1,2,0).squeeze().detach().numpy()
        st.image(temp, clamp=True)


model = load_model()
images, labels = load_data()

rule_1 = labels==7
rule_2 = labels==9

a = torch.tensor(images[rule_1][:1])/255
b = torch.tensor(images[rule_2][:1])/255

with torch.no_grad():
    l1 = model.encoder(a)
    l2 = model.encoder(b)
    l_avg = (l1+l2)/2

    o1 = model.decoder(l1)
    o2 = model.decoder(l2)
    o3 = model.decoder(l_avg)

    display_img(o1)
    display_img(o2)
    display_img(o3)

print("DONE!")