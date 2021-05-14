# mnist-Muddle
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/vdivakar/mnistmuddle/br_streamlit/app.py)

Generating poorly written digits to prove humans' handwriting is still good!

<img src="https://github.com/vdivakar/mnistMuddle/blob/master/images/sample_output.png" width="120">

### Project Demo Page - [Link](https://share.streamlit.io/vdivakar/mnistmuddle/br_streamlit/app.py)
[![project demo page][1]][2]

[1]:  https://github.com/vdivakar/mnistMuddle/blob/master/images/mnistMuddleCast.gif
[2]:  https://share.streamlit.io/vdivakar/mnistmuddle/br_streamlit/app.py

### Details and description - [Blog post](https://www.divakar-verma.com/post/mnist-muddle)

#### Files:

`train_model.py` - Train your own model from scratch. <br>
`model_class.py` - AutoEncoder model <br>
`inference.ipynb` - Jupyter notebook to generate output images and plot charts<br>
`experiments.ipynb` - Jupyter notebook containing experimental code<br>

For __Streamlit app__, checkout `br_streamlit` branch
```
git checkout br_streamlit
```

This project was build using these Requirements:
```
torch==1.7.0
pandas==1.2.4
numpy==1.19.4
streamlit==0.80.0
python-mnist
```



