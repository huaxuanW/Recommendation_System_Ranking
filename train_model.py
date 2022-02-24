import torch
from deepctr_torch.models import DeepFM
from data_preprocessing import get_data

dnn_feature_columns, linear_feature_columns, train_model_input, test_model_input, y_train, y_test = get_data()

device = 'cpu'
use_cuda = True
if use_cuda and torch.cuda.is_available():
    print('cuda ready...')
    device = 'cuda'

model = DeepFM(
    linear_feature_columns=linear_feature_columns, 
    dnn_feature_columns=dnn_feature_columns,
    task='binary',
    device=device
)

model.compile(
    optimizer="adam", 
    loss="binary_crossentropy",
    metrics=["acc"])

model.fit(
    train_model_input,
    y_train,
    batch_size=32,
    epochs=5,
    verbose=2,
    validation_data=(test_model_input, y_test)
)