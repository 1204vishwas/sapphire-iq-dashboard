import pickle as pkl

model = None

with open(r'models\return_prediction_model.pkl', 'rb') as f:
    model = pkl.load(f)

print(model)
