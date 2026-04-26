import pickle
from src.train import prepare_data
from src.evaluate import evaluate_all_endpoints
from src.featurize import TARGET_COLS
import torch

with open('models/model_artifact.pkl', 'rb') as f:
    m = pickle.load(f)
    
model = m['mondrian_predictor'].base_model
_, _, _, _, _, _, X_t, Y_t, _, _ = prepare_data()
model.eval()
p = torch.sigmoid(model(torch.tensor(X_t, dtype=torch.float32))).detach().numpy()
res = evaluate_all_endpoints(Y_t, p, TARGET_COLS)
print(res.iloc[-1])
