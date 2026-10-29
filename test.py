import xgboost as xgb
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

# 1. Prepare data and split into training/test sets
X, y = make_classification(n_samples=3000, n_features=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Simulate Horizontal Partitioning among 3 clients using clean array slicing
num_clients = 3
chunk_size = len(X_train) // num_clients
client_dmatrices = []

for i in range(num_clients):
    start_idx = i * chunk_size
    end_idx = (i + 1) * chunk_size if i < num_clients - 1 else len(X_train)
    
    cx = X_train[start_idx:end_idx]
    cy = y_train[start_idx:end_idx]
    
    client_dmatrices.append(xgb.DMatrix(cx, label=cy))

test_dmatrix = xgb.DMatrix(X_test, label=y_test)

# 3. Define the Custom Objective Function for Gradients and Hessians
def custom_logistic_objective(preds, dtrain):
    labels = dtrain.get_label()
    p = 1.0 / (1.0 + np.exp(-preds))
    grad = p - labels
    hess = p * (1.0 - p)
    return grad, hess

# 4. Server-side Aggregation Function
def server_aggregate(client_grads, client_hess):
    global_grads = np.concatenate(client_grads)
    global_hess = np.concatenate(client_hess)
    return global_grads, global_hess

# 5. Federated Training Configuration
num_communication_rounds = 100
local_boosting_rounds = 1   # Trees built locally per client per round

params = {
    'max_depth': 4,
    'eta': 0.1,
    'disable_default_eval_metric': 1
}

global_model = None
round_accuracies = []

# 6. Main Federated Loop with Evaluation Tracking
for round_idx in range(num_communication_rounds):
    client_gradients = []
    client_hessians = []
    local_boosters = []
    
    # --- Client Local Step ---
    for client_id, dmatrix in enumerate(client_dmatrices):
        if global_model is None:
            bst = xgb.train(params, dmatrix, num_boost_round=local_boosting_rounds, obj=custom_logistic_objective)
        else:
            bst = xgb.train(params, dmatrix, num_boost_round=local_boosting_rounds, xgb_model=global_model, obj=custom_logistic_objective)
            
        local_boosters.append(bst)
        
        preds = bst.predict(dmatrix, output_margin=True)
        grad, hess = custom_logistic_objective(preds, dmatrix)
        
        client_gradients.append(grad)
        client_hessians.append(hess)

    # --- Server Aggregation Step ---
    agg_grads, agg_hess = server_aggregate(client_gradients, client_hessians)
    global_model = local_boosters[-1]

    # --- Evaluation Phase (Every 10 rounds & on the final round) ---
    if (round_idx + 1) % 10 == 0 or round_idx == num_communication_rounds - 1:
        test_preds_margin = global_model.predict(test_dmatrix, output_margin=True)
        test_preds_prob = 1.0 / (1.0 + np.exp(-test_preds_margin))
        test_preds_label = [1 if p > 0.5 else 0 for p in test_preds_prob]
        
        acc = accuracy_score(y_test, test_preds_label)
        loss = log_loss(y_test, test_preds_prob)
        round_accuracies.append(acc)
        
        print(f"Round {round_idx + 1:3d}/{num_communication_rounds} | Test Accuracy: {acc * 100:.2f}% | Test LogLoss: {loss:.4f}")

# 7. Final Model Summary Evaluation
print("\n--- Final Federated Model Performance ---")
final_preds_margin = global_model.predict(test_dmatrix, output_margin=True)
final_preds_prob = 1.0 / (1.0 + np.exp(-final_preds_margin))
final_preds_label = [1 if p > 0.5 else 0 for p in final_preds_prob]

final_accuracy = accuracy_score(y_test, final_preds_label)
print(f"Final Model Test Accuracy: {final_accuracy * 100:.2f}%")