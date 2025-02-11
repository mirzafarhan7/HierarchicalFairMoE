import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import functional as f
from tqdm import tqdm

import helpers_adult

def load_adult_dataset():
    """Load and preprocess Adult dataset"""
    # Load data from UCI ML repository
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                    'marital-status', 'occupation', 'relationship', 'race', 'sex',
                    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

    train_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
                             names=column_names, sep=',\s*', engine='python')
    test_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test',
                            names=column_names, sep=',\s*', engine='python', skiprows=1)

    # Combine train and test
    data = pd.concat([train_data, test_data])

    # Create binary labels
    data['Labels'] = (data['income'].str.contains('>50K')).astype(int)

    # Select protected attributes
    protected_attributes = ['race', 'sex']

    # Select features for the model
    numerical_columns = ['age', 'fnlwgt', 'education-num', 'capital-gain',
                         'capital-loss', 'hours-per-week']
    categorical_columns = ['workclass', 'education', 'marital-status', 'occupation',
                           'relationship', 'native-country']

    # Preprocessing
    # Handle missing values
    data = data.replace('?', np.nan)

    # Convert numerical columns
    for col in numerical_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(data[col].mean())

    # Handle categorical columns
    for col in categorical_columns + protected_attributes:
        data[col] = data[col].fillna(data[col].mode()[0])

    # Create dummy variables for categorical features
    data_encoded = pd.get_dummies(data, columns=categorical_columns + protected_attributes)

    # Ensure all columns are numeric
    for col in data_encoded.columns:
        if data_encoded[col].dtype == 'object':
            data_encoded[col] = pd.to_numeric(data_encoded[col], errors='coerce')

    # Normalize numerical features
    scaler = StandardScaler()
    data_encoded[numerical_columns] = scaler.fit_transform(data_encoded[numerical_columns])

    # Convert all columns to float32 for better compatibility with PyTorch
    data_encoded = data_encoded.astype('float32')

    # Separate features and labels
    X = data_encoded.drop(['Labels', 'income'], axis=1)
    y = data_encoded['Labels']

    # Get protected attribute columns
    gender_columns = [col for col in X.columns if col.startswith('sex_')]
    race_columns = [col for col in X.columns if col.startswith('race_')]

    # Combine protected columns
    protected = pd.concat([
        data_encoded[gender_columns],
        data_encoded[race_columns]
    ], axis=1)

    return X, y, protected, len(gender_columns), len(race_columns)


class FairnessAnalyzer:
    def __init__(self):
        self.fairness_flags = {
            'gender': {'spd': False, 'aod': False},
            'race': {'spd': False, 'aod': False}
        }
        self.thresholds = {
            'spd': 0.1,
            'aod': 0.1
        }
        self.metric_scores = {}
        self.worst_metrics = []

    def analyze_fairness(self, predictions, labels, protected_attributes, attribute_slices):
        """Analyze fairness issues and set flags"""
        # If predictions is None, return current flags and worst metrics
        if predictions is None:
            return self.fairness_flags, self.worst_metrics

        metric_values = {}

        for attr_name, attr_slice in attribute_slices.items():
            curr_protected = protected_attributes[:, attr_slice]
            spd = helpers_adult.calculate_spd(predictions, curr_protected)
            aod = helpers_adult.calculate_aod(predictions, labels, curr_protected)

            metric_values[f"{attr_name}_spd"] = abs(spd)
            metric_values[f"{attr_name}_aod"] = abs(aod)

        # Sort metrics by value to identify worst performers
        sorted_metrics = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
        self.worst_metrics = [metric[0] for metric in sorted_metrics[:2]]

        # Set flags based on worst metrics
        updated_flags = {
            'gender': {'spd': False, 'aod': False},
            'race': {'spd': False, 'aod': False}
        }

        for metric in self.worst_metrics:
            attr, metric_type = metric.split('_')
            updated_flags[attr][metric_type] = True

        self.fairness_flags = updated_flags
        self.metric_scores = metric_values
        return updated_flags, self.worst_metrics

class ExpertNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.requires_focus = False
        self.specialization = None

        # Initialize with larger weights to prevent vanishing gradients
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=2.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)

        self.spd_score = 0.0
        self.aod_score = 0.0
        self.update_count = 0

    def forward(self, x):
        logits = self.network(x)
        return torch.sigmoid(logits)


    def update_metrics(self, spd, aod):
        # Safety check for invalid values
        if torch.isnan(torch.tensor([spd, aod])).any() or \
           torch.isinf(torch.tensor([spd, aod])).any():
            print(f"Invalid metrics detected: SPD={spd}, AOD={aod}")

        if self.requires_focus:
            # Apply stronger updates for focused training
            self.spd_score *= 1.2
            self.aod_score *= 1.2
            return

        alpha = 0.9
        self.spd_score = alpha * self.spd_score + (1 - alpha) * float(abs(spd))
        self.aod_score = alpha * self.aod_score + (1 - alpha) * float(abs(aod))
        # self.intersectional_tpr_score = alpha * self.intersectional_tpr_score + (1 - alpha) * float(abs(intersectional_tpr))
        self.update_count += 1

    def get_specialization(self):
        """Return the metric this expert is best at"""
        if self.update_count < 100:  # Require minimum updates
            return None

        metrics = {
            'spd': self.spd_score,
            'aod': self.aod_score
        }

        # Find minimum score (best performance)
        min_metric = min(metrics.items(), key=lambda x: x[1])

        # Only return specialization if scores are valid
        if min_metric[1] == float('inf') or min_metric[1] == 0.0:
            return None

        return min_metric[0]


def evaluate_fairness(model, test_loader):
    """Evaluate model fairness metrics"""
    model.eval()
    fairness_metrics = {
        'gender': {'spd': [], 'aod': []},
        'race': {'spd': [], 'aod': []}
    }

    with torch.no_grad():
        for batch_data, batch_labels, batch_protected in test_loader:
            device = next(model.parameters()).device
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            batch_protected = batch_protected.to(device)

            outputs = model(batch_data, batch_labels, batch_protected)
            branch_preds = outputs['branch_preds']  # Shape: [batch_size, 2]
            final_pred = torch.sum(outputs['branch_weights'] * branch_preds, dim=1)

            # Calculate metrics for each branch using correct column ranges
            branch_configs = [
                ('gender', model.gender_cols),
                ('race', model.race_cols)
            ]

            for branch_name, cols in branch_configs:
                # Calculate SPD and AOD using helpers
                spd = helpers_adult.calculate_spd(
                    final_pred,
                    batch_protected[:, cols]
                )
                aod = helpers_adult.calculate_aod(
                    final_pred,
                    batch_labels,
                    batch_protected[:, cols]
                )

                fairness_metrics[branch_name]['spd'].append(spd.item())
                fairness_metrics[branch_name]['aod'].append(aod.item())

            # Optional: Log summary statistics for this batch
            if hasattr(model, 'current_epoch') and model.current_epoch % 10 == 0:
                print(f"\nBatch Fairness Summary:")
                for branch_name in ['gender', 'race']:
                    print(f"\n{branch_name.capitalize()} Metrics:")
                    print(f"SPD: {fairness_metrics[branch_name]['spd'][-1]:.4f}")
                    print(f"AOD: {fairness_metrics[branch_name]['aod'][-1]:.4f}")

    # Calculate average metrics across all batches
    avg_metrics = {
        branch: {
            metric: sum(values) / len(values)
            for metric, values in metrics.items()
        }
        for branch, metrics in fairness_metrics.items()
    }

    return avg_metrics

class FairnessBranch(nn.Module):
    def __init__(self, input_dim: int, attribute_name: str, num_categories: int, device=None):
        super().__init__()
        self.device = device
        self.attribute_name = attribute_name
        self.num_categories = num_categories

        self.num_experts = 16
        self.experts = nn.ModuleList([
            ExpertNetwork(input_dim) for _ in range(self.num_experts)
        ])

        self.router = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_experts),
            nn.Softmax(dim=1)
        )

        self.attribute_weight = nn.Parameter(torch.ones(1))
        self.to(self.device)

    def forward(self, x: torch.Tensor):
        x = x.to(self.device)
        routing_weights = self.router(x)

        expert_outputs = []
        for expert in self.experts:
            pred = expert(x).squeeze(-1)
            expert_outputs.append(pred)

        expert_outputs = torch.stack(expert_outputs, dim=1)
        weighted_pred = torch.sum(routing_weights * expert_outputs, dim=1)

        return weighted_pred, routing_weights, expert_outputs

    @staticmethod
    def calculate_fairness_losses(predictions, labels, protected_attributes):
        epsilon = 1e-8
        predictions = torch.clamp(predictions.squeeze(), epsilon, 1 - epsilon)
        labels = labels.squeeze()

        # Calculate regular fairness metrics
        pred_rates = []
        tpr_values = []
        fpr_values = []

        num_categories = protected_attributes.shape[1]
        for i in range(num_categories):
            mask_i = (protected_attributes[:, i] == 1)
            if mask_i.sum() == 0:
                continue

            # Calculate prediction rate for SPD
            pred_i = predictions[mask_i].mean()
            pred_rates.append(pred_i)

            # Calculate TPR and FPR for individual attributes
            mask_i_pos = mask_i & (labels == 1)
            mask_i_neg = mask_i & (labels == 0)

            if mask_i_pos.sum() > 0:
                tpr_i = torch.sum(predictions[mask_i_pos]) / (mask_i_pos.sum() + epsilon)
                tpr_values.append(tpr_i)

            if mask_i_neg.sum() > 0:
                fpr_i = torch.sum(predictions[mask_i_neg]) / (mask_i_neg.sum() + epsilon)
                fpr_values.append(fpr_i)

        # Calculate metrics
        pred_rates = torch.stack(pred_rates) if pred_rates else torch.tensor([0.0])
        tpr_values = torch.stack(tpr_values) if tpr_values else torch.tensor([0.0])
        fpr_values = torch.stack(fpr_values) if fpr_values else torch.tensor([0.0])

        spd_loss = torch.max(pred_rates) - torch.min(pred_rates)
        tpr_diff = torch.max(tpr_values) - torch.min(tpr_values)
        fpr_diff = torch.max(fpr_values) - torch.min(fpr_values)
        aod_loss = (tpr_diff + fpr_diff) / 2

        return spd_loss, aod_loss

class HierarchicalFairnessMoE(nn.Module):
    def __init__(self, input_dim, device, gender_categories, race_categories):
        super().__init__()
        self.device = device
        self.current_epoch = 0

        # Define slice objects for each attribute
        current_idx = 0
        self.gender_cols = slice(current_idx, current_idx + gender_categories)
        current_idx += gender_categories
        self.race_cols = slice(current_idx, current_idx + race_categories)

        # Initialize branch weights
        self.branch_weights = nn.Parameter(torch.tensor([0.5, 0.5], device=self.device))

        # Create branches
        self.gender_branch = FairnessBranch(input_dim, "gender", num_categories=gender_categories,
                                          device=self.device)
        self.race_branch = FairnessBranch(input_dim, "race", num_categories=race_categories,
                                         device=self.device)

        # Add batch normalization for final output
        self.final_bn = nn.BatchNorm1d(1)

        # Initialize optimizers
        self.branch_optimizers = {
            'gender': torch.optim.AdamW(
                self.gender_branch.parameters(),
                lr=0.0001,
                weight_decay=0.01,
                betas=(0.9, 0.999)
            ),
            'race': torch.optim.AdamW(
                self.race_branch.parameters(),
                lr=0.0001,
                weight_decay=0.01,
                betas=(0.9, 0.999)
            )
        }

        # Initialize fairness analyzer
        self.fairness_analyzer = FairnessAnalyzer()
        self.attribute_slices = {
            'gender': self.gender_cols,
            'race': self.race_cols
        }

    def get_expert_specializations(self):
        total_stats = {
            'spd': 0,
            'aod': 0
        }

        branch_stats = {
            'gender': {'spd': 0, 'aod': 0},
            'race': {'spd': 0, 'aod': 0}
        }

        performance_stats = {
            'gender': {'spd': [], 'aod': []},
            'race': {'spd': [], 'aod': []}
        }

        for branch_name, branch in [
            ('gender', self.gender_branch),
            ('race', self.race_branch)
        ]:
            for i, expert in enumerate(branch.experts):
                print(f"Expert {i}: Updates={expert.update_count}, "
                      f"SPD={expert.spd_score:.4f}, "
                      f"AOD={expert.aod_score:.4f}")

                if expert.update_count > 0:
                    metrics = {
                        'spd': expert.spd_score,
                        'aod': expert.aod_score
                    }

                    # Find the metric this expert is best at
                    best_metric = min(metrics.items(), key=lambda x: abs(x[1]))[0]

                    # Update counters
                    total_stats[best_metric] += 1
                    branch_stats[branch_name][best_metric] += 1

                    # Store performance scores
                    for metric, score in metrics.items():
                        performance_stats[branch_name][metric].append(score)

        # Calculate average performance for each metric in each branch
        avg_performance = {
            branch: {
                metric: sum(scores) / len(scores) if scores else 0
                for metric, scores in metrics.items()
            }
            for branch, metrics in performance_stats.items()
        }

        return {
            'total_stats': total_stats,
            'branch_stats': branch_stats,
            'avg_performance': avg_performance
        }

    def freeze_weights(self):
        """Freeze weights for inference"""
        for param in self.parameters():
            param.requires_grad = False

        # Keep router trainable
        for branch in [self.gender_branch, self.race_branch]:
            for param in branch.router.parameters():
                param.requires_grad = True

    def unfreeze_weights(self):
        """Unfreeze weights for training"""
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor, labels=None, protected_attributes=None):
        batch_size = x.shape[0]
        orig_branch_weights = self.branch_weights.clone().to(self.device)

        # Get predictions from each branch
        gender_pred, gender_weights, gender_experts = self.gender_branch(x)
        race_pred, race_weights, race_experts = self.race_branch(x)

        # Combine predictions using branch weights
        branch_preds = torch.stack([gender_pred, race_pred], dim=1).to(self.device)
        expanded_weights = orig_branch_weights.unsqueeze(0).expand(batch_size, -1)
        final_pred = torch.sum(expanded_weights * branch_preds, dim=1)

        # Update branch weights during training based on fairness flags
        if self.training and all(v is not None for v in [labels, protected_attributes]):
            with torch.no_grad():
                fairness_flags, worst_metrics = self.fairness_analyzer.analyze_fairness(
                    final_pred,
                    labels,
                    protected_attributes,
                    self.attribute_slices
                )

                # Update worst metrics for branches
                self.gender_branch.update_worst_metrics(
                    [m.split('_')[1] for m in worst_metrics if m.split('_')[0] == 'gender']
                )
                self.race_branch.update_worst_metrics(
                    [m.split('_')[1] for m in worst_metrics if m.split('_')[0] == 'race']
                )

                # Adjust weights based on fairness violations
                new_weights = orig_branch_weights.clone()
                for i, attr_name in enumerate(['gender', 'race']):
                    if attr_name in [m.split('_')[0] for m in worst_metrics]:
                        new_weights[i] *= 1.2

                new_weights = new_weights / new_weights.sum()
                self.branch_weights.data.copy_(new_weights)

        return {
            'final_pred': final_pred,
            'branch_preds': branch_preds,
            'branch_weights': orig_branch_weights,
            'expert_outputs': {
                'gender': gender_experts,
                'race': race_experts
            }
        }

    def update_epoch(self, epoch):
        self.current_epoch = epoch

    # def calculate_intersectional_fairness(self, outputs, labels, protected_attributes):
    #     predictions = outputs['final_pred']
    #     epsilon = 1e-8
    #     tpr_values = []
    #
    #     # Get all possible intersectional groups
    #     all_combinations = torch.unique(protected_attributes, dim=0)
    #
    #     # Calculate TPR for each intersectional group
    #     for group in all_combinations:
    #         group_tensor = torch.tensor(group, device=protected_attributes.device)
    #         mask = torch.all(torch.eq(protected_attributes, group_tensor), dim=1)
    #         mask_pos = mask & (labels == 1)
    #         if mask_pos.sum() > 0:
    #             tpr = predictions[mask_pos].sum() / (mask_pos.sum() + epsilon)
    #             tpr_values.append(tpr)
    #
    #     # Calculate maximum difference between any two groups
    #     if len(tpr_values) > 1:  # Need at least 2 groups to calculate difference
    #         tpr_values = torch.stack(tpr_values)
    #         max_tpr_diff = torch.max(tpr_values) - torch.min(tpr_values)
    #         return max_tpr_diff
    #     else:
    #         return torch.tensor(0.0, device=self.device)

    def training_step(self, batch_data, batch_labels, protected_attributes):
        # Zero gradients at the start
        for optimizer in self.branch_optimizers.values():
            optimizer.zero_grad()

        outputs = self(batch_data, batch_labels, protected_attributes)
        final_pred = outputs['final_pred']

        # Get the current worst metrics
        fairness_flags, worst_metrics = self.fairness_analyzer.analyze_fairness(
            final_pred,  # Using current predictions
            batch_labels,
            protected_attributes,
            {'gender': self.gender_cols, 'race': self.race_cols}
        )

        # Ensure proper shapes
        batch_labels = batch_labels.float().view(-1)
        final_pred = final_pred.view(-1)

        # Base prediction loss (BCE)
        criterion = nn.BCELoss()
        pred_loss = criterion(final_pred, batch_labels)

        # Apply sigmoid to get probabilities
        sigmoid_preds = torch.sigmoid(final_pred)

        # Fairness loss focusing on worst metrics
        fairness_loss = torch.tensor(0.0, device=self.device)
        if self.current_epoch >= 50:  # Start fairness training after 50 epochs
            for branch_name, branch, cols in [
                ('gender', self.gender_branch, self.gender_cols),
                ('race', self.race_branch, self.race_cols)
            ]:
                if any(fairness_flags[branch_name].values()):
                    # Calculate losses only for the worst metrics
                    spd, aod = branch.calculate_fairness_losses(
                        sigmoid_preds,
                        batch_labels,
                        protected_attributes[:, cols]
                    )

                    # Add losses based on which metrics are worst
                    for metric in worst_metrics:
                        attr, metric_type = metric.split('_')
                        if attr == branch_name:
                            if metric_type == 'spd':
                                fairness_loss += 2.0 * spd  # Higher weight for worst metric
                            elif metric_type == 'aod':
                                fairness_loss += 2.0 * aod  # Higher weight for worst metric

        # Adaptive weighting based on epoch and worst metrics
        base_fairness_weight = min(0.05, self.current_epoch / 100)
        fairness_weight = base_fairness_weight * (
                    1 + len(worst_metrics) * 0.2)  # Increase weight based on number of worst metrics

        # Combined loss
        total_loss = pred_loss + fairness_weight * fairness_loss

        # Backward pass and optimization
        total_loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        # Update weights with focus on experts handling worst metrics
        for branch_name, branch in [
            ('gender', self.gender_branch),
            ('race', self.race_branch)
        ]:
            if any(fairness_flags[branch_name].values()):
                for i, expert in enumerate(branch.experts):
                    if expert.get_specialization() in [m.split('_')[1] for m in worst_metrics]:
                        # Apply stronger learning rate for experts handling worst metrics
                        for param in expert.parameters():
                            if param.grad is not None:
                                param.data -= 1.2 * self.branch_optimizers[branch_name].param_groups[0][
                                    'lr'] * param.grad

        # Normal update for other parameters
        for optimizer in self.branch_optimizers.values():
            optimizer.step()
            optimizer.zero_grad()

        return {
            'final_pred': sigmoid_preds.detach(),
            'total_loss': total_loss.item(),
            'pred_loss': pred_loss.item(),
            'fairness_loss': fairness_loss.item(),
            'worst_metrics': worst_metrics
        }


def main():
    # Load Adult dataset
    X, y, protected, num_gender_categories, num_race_categories = load_adult_dataset()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Calculate batch size
    dataset_size = len(X)
    batch_size = min(2 ** int(np.log2(dataset_size * 0.01)), 256)
    batch_size = max(32, batch_size)  # Minimum batch size of 32
    print(f"Dataset size: {dataset_size}, Selected batch size: {batch_size}")

    # Modify the data splitting
    train_idx, temp_idx = train_test_split(
        range(len(X)),
        test_size=0.3,  # 30% for temp
        stratify=y,
        random_state=42
    )

    temp_labels = y.iloc[temp_idx]  # Using iloc for integer-based indexing
    unique_labels = temp_labels.unique()  # Using pandas unique() for Series
    if len(unique_labels) > 1 and all(temp_labels.value_counts() >= 2):
        # If we have enough samples of each class, use stratification
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.5,  # 15% each for validation and test
            stratify=temp_labels,
            random_state=42
        )
    else:
        # If stratification is not possible, do regular split
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.5,
            random_state=42
        )

    # Convert to tensors and move to device
    X_train = torch.FloatTensor(X.iloc[train_idx].values).to(device)
    y_train = torch.FloatTensor(y.iloc[train_idx].values).to(device)
    protected_train = torch.FloatTensor(protected.iloc[train_idx].values).to(device)

    X_val = torch.FloatTensor(X.iloc[val_idx].values).to(device)
    y_val = torch.FloatTensor(y.iloc[val_idx].values).to(device)
    protected_val = torch.FloatTensor(protected.iloc[val_idx].values).to(device)

    X_test = torch.FloatTensor(X.iloc[test_idx].values).to(device)
    y_test = torch.FloatTensor(y.iloc[test_idx].values).to(device)
    protected_test = torch.FloatTensor(protected.iloc[test_idx].values).to(device)

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train, protected_train)
    val_dataset = TensorDataset(X_val, y_val, protected_val)
    test_dataset = TensorDataset(X_test, y_test, protected_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    input_dim = X.shape[1]
    model = HierarchicalFairnessMoE(
        input_dim=input_dim,
        device=device,
        gender_categories=num_gender_categories,
        race_categories=num_race_categories
    )

    # Training parameters
    num_epochs = 150
    patience = 10
    patience_counter = 0
    best_val_loss = float('inf')
    best_epoch = 0

    metrics_history = {
        'train_loss': [],
        'val_loss': [],
        'fairness_metrics': [],
        'expert_specializations': []
    }

    print("\nStarting training...")

    with tqdm(total=num_epochs, desc="Training") as pbar:
        for epoch in range(num_epochs):
            model.update_epoch(epoch)
            model.train()
            epoch_losses = []

            # Training loop
            for batch_data, batch_labels, batch_protected in train_loader:
                # Perform training step
                results = model.training_step(
                    batch_data=batch_data,
                    batch_labels=batch_labels,
                    protected_attributes=batch_protected
                )
                epoch_losses.append(results['total_loss'])

            # Calculate average loss for epoch
            avg_loss = np.mean(epoch_losses)
            metrics_history['train_loss'].append(avg_loss)

            # Evaluate every 10 epochs
            if epoch % 10 == 0:
                model.eval()
                print(f"\nEvaluating model at epoch {epoch}")

                # Validation loop
                val_losses = []
                correct = 0
                total = 0
                predictions = []
                true_labels = []

                with torch.no_grad():
                    for batch_data, batch_labels, batch_protected in val_loader:
                        outputs = model(batch_data, batch_labels, batch_protected)
                        branch_preds = outputs['branch_preds']  # Shape: [batch_size, 2]
                        branch_weights = outputs['branch_weights']
                        final_pred = torch.sum(branch_weights * branch_preds, dim=1)

                        # Calculate validation loss
                        val_loss = torch.nn.functional.binary_cross_entropy(final_pred, batch_labels)
                        val_losses.append(val_loss.item())

                        # Calculate accuracy
                        pred = (final_pred > 0.5).float()
                        correct += (pred == batch_labels).sum().item()
                        total += batch_labels.size(0)

                        predictions.extend(pred.cpu().numpy())
                        true_labels.extend(batch_labels.cpu().numpy())

                # Calculate validation metrics
                current_val_loss = np.mean(val_losses)
                accuracy = correct / total
                metrics_history['val_loss'].append(current_val_loss)

                print(f"\nValidation Loss: {current_val_loss:.4f}")
                print(f"Validation Accuracy: {accuracy:.4f}")

                # Early stopping check
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    best_accuracy = accuracy
                    best_epoch = epoch
                    patience_counter = 0
                    torch.save(model.state_dict(), 'best_model.pt')
                else:
                    patience_counter += 1

                # Get fairness metrics
                fairness_metrics = evaluate_fairness(model, val_loader)

                # Print fairness metrics
                print("\nFairness Metrics:")
                for branch in ['gender', 'race']:
                    print(f"\n{branch.capitalize()} Branch:")
                    for metric in ['spd', 'aod']:
                        values = fairness_metrics[branch][metric]
                        if isinstance(values, list) and values:
                            avg_value = sum(values) / len(values)
                            print(f"  {metric.upper()}: {avg_value:.4f}")
                        else:
                            print(f"  {metric.upper()}: {values:.4f}")

                # Early stopping
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    break

            pbar.update(1)
            pbar.set_postfix({'loss': avg_loss})

    model.freeze_weights()
    torch.save({
        'model_state_dict': model.state_dict(),
        'fairness_analyzer_state': model.fairness_analyzer.metric_scores,
        'worst_metrics': model.fairness_analyzer.worst_metrics
    }, 'model_checkpoint.pt')

    # Final evaluation on test set
    print("\nFinal Model Evaluation:")
    final_metrics = helpers_adult.evaluate_model(model, test_loader, num_epochs)
    final_fairness = evaluate_fairness(model, test_loader)

    # Print comprehensive final metrics
    print("\n" + "=" * 50)
    print("Final Model Performance Summary")
    print("=" * 50)
    print(f"Best epoch: {best_epoch}")
    print(f"Model Performance Metrics:")
    print(f"Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"AUROC Score: {final_metrics['auroc']:.4f}")
    print(f"True Positives: {final_metrics['tp']}")
    print(f"True Negatives: {final_metrics['tn']}")
    print(f"False Positives: {final_metrics['fp']}")
    print(f"False Negatives: {final_metrics['fn']}")

    print("\nFairness Metrics:")
    for branch in ['gender', 'race']:
        print(f"\n{branch.capitalize()} Branch:")
        for metric in ['spd', 'aod']:
            values = final_fairness[branch][metric]
            if isinstance(values, list) and values:
                avg_value = sum(values) / len(values)
                print(f"  {metric.upper()}: {avg_value:.4f}")
            else:
                print(f"  {metric.upper()}: {values:.4f}")

    # Analyze expert specializations
    print("\nExpert Specialization Analysis:")
    expert_analysis = model.get_expert_specializations()
    for branch in ['gender', 'race']:
        print(f"\n{branch.capitalize()} Branch Experts:")
        for metric, count in expert_analysis['branch_stats'][branch].items():
            print(f"  {metric.upper()}: {count} experts")


if __name__ == "__main__":
    main()





from typing import Dict
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

def print_label_distributions(df):
    print("\nLabel Distribution by Protected Attributes:")

    # Overall distribution
    print("\nOverall Label Distribution:")
    print(df['Labels'].value_counts(normalize=True))

    # Gender distribution
    print("\nLabel Distribution by Gender:")
    gender_dist = df.groupby(['gender', 'Labels']).size().unstack()
    gender_dist_pct = gender_dist.div(gender_dist.sum(axis=1), axis=0)
    print("\nCounts:")
    print(gender_dist)
    print("\nPercentages:")
    print(gender_dist_pct)

    # Ethnicity distribution
    print("\nLabel Distribution by Ethnicity:")
    ethnicity_dist = df.groupby(['ethnicity', 'Labels']).size().unstack()
    ethnicity_dist_pct = ethnicity_dist.div(ethnicity_dist.sum(axis=1), axis=0)
    print("\nCounts:")
    print(ethnicity_dist)
    print("\nPercentages:")
    print(ethnicity_dist_pct)

    # Insurance distribution
    print("\nLabel Distribution by Insurance:")
    insurance_dist = df.groupby(['insurance', 'Labels']).size().unstack()
    insurance_dist_pct = insurance_dist.div(insurance_dist.sum(axis=1), axis=0)
    print("\nCounts:")
    print(insurance_dist)
    print("\nPercentages:")
    print(insurance_dist_pct)

def create_balanced_batch(X, y, protected, batch_size):
    """Create balanced batch with stratification"""
    device = X.device

    # Get indices for positive and negative samples
    pos_idx = torch.where(y == 1)[0]
    neg_idx = torch.where(y == 0)[0]

    # Calculate sizes for balanced batch
    pos_size = batch_size // 2
    neg_size = batch_size - pos_size

    # Sample indices
    pos_batch_idx = pos_idx[torch.randperm(len(pos_idx))[:pos_size]]
    neg_batch_idx = neg_idx[torch.randperm(len(neg_idx))[:neg_size]]

    # Combine indices
    batch_idx = torch.cat([pos_batch_idx, neg_batch_idx])

    # Shuffle combined indices
    batch_idx = batch_idx[torch.randperm(len(batch_idx))]

    return (
        X[batch_idx].to(device),
        y[batch_idx].to(device),
        protected[batch_idx].to(device)
    )

# def calculate_aod(predictions, labels, protected_attributes):
#     """Calculate Average Odds Difference using PyTorch tensors"""
#     tpr_values = []
#     fpr_values = []
#     epsilon = 1e-8  # Small value to prevent division by zero
#
#     # For each category in protected attribute
#     for i in range(protected_attributes.shape[1]):
#         group_mask = (protected_attributes[:, i] == 1)
#
#         # True Positive Rate
#         positive_mask = (labels == 1)
#         group_positive_mask = group_mask & positive_mask
#         if positive_mask.sum() > 0:
#             tpr = (predictions[group_positive_mask].sum() + epsilon) / (positive_mask.sum() + epsilon)
#             tpr_values.append(tpr)
#
#         # False Positive Rate
#         negative_mask = (labels == 0)
#         group_negative_mask = group_mask & negative_mask
#         if negative_mask.sum() > 0:
#             fpr = (predictions[group_negative_mask].sum() + epsilon) / (negative_mask.sum() + epsilon)
#             fpr_values.append(fpr)
#
#     # Calculate AOD
#     if tpr_values and fpr_values:
#         tpr_values = torch.stack(tpr_values)
#         fpr_values = torch.stack(fpr_values)
#
#         tpr_diff = torch.max(tpr_values) - torch.min(tpr_values)
#         fpr_diff = torch.max(fpr_values) - torch.min(fpr_values)
#
#         aod = (tpr_diff + fpr_diff) / 2
#         return aod
#
#     return torch.tensor(0.0, device=predictions.device)

# def calculate_spd(predictions, protected_attributes):
#     """Calculate Statistical Parity Difference using PyTorch tensors"""
#     selection_rates = []
#     # For each category in protected attribute
#     for i in range(protected_attributes.shape[1]):
#         # Create mask for current group
#         group_mask = (protected_attributes[:, i] == 1)
#         if group_mask.sum() > 0:
#             # Calculate selection rate for this group
#             selection_rate = predictions[group_mask].mean()
#             selection_rates.append(selection_rate)
#
#     # Convert to tensor and calculate SPD
#     if selection_rates:
#         selection_rates = torch.stack(selection_rates)
#         spd = torch.max(selection_rates) - torch.min(selection_rates)
#         return spd
#     return torch.tensor(0.0, device=predictions.device)

def calculate_aod(y_true, y_pred, protected_attribute, epsilon=1e-8):
    # Convert everything to numpy arrays and ensure correct shape
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    protected_attribute = np.asarray(protected_attribute).flatten()

    group_0 = protected_attribute == 0
    group_1 = protected_attribute == 1

    # Calculate components
    tpr_0_num = np.sum((y_pred == 1) & (y_true == 1) & group_0)
    tpr_0_den = np.sum((y_true == 1) & group_0)
    tpr_0 = tpr_0_num / max(tpr_0_den, epsilon)

    tpr_1_num = np.sum((y_pred == 1) & (y_true == 1) & group_1)
    tpr_1_den = np.sum((y_true == 1) & group_1)
    tpr_1 = tpr_1_num / max(tpr_1_den, epsilon)

    fpr_0_num = np.sum((y_pred == 1) & (y_true == 0) & group_0)
    fpr_0_den = np.sum((y_true == 0) & group_0)
    fpr_0 = fpr_0_num / max(fpr_0_den, epsilon)

    fpr_1_num = np.sum((y_pred == 1) & (y_true == 0) & group_1)
    fpr_1_den = np.sum((y_true == 0) & group_1)
    fpr_1 = fpr_1_num / max(fpr_1_den, epsilon)

    # Add debug prints
    print(f"\nTPR group 0: {tpr_0:.4f} ({tpr_0_num}/{tpr_0_den})")
    print(f"TPR group 1: {tpr_1:.4f} ({tpr_1_num}/{tpr_1_den})")
    print(f"FPR group 0: {fpr_0:.4f} ({fpr_0_num}/{fpr_0_den})")
    print(f"FPR group 1: {fpr_1:.4f} ({fpr_1_num}/{fpr_1_den})")

    aod = abs((tpr_0 - tpr_1) + (fpr_0 - fpr_1)) / 2
    return aod

def calculate_spd(y_pred, protected_attribute):
    group_0 = y_pred[protected_attribute == 0]
    group_1 = y_pred[protected_attribute == 1]
    return abs(group_0.mean() - group_1.mean())

def calculate_inter_tpr(predictions, labels, protected_attributes, group_names=None):
    epsilon = 1e-8
    # Get unique combinations
    unique_combinations = torch.unique(protected_attributes, dim=0)
    results = []

    for combination in unique_combinations:
        intersectional_mask = torch.all(protected_attributes == combination, dim=1)
        group_size = intersectional_mask.sum().item()

        if group_size > 0:
            # Get positive and negative masks
            pos_mask = intersectional_mask & (labels == 1)
            neg_mask = intersectional_mask & (labels == 0)

            # Calculate metrics
            pos_count = pos_mask.sum().item()
            neg_count = neg_mask.sum().item()

            if pos_count > 0:
                tpr = (torch.sum(predictions[pos_mask]) / (pos_count + epsilon)).item()
            else:
                tpr = 0.0

            if neg_count > 0:
                fpr = (torch.sum(predictions[neg_mask]) / (neg_count + epsilon)).item()
            else:
                fpr = 0.0

            pred_rate = predictions[intersectional_mask].mean().item()

            # Format group description
            if group_names:
                group_desc = ", ".join([
                    f"{name}={val.item()}"
                    for name, val in zip(group_names, combination)
                    if val.item() == 1
                ])
            else:
                group_desc = str(combination.tolist())

            results.append({
                'group': group_desc,
                'size': group_size,
                'positive_samples': pos_count,
                'tpr': tpr,
                'fpr': fpr,
                'prediction_rate': pred_rate
            })
            return results


def calculate_intersectional_tpr(y_true, y_pred, protected_attributes, attribute_names=None):
    """
    Calculate and print TPR for each intersectional group (e.g., black female, white male, etc.)

    Args:
        y_true: True labels
        y_pred: Predicted labels
        protected_attributes: Matrix of protected attributes
        attribute_names: List of lists containing attribute names for each protected category
                        e.g., [['Male', 'Female'], ['White', 'Black', 'Asian', ...]]
    """
    epsilon = 1e-8
    tpr_values = {}

    # Get all unique combinations
    unique_combinations = torch.unique(protected_attributes, dim=0)

    # Create attribute name mappings if not provided
    if attribute_names is None:
        attribute_names = [
            ['Male', 'Female'],  # Gender
            ['White', 'Black', 'Asian', 'Other']  # Race - adjust based on your categories
        ]

    for combination in unique_combinations:
        # Create intersectional group mask
        mask = torch.all(protected_attributes == combination, dim=1)
        pos_mask = mask & (y_true == 1)

        if pos_mask.sum() > 0:
            tpr = (torch.sum((y_pred == 1)[pos_mask]) + epsilon) / (pos_mask.sum() + epsilon)

            # Create group name
            group_name = []
            for i, val in enumerate(combination):
                category_idx = val.item()
                if category_idx < len(attribute_names[i]):
                    group_name.append(attribute_names[i][category_idx])

            group_key = " ".join(group_name)
            tpr_values[group_key] = tpr.item()

            # Print detailed statistics
            print(f"\nGroup: {group_key}")
            print(f"Total positive samples: {pos_mask.sum().item()}")
            print(f"True Positives: {torch.sum((y_pred == 1)[pos_mask]).item()}")
            print(f"TPR: {tpr:.4f}")

    # Print TPR differences between all pairs
    print("\nTPR Differences between groups:")
    groups = list(tpr_values.keys())
    max_diff = 0
    max_diff_groups = None

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            diff = abs(tpr_values[groups[i]] - tpr_values[groups[j]])
            print(f"{groups[i]} vs {groups[j]}: {diff:.4f}")

            if diff > max_diff:
                max_diff = diff
                max_diff_groups = (groups[i], groups[j])

    print(f"\nMaximum TPR difference: {max_diff:.4f}")
    if max_diff_groups:
        print(f"Between groups: {max_diff_groups[0]} and {max_diff_groups[1]}")

    return max_diff, tpr_values

def calculate_initial_fairness_metrics(df):
    metrics = {}

    def calculate_metrics_for_attribute(df, attribute):
        # Get unique values for the attribute
        groups = df[attribute].unique()

        # Calculate SPD (Statistical Parity Difference)
        selection_rates = []
        for group in groups:
            group_mask = df[attribute] == group
            selection_rate = df[group_mask]['Labels'].mean()
            selection_rates.append(selection_rate)
        spd = max(selection_rates) - min(selection_rates)

        # Calculate AOD (Average Odds Difference)
        tpr_values = []
        fpr_values = []
        for group in groups:
            group_mask = df[attribute] == group

            # True Positive Rate
            positive_mask = df['Labels'] == 1
            group_positive_mask = group_mask & positive_mask
            if positive_mask.sum() > 0:
                tpr = (group_positive_mask & positive_mask).sum() / positive_mask.sum()
                tpr_values.append(tpr)

            # False Positive Rate
            negative_mask = df['Labels'] == 0
            group_negative_mask = group_mask & negative_mask
            if negative_mask.sum() > 0:
                fpr = (group_negative_mask & negative_mask).sum() / negative_mask.sum()
                fpr_values.append(fpr)

        # Calculate AOD
        tpr_diff = max(tpr_values) - min(tpr_values)
        fpr_diff = max(fpr_values) - min(fpr_values)
        aod = (tpr_diff + fpr_diff) / 2

        return {
            'spd': spd,
            'aod': aod,
            'tpr_diff': tpr_diff,
            'selection_rates': dict(zip(groups, selection_rates)),
            'tpr_values': dict(zip(groups, tpr_values)),
            'fpr_values': dict(zip(groups, fpr_values))
        }

    # Calculate metrics for each protected attribute
    metrics['gender'] = calculate_metrics_for_attribute(df, 'gender')
    metrics['race'] = calculate_metrics_for_attribute(df, 'race')

    # Calculate maximum TPR difference across all intersectional groups
    def get_intersectional_groups(row):
        return f"{row['gender']}_{row['race']}"

    df['intersectional_group'] = df.apply(get_intersectional_groups, axis=1)
    intersectional_groups = df['intersectional_group'].unique()

    intersectional_tpr_values = []
    for group in intersectional_groups:
        group_mask = df['intersectional_group'] == group
        positive_mask = df['Labels'] == 1
        group_positive_mask = group_mask & positive_mask

        if positive_mask.sum() > 0:
            tpr = (group_positive_mask & positive_mask).sum() / positive_mask.sum()
            intersectional_tpr_values.append(tpr)

    if intersectional_tpr_values:
        metrics['max_intersectional_tpr_diff'] = max(intersectional_tpr_values) - min(intersectional_tpr_values)
    else:
        metrics['max_intersectional_tpr_diff'] = 0.0

    # Add summary statistics
    metrics['summary'] = {
        'overall_label_distribution': df['Labels'].value_counts(normalize=True).to_dict(),
        'total_samples': len(df),
        'positive_rate': df['Labels'].mean()
    }

    return metrics

def calculate_fairness_losses(predictions, labels, protected_attributes):
    spd = calculate_spd(predictions, protected_attributes)
    aod = calculate_aod(predictions, labels,  protected_attributes)
    return spd,aod

def evaluate_model(model, test_loader, epoch):
    """Evaluate model performance with additional metrics"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_protected = []
    all_raw_predictions = []

    with torch.no_grad():
        for batch_data, batch_labels, batch_protected in test_loader:
            device = next(model.parameters()).device
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            batch_protected = batch_protected.to(device)

            outputs = model(batch_data, batch_labels, batch_protected)
            branch_preds = outputs['branch_preds']
            final_pred = torch.mean(branch_preds, dim=1)

            all_raw_predictions.extend(final_pred.cpu().numpy())
            pred = (final_pred > 0.5).float()
            all_predictions.extend(pred.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            all_protected.append(batch_protected.cpu().numpy())

    predictions = np.array(all_predictions)
    raw_predictions = np.array(all_raw_predictions)
    labels = np.array(all_labels)
    protected = np.vstack(all_protected)

    # Basic metrics
    tp = np.sum((predictions == 1) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))

    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # AUROC
    auroc = roc_auc_score(labels, raw_predictions)

    # Fairness metrics with safety checks
    spd_scores = []
    aod_scores = []

    for attr_idx in range(protected.shape[1]):
        group_0_mask = (protected[:, attr_idx] == 0)
        group_1_mask = (protected[:, attr_idx] == 1)

        # SPD calculation
        if np.any(group_0_mask) and np.any(group_1_mask):
            prob_pos_0 = np.mean(predictions[group_0_mask])
            prob_pos_1 = np.mean(predictions[group_1_mask])
            spd = abs(prob_pos_0 - prob_pos_1)
            spd_scores.append(spd)

        # AOD calculation
        pos_0 = np.sum(labels[group_0_mask] == 1)
        pos_1 = np.sum(labels[group_1_mask] == 1)
        neg_0 = np.sum(labels[group_0_mask] == 0)
        neg_1 = np.sum(labels[group_1_mask] == 0)

        if pos_0 > 0 and pos_1 > 0 and neg_0 > 0 and neg_1 > 0:
            tpr_0 = np.sum((predictions[group_0_mask] == 1) & (labels[group_0_mask] == 1)) / pos_0
            tpr_1 = np.sum((predictions[group_1_mask] == 1) & (labels[group_1_mask] == 1)) / pos_1
            fpr_0 = np.sum((predictions[group_0_mask] == 1) & (labels[group_0_mask] == 0)) / neg_0
            fpr_1 = np.sum((predictions[group_1_mask] == 1) & (labels[group_1_mask] == 0)) / neg_1
            aod = abs((tpr_0 - tpr_1) + (fpr_0 - fpr_1)) / 2
            aod_scores.append(aod)

    avg_spd = np.mean(spd_scores) if spd_scores else 0
    avg_aod = np.mean(aod_scores) if aod_scores else 0

    # Print metrics
    print("\nFinal Evaluation Metrics:")
    print(f"True Positives (TP): {tp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUROC Score: {auroc:.4f}")
    print(f"Average SPD Score: {avg_spd:.4f}")
    print(f"Average AOD Score: {avg_aod:.4f}")

    return {
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'accuracy': accuracy, 'auroc': auroc,
        'avg_spd': avg_spd, 'avg_aod': avg_aod
    }

def print_fairness_metrics(metrics):
    """
    Print the fairness metrics in a readable format.
    """
    print("\n=== Fairness Metrics Analysis ===\n")

    # Print summary statistics
    print("Overall Statistics:")
    print(f"Total Samples: {metrics['summary']['total_samples']}")
    print(f"Overall Positive Rate: {metrics['summary']['positive_rate']:.4f}")
    print("\nLabel Distribution:")
    for label, prop in metrics['summary']['overall_label_distribution'].items():
        print(f"  Class {label}: {prop:.4f}")

    # Print metrics for each protected attribute
    for attribute in ['gender', 'ethnicity', 'insurance']:
        print(f"\n{attribute.capitalize()} Metrics:")
        print(f"Statistical Parity Difference (SPD): {metrics[attribute]['spd']:.4f}")
        print(f"Average Odds Difference (AOD): {metrics[attribute]['aod']:.4f}")
        print(f"TPR Difference: {metrics[attribute]['tpr_diff']:.4f}")

        print(f"\nSelection Rates by {attribute}:")
        for group, rate in metrics[attribute]['selection_rates'].items():
            print(f"  {group}: {rate:.4f}")

        print(f"\nTrue Positive Rates by {attribute}:")
        for group, rate in metrics[attribute]['tpr_values'].items():
            print(f"  {group}: {rate:.4f}")

        print(f"\nFalse Positive Rates by {attribute}:")
        for group, rate in metrics[attribute]['fpr_values'].items():
            print(f"  {group}: {rate:.4f}")

    print("\nIntersectional Fairness:")
    print(f"Maximum TPR Difference across all intersectional groups: {metrics['max_intersectional_tpr_diff']:.4f}")

def print_expert_analysis(analysis_results):
    print("\n=== Expert Specialization Analysis ===")

    # Print overall statistics with validation
    print("\nOverall Expert Distribution:")
    total_experts = sum(analysis_results['total_stats'].values())
    if total_experts == 0:
        print("No experts have specialized yet")
    else:
        for metric, count in analysis_results['total_stats'].items():
            print(f"{metric.upper()} Experts: {count} ({count / total_experts * 100:.1f}%)")

    # Print branch-specific statistics
    print("\nDetailed Branch-Specific Distribution:")
    for branch, stats in analysis_results['branch_stats'].items():
        print(f"\n{branch.capitalize()} Branch:")
        branch_total = sum(stats.values())
        if branch_total == 0:
            print("  No experts have specialized in this branch")
        else:
            for metric, count in stats.items():
                if count > 0:  # Only show non-zero counts
                    print(f"  {metric.upper()} Experts: {count} ({count / 16 * 100:.1f}% of branch)")

    # Print average scores only if they're valid
    print("\nBranch Performance Scores:")
    for branch, metrics in analysis_results['avg_performance'].items():
        print(f"\n{branch.capitalize()} Branch:")
        for metric, score in metrics.items():
            if score != float('inf') and score != 0.0:
                print(f"  {metric.upper()} Score: {score:.4f}")
            else:
                print(f"  {metric.upper()} Score: Not yet stabilized")




