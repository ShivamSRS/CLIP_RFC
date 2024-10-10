import torch
import wandb
import datetime as dt
from tqdm import tqdm
import clip
import torch.nn as nn
import sklearn.metrics as metrics
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import torch.nn.functional as F


def print_metrices(y_true, y_pred, y_score, y_class_score):
    """
    print the metrices
    @param y_true: the true label
    @param y_pred: the predicted label
    @param y_score: the predicted score
    """
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)
    AUC = metrics.roc_auc_score(y_true, y_score)

    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)
    print("AUC: ", AUC)

    wandb.log({"Test Accuracy": accuracy, "Test Precision": precision,
              "Test Recall": recall, "Test F1": f1, "Test AUC": AUC})
    
    # following https://docs.wandb.ai/guides/track/log/plots
    wandb.log({"roc": wandb.plot.roc_curve(
        y_true, y_class_score, labels={"healthy lymph node tissue", "lymph node tumor tissue"})})

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 2048, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x  # Output shape: [N, 2048, 7, 7]



class CustomCLIP(nn.Module):
    def __init__(self, config, in_features, reduction=4):
        """
        @param config: config file for running CLIP + Residual Feature Connection
        @param in_features: the input feature size
        @param reduction: the reduction factor
        """
        super(CustomCLIP, self).__init__()
        
        # define the hyperparameters from config
        self.CLIP = config.CLIP
        self.scaler = config.scaler
        self.softmax = nn.Softmax(dim=1)
        self.epochs = config.epochs
        self.device = config.device
        self.loss_fn = config.loss_fn
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.batch_size = config.batch_size
        self.alpha = config.alpha
        self.percent_train = config.percent_training_set

        # define the dataset from config
        self.train_dataset = config.train_dataset
        self.test_dataset = config.test_dataset
        self.valid_dataset = config.valid_dataset

        # TODO add text input to match different dataset
        self.text_input = torch.cat([clip.tokenize(
            "this is a photo of healthy lymph node tissue"), clip.tokenize("this is a photo of lymph node tumor tissue")]).to(self.device)

        self.fc = nn.Sequential(
            nn.Linear(in_features, in_features // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_features // reduction, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, in_features // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_features // reduction, in_features, bias=False),
            nn.ReLU(),
        )

        # CNN-based autoencoder
        self.autoencoder = AutoEncoder()
        self.autoencoder.to(self.device)

        # Text projection layer to match channel dimension
        self.text_projection = nn.Linear(1024, 2048)
        self.text_projection.to(self.device)

        # Linear layers for learned mean and stddev
        self.pred_image_fc = nn.Linear(2048, 2)
        self.text_features_fc = nn.Linear(2048, 2)

    def extract_multidim_image_features(self,model, images):
        with torch.no_grad():
            x = images.type(model.visual.conv1.weight.dtype)

            # Stem
            x = model.visual.conv1(x)
            x = model.visual.bn1(x)
            x = model.visual.relu1(x)
            x = model.visual.conv2(x)
            x = model.visual.bn2(x)
            x = model.visual.relu2(x)
            x = model.visual.conv3(x)
            x = model.visual.bn3(x)
            x = model.visual.relu3(x)
            x = model.visual.avgpool(x)

            # ResNet layers
            x = model.visual.layer1(x)
            x = model.visual.layer2(x)
            x = model.visual.layer3(x)
            x = model.visual.layer4(x)

            # x is now of shape (N, C, H, W)
            return x
    
    def forward(self, image_input, label,train=True):
        N = image_input.size(0)  # Batch size
        M = self.text_input.size(0)  # Number of text classes

        with autocast():
            # Extract multidimensional image features from CLIP
            image_features_multi = self.extract_multidim_image_features(self.CLIP, image_input)  # Shape: [N, 2048, 7, 7]

            # Pass the multidimensional features through the autoencoder
            pred_image_features = self.autoencoder(image_features_multi)  # Shape: [N, 2048, 7, 7]

            # Retain gradients for pred_image_features to access grad later
            pred_image_features = pred_image_features / pred_image_features.norm(dim=1, keepdim=True)
            if train==True:
                pred_image_features.retain_grad()

            # Normalize features over the channel dimension
            
            image_features_multi = image_features_multi / image_features_multi.norm(dim=1, keepdim=True)

            # Combine features
            combined_features = self.alpha * pred_image_features + (1 - self.alpha) * image_features_multi  # Shape: [N, 2048, 7, 7]

            # Encode text features and project to match channel dimension
            with torch.no_grad():
                text_features = self.CLIP.encode_text(self.text_input)  # Shape: [M, 512]
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # Normalize

            text_features_projected = self.text_projection(text_features)  # Shape: [M, 2048]

            # Expand dimensions to match for broadcasting
            text_features_expanded = text_features_projected.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # Shape: [1, M, 2048, 1, 1]
            combined_features_expanded = combined_features.unsqueeze(1)  # Shape: [N, 1, 2048, 7, 7]

            # Compute similarity map over channel dimension
            similarity_map = (combined_features_expanded * text_features_expanded).sum(dim=2)  # Shape: [N, M, 7, 7]

            # Average over spatial dimensions to get logits
            logits = similarity_map.mean(dim=[2, 3])  # Shape: [N, M]

            # Apply logit scaling from CLIP
            logit_scale = self.CLIP.logit_scale.exp()
            logits = logit_scale * logits  # Shape: [N, M]

            # Compute the classification loss
            classification_loss = self.loss_fn(logits, label).to(self.device)

            # Now compute the learned mean and stddev for uncertainty estimation

            # For pred_image_features
            # Apply global average pooling over spatial dimensions
            pooled_pred_image_features = torch.mean(pred_image_features, dim=[2, 3])  # Shape: [N, 2048]

            # Pass through linear layer to get mean and log variance
            pred_image_params = self.pred_image_fc(pooled_pred_image_features)  # Shape: [N, 2]
            pred_image_mean = pred_image_params[:, 0]  # Shape: [N]
            pred_image_logvar = pred_image_params[:, 1]  # Shape: [N]

            # Use softplus to ensure positivity of variance
            pred_image_var = F.softplus(pred_image_logvar) + 1e-6  # Shape: [N]

            # Similarly for text_features_projected
            text_features_params = self.text_features_fc(text_features_projected)  # Shape: [M, 2]
            text_features_mean = text_features_params[:, 0]  # Shape: [M]
            text_features_logvar = text_features_params[:, 1]  # Shape: [M]
            text_features_var = F.softplus(text_features_logvar) + 1e-6  # Shape: [M]

            # Expand dimensions to compute pairwise KL divergence
            mu_p = pred_image_mean.unsqueeze(1)  # Shape: [N, 1]
            var_p = pred_image_var.unsqueeze(1)  # Shape: [N, 1]

            mu_q = text_features_mean.unsqueeze(0)  # Shape: [1, M]
            var_q = text_features_var.unsqueeze(0)  # Shape: [1, M]

            # Compute KL divergence between each pair
            kl_div = 0.5 * (
                (var_p / var_q) +
                ((mu_q - mu_p) ** 2) / var_q -
                1 +
                torch.log(var_q / var_p)
            )  # Shape: [N, M]

            kl_loss = kl_div.mean()

            # Combine losses
            beta = 1.0  # Adjust this hyperparameter as needed
            total_loss = classification_loss + beta * kl_loss

        return total_loss, logits, pred_image_features


    def save(self):
        date = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = '/cache/Shivam/clipadapter/CLIP_RFC/Pcam/src/model/checkpoints/:' + date + '.pt'
        torch.save(self.state_dict(), path)

    

    def train(self):
        self.autoencoder.train()
        self.text_projection.train()
        self.pred_image_fc.train()
        self.text_features_fc.train()

        # Freeze CLIP parameters
        # self.CLIP.eval()
        # for param in self.CLIP.parameters():
        #     param.requires_grad = False

        optimizer = torch.optim.Adam(
            list(self.autoencoder.parameters()) +
            list(self.text_projection.parameters()) +
            list(self.pred_image_fc.parameters()) +
            list(self.text_features_fc.parameters()),
            lr=self.lr, weight_decay=self.weight_decay
        )

        for epoch in range(self.epochs):
            running_loss = 0.0
            pred = []
            true = []
            for images, labels, index in tqdm(DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)):
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                with autocast():
                    total_loss, logits, pred_image_features = self.forward(images, labels)

                # First backward pass to compute gradients for pred_image_features
                self.scaler.scale(total_loss).backward(retain_graph=True)

                # Extract gradients to form F_att
                F_att = pred_image_features.grad.detach().abs()  # Shape: [N, 2048, 7, 7]

                # Normalize F_att
                F_att = F_att / F_att.sum(dim=[1, 2, 3], keepdim=True)

                # Compute optimal_transport via element-wise multiplication
                optimal_transport = pred_image_features * F_att  # Shape: [N, 2048, 7, 7]

                # Aggregate spatial dimensions
                pooled_optimal_transport = torch.mean(optimal_transport, dim=[2, 3])  # Shape: [N, 2048]

                # Normalize pooled_optimal_transport
                pooled_optimal_transport = pooled_optimal_transport / pooled_optimal_transport.norm(dim=-1, keepdim=True)

                # Compute similarities with text_features_projected
                text_features = self.CLIP.encode_text(self.text_input)  # Shape: [M, 512]
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # Normalize

                text_features_projected = self.text_projection(text_features)  # Shape: [M, 2048]

                similarities_ot = pooled_optimal_transport @ text_features_projected.T  # Shape: [N, M]

                # Compute logits_ot with CLIP's logit_scale
                logit_scale = self.CLIP.logit_scale.exp()
                logits_ot = logit_scale * similarities_ot  # Shape: [N, M]

                ot_loss = self.loss_fn(logits_ot, labels).to(self.device)

                # Combine ot_loss with total_loss
                gamma = 1.0  # Adjust as needed
                final_loss = total_loss + gamma * ot_loss

                # Zero gradients again before second backward pass
                optimizer.zero_grad()

                # Backward pass on the combined loss
                self.scaler.scale(final_loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

                # Logging and metrics computation
                val, predicted = torch.max(logits.data, 1)
                pred.extend(predicted.cpu().numpy())
                true.extend(labels.cpu().numpy())
                running_loss += final_loss.item()

                # Log training loss and accuracy to WandB
                wandb.log({
                    "Training loss - Step": final_loss.item(),
                    "Training accuracy - Step": metrics.accuracy_score(true, pred)
                })

            # Calculate epoch metrics
            epoch_loss = running_loss / len(self.train_dataset)
            epoch_accuracy = metrics.accuracy_score(true, pred)
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy}")
            wandb.log({"Training loss - Epoch": epoch_loss, "Training accuracy": epoch_accuracy})

        # Save the model at the end of training
        self.save()

    def test(self):
        self.autoencoder.eval()
        self.text_projection.eval()
        self.pred_image_fc.eval()
        self.text_features_fc.eval()
        # self.CLIP.eval()
        # for param in self.CLIP.parameters():
        #     param.requires_grad = False

        pred = []
        true = []
        score = []       # For ROC AUC metric, only record prob of positive class in binary classification
        class_score = [] # For ROC curve, record probabilities of both classes

        with torch.no_grad():
            for images, labels, index in tqdm(DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)):
                images = images.to(self.device)
                labels = labels.to(self.device)

                with autocast():
                    # Forward pass
                    total_loss, logits, _ = self.forward(images, labels,train=False)

                # Get predicted class
                _, predicted = torch.max(logits, 1)
                pred.extend(predicted.cpu().numpy())
                true.extend(labels.cpu().numpy())

                # Calculate probabilities for each class
                softmax_scores = self.softmax(logits)
                probs = softmax_scores.cpu().numpy()

                # For ROC AUC and ROC curve
                class_score.extend(probs.tolist())
                score.extend(probs[:, 1].tolist())  # Assuming class 1 is the positive class

                # Log test loss and accuracy to WandB
                wandb.log({
                    "Test loss - Step": total_loss.item(),
                    "Test accuracy - Step": metrics.accuracy_score(true, pred)
                })

        # Calculate and print metrics
        print_metrices(true, pred, score, class_score)
