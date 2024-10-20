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

from sklearn.decomposition import PCA


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
        
        # exit()
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
        self.varproject = nn.Linear(512,2048)
        # CNN-based autoencoder
        self.autoencoder = AutoEncoder()
        self.autoencoder.to(self.device)

        # Text projection layer to match channel dimension
        self.text_projection = nn.Linear(1024, 2048)
        self.text_projection.to(self.device)

        # Linear layers for learned mean and stddev
        # self.mean_image_fc = nn.Linear(2048, 2)
        # self.text_features_fc = nn.Linear(2048, 2)
        self.linear_mean_img = nn.Linear(1024, 512)  # Projection layer for mean
        self.linear_stddev_img = nn.Linear(1024, 512)  # Projection layer for stddev
        self.linear_mean_txt = nn.Linear(49, 512)  # Projection layer for mean
        self.linear_stddev_txt = nn.Linear(49, 512) 
        self.fit_pca_on_text_features()

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
    
    def extract_multidim_text_features(self,model, text_tokens):
        with torch.no_grad():
            # Get token embeddings
            x = model.token_embedding(text_tokens)  # Shape: [batch_size, seq_length, embed_dim]

            # Add positional embeddings
            x = x + model.positional_embedding[:x.size(1), :]  # Shape: [batch_size, seq_length, embed_dim]

            # Transpose for transformer: [batch_size, seq_length, embed_dim] -> [seq_length, batch_size, embed_dim]
            x = x.permute(1, 0, 2)

            # Pass through Transformer layers
            x = model.transformer(x)  # Shape: [seq_length, batch_size, embed_dim]

            # Transpose back: [batch_size, seq_length, embed_dim]
            x = x.permute(1, 0, 2)

            # Apply final LayerNorm
            x = model.ln_final(x)  # Shape: [batch_size, seq_length, embed_dim]

            # x is now the per-token embeddings after the Transformer
            return x  # Shape: [batch_size, seq_length, embed_dim]
    def fit_pca_on_text_features(self):
        with torch.no_grad():
            # Extract per-token embeddings for both text inputs
            text_features_multi = self.extract_multidim_text_features(self.CLIP, self.text_input)  # Shape: [2, 77, 1024]

            # Reshape to [154, 1024]
            M, seq_length, embed_dim = text_features_multi.shape  # M=2, seq_length=77, embed_dim=1024
            text_features_multi_flat = text_features_multi.view(-1, embed_dim).cpu().numpy()  # Shape: [154, 1024]

            # Fit PCA to reduce dimensionality to 49
            self.pca = PCA(n_components=49)
            self.pca.fit(text_features_multi_flat)

            # Transform the embeddings
            text_features_pca = self.pca.transform(text_features_multi_flat)  # Shape: [154, 49]
            text_features_pca = torch.from_numpy(text_features_pca).to(self.device)  # Convert back to tensor

            # Reshape back to [2, 77, 49]
            self.text_features_pca = text_features_pca.view(M, seq_length, -1)  # Shape: [2, 77, 49]

            # Normalize the embeddings
            self.text_features_pca = self.text_features_pca / self.text_features_pca.norm(dim=-1, keepdim=True)

    def loss_fn_selector(self,s_ij,labels,num_classes,batch_size,device,selection="CLIP"):
        if selection=="CLIP":
            # For the image-to-text direction
            labels_i2t = labels  # [batch_size]
            logits_i2t = s_ij    # [batch_size, num_classes]

            # For the text-to-image direction, we need to compute similarities between texts and images
            # Compute the similarity matrix s_ji (transpose of s_ij)
            s_ji = s_ij.transpose(0, 1)  # [num_classes, batch_size]

            # For each text (class), the positive images are those with matching labels
            # Create targets for text-to-image
            labels_t2i = []
            for class_idx in range(num_classes):
                # Get indices of images belonging to this class
                positive_indices = (labels == class_idx).nonzero(as_tuple=True)[0]
                labels_t2i.append(positive_indices)

            # Build logits and targets for text-to-image direction
            logits_t2i = s_ji  # [num_classes, batch_size]

            # Compute the loss in both directions
            loss_i2t = F.cross_entropy(logits_i2t, labels_i2t)
            loss_t2i = 0.0

            for class_idx in range(num_classes):
                if labels_t2i[class_idx].numel() > 0:
                    # Get logits for this class
                    logits = logits_t2i[class_idx]  # [batch_size]
                    # Create targets: positive indices are where labels == class_idx
                    targets = torch.zeros(batch_size, device=device, dtype=torch.float)
                    targets[labels == class_idx] = 1  # Positive examples
                    # print(logits.shape,type(logits),targets.shape,type(targets))
                    loss = F.cross_entropy(logits.unsqueeze(0), targets.unsqueeze(0))
                    loss_t2i += loss

            # Average the text-to-image loss over the number of classes with positive examples
            loss_t2i = loss_t2i / num_classes

            # Total loss
            # print(loss_t2i,loss_i2t)
            loss = (loss_i2t + loss_t2i) / 2.0
            return loss

    def uncertainty_aware_clip_loss(self,image_means, image_stddevs, text_means, text_stddevs, labels, temperature=0.07,selection="CLIP"):
        """
        Computes the Uncertainty-Aware CLIP loss adjusted for class-based text embeddings.

        Args:
            image_means (Tensor): Mean embeddings of images (batch_size x embedding_dim).
            image_stddevs (Tensor): Std devs of image embeddings (batch_size x embedding_dim).
            text_means (Tensor): Mean embeddings of texts (num_classes x token_len x embedding_dim).
            text_stddevs (Tensor): Std devs of text embeddings (num_classes x token_len x embedding_dim).
            labels (Tensor): Class labels of images (batch_size).
            temperature (float): Temperature parameter for scaling.

        Returns:
            loss (Tensor): The computed Uncertainty-Aware CLIP loss.
        """

        eps = 1e-8
        batch_size, embedding_dim = image_means.size()
        num_classes = text_means.size(0)

        device = image_means.device

        # Ensure stddevs are positive
        #squaring the variance to get stddeviation

        
        image_means = image_means / image_means.norm(dim=1, keepdim=True)
        image_stddevs = image_stddevs / image_stddevs.norm(dim=1, keepdim=True)
        # print(text_means.shape)
        text_means = text_means / text_means.norm(dim=1, keepdim=True)
        text_variances = text_stddevs / text_stddevs.norm(dim=1, keepdim=True)
        # print(text_means.shape)

        image_variances = image_stddevs#.pow(2) + eps     # [batch_size, embedding_dim]
        text_variances = text_stddevs#.pow(2) + eps # [num_classes, embedding_dim]

        # Expand dimensions for broadcasting
        image_means_exp = image_means.unsqueeze(1)       # [batch_size, 1, embedding_dim]
        text_means_exp = text_means.unsqueeze(0)  # [1, num_classes, embedding_dim]
        image_vars_exp = image_variances.unsqueeze(1)    # [batch_size, 1, embedding_dim]
        text_vars_exp = text_variances.unsqueeze(0)      # [1, num_classes, embedding_dim]
        # print(text_means.shape)
        # exit()
        # Sum of variances
        var_sum = image_vars_exp + text_vars_exp         # [batch_size, num_classes, embedding_dim]

        # Difference of means squared
        mean_diff_squared = (image_means_exp - text_means_exp).pow(2)  # [batch_size, num_classes, embedding_dim]

        # Compute the probabilistic similarity matrix s_ij
        new_pi = 2 * torch.tensor(torch.pi) 
        const = 0.5 * embedding_dim * torch.log(new_pi)
        s_ij = -0.5 * torch.sum(mean_diff_squared / var_sum + torch.log(var_sum), dim=2) - const  # [batch_size, num_classes]

        # Scale similarities by temperature
        # s_ij = s_ij / temperature
        # # Apply logit scaling from CLIP
        logit_scale = self.CLIP.logit_scale.exp()
        s_ij = logit_scale * s_ij 

        # Compute the cross-entropy loss
        if selection=="CLIP":
            loss = self.loss_fn_selector(s_ij,labels,num_classes,batch_size,device)
        else:
            loss = F.cross_entropy(s_ij, labels)
        # print(loss)
        # exit()

        return loss,s_ij


    def compute_squared_euclidean_distances(self,X, S):
        """
        Computes the squared Euclidean distances between all pairs of vectors from X and S.
        
        Args:
            X (torch.Tensor): A tensor of shape [batch_size_X, N, d] where each vector X_i has dimension d.
            S (torch.Tensor): A tensor of shape [batch_size_S, M, d] where each vector S_j has dimension d.
        
        Returns:
            torch.Tensor: A tensor of shape [batch_size_X, N, M] containing the squared distances between all pairs X_i and S_j.
        """
        # Get the batch sizes
        batch_size_X = X.shape[0]
        batch_size_S = S.shape[0]

        # Compute how many times to repeat S along the batch dimension to match X
        repeats = batch_size_X // batch_size_S

        # If X's batch size is not divisible by S's batch size, increase the repeat count
        if batch_size_X % batch_size_S != 0:
            repeats += 1

        # Repeat S along the batch dimension to match X
        S_expanded = S.repeat(repeats, 1, 1)

        # Truncate S_expanded if it's larger than X's batch size
        S_expanded = S_expanded[:batch_size_X, :, :]  # Shape: (batch_size_X, M, d)

        # Compute squared norms of X and S
        X_norm = (X ** 2).sum(dim=2, keepdim=True)  # Shape: (batch_size_X, N, 1)
        S_norm = (S_expanded ** 2).sum(dim=2, keepdim=True)  # Shape: (batch_size_X, M, 1)

        # Compute the cross-term using batch matrix multiplication
        cross_term = torch.bmm(X, S_expanded.transpose(1, 2))  # Shape: (batch_size_X, N, M)

        # Compute the squared Euclidean distances
        C = X_norm + S_norm.transpose(1, 2) - 2 * cross_term  # Shape: (batch_size_X, N, M)

        return C


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
            

            # Normalize features over the channel dimension
            
            image_features_multi = image_features_multi / image_features_multi.norm(dim=1, keepdim=True)

            # Combine features
            combined_features = self.alpha * pred_image_features + (1 - self.alpha) * image_features_multi  # Shape: [N, 2048, 7, 7]
            combined_features = combined_features/combined_features.norm(dim=1,keepdim=True)
            if train==True:
                combined_features.retain_grad()

            # Encode text features and project to match channel dimension
            
            # Use the precomputed PCA-transformed text features
            text_features_pca = self.text_features_pca  # Shape: [2, 77, 49]
            
            
            combined_features_pooled = self.CLIP.visual.attnpool(combined_features)
            


            mu_img = self.linear_mean_img(combined_features_pooled)  # Shape: [N]
            # we predict the variance of each tensor 
            pred_image_logvar = self.linear_stddev_img(combined_features_pooled) # Shape: [N]
            var_img = F.softplus(pred_image_logvar) + 1e-6  

            mu_text = self.linear_mean_txt(text_features_pca)  # Shape: [N]
            # we predict the variance of each tensor 
            pred_text_logvar = self.linear_stddev_txt(text_features_pca) # Shape: [N]
            var_text = F.softplus(pred_text_logvar) + 1e-6  
            
            mu_text = mu_text.mean(dim=1)       # [num_classes, embedding_dim]
            var_text = var_text.mean(dim=1) 

            
            
            unc_clip_loss,logits = self.uncertainty_aware_clip_loss(mu_img,var_img,mu_text,var_text,label)
            
            
            
            

             # Shape: [N, M]


            beta = 0.8  # Adjust this hyperparameter as needed
            clip_probsim_loss = beta * unc_clip_loss

        return clip_probsim_loss, logits, combined_features,var_img


    def save(self):
        date = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = '/cache/Shivam/clipadapter/CLIP_RFC/Pcam/src/model/checkpoints/Shivam:' + date + '.pt'
        torch.save(self.state_dict(), path)

    import torch

    def compute_T(self,X, S):
        """
        Computes the matrix T of shape [Channels, D], where Channels is 2048 and D is 77,
        via an einsum multiplication of X and S.

        Args:
            X (torch.Tensor): Tensor of shape [batch_size_X, Channels, d]
            S (torch.Tensor): Tensor of shape [batch_size_S, D, d]

        Returns:
            T (torch.Tensor): Tensor of shape [Channels, D]
        """
        batch_size_X, Channels, d = X.shape
        batch_size_S, D, _ = S.shape

        # Compute repeats needed to expand S to match batch_size_X
        repeats = batch_size_X // batch_size_S
        if batch_size_X % batch_size_S != 0:
            repeats += 1

        # Repeat S along the batch dimension
        S_expanded = S.repeat(repeats, 1, 1)  # Shape: (batch_size_S * repeats, D, d)
        S_expanded = S_expanded[:batch_size_X, :, :]  # Truncate to match batch_size_X

        # Now compute T using einsum
        # Sum over batch dimension and feature dimension k
        T = torch.einsum('bik,bjk->ij', X, S_expanded)  # Resulting shape: [Channels, D]

        return T

    def compute_C(self,X, S):
        """
        Computes the cost matrix C of shape [Channels, D], where Channels is 2048 and D is 77.

        Args:
            X (torch.Tensor): Tensor of shape [batch_size_X, Channels, d]
            S (torch.Tensor): Tensor of shape [batch_size_S, D, d]

        Returns:
            C (torch.Tensor): Tensor of shape [Channels, D]
        """
        # Average over batch dimensions
        
        X_mean = X.mean(dim=0)  # Shape: [Channels, d]
        S_mean = S.mean(dim=0)  # Shape: [D, d]
        X_mean = torch.nn.functional.normalize(X_mean, p=2, dim=1)
        S_mean = torch.nn.functional.normalize(S_mean, p=2, dim=1)


        # Compute squared norms
        
        X_norm = (X_mean ** 2).sum(dim=1).unsqueeze(1)  # Shape: [Channels, 1]
        S_norm = (S_mean ** 2).sum(dim=1).unsqueeze(0)  # Shape: [1, D]

        # Compute cross-term
        cross_term = torch.mm(X_mean, S_mean.t()) / (X_mean.size(1))  # Shape: [Channels, D]

        # Compute cost matrix C
        
        
        C = X_norm + S_norm - 2 * cross_term  # Shape: [Channels, D]
        
        # exit()
        return C

    def compute_loss(self,T, C, lambda_param):
        """
        Computes the loss function.

        Args:
            T (torch.Tensor): Transport plan of shape [Channels, D].
            C (torch.Tensor): Cost matrix of shape [Channels, D].
            lambda_param (float): Regularization parameter.

        Returns:
            loss (torch.Tensor): Scalar loss value.
        """
        # Avoid log(0) by adding a small epsilon
        epsilon = 1e-8
        # T_safe = T + epsilon

        # Compute the entropy H(T)
        # print(T,torch.log(F.relu(T)+epsilon) )
        # exit()
        relu_T = F.relu(T)
        relu_T = torch.clamp(relu_T, min=1e-6)  # Add a small clamp to avoid very small values
        product = T * torch.log(relu_T + epsilon)

        # product = T * (torch.log(F.relu(T)+epsilon))
        
        product = product/(product.norm(dim=1,keepdim=True) + 1e-6 )
        # product = torch.clamp(product, min=-1e6, max=1e6)  # Clamping to avoid extreme values
        H_T = - (product - 1).sum()
        # print(H_T)
        # exit()
        
        # exit()
        # Compute the loss L = sum_{i,j} T_{ij} * C_{ij} - lambda * H(T)
        loss = (T * C).sum() - lambda_param * H_T
        return loss

    def train(self):
        self.autoencoder.train()
        # self.text_projection.train()
        self.linear_mean_img.train()  # Projection layer for mean
        self.linear_stddev_img.train() 
        self.linear_mean_txt.train()  # Projection layer for mean
        self.linear_stddev_txt.train() 

        # Freeze CLIP parameters
        # self.CLIP.eval()
        # for param in self.CLIP.parameters():
        #     param.requires_grad = False

        optimizer = torch.optim.Adam(
            list(self.autoencoder.parameters()) +
            # list(self.text_projection.parameters()) +
            list(self.linear_mean_img.parameters()) +
            list(self.linear_stddev_img.parameters()) +
            list(self.linear_mean_txt.parameters()) +
            list(self.linear_stddev_txt.parameters()) ,
            # list(self.text_features_fc.parameters()),
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
                    total_loss, logits, combined_features,var_img = self.forward(images, labels)

                # First backward pass to compute gradients for pred_image_features
                self.scaler.scale(total_loss).backward(retain_graph=True)
                
                # exit()
                # torch.nn.utils.clip_grad_norm_(optimizer.parametess(), 5)
                gradget = combined_features.grad.detach().abs()
                gradget=gradget.clamp(min=-5,max=5)
                # Extract gradients to form F_att
                if torch.isnan(gradget).any():
                    print("NaNs detected in combined_features gradient")
                    exit()
                F_att = -gradget # Shape: [N, 2048, 7, 7]
                
                # Normalize F_att
                F_att = F_att.clamp(min=1e-6,max=1e6)
                F_att = F_att / F_att.sum(dim=[1, 2, 3], keepdim=True) + 1e-6 
                # F_att = F_att.clamp(min=1e-6,max=1e6)

                # print("cmbf",combined_features)#x,"\n\n\F\n",F_att)
                if torch.isnan(F_att).any():
                    print("NaNs detected in F_att ")
                    exit()
                if torch.isnan(combined_features).any():
                    print("NaNs detected in combined_features")
                    exit()
                # if torch.isnan(combined_features).any():
                #     print("NaNs detected in combined_features")

                # exit()
                # Compute optimal_transport via element-wise multiplication
                # combined_features = torch.clamp(combined_features, max=1e6)
                combined_features_weighted_ = combined_features * F_att + 1e6 # Shape: [N, 2048, 7, 7]
                combined_features_weighted_ = torch.clamp(combined_features_weighted_,min=1e-6, max=1e6)
                combined_features_weighted_ = combined_features_weighted_/combined_features_weighted_.norm(dim=1,keepdim=True)
                
                
                    
                
                # print(combined_features_weighted)
                # exit()
                c =100
                combined_features_weighted_ai = F.relu(combined_features_weighted_) + c*F.relu(-combined_features_weighted_)
                var_img2 = self.varproject(var_img)
                print(F.softmax(combined_features_weighted_ai).shape,combined_features_weighted_ai.shape,(1-var_img2).shape)
                w_i = torch.einsum("ab,abcd->abcd", (1-var_img2),F.softmax(combined_features_weighted_ai))
                combined_features_weighted = w_i*combined_features
                print(combined_features_weighted.shape)
                if torch.isnan(combined_features_weighted).any():
                    print("NaNs detected in combined_features weight")
                    exit()
                # exit()
                combined_features_weighted = combined_features_weighted.reshape((combined_features_weighted.shape[0],combined_features_weighted.shape[1],combined_features_weighted.shape[2]*combined_features_weighted.shape[3]))
                



                C_cost = self.compute_C(combined_features_weighted, self.text_features_pca)#self.compute_squared_euclidean_distances(combined_features_weighted,self.text_features_pca )#torch.sum((combined_features_weighted_exp - self.text_features_pca_exp ) ** 2, dim=-1) 
                



                OT = self.compute_T(combined_features_weighted, self.text_features_pca)
                OT = OT/OT.norm(dim=1,keepdim=True)
                print(C_cost.shape,OT.shape)
                # exit()

                # OT = torch.einsum("abc,adc->abd",combined_features_weighted,self.text_features_pca )
                # Compute logits_ot with CLIP's logit_scale
                lamba=0.9
                OT_loss = self.compute_loss(OT, C_cost, 0.9)
                # print("Optimal transport and loss",OT.shape,total_loss,OT_loss)
                # print(OT)


                if torch.isnan(C_cost).any():
                    print("NaNs detected in C_cost ")
                    exit()
                if torch.isnan(OT).any():
                    print("NaNs detected in OT ")
                    exit()
                if torch.isnan(OT_loss).any():
                    print("NaNs detected in OT loss ")
                    exit()
                
                
                # 222wsssseszxcd

                # Combine ot_loss with total_loss
                gamma = 0.8  # Adjust as needed
                final_loss = total_loss + gamma * OT_loss

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
                # print("running loss is",running_loss)
                # exit()
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
        # self.text_projection.eval()
        self.linear_mean_img.eval()  # Projection layer for mean
        self.linear_stddev_img.eval() 
        self.linear_mean_txt.eval()  # Projection layer for mean
        self.linear_stddev_txt.eval()
        # self.CLIP.eval()
        # for param in self.CLIP.parameters():
        #     param.requires_grad = False

        pred = []
        true = []
        score = []       # For ROC AUC metric, only record prob of positive class in binary classification
        class_score = [] # For ROC curve, record probabilities of both classes

        with torch.no_grad():
            print(len(self.test_dataset))
            # exit()
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
