import sys
import torch
import torchvision
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import HuberLoss

import albumentations as A
from albumentations.pytorch import ToTensorV2

class IrradianceModel(LightningModule):

    def __init__(self, d_input, d_output, eve_norm, model='efficientnet_b0', dp=0.75):
        super().__init__()
        self.eve_norm = eve_norm

        if model == 'efficientnet_b0':
            model = torchvision.models.efficientnet_b0(pretrained=True)
        elif model == 'efficientnet_b1': 
            model = torchvision.models.efficientnet_b1(pretrained=True)
        elif model == 'efficientnet_b2': 
            model = torchvision.models.efficientnet_b2(pretrained=True)
        elif model == 'efficientnet_b3':
            model = torchvision.models.efficientnet_b3(pretrained=True)
        elif model == 'efficientnet_b4': 
            model = torchvision.models.efficientnet_b4(pretrained=True)
        elif model == 'efficientnet_b5': 
            model = torchvision.models.efficientnet_b5(pretrained=True)
        elif model == 'efficientnet_b6': 
            model = torchvision.models.efficientnet_b6(pretrained=True)
        elif model == 'efficientnet_b7': 
            model = torchvision.models.efficientnet_b7(pretrained=True)
        conv1_out = model.features[0][0].out_channels
        model.features[0][0] = nn.Conv2d(d_input, conv1_out, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        lin_in = model.classifier[1].in_features
        # consider adding average pool of full image(s)
        classifier = nn.Sequential(nn.Dropout(p=dp, inplace=True),
                                   nn.Linear(in_features=lin_in, out_features=d_output, bias=True))
        model.classifier = classifier
        # set all dropouts to 0.75
        # TODO: other dropout values?
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.p = dp
        self.model = model
        self.loss_func = HuberLoss() # consider MSE

    def forward(self, x):
        x = self.model(x)
        return x

    def forward_unnormalize(self, x):
        x = self.forward(x)
        return unnormalize(x, self.eve_norm)
        
    def training_step(self, batch, batch_nb):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_func(y_pred, y)

        y = unnormalize(y, self.eve_norm)
        y_pred = unnormalize(y_pred, self.eve_norm)

        epsilon = sys.float_info.min
        rae = torch.abs((y - y_pred) / (torch.abs(y) + epsilon)) * 100
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_RAE", rae.mean(), on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_func(y_pred, y)

        y = unnormalize(y, self.eve_norm) 
        y_pred = unnormalize(y_pred, self.eve_norm)

        #computing relative absolute error
        epsilon = sys.float_info.min
        rae = torch.abs((y - y_pred) / (torch.abs(y) + epsilon)) * 100
        av_rae = rae.mean()
        av_rae_wl = rae.mean(0)
        # compute average cross-correlation
        cc = torch.tensor([torch.corrcoef(torch.stack([y[i], y_pred[i]]))[0, 1] for i in range(y.shape[0])]).mean()
        # mean absolute error
        mae = torch.abs(y - y_pred).mean()

        self.log("valid_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_MAE", mae, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_RAE", av_rae, on_epoch=True, prog_bar=True, logger=True)
        [self.log(f"valid_RAE_{i}", err, on_epoch=True, prog_bar=True, logger=True) for i, err in enumerate(av_rae_wl)]
        self.log("valid_correlation_coefficient", cc, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_func(y_pred, y)

        y = unnormalize(y, self.eve_norm) 
        y_pred = unnormalize(y_pred, self.eve_norm)

        #computing relative absolute error
        epsilon = sys.float_info.min
        rae = torch.abs((y - y_pred) / (torch.abs(y) + epsilon)) * 100
        av_rae = rae.mean()
        av_rae_wl = rae.mean(0)
        # compute average cross-correlation
        cc = torch.tensor([torch.corrcoef(torch.stack([y[i], y_pred[i]]))[0, 1] for i in range(y.shape[0])]).mean()
        # mean absolute error
        mae = torch.abs(y - y_pred).mean()

        self.log("valid_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_MAE", mae, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_RAE", av_rae, on_epoch=True, prog_bar=True, logger=True)
        [self.log(f"valid_RAE_{i}", err, on_epoch=True, prog_bar=True, logger=True) for i, err in enumerate(av_rae_wl)]
        self.log("valid_correlation_coefficient", cc, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

def unnormalize(y, eve_norm):
    eve_norm = torch.tensor(eve_norm).float()
    norm_mean = eve_norm[0]
    norm_stdev = eve_norm[1]
    y = y * norm_stdev[None].to(y) + norm_mean[None].to(y)
    return y


class ChoppedAlexnetBN(LightningModule):

    # def __init__(self, numlayers, n_channels, outSize, dropout):
    def __init__(self, d_input, d_output, eve_norm, numLayers=3, dropout=0):
        super(ChoppedAlexnetBN, self).__init__()
        self.eve_norm = eve_norm
        self.numLayers = numLayers
        self.n_channels = d_input
        self.outSize = d_output
        self.loss_func = HuberLoss() # consider MSE

        layers, channelSize = self.getLayers(self.numLayers, self.n_channels)
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.model = nn.Sequential(nn.Dropout(p=dropout),
                                   nn.Linear(channelSize, self.outSize))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def getLayers(self, numLayers, n_channels):
        """Returns a list of layers + the feature size coming out"""
        layers = [nn.Conv2d(n_channels, 64, kernel_size=11, stride=4, padding=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True), ]
        if numLayers == 1:
            return (layers, 64)
        layers += [nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(64, 192, kernel_size=5, padding=2), nn.BatchNorm2d(192), nn.ReLU(inplace=True), ]
        if numLayers == 2:
            return (layers, 192)
        layers += [nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.BatchNorm2d(384), nn.ReLU(inplace=True)]
        if numLayers == 3:
            return (layers,384)

        layers += [nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(256)]
        return (layers,256)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).view(x.size(0),-1)
        x = self.model(x)
        return x
    
    def forward_unnormalize(self, x):
        x = self.forward(x)
        return unnormalize(x, self.eve_norm)
        
    def training_step(self, batch, batch_nb):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_func(y_pred, y)

        y = unnormalize(y, self.eve_norm)
        y_pred = unnormalize(y_pred, self.eve_norm)

        epsilon = sys.float_info.min
        rae = torch.abs((y - y_pred) / (torch.abs(y) + epsilon)) * 100
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_RAE", rae.mean(), on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_func(y_pred, y)

        y = unnormalize(y, self.eve_norm) 
        y_pred = unnormalize(y_pred, self.eve_norm)

        #computing relative absolute error
        epsilon = sys.float_info.min
        rae = torch.abs((y - y_pred) / (torch.abs(y) + epsilon)) * 100
        av_rae = rae.mean()
        av_rae_wl = rae.mean(0)
        # compute average cross-correlation
        cc = torch.tensor([torch.corrcoef(torch.stack([y[i], y_pred[i]]))[0, 1] for i in range(y.shape[0])]).mean()
        # mean absolute error
        mae = torch.abs(y - y_pred).mean()

        self.log("valid_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_MAE", mae, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_RAE", av_rae, on_epoch=True, prog_bar=True, logger=True)
        [self.log(f"valid_RAE_{i}", err, on_epoch=True, prog_bar=True, logger=True) for i, err in enumerate(av_rae_wl)]
        self.log("valid_correlation_coefficient", cc, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def test_step(self, batch, batch_nb):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_func(y_pred, y)

        y = unnormalize(y, self.eve_norm) 
        y_pred = unnormalize(y_pred, self.eve_norm)

        #computing relative absolute error
        epsilon = sys.float_info.min
        rae = torch.abs((y - y_pred) / (torch.abs(y) + epsilon)) * 100
        av_rae = rae.mean()
        av_rae_wl = rae.mean(0)
        # compute average cross-correlation
        cc = torch.tensor([torch.corrcoef(torch.stack([y[i], y_pred[i]]))[0, 1] for i in range(y.shape[0])]).mean()
        # mean absolute error
        mae = torch.abs(y - y_pred).mean()

        self.log("valid_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_MAE", mae, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_RAE", av_rae, on_epoch=True, prog_bar=True, logger=True)
        [self.log(f"valid_RAE_{i}", err, on_epoch=True, prog_bar=True, logger=True) for i, err in enumerate(av_rae_wl)]
        self.log("valid_correlation_coefficient", cc, on_epoch=True, prog_bar=True, logger=True)

    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
    

class LinearIrradianceModel(LightningModule):

    def __init__(self, d_input, d_output, eve_norm):
        super().__init__()
        self.eve_norm = eve_norm
        self.n_channels = d_input
        self.outSize = d_output        

        self.model = nn.Linear(2*self.n_channels, self.outSize)
        self.loss_func = HuberLoss() # consider MSE

    def forward(self, x):
        mean_irradiance = torch.torch.mean(x, dim=(2,3))
        std_irradiance = torch.torch.std(x, dim=(2,3))
        x = self.model(torch.cat((mean_irradiance, std_irradiance), dim=1))
        return x

    def forward_unnormalize(self, x):
        mean_irradiance = torch.torch.mean(x, dim=(2,3))
        std_irradiance = torch.torch.std(x, dim=(2,3))
        x = self.model(torch.cat((mean_irradiance, std_irradiance), dim=1))
        return unnormalize(x, self.eve_norm)
        
    def training_step(self, batch, batch_nb):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_func(y_pred, y)

        y = unnormalize(y, self.eve_norm)
        y_pred = unnormalize(y_pred, self.eve_norm)

        epsilon = sys.float_info.min
        rae = torch.abs((y - y_pred) / (torch.abs(y) + epsilon)) * 100
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_RAE", rae.mean(), on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_func(y_pred, y)

        y = unnormalize(y, self.eve_norm) 
        y_pred = unnormalize(y_pred, self.eve_norm)

        #computing relative absolute error
        epsilon = sys.float_info.min
        rae = torch.abs((y - y_pred) / (torch.abs(y) + epsilon)) * 100
        av_rae = rae.mean()
        av_rae_wl = rae.mean(0)
        # compute average cross-correlation
        cc = torch.tensor([torch.corrcoef(torch.stack([y[i], y_pred[i]]))[0, 1] for i in range(y.shape[0])]).mean()
        # mean absolute error
        mae = torch.abs(y - y_pred).mean()

        self.log("valid_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_MAE", mae, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_RAE", av_rae, on_epoch=True, prog_bar=True, logger=True)
        [self.log(f"valid_RAE_{i}", err, on_epoch=True, prog_bar=True, logger=True) for i, err in enumerate(av_rae_wl)]
        self.log("valid_correlation_coefficient", cc, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_func(y_pred, y)

        y = unnormalize(y, self.eve_norm) 
        y_pred = unnormalize(y_pred, self.eve_norm)

        #computing relative absolute error
        epsilon = sys.float_info.min
        rae = torch.abs((y - y_pred) / (torch.abs(y) + epsilon)) * 100
        av_rae = rae.mean()
        av_rae_wl = rae.mean(0)
        # compute average cross-correlation
        cc = torch.tensor([torch.corrcoef(torch.stack([y[i], y_pred[i]]))[0, 1] for i in range(y.shape[0])]).mean()
        # mean absolute error
        mae = torch.abs(y - y_pred).mean()

        self.log("valid_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_MAE", mae, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_RAE", av_rae, on_epoch=True, prog_bar=True, logger=True)
        [self.log(f"valid_RAE_{i}", err, on_epoch=True, prog_bar=True, logger=True) for i, err in enumerate(av_rae_wl)]
        self.log("valid_correlation_coefficient", cc, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-2)
    

class HybridIrradianceModel(LightningModule):

    def __init__(self, d_input, d_output, eve_norm, cnn_model='resnet', ln_model=True, ln_params=None, lr=1e-4, cnn_dp=0.75):
        super().__init__()
        self.eve_norm = eve_norm
        self.n_channels = d_input
        self.outSize = d_output
        self.ln_params = ln_params  
        self.lr = lr      

        # Linear model
        self.ln_model = None      
        if ln_model:
            self.ln_model = LinearIrradianceModel(d_input, d_output, eve_norm)
        if self.ln_params is not None:
            self.ln_model.weight = torch.nn.Parameter(self.ln_params['weight'])
            self.ln_model.bias = torch.nn.Parameter(self.ln_params['bias'])
        
        # CNN model
        efficientnets = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
                          'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7']
        self.cnn_model = None
        self.cnn_lambda = 1.
        if cnn_model == 'resnet':
            self.cnn_model = ChoppedAlexnetBN(d_input, d_output, eve_norm)
        elif cnn_model in efficientnets:
            self.cnn_model = IrradianceModel(d_input, d_output, eve_norm, model=cnn_model, dp=cnn_dp)

        # Error
        if self.ln_model is None and self.cnn_model is None:
            raise ValueError('Please pass at least one model.')

        # Loss function
        self.loss_func = HuberLoss() # consider MSE

    def forward(self, x):
        # Hybrid model
        if self.ln_model is not None and self.cnn_model is not None:
            return self.ln_model.forward(x) + self.cnn_lambda * self.cnn_model.forward(x)
        # Linear model only
        elif self.ln_model is not None:
            return self.ln_model.forward(x)
        # CNN model only
        elif self.cnn_model is not None:
            return self.cnn_model.forward(x)


    def forward_unnormalize(self, x):
        return unnormalize(self.forward(x), self.eve_norm)
        
        
    def training_step(self, batch, batch_nb):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_func(y_pred, y)

        y = unnormalize(y, self.eve_norm)
        y_pred = unnormalize(y_pred, self.eve_norm)

        epsilon = sys.float_info.min
        rae = torch.abs((y - y_pred) / (torch.abs(y) + epsilon)) * 100
        
        # Logging
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_RAE", rae.mean(), on_epoch=True, prog_bar=True, logger=True)
        self.log('lambda_cnn', self.cnn_lambda, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_func(y_pred, y)

        y = unnormalize(y, self.eve_norm) 
        y_pred = unnormalize(y_pred, self.eve_norm)

        #computing relative absolute error
        epsilon = sys.float_info.min
        rae = torch.abs((y - y_pred) / (torch.abs(y) + epsilon)) * 100
        av_rae = rae.mean()
        av_rae_wl = rae.mean(0)
        # compute average cross-correlation
        cc = torch.tensor([torch.corrcoef(torch.stack([y[i], y_pred[i]]))[0, 1] for i in range(y.shape[0])]).mean()
        # mean absolute error
        mae = torch.abs(y - y_pred).mean()

        # Logging
        self.log("valid_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_MAE", mae, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_RAE", av_rae, on_epoch=True, prog_bar=True, logger=True)
        [self.log(f"valid_RAE_{i}", err, on_epoch=True, prog_bar=True, logger=True) for i, err in enumerate(av_rae_wl)]
        self.log("valid_correlation_coefficient", cc, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_func(y_pred, y)

        y = unnormalize(y, self.eve_norm) 
        y_pred = unnormalize(y_pred, self.eve_norm)

        #computing relative absolute error
        epsilon = sys.float_info.min
        rae = torch.abs((y - y_pred) / (torch.abs(y) + epsilon)) * 100
        av_rae = rae.mean()
        av_rae_wl = rae.mean(0)
        # compute average cross-correlation
        cc = torch.tensor([torch.corrcoef(torch.stack([y[i], y_pred[i]]))[0, 1] for i in range(y.shape[0])]).mean()
        # mean absolute error
        mae = torch.abs(y - y_pred).mean()

        # Logging
        self.log("valid_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_MAE", mae, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_RAE", av_rae, on_epoch=True, prog_bar=True, logger=True)
        [self.log(f"valid_RAE_{i}", err, on_epoch=True, prog_bar=True, logger=True) for i, err in enumerate(av_rae_wl)]
        self.log("valid_correlation_coefficient", cc, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def set_train_mode(self, mode):
        if mode == 'linear':
            self.cnn_lambda = 0
            self.cnn_model.freeze()
            self.ln_model.unfreeze()
        elif mode == 'cnn':
            self.cnn_lambda = 0.01
            self.cnn_model.unfreeze()
            self.ln_model.freeze()
        elif mode == 'both':
            self.cnn_lambda = 0.01
            self.cnn_model.unfreeze()
            self.ln_model.unfreeze()
        else:
            raise NotImplemented(f'Mode not supported: {mode}')
    