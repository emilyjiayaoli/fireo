import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, modality, cfg):
        super(Encoder, self).__init__()
        if modality == "rgb":
            in_channels = 3
        if modality == "ir":
            in_channels = 1
            
        hidden_dims = cfg.baseline_encoder_hidden_dims # list storing the out_channels at each layer in cnn encoder
        self.encoder = nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=hidden_dims[0], kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(hidden_dims[0]),
            torch.nn.GELU(),
            torch.nn.Conv2d(in_channels=hidden_dims[0], out_channels=hidden_dims[1], kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(hidden_dims[1]),
            torch.nn.GELU(),
            torch.nn.Conv2d(in_channels=hidden_dims[1], out_channels=hidden_dims[2], kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(hidden_dims[2]),
            torch.nn.GELU(),
            torch.nn.Conv2d(in_channels=hidden_dims[2], out_channels=hidden_dims[3], kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(hidden_dims[3]),
            torch.nn.GELU(),
            torch.nn.Conv2d(in_channels=hidden_dims[3], out_channels=hidden_dims[3], kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(hidden_dims[3]),
            torch.nn.GELU(),
        )

    def forward(self, x):
        local_var = "hi im a local var"
        z = self.encoder(x)
        
        return z
    
class MLP_Baseline(nn.Module):
    def __init__(self, cfg):
        super(MLP_Baseline, self).__init__()
        self.cfg = cfg
        self.rgb_encoder = Encoder("rgb", cfg)
        self.ir_encoder = Encoder("ir", cfg)

        with torch.no_grad():
            encoder_out = self.rgb_encoder(torch.zeros(self.cfg.batch_size, self.cfg.model_rgb_encoder_image_shape[0], self.cfg.model_rgb_encoder_image_shape[1], self.cfg.model_rgb_encoder_image_shape[2]))
            hidden_size = encoder_out.size()[-1] * encoder_out.size()[-2] * encoder_out.size()[-3]

        if cfg.model_param_size == '5m':
            self.projection = nn.Sequential(
                torch.nn.Linear(2 * hidden_size, cfg.baseline_mlp_hidden_size[2]),
                nn.GELU(),
                torch.nn.Linear(cfg.baseline_mlp_hidden_size[2], cfg.model_decoder_output_class_num),
                )


        elif cfg.model_param_size == '10m':
            self.projection = nn.Sequential(
                torch.nn.Linear(2 * hidden_size, cfg.baseline_mlp_hidden_size[0]),
                nn.GELU(),
                torch.nn.Linear(cfg.baseline_mlp_hidden_size[0], cfg.baseline_mlp_hidden_size[1]),
                nn.GELU(),
                torch.nn.Linear(cfg.baseline_mlp_hidden_size[1], cfg.baseline_mlp_hidden_size[2]),
                nn.GELU(),
                torch.nn.Linear(cfg.baseline_mlp_hidden_size[2], cfg.model_decoder_output_class_num),
                )   

    def forward(self, rgb_input, ir_input):
        
        rgb_z = self.rgb_encoder(rgb_input)
        rgb_z = rgb_z.view(rgb_z.size(0), -1)

        ir_z = self.ir_encoder(ir_input)
        ir_z = ir_z.view(ir_z.size(0), -1)

        z = torch.cat((rgb_z, ir_z), dim=-1)

        return self.projection(z)

