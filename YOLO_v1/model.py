import torch
import pandas as pd


class CNNBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = torch.nn.BatchNorm2d(out_channels)
        self.leakyrelu = torch.nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))
    

class Yolov1(torch.nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture_path = 'Darknet_architecture.csv'
        self.architecture = self._call_architecture_as_lst()
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def _call_architecture_as_lst(self, head_label=False):
        df = pd.read_csv(self.architecture_path)
        df['Filter Number'] = df['Filter Number'].fillna(0).astype('Int64')
        architecture_arr = df.to_numpy()
        if head_label:
            return architecture_arr, list(df.columns)
        return architecture_arr
    
    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for a_layer in architecture:
            if a_layer[0] == 'Conv':
                layers += [CNNBlock(in_channels, a_layer[2], kernel_size=a_layer[1],
                                    stride=a_layer[3], padding=a_layer[4],)]
                in_channels = a_layer[2]
            elif a_layer[0] == 'Max_Pool':
                layers += [torch.nn.MaxPool2d(kernel_size=(a_layer[1], a_layer[1]), 
                                              stride=(a_layer[3], a_layer[3]))]
        return torch.nn.Sequential(*layers)
    
    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(1024 * S * S, 496), # Original paper this should be 4096
            torch.nn.Dropout(0.0),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(496, S * S * (C + B *5)), # (S, S, 30) where C + B * 5 = 30
        )
                
    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))
    

def test(S=7, B=2, C=20):
    model = Yolov1(split_size=S, num_boxes=B, num_classes=C)
    print('Model: \n', model)
    x = torch.rand((2, 448, 448, 3)).permute(0, 3, 1, 2) # torch (batch_size, channel, height, width)
    print(model(x).shape) # x was passed through the forward function


if __name__ == '__main__':
    test()



