import torch
import torch.nn as nn



class CNN_LSTM_Model(nn.Module):
    def __init__(self, input_channels, output_features, num_layers, hidden_size, output_size):
        super(CNN_LSTM_Model, self).__init__()

        # 1. Definir la Red Neuronal Convolucional (CNN)
        # Esto será el extractor de características para cada fotograma (paso de tiempo).
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(),

            nn.Conv2d(8, output_features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)


            # La salida de esto será (Batch_Size, output_features, H', W')
        )
        self.dropout = nn.Dropout(0.25) # Define dropout layer

        #print("fin de conv2d")
        # Calcular el tamaño del vector de características aplanado
        # (Necesitarás hacer un forward pass con un tensor simulado para obtener el tamaño exacto
        # si las dimensiones de entrada H y W son variables).
        # Por simplicidad, asumiremos que H' * W' es 'feature_dim'
        feature_dim = output_features * 32 * 32  # Ajusta estos números según tu entrada real (H' x W')

        # 2. Definir la capa LSTM
        self.lstm = nn.LSTM(
            input_size = feature_dim, # La entrada es el vector aplanado de la CNN
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True        # El formato de entrada será (B, T, F)
        )

        # 3. Definir la capa de Salida
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x.shape es (B, T, C, H, W)
        B, T, C, H, W = x.size()
        #print("x size",x.shape)
        # Reestructurar: Combinar Batch y Tiempo para aplicar la CNN
        # Ahora x.shape es (B * T, C, H, W)
        c_in = x.view(B * T, C, H, W)

        # Aplicar la CNN
        c_out = self.cnn(c_in)
        # c_out.shape es (B * T, C', H', W')

        #c_out = self.dropout(c_in)

        # Aplanar la salida de la CNN para que sea un vector (F)
        # Ahora c_out.shape es (B * T, Feature_Dim)
        r_in = c_out.view(B * T, -1)

        # Reestructurar de nuevo: Separar Batch y Tiempo para la LSTM
        # Ahora r_in.shape es (B, T, Feature_Dim)
        r_in = r_in.view(B, T, -1)

        # Aplicar la LSTM
        r_out, (h_n, h_c) = self.lstm(r_in)
        # r_out.shape es (B, T, Hidden_Size) si return_sequences=True (por defecto)

        # Usar solo la salida del último paso de tiempo para la clasificación/regresión
        # Ahora r_out.shape es (B, Hidden_Size)
        r_out = r_out[:, -1, :]


        # Aplicar la capa de Salida
        output = self.fc(r_out)
        return output #r_out

