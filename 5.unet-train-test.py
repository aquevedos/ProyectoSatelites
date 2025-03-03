import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms import CenterCrop
from torch.nn import ModuleList
from torch.nn import functional as F
import matplotlib.pyplot as plt

import rasterio
import numpy as np
import os
import time
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split


class Block(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(Block, self).__init__()
        # convolution and RELU layers
        self.conv1 = nn.Conv2d(inChannels, outChannels, kernel_size=3)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outChannels, outChannels, kernel_size=3)

    def forward(self, x):
        # apply CONV => RELU => CONV block to the inputs and return it
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        return x
    
class Encoder(nn.Module):
    def __init__(self, channels=(3, 16, 32, 64)):
            super().__init__()
            # create the encoder blocks and maxpooling layer
            self.encBlocks = nn.ModuleList([
               Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)
            ])
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # initialize an empty list to store the intermediate outputs
        blockOutputs = []
        # loop through the encoder blocks and update the blockOutputs list
        for block in self.encBlocks:
            x = block(x)
            blockOutputs.append(x)
            x = self.pool(x)

        return blockOutputs

class Decoder(nn.Module):
    def __init__(self, channels=(64, 32, 16)):
          super().__init__()
          # initialize the number of channels, upsampler blocks, and decoder blocks
          self.upconvs = nn.ModuleList([
              nn.ConvTranspose2d(channels[i], channels[i+1], kernel_size=2, stride=2)
              for i in range(len(channels)-1)
          ])
          self.dec_blocks = nn.ModuleList([
              Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)
          ])

    def forward(self, x, encFeatures):
      # loop through the number of channels
        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)
            # Crop the current features from the encoder blocks,
            # concatenate them with the current upsampled features,
            # and pass the concatenated output through the current
            # decoder block
            encFeat = self.crop(encFeatures[i], x)
            x = torch.cat([x, encFeat], dim=1)
            x = self.dec_blocks[i](x)

        # return the final decoder output
        return x

    def crop(self, encFeatures, x):
          # grab the dimensions of the inputs, and crop the encoder
          # features to match the dimensions
          (_, _, H, W) = x.shape
          encFeatures = CenterCrop([H, W])(encFeatures)
          # return the cropped features
          return encFeatures

class UNet(nn.Module):
    def __init__(self, encChannels=(3, 16, 32, 64),
          decChannels=(64, 32, 16),
          nbClasses=12, retainDim=True,
          outSize=(1000, 1000)):

        super().__init__()
        # initialize the encoder and decoder
        self.encoder = Encoder(encChannels)
        self.decoder = Decoder(decChannels)

        # initialize the regression head and store the class variables
        self.head = nn.Conv2d(decChannels[-1], nbClasses, kernel_size=1)
        self.retainDim = retainDim
        self.outSize = outSize

    def forward(self, x):
        # grab the features from the encoder
        encFeatures = self.encoder(x)

        # pass the encoder features through decoder making sure that
        # their dimensions are suited for concatenation
        decFeatures = self.decoder(encFeatures[::-1][0], encFeatures[::-1][1::])

        # pass the decoder features through the regression head to
        # obtain the segmentation mask
        map = self.head(decFeatures)

        # check to see if we are retaining the original output
        # dimensions and if so, then resize the output to match them
        if self.retainDim:
          map = F.interpolate(map, self.outSize)

        # return the segmentation map
        return map


class GeotiffDataset(Dataset):
    # Cal que la imatge i la màscara tinguin el mateix nom
    def __init__(self, imagePaths, maskPaths, transforms = None):
        
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.transforms = transforms

    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.imagePaths)
    
    def check_image_mask_alignment(self, image_path, mask_path):
        # Comprovo que la imatge i la mask siguin coherents
        if "august" in image_path:
            assert ("august" in image_path and "august" in mask_path), "🚨 No coincideix la mascara amb el tile!"
        else:
            assert ("april" in image_path and "april" in mask_path), "🚨 No coincideix la mascara amb el tile!"

        # Extract X and Y from the image filename
        filename = os.path.basename(image_path)
        parts = filename.split('_')
        iX = parts[-2]
        iY = parts[-1].split('.')[0]

        filename = os.path.basename(mask_path)
        parts = filename.split('_')
        mX = parts[-2]
        mY = parts[-1].split('.')[0]

        assert (iX == mX) and (iY == mY), "🚨 Les coordenades de la imatge i la mask no coincideixen!   "
    
    def __getitem__(self, idx):
        
        # Per a retornar la imatge correcta cal tenir present q tenim les de August i les de April
        # Son 386 per a cada mes pero la 1a de April i la 1a de August utilitzen la mateixa mascara
        # Per tant, cal aplicar el modul 386 per a obtenir la imatge correcta	

        imagePath = self.imagePaths[idx]
        maskPath = self.maskPaths[idx]

        self.check_image_mask_alignment(imagePath, maskPath)

        # Load the image from disk and read the associated mask 
        image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

        with rasterio.open(maskPath) as mask_src:
            mask = mask_src.read(1)  # Read the first (and only) band
        
        # Ensure mask is an integer tensor
        mask = torch.tensor(mask, dtype=torch.long)  # class labels

        #print(f"Image shape: {image.shape}, Mask shape: {mask.shape}")

        # Ja estic passant una transformació de la imatge a tensor, sinó caldria fer-ho aquí
        image = self.transforms(image)

        return (image, mask)

def iou_score(preds, labels, num_classes=12):
    preds = torch.argmax(preds, dim=1)
    ious = []
    for cls in range(num_classes):
        pred_cls = preds == cls
        label_cls = labels == cls
        intersection = (pred_cls & label_cls).sum().float()
        union = (pred_cls | label_cls).sum().float()
        ious.append(intersection / (union + 1e-8))
    return torch.mean(torch.tensor(ious))

def dice_score(preds, labels, num_classes=12):
    preds = torch.argmax(preds, dim=1)
    dice = []
    for cls in range(num_classes):
        pred_cls = preds == cls
        label_cls = labels == cls
        intersection = (pred_cls & label_cls).sum().float()
        dice.append(2. * intersection / (pred_cls.sum().float() + label_cls.sum().float() + 1e-8))
    return torch.mean(torch.tensor(dice))


if __name__ == '__main__':

    torch.cuda.empty_cache() # Clean the Cuda cache

    NUM_CHANNELS = 3
    NUM_CLASSES = 12

    INIT_LR = 0.001
    NUM_EPOCHS = 20
    BATCH_SIZE = 4
    THRESHOLD = 0.5 # define threshold to filter weak predictions
    
    # define the input image dimensions
    INPUT_IMAGE_WIDTH = 1000
    INPUT_IMAGE_HEIGHT = 1000

    # define the path to the images and masks dataset
    #IMAGE_DATASET_PATH = r'D:\aidl_projecte\sentinel2april\tile_norm'
    IMAGE_DATASET_PATH = r'D:\aidl_projecte\101.tiles_for_training'
    MASK_DATASET_PATH = r'D:\aidl_projecte\100.masks_for_training'
    TEST_SPLIT = 0.15
    TEST_PATHS_LIST = os.path.sep.join([IMAGE_DATASET_PATH + r"\output", 'test_paths.txt'])

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PIN_MEMORY = True if DEVICE == "cuda" else False
    print(f"Using device {DEVICE} and Pinned memory {PIN_MEMORY}")

    image_paths = [
            os.path.join(IMAGE_DATASET_PATH, file)
            for file in os.listdir(IMAGE_DATASET_PATH)
            if file.lower().endswith('png') and os.path.isfile(os.path.join(IMAGE_DATASET_PATH, file))
    ]

    masks_path = [
            os.path.join(MASK_DATASET_PATH, file)
            for file in os.listdir(MASK_DATASET_PATH)
            if file.lower().endswith('tif') and os.path.isfile(os.path.join(MASK_DATASET_PATH, file))
    ]

    # Split training and testing splits using 85%
    split = train_test_split(image_paths, masks_path, test_size=TEST_SPLIT)
    (trainImages, testImages) = split[:2]
    (trainMasks, testMasks) = split[2:]

    print(f"Imatges per a training: {len(trainImages)} i per a testing: {len(testImages)} (haurien de ser 328*2 i 58*2)")
    print(f"Masks per a training : {len(trainMasks)} i per a testing: {len(testMasks)} haurien de ser 328* i 58*2")

    # write the testing image paths to disk so that we can use then
    # when evaluating/testing our model
    print("Guardo paths images tests per a mostrar imatges vs preds vs groundtruth...")
    f = open(TEST_PATHS_LIST, "w")
    f.write("\n".join(testImages))
    f.close()

    # Cal transformar imatge a Tensor
    trans = transforms.Compose([transforms.ToPILImage(),
            transforms.Resize((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)),
            transforms.ToTensor()])

    # Creació de datasets i dataloaders
    trainDS = GeotiffDataset(imagePaths=trainImages, maskPaths=trainMasks, transforms=trans)
    testDS = GeotiffDataset(imagePaths=testImages, maskPaths=testMasks, transforms=trans)
    trainLoader = DataLoader(trainDS, shuffle=True, batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY, num_workers=0)
    testLoader = DataLoader(testDS, shuffle=False, batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY, num_workers=0)
    print(f"{len(trainDS)} examples in the training set...")
    print(f"{len(testDS)} examples in the test set...")

    # Initialize model, loss, and optimizer
    unet = UNet()
    unet = unet.to(DEVICE)

    #lossFunc = nn.BCEWithLogitsLoss() és clasificacion binaria
    lossFunc = nn.CrossEntropyLoss()
    opt = optim.Adam(unet.parameters(), lr=INIT_LR)

    # Càlcul de steps per epoch
    trainSteps = len(trainDS) // BATCH_SIZE
    testSteps = len(testDS) // BATCH_SIZE

    #Diccionari a on guardo evolució dels resultats de l'entrenament
    H = {"train_loss": [], "test_loss": [], "train_iou": [], "test_iou": [], "train_dice": [], "test_dice": []}

    # loop over epochs
    startTime = time.time()
    startTimeFormatted = time.strftime('%H:%M:%S', time.localtime(startTime))
    print(f"Inici del loop de training i testing: {startTimeFormatted}")

    for epoch in tqdm(range(NUM_EPOCHS)):
        unet.train()
        
        totalTrainLoss, totalTrainIoU, totalTrainDice = 0, 0, 0
        totalTestLoss, totalTestIoU, totalTestDice = 0, 0, 0
        
        for (i, (x, y)) in enumerate(trainLoader):
            
            (x, y) = (x.to(DEVICE), y.to(DEVICE))
            pred = unet(x)
            
            # Detectar problemes en la prediccio
            assert not (torch.isnan(pred).any() and torch.isinf(pred).any()), "🚨 Problemes en la predicció"
            
            # El dataset té clases de 1 a 12 no de 0 a 11, com espera CrossEntropyLoss. 
            # Per tant, cal restar 1 a las clases para que estén en el rango
            y = y -1 

            # També he trobat valores de classe > 12, q donaven errors CrossEntropyLoss i CUDA
            # Decideixo fer un clamp -> és una decissió discutible
            y = torch.clamp(y, min=0, max=11) #los valores menores de 0 los pongo a , y los mayores de 11 a 11
            assert torch.all((y >= 0) & (y <= 11)), "🚨 Found invalid class labels in y!"
            
            loss = lossFunc(pred, y)

            # Típic: eliminar gradients anterios i backpropagation
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Aquí acumulem la loss per a cada batch, per a calcular la loss mitjana
            totalTrainLoss += loss 
            totalTrainIoU += iou_score(pred, y)
            totalTrainDice += dice_score(pred, y)

        # Fem testing
        with torch.no_grad():
            unet.eval()
            
            for (x, y) in testLoader:
                  (x, y) = (x.to(DEVICE), y.to(DEVICE))
                  pred = unet(x)
                  
                  # El mateix raonament que abans
                  y = y -1 
                  y = torch.clamp(y, min=0, max=11) #los valores menores de 0 los pongo a , y los mayores de 11 a 11
                  assert torch.all((y >= 0) & (y <= 11)), "🚨 Found invalid class labels in y!"
                  
                  totalTestLoss += lossFunc(pred, y) 
                  totalTestIoU += iou_score(pred, y)
                  totalTestDice += dice_score(pred, y)
                  
        H["train_loss"].append((totalTrainLoss / trainSteps).cpu().detach().numpy())
        H["test_loss"].append((totalTestLoss / testSteps).cpu().detach().numpy())
        H["train_iou"].append((totalTrainIoU / trainSteps).cpu().detach().numpy())
        H["test_iou"].append((totalTestIoU / testSteps).cpu().detach().numpy())
        H["train_dice"].append((totalTrainDice / trainSteps).cpu().detach().numpy())
        H["test_dice"].append((totalTestDice / testSteps).cpu().detach().numpy())
        
        print(f"Epoch: {epoch+1}/{NUM_EPOCHS}")
        print(f"Valors obtinguts -> Train Loss= {H['train_loss'][-1]:.3f}, Test Loss = {H['test_loss'][-1]:.3f}, Train IoU = {H['train_iou'][-1]:.3f}, Test IoU  = {H['test_iou'][-1]:.3f}, Train Dice= {H['train_dice'][-1]:.3f}, Test Dice = {H['test_dice'][-1]:.3f}")
            
        if (epoch +1) == NUM_EPOCHS:
            print(f"Temps total:{(time.time() - startTime):.2f}  segons, en una GPU RTX3050 8GB VRAM")

    
    #Guardo el grafic, i el model a disc per a utilizar-lo posteriorment
    current_time = time.strftime('%Y%m%d-%H%M%S')
    BASE_OUTPUT = os.path.join(r'D:\aidl_projecte\resultados', current_time)
    PLOT_PATH = os.path.join(BASE_OUTPUT, "resultados.png")
    if not os.path.exists(BASE_OUTPUT):
        os.makedirs(BASE_OUTPUT)
    MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_satelitales.pth")
    TEST_PATHS = os.path.join(BASE_OUTPUT, "test_paths.txt")
    
    plt.figure()
    plt.title("Evolució dels Params")
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["test_loss"], label="test_loss")
    plt.plot(H["train_iou"], label="train_iou")
    plt.plot(H["test_iou"], label="test_iou")
    plt.plot(H["train_dice"], label="train_dice")
    plt.plot(H["test_dice"], label="test_dice")
    plt.xlabel("Epoch #")
    plt.ylabel("Metrics")
    plt.legend()
    plt.show()
    plt.savefig(PLOT_PATH)
    
    torch.save(unet, MODEL_PATH)