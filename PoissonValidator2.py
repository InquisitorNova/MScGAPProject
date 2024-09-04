import torch
import numpy as np
from torchvision import transforms
import os
from PIL import Image
import matplotlib.pyplot as plt

# Define the class for Generating the NoisyFaces Dataset:

class ValidationSetGenerator(torch.utils.data.Dataset):
    def __init__(self, root_dir, out_dir, names, minPSNR, targetPSNR, augment = True, 
                 maxProb = 0.99, sample_size = None, grayscale = True, amplification_factor = 1.0,
                 ):
        super(ValidationSetGenerator, self).__init__()
        # Initialize the dataset:
        self.root_dir = root_dir
        self.out_dir = out_dir
        self.names = names
        self.minPSNR = minPSNR
        self.targetPSNR = targetPSNR
        self.augment = augment
        self.maxProb = maxProb
        self.sample_size = sample_size
        self.grayscale = grayscale
        self.amplification_factor = torch.FloatTensor([amplification_factor])
        self.Image_Paths = [os.path.join(self.root_dir, name) for name in self.names]
        self.length = len(self.Image_Paths)
        
        # Preprocessing Transformations
        self.crop = transforms.CenterCrop((200,200))
        self.resize = transforms.Resize((128,128))
        self.tensorize = transforms.PILToTensor()

        # Random Transformations
        self.flipH = transforms.RandomHorizontalFlip(p = 0.5)
        #self.flipV = transforms.RandomVerticalFlip(p = 0.5)
        #self.scale = transforms.RandomAffine(degrees = 0, translate = (0,0), scale = (1.0,1.3), shear = 0)
        #self.rotate = transforms.RandomAffine(degrees = 5, scale = (1,1), shear = 0)

    def __len__(self):
        if self.virtSize is not None:
            return self.virtSize
        else:
            return self.length
        
    def __GenerateNoisyImage__(self, idx, psnr_level):
        # Load the image:
        idx_ = idx
        image = Image.open(self.Image_Paths[idx_])
        if self.grayscale:
            image = image.convert('L')
        image = self.crop(image)
        image = self.resize(image)
        image = self.tensorize(image)
        
        # Apply augmentations:
        if self.augment:
            if torch.rand(1) < self.maxProb:
                image = self.flipH(image)
            #if torch.rand(1) < self.maxProb:
                #image = self.flipV(image)
            #if torch.rand(1) < self.maxProb:
                #image = self.scale(image)
            #if torch.rand(1) < self.maxProb:
                #image = self.rotate(image)
        
        # Generate the noisy image:
        image = image.type(torch.float32)
        image *= self.amplification_factor
        
        Image_Target = image/(torch.mean(image, dim =(-1,-2,-3), keepdims = True))
        Image_Target.type(torch.float32)

        uniform = psnr_level
        level = (10.0**(uniform/10.0))

        #print(image.dtype, self.amplification_factor.dtype)
        ImageNoise = torch.poisson((image/(torch.mean(image, dim =(-1,-2,-3), keepdims = True))) * level)
        ImageNoise = ImageNoise.type(torch.float32)

        psnr = torch.FloatTensor([uniform])
        psnr_map = psnr.unsqueeze(-1).unsqueeze(-1).expand(image.shape).type(torch.float32)

        #return image, image_input, ImageNoise, psnr_map
        return ImageNoise, psnr_map, Image_Target
        
    def Generate_Noisy_Faces(self, psnr_label):
        Noisy_Images = []
        Target_Images = []
        Psnr_Maps = []
        sample_size = self.sample_size

        uniform_range = np.linspace(start = self.minPSNR, stop = self.targetPSNR, num = sample_size)
        index = 0
        while index <= sample_size -1:
            psnr_level = uniform_range[index]
            ImageNoise, Psnr_Map, Image_Target = self.__GenerateNoisyImage__(index, psnr_level)
            Noisy_Images.append(ImageNoise)
            Psnr_Maps.append(Psnr_Map)
            Target_Images.append(Image_Target)
            index+=1

        Noisy_Images = torch.tensor(np.array(Noisy_Images)).type(torch.float32)
        Psnr_Maps = torch.tensor(np.array(Noisy_Images)).type(torch.float32)
        Target_Images = torch.tensor(np.array(Target_Images)).type(torch.float32)
        torch.save(Noisy_Images, self.out_dir + "\\" + f"{psnr_label}NI.pth")
        torch.save(Psnr_Maps, self.out_dir + "\\" + f"{psnr_label}PSM.pth")
        torch.save(Target_Images, self.out_dir +"\\" + f"{psnr_label}IT.pth")

        return self.out_dir + "\\" + f"{psnr_label}NI.pth", self.out_dir + "\\" +f"{psnr_label}PSM.pth", self.out_dir + "\\" + f"{psnr_label}IT.pth"

    def Load_Noisy_Faces(self, file_path_NI, file_path_PSM, file_path_IT, device):
        
        Noisy_Images = torch.load(file_path_NI)
        Psnr_Maps = torch.load(file_path_PSM)
        Target_images = torch.load(file_path_IT)

        return Noisy_Images.to(device), Psnr_Maps.to(device), Target_images.to(device)

   