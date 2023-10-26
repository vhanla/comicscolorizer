import os
from delphifmx import *
import torch
import numpy as np
import matplotlib.pyplot as plt
from tkinter.filedialog import askdirectory
from colorizator import MangaColorizator
import tempfile
import time
import cv2

CUDA_AVAILABLE = torch.cuda.device_count() > 0
colorizer = MangaColorizator('cuda', 'networks/generator.zip', 'networks/extractor.pth')

def pastel(input, factor=1.5):
    lab = cv2.cvtColor(input, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    mask = cv2.inRange(l, 0, 255)
    a = cv2.multiply(a, factor)
    b = cv2.multiply(b, factor)
    l = cv2.addWeighted(l, 1, a, 1, 0)
    l = cv2.addWeighted(l, 1, b, 1, 0)
    lab = cv2.merge((l, a, b))
    pim = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    pim = cv2.bitwise_and(pim, pim, mask=255-mask)
    pim += cv2.bitwise_and(input, input, mask=mask)
    return pim

class mainForm(Form):

    def __init__(self, owner):
        self.Panel1 = None
        self.StatusBar1 = None
        self.StyleBook1 = None
        self.Panel2 = None
        self.edInputDir = None
        self.btnDirPath = None
        self.ImageViewer1 = None
        self.ListBox1 = None
        self.chkGrayFirst = None
        self.Splitter1 = None
        self.Label1 = None
        self.Label2 = None
        self.edOutputDir = None
        self.btnOutPath = None
        self.ProgressBar1 = None
        self.chkBlendGray = None
        self.chkUpscale = None
        self.lblGPU = None
        self.btnColorizw = None
        self.ImageViewer2 = None
        self.ScrollBox1 = None
        self.LoadProps(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Unit1.pyfmx"))

        self.btnDirPath.OnClick = self.__dirClick
        self.ListBox1.OnClick = self.__listClick

        if CUDA_AVAILABLE:
            gpu_info = torch.cuda.get_device_properties(0)
            self.lblGPU.SetProps(width=400, AutoSize=True)
            self.lblGPU.Text = f"{gpu_info.name} {gpu_info.total_memory / 1024**3:.2f} GB"

        self.btnColorizw.OnClick = self.__colorize


    def __dirClick(self, sender):
        dir = askdirectory()
        self.edInputDir.Text = dir
        self.ListBox1.Items.Clear()
        for filename in os.listdir(dir):
            if filename.endswith((".jpg", ".jpeg", ".png", ".webp")):
                image_path = os.path.join(dir, filename)
                self.ListBox1.Items.Add(filename)

    def __listClick(self, sender):
        file = os.path.join(self.edInputDir.Text, self.ListBox1.Items[self.ListBox1.ItemIndex])
        if os.path.exists(file):
            self.ImageViewer1.Bitmap.LoadFromFile(file)

    def __colorize(self, sender):
        file = os.path.join(self.edInputDir.Text, self.ListBox1.Items[self.ListBox1.ItemIndex])
        if os.path.exists(file):
            outfile = ""
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
                _, mask = cv2.threshold(image, 118, 255, cv2.THRESH_BINARY)
                kernel = np.ones((1,1), np.uint8)
                dilated_mask = cv2.dilate(mask, kernel, iterations=1)
                lineart = np.ones_like(image) * 255
                lineart[dilated_mask == 0] = image[dilated_mask == 0]
                h, w = image.shape[:2]
                colorizer.set_image(image, 576, True, 25)
                result = colorizer.colorize()
                image = cv2.cvtColor(lineart, cv2.COLOR_GRAY2BGR)
                image = pastel(image)
                image = image.astype('float32')
                _image = cv2.resize(result, (w, h))
                _image = _image.astype('float32')
                blended = cv2.multiply(image, _image, dtype=cv2.CV_32F)
                blended = blended.astype('uint8')

                plt.imsave(temp_file.name, blended)
                outfile = temp_file.name
            self.ImageViewer2.Bitmap.LoadFromFile(outfile)