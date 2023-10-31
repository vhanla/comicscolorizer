import os
from delphifmx import * #Button, OpenDialog, ListBox, Form
import torch
import numpy as np
import matplotlib.pyplot as plt
#from tkinter.filedialog import askdirectory, Tk
from colorizator import MangaColorizator
import tempfile
import time
import cv2
import zipfile
import rarfile
import py7zr
import io
from scipy.interpolate import UnivariateSpline
import threading
from enum import Enum

CUDA_AVAILABLE = torch.cuda.device_count() > 0
colorizer = None

# constants
DESGREENS = 1
DESREDS = 2
DESBLUES = 4

# App Mode
class AppMode(Enum):
    COMICBOOK = 1
    DIRECTORY = 2
    IMAGEONLY = 3

def sepia(img):
    img_sepia = np.array(img, dtype=np.float64) # converting to float to prevent loss
    img_sepia = cv2.transform(img_sepia, np.matrix([[0.272, 0.534, 0.131],
                                    [0.349, 0.686, 0.168],
                                    [0.393, 0.769, 0.189]])) # multipying image with special sepia matrix
    img_sepia[np.where(img_sepia > 255)] = 255 # normalizing values greater than 255 to 255
    img_sepia = np.array(img_sepia, dtype=np.uint8)
    return img_sepia

#sharp effect
def sharpen(img):
    kernel = np.array([[-1, -1, -1], [-1, 9.5, -1], [-1, -1, -1]])
    img_sharpen = cv2.filter2D(img, -1, kernel)
    return img_sharpen

# brightness adjustment
def bright(img, beta_value ):
    img_bright = cv2.convertScaleAbs(img, beta=beta_value)
    return img_bright

def LookupTable(x, y):
    spline = UnivariateSpline(x, y)
    return spline(range(256))

def Summer(img):
    increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    blue_channel, green_channel,red_channel  = cv2.split(img)
    red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
    sum= cv2.merge((blue_channel, green_channel, red_channel ))
    return sum

def Winter(img):
    increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    blue_channel, green_channel,red_channel = cv2.split(img)
    red_channel = cv2.LUT(red_channel, decreaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
    win= cv2.merge((blue_channel, green_channel, red_channel))
    return win

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
        self.__sm = StyleManager()
        self.__sm.SetStyleFromFile(os.path.join(os.getcwd(), "styles/Dark.style"))

        self.AppMode = None

        self.Panel1 = None
        self.ListBox1 = None
        self.StatusBar1 = None
        self.StyleBook1 = None
        self.TabControl1 = None
        self.TabItem1 = None
        self.Label3 = None
        self.btnOpenArch = None
        self.btnSaveArch = None
        self.Label4 = None
        self.edInputArch = None
        self.edOutputArch = None
        self.CheckBox1 = None
        self.TabItem2 = None
        self.btnDirPath = None
        self.btnOutPath = None
        self.edInputDir = None
        self.edOutputDir = None
        self.Label1 = None
        self.Label2 = None
        self.TabItem3 = None
        self.Label5 = None
        self.btnOpenImg = None
        self.btnSaveImg = None
        self.Label6 = None
        self.edInputImg = None
        self.edOutputImg = None
        self.Panel3 = None
        self.chkBlendGray = None
        self.chkGrayFirst = None
        self.chkUpscale = None
        self.btnColorize = None
        self.ProgressBar1 = None
        self.Panel4 = None
        self.ImageViewer1 = None
        self.ImageViewer2 = None
        self.Splitter1 = None
        self.lblGPU = None
        self.swThreshold = None
        self.Label7 = None
        self.tbThreshold = None
        self.lbThreshold = None
        self.swLineart = None
        self.lblLineartSw = None
        self.OpenDialog1 = None
        self.LoadProps(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Unit1.pyfmx"))

        self.btnDirPath.OnClick = self.__openDirectory
        self.ListBox1.OnClick = self.__listClick

        if CUDA_AVAILABLE:
            gpu_info = torch.cuda.get_device_properties(0)
            self.lblGPU.SetProps(width=400, AutoSize=True)
            self.lblGPU.Text = f"{gpu_info.name} {gpu_info.total_memory / 1024**3:.2f} GB"
        else:
            self.lblGPU.Text = "No CUDA"

        self.btnColorize.OnClick = self.hilazo#__colorize
        self.tbThreshold.OnClick = self.__thresholdPreview
        self.swThreshold.OnClick = self.__thresholdPreview
        self.tbThreshold.OnChange = self.__thresholdChange
        self.ImageViewer1.OnMouseWheel = self.__viewresize
        self.ImageViewer1.OnHScrollChange = self.__scrolling
        self.ImageViewer1.OnVScrollChange = self.__scrolling

        # current image file
        self.curimg = None
        self.curlineart = None
        self.curcolored = None
        #self.SetProps(position=4)
        self.ImageViewer1.BackgroundFill.Color = 0.0
        self.ImageViewer2.BackgroundFill.Color = 0.0

        # archives
        self.fArchiveFile = None
        self.btnOpenArch.OnClick = self.__openArchive

    def __openArchive(self, sender):
        self.OpenDialog1.Title = "Select a comic archive file."
        self.OpenDialog1.Filter = "Comic Book files (*.cbr;*.cbz;*.cb7)|*.cbr;*.cbz;*.cb7|All files (*.*)|*.*"
        #dir = askdirectory()
        #root.destroy()
        if self.OpenDialog1.Execute():
            pictypes = ['.webp', '.png', '.jpeg', '.jpg']
            self.fArchiveFile = self.OpenDialog1.FileName
            comic_file = self.fArchiveFile
            self.ListBox1.Clear();
            if comic_file.endswith((".cbr")): # rar file
                with rarfile.RarFile(comic_file) as cbr:
                    for rarinfo in cbr.infolist():
                        if any(rarinfo.filename.lower().endswith(ext) for ext in pictypes):
                            self.ListBox1.Items.Add(rarinfo.filename)
                    # rearrange the list
                    self.ListBox1.Sorted = True
                    if self.ListBox1.Items.Count > 0:
                        self.AppMode = AppMode.COMICBOOK



    def __viewresize(self, sender, old, new, size):
        if self.ImageViewer2.Bitmap is not None:
            self.ImageViewer2.BitmapScale = self.ImageViewer1.BitmapScale

    def __scrolling(self, sender):
        self.ImageViewer2.ViewportPosition = self.ImageViewer1.ViewportPosition

    def getlineart(self, image, threshold):
        if len(image.shape) == 2:
            gray_image = image
        else:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

        binary_mask = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)[1]

        kernel = np.ones((1,1), np.uint8)
        dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)

        lineart = np.ones_like(gray_image) * 255
        lineart[dilated_mask == 0] = gray_image[dilated_mask == 0]
        return lineart

    def desaturateColor(self, img, color_ranges, percent):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        mask = None

        if color_ranges & DESGREENS:
            # Create a 1D LUT for green
            # (120 out of 360) = (60 out of 180) +- 25
            lut = np.zeros((1, 256), dtype=np.uint8)
            white = np.full((1, 50), 255, dtype=np.uint8)
            lut[0:1, 35:85] = white

            # Apply the green LUT to the hue channel as a mask
            mask_green = cv2.LUT(h, lut)
            mask_green = mask_green.astype(np.float32) / 255

            if mask is None:
                mask = mask_green
            else:
                mask = cv2.bitwise_or(mask, mask_green)
        if color_ranges & DESREDS:
            # Define a range for red color in HSV
            lower_red = np.array([0, 100, 100])
            upper_red = np.array([10, 255, 255])

            # Create a mask for red pixels
            mask_red = cv2.inRange(hsv, lower_red, upper_red)

            # Invert the mask to keep non-red pixels
            mask_red = cv2.bitwise_not(mask_red)

            # Create a 1D LUT for red
            lut = np.zeros((1, 256), dtype=np.uint8)
            lut[0, 100:256] = 255

            # Apply the red LUT to the hue channel as a mask
            mask_red = cv2.LUT(h, lut)
            mask_red = mask_red.astype(np.float32) / 255

            if mask is None:
                mask = mask_red
            else:
                mask = cv2.bitwise_or(mask, mask_red)
        if color_ranges & DESBLUES:
            # Define a range for blue color in HSV
            lower_blue = np.array([100, 100, 100])
            upper_blue = np.array([130, 255, 255])

            # Create a mask for blue pixels
            mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

            # Invert the mask to keep non-blue pixels
            mask_blue = cv2.bitwise_not(mask_blue)

            # Create a 1D LUT for blue
            lut = np.zeros((1, 256), dtype=np.uint8)
            lut[0, 100:256] = 255

            # Apply the blue LUT to the hue channel as a mask
            mask_blue = cv2.LUT(h, lut)
            mask_blue = mask_blue.astype(np.float32) / 255

            if mask is None:
                mask = mask_blue
            else:
                mask = cv2.bitwise_or(mask, mask_blue)
        if mask is None:
            raise ValueError("No color ranges selected")

        # Desaturate the selected color ranges
        s_desat = cv2.multiply(s, percent).astype(np.uint8)

        # Merge the desaturated channel with the original image
        hsv_new = cv2.merge([h, s_desat, v])
        bgr_desat = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)

        # Resize the mask to match the dimensions of the input image
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

        # mask bgr_desat and img
        result = mask[:, :, np.newaxis] * bgr_desat + (1 - mask[:, :, np.newaxis]) * img
        #result = mask * bgr_desat + (1 - mask) * img
        #result = result.clip(0, 255).astype(np.uint8)

        return result


    def __thresholdPreview(self, sender):
        print(self.swThreshold.IsChecked)
        if self.curimg is not None:
            print("Getting lineart...")
            if self.swThreshold.IsChecked:
                lineart = self.getlineart(self.curimg, round(self.tbThreshold.Value))

                # CLAHE contrast enhancement 2.0
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                lineart = clahe.apply(lineart)

                # translucent
                #mask = cv2.bitwise_not(mask)
                #alpha = cv2.cvtColor(lineart, cv2.COLOR_GRAY2BGRA)
                #alpha[mask == 0] = [255, 255, 255, 0]

                #h, w = image.shape[:2]
                '''
                colorizer.set_image(image, 576, True, 25)
                result = colorizer.colorize()
                '''
                #image = cv2.cvtColor(lineart, cv2.COLOR_GRAY2BGR)
                #image = pastel(image)
                #image = image.astype('float32')
                '''
                _image = cv2.resize(result, (w, h))
                _image = _image.astype('float32')
                blended = cv2.multiply(image, _image, dtype=cv2.CV_32F)
                blended = blended.astype('uint8')
                '''

                '''
                bgr_image_with_alpha = cv2.cvtColor(_image, cv2.COLOR_BGR2BGRA)
                b, g, r, a = cv2.split(alpha)
                b = cv2.bitwise_and(b, b, mask=a)
                g = cv2.bitwise_and(g, g, mask=a)
                r = cv2.bitwise_and(r, r, mask=a)
                alpha = cv2.merge([b,g,r,a])
                alpha = alpha.astype(float)/255 # normalize the alpha mask to keep intensity between 0 and 1
                #linearted = cv2.addWeighted(bgr_image_with_alpha, 1, alpha[:,:,:3], 1, 0, dtype=cv2.CV_8U)
                lineart = cv2.cvtColor(lineart, cv2.COLOR_BGR2BGRA)
                foreground = cv2.multiply(alpha, lineart, dtype=cv2.CV_32F)
                background = cv2.multiply(1.0 - alpha, bgr_image_with_alpha, dtype=cv2.CV_32F)
                linearted = cv2.add(foreground, background)
                '''

                #plt.imsave(temp_file.name, blended) #linearted.astype('uint8'))
                #outfile = temp_file.name

                image_bytes = cv2.imencode('.bmp', lineart)[1].tobytes()
                self.ImageViewer1.Bitmap.LoadFromStream(BytesStream(image_bytes))
            else:
                image_bytes = cv2.imencode('.bmp', self.curimg)[1].tobytes()
                self.ImageViewer1.Bitmap.LoadFromStream(BytesStream(image_bytes))



    def __thresholdChange(self, sender):
        self.lbThreshold.Text = round(self.tbThreshold.Value)

    def __openDirectory(self, sender):
        #root = Tk()
        #root.withdraw()
        self.OpenDialog1.Title = "Select any picture file."
        self.OpenDialog1.Filter = "Image files (*.jpg;*.jpeg;*.png;*.gif;*.bmp, *.webp)|*.jpg;*.jpeg;*.png;*.gif;*.bmp;*.webp|All files (*.*)|*.*"
        #dir = askdirectory()
        #root.destroy()
        if self.OpenDialog1.Execute():
            dir = os.path.dirname(self.OpenDialog1.FileName)
            self.edInputDir.Text = dir
            self.ListBox1.Items.Clear()
            for filename in os.listdir(dir):
                if filename.endswith((".jpg", ".jpeg", ".png", ".webp")):
                    image_path = os.path.join(dir, filename)
                    self.ListBox1.Items.Add(filename)
            if self.ListBox1.Items.Count > 0:
                self.AppMode = AppMode.DIRECTORY

    def __listClick(self, sender):
        if self.ListBox1.Items.Count > 0:
            if self.AppMode == AppMode.COMICBOOK:
                comic_file = self.fArchiveFile
                stream = io.BytesIO()
                file_to_extract = self.ListBox1.Items[self.ListBox1.ItemIndex]
                #CBR
                if comic_file.lower().endswith((".cbr")):
                    with rarfile.RarFile(comic_file) as cbr:
                        if file_to_extract in cbr.namelist():
                            binary = cbr.open(file_to_extract)
                            stream.write(binary.read())
                            cbr.close()
                            stream.seek(0)
                            image_data = np.frombuffer(stream.read(), np.uint8)
                            self.curimg = cv2.imdecode(image_data, cv2.IMREAD_UNCHANGED)
                            if self.swThreshold.IsChecked:
                                self.__thresholdPreview(sender)
                            else:
                                image_bytes = cv2.imencode('.bmp', self.curimg)[1].tobytes()
                                self.ImageViewer1.Bitmap.LoadFromStream(BytesStream(image_bytes))


            elif self.AppMode == AppMode.DIRECTORY:
                file = os.path.join(self.edInputDir.Text, self.ListBox1.Items[self.ListBox1.ItemIndex])
                if os.path.exists(file):
                    self.curimg = cv2.imread(file, cv2.IMREAD_UNCHANGED) #preload for other calls
                    if self.swThreshold.IsChecked:
                        self.__thresholdPreview(sender)
                    else:
                        self.ImageViewer1.Bitmap.LoadFromFile(file)
            elif self.AppMode == AppMode.IMAGEONLY:
                print("Reading a single image file")
            else:
                print("Invalid mode!")


    def hilo(self):
        print("hilando")
        self.btnColorize.Enabled = False
        self.__colorize()
        #self.btnColorize.Enabled = True

    def hilazo(self, sender):
        self.btnColorize.Enabled = False
        hl = threading.Thread(target=self.hilo)
        hl.start()
        hl.join()

    def __colorize(self):
        global colorizer

        #file = os.path.join(self.edInputDir.Text, self.ListBox1.Items[self.ListBox1.ItemIndex])

        #if os.path.exists(file):
        if self.curimg is not None:
            #input_image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            input_image = cv2.cvtColor(self.curimg, cv2.COLOR_BGR2GRAY)
            h, w = input_image.shape[:2]
            _, mask = cv2.threshold(input_image, round(self.tbThreshold.Value), 255, cv2.THRESH_BINARY)
            kernel = np.ones((1,1), np.uint8)
            dilated_mask = cv2.dilate(mask, kernel, iterations=1)
            lineart = np.ones_like(input_image) * 255
            lineart[dilated_mask == 0] = input_image[dilated_mask == 0]
            # translucent
            #mask = cv2.bitwise_not(mask)
            #alpha = cv2.cvtColor(lineart, cv2.COLOR_GRAY2BGRA)
            #alpha[mask == 0] = [255, 255, 255, 0]

            if colorizer is None:
                colorizer = MangaColorizator('cuda', 'networks/generator.zip', 'networks/extractor.pth')

            if self.swLineart.IsChecked:
                #self.getlineart(image, round(self.tbThreshold.Value))
                colorizer.set_image(lineart, 576, True, 25)
            else:
                colorizer.set_image(input_image, 576, True, 25)

            colorized = colorizer.colorize()

            # gray to bgr
            lineart = cv2.cvtColor(lineart, cv2.COLOR_GRAY2BGR)
            #image = pastel(image)
            #image = image.astype('float32')

            resized_image = cv2.resize(colorized, (w, h))
            resized_image = resized_image.astype('float32')
            if self.chkBlendGray.IsChecked:
                result = cv2.multiply(lineart, resized_image, dtype=cv2.CV_32F)
            else:
                result = cv2.multiply(resized_image, resized_image, dtype=cv2.CV_32F)

            result = result.astype('uint8')

            #desaturation
            #blended = self.desaturateColor(blended, DESGREENS | DESREDS | DESBLUES, 0.5)
            if self.chkUpscale.IsChecked:
                result = cv2.detailEnhance(result, sigma_s=3, sigma_r=0.15)
                result = Summer(result)

            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

            image_bytes = cv2.imencode('.bmp', result)[1].tobytes()
            self.ImageViewer2.Bitmap.LoadFromStream(BytesStream(image_bytes))

            # save file
            #with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                #plt.imsave(temp_file.name, result) #linearted.astype('uint8'))
                #outfile = temp_file.name
            #self.ImageViewer2.Bitmap.LoadFromFile(outfile)
        self.btnColorize.Enabled = True