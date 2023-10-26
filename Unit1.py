import os
from delphifmx import *

class mainForm(Form):

    def __init__(self, owner):
        self.Panel1 = None
        self.ListView1 = None
        self.StatusBar1 = None
        self.StyleBook1 = None
        self.Panel2 = None
        self.Edit1 = None
        self.btnDirPath = None
        self.ImageViewer1 = None
        self.LoadProps(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Unit1.pyfmx"))