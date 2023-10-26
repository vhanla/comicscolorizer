from delphifmx import *
from Unit1 import mainForm

def main():
    Application.Initialize()
    Application.Title = 'Comics Colorizer'
    Application.MainForm = mainForm(Application)
    Application.MainForm.Show()
    Application.Run()
    Application.MainForm.Destroy()

if __name__ == '__main__':
    main()
