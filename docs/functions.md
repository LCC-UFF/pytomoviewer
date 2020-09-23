# pytomoviewer

A simple GUI that allows to load and visualize a stack of images (slices) from micro-computed tomography converted to 2D (or 3D) Numpy arrays.
Basic current capabilities:
- Visualize the image stack;
- Plot image histogram;
- Resample the data;
- Export to RAW (JSON) format;
- Identify unconnected pores;
- Replicate a single image;

# Functions

This function was adapted from this [GitHub](https://github.com/Entscheider/SeamEater/blob/master/gui/QtTool.py),
Project: SeamEater; Author: Entscheider; File: QtTool.py; GNU General Public License v3.0.

The original function name is ```qimage2numpy(qimg)```. We consider just 8 bits images and convert to single depth:

def convertQImageToNumpy (_qimg)
```
def convertQImageToNumpy(_qimg):
    h = _qimg.height()
    w = _qimg.width()
    ow = _qimg.bytesPerLine() * 8 // _qimg.depth()
    d = 0
    if _qimg.format() in (QtGui.QImage.Format_ARGB32_Premultiplied,
                          QtGui.QImage.Format_ARGB32,
                          QtGui.QImage.Format_RGB32):
        d = 4 # formats: 6, 5, 4.
    elif _qimg.format() in (QtGui.QImage.Format_Indexed8,
                            QtGui.QImage.Format_Grayscale8):
        d = 1 # formats: 3, 24.
    else:
        raise ValueError(".ERROR: Unsupported QImage format!")
    buf = _qimg.bits().asstring(_qimg.byteCount())
    res = np.frombuffer(buf, 'uint8')
    res = res.reshape((h,ow,d)).copy()
    if w != ow:
        res = res[:,:w] 
    if d >= 3:
        res = res[:,:,0].copy()
    else:
        res = res[:,:,0] 
    return res 
```

Main function that starts the GUI

def main()
```
def main():
    # To ensure that every time you call QSettings not enter the data of your application, 
    # which will be the settings, you can set them globally for all applications   
    QtCore.QCoreApplication.setApplicationName(ORGANIZATION_NAME)
    QtCore.QCoreApplication.setOrganizationDomain(ORGANIZATION_DOMAIN)
    QtCore.QCoreApplication.setApplicationName(APPLICATION_NAME)
    # create pyqt5 app
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    # create the instance of our Window
    mw = TomoViewer()
    # showing all the widgets
    mw.show()
    # start the app
    sys.exit(app.exec_())
```


