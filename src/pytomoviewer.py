"""
A simple GUI that allows to load and visualize a stack of images (slices)
from micro-computed tomography converted to 2D (or 3D) Numpy arrays.
Basic current capabilities:
- Visualize the image stack;
- Plot image histogram;
- Resample the data;
- Export to RAW (JSON) format;
- Identify unconnected pores;
- Replicate a single image;
"""
import os
import sys
import json
import numpy as np
import matplotlib as mpl
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg, NavigationToolbar2QT)
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets 
from scipy import ndimage

ORGANIZATION_NAME = 'LCC-IC-UFF'
ORGANIZATION_DOMAIN = 'www.ic.uff.br/~andre/'
APPLICATION_NAME = 'pyTomoViewer'
SETTINGS_TRAY = 'settings/tray'

# Inherit from QDialog
#class TomoViewer(QtWidgets.QDialog):
class TomoViewer(QtWidgets.QMainWindow):
    # Override the class constructor
    def __init__(self, parent=None):
        super(TomoViewer, self).__init__(parent)
        # setting main Widget 
        self.main = QtWidgets.QWidget()
        self.setCentralWidget(self.main)        
        # setting title 
        self.setWindowTitle(APPLICATION_NAME) 
        # setting geometry and minimum size
        self.setGeometry(100, 100, 600, 600) 
        self.setMinimumSize(QtCore.QSize(400, 400))
        # a figure instance to plot on
        self.figure = Figure()
        # this is the Canvas widget that displays the 'figure'
        self.canvas = FigureCanvasQTAgg(self.figure)
        mpl.rcParams['image.cmap'] = 'gray' # magma, seismic
        # this is the Navigation widget
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        # this is the Menu bar 
        bar = self.menuBar()
        mfile = bar.addMenu("&File")
        open_action = QtWidgets.QAction("&Open...", self)
        open_action.triggered.connect(self.openImage)
        save_action = QtWidgets.QAction("&Export...", self)
        save_action.triggered.connect(self.exportImage)
        exit_action = QtWidgets.QAction("&Exit", self)
        exit_action.setShortcut('Ctrl+Q') 
        exit_action.triggered.connect(QtWidgets.qApp.quit)
        mfile.addAction(open_action)
        mfile.addAction(save_action)
        mfile.addAction(exit_action)
        mcomp = bar.addMenu("&Compute")
        replc_action = QtWidgets.QAction("&Replicate Image...", self)
        replc_action.triggered.connect(self.replicateImage)
        remov_action = QtWidgets.QAction("&Remove Unconnected Pores...", self)
        remov_action.triggered.connect(self.removeUnconnected)
        binar_action = QtWidgets.QAction("&Convert to Binary Image...", self)
        binar_action.triggered.connect(self.convertToBinary)
        mcomp.addAction(replc_action)
        mcomp.addAction(remov_action)
        mcomp.addAction(binar_action)
        mhelp = bar.addMenu("&Help")
        about_action = QtWidgets.QAction("&About...", self)
        about_action.triggered.connect(self.aboutDlg)
        mhelp.addAction(about_action)
        # these are the app widgets connected to their slot methods
        self.slideBar = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slideBar.setMinimum(0)
        self.slideBar.setTickPosition(QtWidgets.QSlider.TicksBothSides)
        self.slideBar.setTickInterval(1)        
        self.slideBar.setSingleStep(1)
        self.slideBar.setEnabled(False)
        self.slideBar.valueChanged[int].connect(self.changeValue)
        self.buttonPlus = QtWidgets.QPushButton('+')
        self.buttonPlus.setMaximumSize(QtCore.QSize(25, 30))
        self.buttonPlus.setEnabled(False)
        self.buttonPlus.clicked.connect(self.slideMoveUp)
        self.buttonMinus = QtWidgets.QPushButton('-')
        self.buttonMinus.setMaximumSize(QtCore.QSize(25, 30))
        self.buttonMinus.setEnabled(False) 
        self.buttonMinus.clicked.connect(self.slideMoveDown)        
        self.buttonPlot = QtWidgets.QPushButton('View Image')
        self.buttonPlot.setEnabled(False)
        self.buttonPlot.clicked.connect(self.plotImage)       
        self.buttonHist = QtWidgets.QPushButton('View Histogram')
        self.buttonHist.setEnabled(False)
        self.buttonHist.clicked.connect(self.plotHistogram)
        self.labelDimensions = QtWidgets.QLabel('[h=0,w=0]')
        self.labelSliceId = QtWidgets.QLabel('Slice = 0')
        self.labelSliceId.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        # set the layouts
        mainLayout = QtWidgets.QVBoxLayout(self.main)
        mainLayout.addWidget(self.toolbar)
        layoutH2 = QtWidgets.QHBoxLayout()
        layoutH3 = QtWidgets.QHBoxLayout()
        layoutH2.addWidget(self.buttonMinus)        
        layoutH2.addWidget(self.slideBar)        
        layoutH2.addWidget(self.buttonPlus)  
        layoutH3.addWidget(self.buttonPlot)
        layoutH3.addWidget(self.buttonHist)
        layoutH3.addWidget(self.labelDimensions)
        layoutH3.addItem(QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.MinimumExpanding))
        layoutH3.addWidget(self.labelSliceId)
        mainLayout.addWidget(self.canvas, QtWidgets.QSizePolicy.MinimumExpanding)
        mainLayout.addLayout(layoutH2)
        mainLayout.addLayout(layoutH3)           
        # initialize the main image data
        self.m_data = None # numpy array
        self.m_image = None # QImage object
        self.m_map = []  # path of all image files 

    def __del__(self):
        # remove temporary data: 
        self.m_data = None
        self.m_image = None
        if len(self.m_map) > 0:
            self.removeTempImagens()

    # @Slot()
    def plotImage(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        img = ax.imshow(self.m_data,vmin=0,vmax=255)      
        self.figure.colorbar(img)
        ax.figure.canvas.draw()
        self.buttonPlot.setEnabled(False)          
        self.buttonHist.setEnabled(True)        

    # @Slot()
    def plotHistogram(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.hist(self.m_data.flatten(), bins=256, fc='k', ec='k')
        ax.set(xlim=(-4, 259))
        ax.figure.canvas.draw()
        self.buttonPlot.setEnabled(True)          
        self.buttonHist.setEnabled(False)   

    # @Slot()
    def changeValue(self, _value):
        filename = self.m_map[_value]
        print(filename)
        self.loadImageData(filename,True)
        self.labelSliceId.setText("Slice = "+str(_value+1))

    # @Slot()
    def slideMoveUp(self):
        self.slideBar.setValue(self.slideBar.value()+1)

    # @Slot()
    def slideMoveDown(self):
        self.slideBar.setValue(self.slideBar.value()-1)

    # @Slot()
    def openImage(self):
        options = QtWidgets.QFileDialog.Options()
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self,"Open Tomo", "","Image Files (*.tif);;Image Files (*.tiff)", options=options)
        if files:
            if len(self.m_map) > 0:
                self.removeTempImagens()
            self.m_map.clear() # remove all items
            for filepath in files:
                self.m_map.append( filepath )
            self.loadImageData(files[0],True)
            self.buttonPlus.setEnabled(True) 
            self.buttonMinus.setEnabled(True) 
            self.slideBar.setMaximum(len(self.m_map)-1)
            self.slideBar.setValue(0)
            self.slideBar.setEnabled(True)
            self.labelSliceId.setText("Slice = 1")

    # @Slot()
    def exportImage(self):
        # check if there is at least one image open, and then proceed:
        if len(self.m_map) == 0:
            return
        options = QtWidgets.QFileDialog.Options()
        filename = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Image Data', '', "Raw Files (*.raw);;Image Files (*.tif)", options=options)
        if filename[1] == 'Raw Files (*.raw)':
            if filename[0][-4:] != '.raw':
                filename = filename[0] + '.raw'
            else:
                filename = filename[0] 
            materials = {} 
            # Save image data in RAW format
            with open(filename, "bw") as file_raw:
                for filepath in self.m_map:
                    self.loadImageData(filepath,False)
                    mat_i, cmat_i = np.unique(self.m_data,return_counts=True)
                    for i in range(len(mat_i)):
                        if mat_i[i] in materials:  
                            materials[mat_i[i]] += cmat_i[i]
                        else:
                            materials[mat_i[i]] = cmat_i[i]
                    # Save image data in binary format
                    self.m_data.tofile(file_raw) 
            self.loadImageData(self.m_map[self.slideBar.value()],True)
            materials = dict(sorted(materials.items(), key=lambda x: x[0]))
            dimensions = np.array([self.m_data.shape[1],self.m_data.shape[0],len(self.m_map)],dtype=int)
            vol = self.m_data.shape[1]*self.m_data.shape[0]*len(self.m_map)
            mat = np.array(list(materials.keys()))  
            cmat = np.array(list(materials.values()))   
            mat = np.vstack((mat, np.zeros((mat.shape[0]),dtype=int))).T
            cmat = cmat*100.0/vol      
            jdata = {}
            jdata["type_of_analysis"] = 0
            jdata["type_of_solver"] = 0
            jdata["type_of_rhs"] = 0
            jdata["voxel_size"] = 1.0
            jdata["solver_tolerance"] = 1.0e-6
            jdata["number_of_iterations"] = 1000
            jdata["image_dimensions"] = dimensions.tolist()          
            jdata["refinement"] = 1
            jdata["number_of_materials"] = mat.shape[0]
            jdata["properties_of_materials"] = mat.tolist()
            jdata["volume_fraction"] = list(np.around(cmat,2))
            jdata["data_type"] = "uint8"
            # Save image data in JSON format
            with open(filename[0:len(filename)-4] + ".json",'w') as file_json:
                json.dump(jdata,file_json,sort_keys=False, indent=4, separators=(',', ': '))
            # Save image data in NF format
            with open(filename[0:len(filename)-4] + ".nf",'w') as file_nf:
                sText = ''
                for k, v in jdata.items():
                    sText += '%' + str(k) + '\n'+ str(v) + '\n\n'
                sText = sText.replace('], ','\n')
                sText = sText.replace('[','')
                sText = sText.replace(']','')
                sText = sText.replace(',','')
                file_nf.write(sText)
        elif filename[1] == 'Image Files (*.tif)' and filename[0][-4:] == '.tif':
            refine, ok = QtWidgets.QInputDialog.getInt(self,"Resampling","Refinement level:", 1, 1, 100, 1)
            if ok:
                filename = filename[0]
                i = 0
                for filepath in self.m_map:
                    self.loadImageData(filepath,False)
                    h = self.m_data.shape[0]
                    w = self.m_data.shape[1]
                    for _ in range(refine):
                        i += 1
                        imgNumber = '{:04d}'.format(i)
                        # Save image data in TIF format
                        self.m_image.scaled(refine*h, refine*w).save(filename[:-4]+"_"+imgNumber+".tif")
                self.loadImageData(self.m_map[self.slideBar.value()], True)

    # @Slot()
    def replicateImage(self):
        # check if there is at least one image open, and then proceed:
        if len(self.m_map) == 0:
            return
        slices, ok = QtWidgets.QInputDialog.getInt(self,"Replicate","Number of slices:", 1, 1, 2024, 1)
        if ok:
            filepath = self.m_map[0]
            self.m_map.clear() # remove all items
            for _ in range(slices):
                self.m_map.append( filepath )
            self.loadImageData(filepath,True)
            self.buttonPlus.setEnabled(True) 
            self.buttonMinus.setEnabled(True) 
            self.slideBar.setMaximum(len(self.m_map)-1)
            self.slideBar.setValue(0)
            self.slideBar.setEnabled(True)
            self.labelSliceId.setText("Slice = 1")

    # @Slot()
    def removeUnconnected(self):
        # check if there is at least one image open, and then proceed:
        if len(self.m_map) == 0:
            return
        dataset = None
        loadedFirst = False
        for filepath in self.m_map:
            self.loadImageData(filepath,False)
            self.m_data[self.m_data>0] = 1
            self.m_data = 1-self.m_data
            if loadedFirst:
                dataset = np.vstack([ dataset, self.m_data[np.newaxis,...] ]) 
            else:
                loadedFirst = True
                dataset = self.m_data[np.newaxis,...] 
        # separate unconnected regions of voxels by labelling each with a different number:    
        imgLabels, numLabels = ndimage.label(dataset)
        # images from all RVE border faces:
        imgXi = imgLabels[:,:,0]
        imgXf = imgLabels[:,:,-1]
        imgYi = imgLabels[:,0]
        imgYf = imgLabels[:,-1]
        # products between opposite faces to find region connections:
        imgXp = imgXi*imgXf
        imgYp = imgYi*imgYf
        # register for each region which other region it is connected with:
        regionsToKeep = []
        for index, e in np.ndenumerate(imgXp):
            if e > 0:
                regionsToKeep.append(imgXi[index[0],index[1]])
                regionsToKeep.append(imgXf[index[0],index[1]])
        for index, e in np.ndenumerate(imgYp):
            if e > 0:
                regionsToKeep.append(imgYi[index[0],index[1]])
                regionsToKeep.append(imgYf[index[0],index[1]])
        # proceed the same way for the case with more than one slice:
        if len(self.m_map) > 1:
            imgZi = imgLabels[0]
            imgZf = imgLabels[-1]                
            imgZp = imgZi*imgZf
            for index, e in np.ndenumerate(imgZp):
                if e > 0:
                    regionsToKeep.append(imgZi[index[0],index[1]])
                    regionsToKeep.append(imgZf[index[0],index[1]])
        # sort and unique data:
        sRTK = set(regionsToKeep)
        # remove the unconnected regions and label the connected porous with 0 and solids with 255
        numRemoved = numLabels-len(sRTK)
        if numRemoved > 0:
            for r in sRTK:
                imgLabels[imgLabels==r] = -1
            imgLabels[imgLabels>0] = 0
            imgLabels += 1
            imgLabels *= 255
            # temporary save:
            nImg = len(self.m_map)
            self.m_map.clear() # remove all items
            for i in range(nImg):
                im = np.uint8(imgLabels[i])
                bytesPerLine = im.shape[1]
                image = QtGui.QImage(im, im.shape[1], im.shape[0], bytesPerLine, QtGui.QImage.Format_Grayscale8)
                imgNumber = '{:04d}'.format(i)
                filepath = "temp_image_"+imgNumber+".tif"
                image.save(filepath)
                self.m_map.append( filepath )
            self.loadImageData(self.m_map[self.slideBar.value()], True)
        sm = "Total number of unconnected regions: "+str(numLabels)+"  \nNumber of regions that were removed: "+str(numRemoved)
        msg = QtWidgets.QMessageBox()
        msg.setText(sm)
        msg.setWindowTitle("Info")
        msg.exec_()

    # @Slot()
    def convertToBinary(self):
        # check if there is at least one image open, and then proceed:
        if len(self.m_map) == 0:
            return
        threshold, ok = QtWidgets.QInputDialog.getInt(self,"Convert image to binary image","Threshold:", 1, 1, 256, 1)
        if ok:
            nImg = len(self.m_map)
            for i in range(nImg):
                self.loadImageData(self.m_map[i],False)
                self.m_data[self.m_data>=threshold] = 255
                self.m_data[self.m_data<threshold]  = 0
                im = np.uint8(self.m_data)
                bytesPerLine = im.shape[1]
                image = QtGui.QImage(im, im.shape[1], im.shape[0], bytesPerLine, QtGui.QImage.Format_Grayscale8)
                imgNumber = '{:04d}'.format(i)
                filepath = "temp_image_"+imgNumber+".tif"
                image.save(filepath)
                self.m_map[i] = filepath 
            self.loadImageData(self.m_map[self.slideBar.value()], True)

    # @Slot()
    def aboutDlg(self):
        sm = """pyTomoViewer\nVersion 1.0.0\n2020\nLicense GPL 3.0\n\nThe authors and the involved Institutions are not responsible for the use or bad use of the program and their results. The authors have no legal dulty or responsability for any person or company for the direct or indirect damage caused resulting from the use of any information or usage of the program available here. The user is responsible for all and any conclusion made with the program. There is no warranty for the program use. """
        msg = QtWidgets.QMessageBox()
        msg.setText(sm)
        msg.setWindowTitle("About")
        msg.exec_()

    # method
    def loadImageData(self, _filepath, _updateWindow):
        self.m_image = QtGui.QImage(_filepath)
        # We perform these conversions in order to deal with just 8 bits images:
        # convert Mono format to Indexed8
        if self.m_image.depth() == 1:
            self.m_image = self.m_image.convertToFormat(QtGui.QImage.Format_Indexed8)
        # convert Grayscale16 format to Grayscale8
        if not self.m_image.format() == QtGui.QImage.Format_Grayscale8:
            self.m_image = self.m_image.convertToFormat(QtGui.QImage.Format_Grayscale8)
        self.m_data = convertQImageToNumpy(self.m_image)
        if _updateWindow:
            self.labelDimensions.setText("[h="+str(self.m_data.shape[0])+",w="+str(self.m_data.shape[1])+"]")
            self.plotImage()     

    # method
    def removeTempImagens(self):
        for filepath in self.m_map:
            if "temp_" in filepath :
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                    except OSError as err:
                        print("Exception handled: {0}".format(err))
                else:
                    print("The file does not exist") 

# This function was adapted from (https://github.com/Entscheider/SeamEater/blob/master/gui/QtTool.py)
# Project: SeamEater; Author: Entscheider; File: QtTool.py; GNU General Public License v3.0 
# Original function name: qimage2numpy(qimg)
# We consider just 8 bits images and convert to single depth:
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

if __name__ == '__main__':
    main()
