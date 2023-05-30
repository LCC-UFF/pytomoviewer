"""
A simple GUI that allows to load and visualize a stack of images (slices)
from micro-computed tomography converted to 2D (or 3D) Numpy arrays.
Basic current capabilities:
- Visualize the image stack;
- Plot image histogram;
- Resample the data;
- Export to RAW, TIF, JSON and NF format;
- Identify and remove unconnected pores;
- Sieves;
- Replicate a single image;
"""
import os
import sys
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg, NavigationToolbar2QT)
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets
from scipy import ndimage
from collections import defaultdict
from skimage import io
from PIL import Image

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
        new2dci_action = QtWidgets.QAction("&2D Circular Inclusion...", self)
        new2dci_action.triggered.connect(self.newImage2D_CI)
        new3dsi_action = QtWidgets.QAction("&3D Spherical Inclusion...", self)
        new3dsi_action.triggered.connect(self.newImage3D_SI)
        new3dbci_action = QtWidgets.QAction("&3D Bidirectional Crossed Inclusion...", self)
        new3dbci_action.triggered.connect(self.newImage3D_BCI)
        open_action = QtWidgets.QAction("&Open...", self)
        open_action.triggered.connect(self.openImage)
        open_action.setShortcut('Ctrl+O')
        save_action = QtWidgets.QAction("&Export...", self)
        save_action.triggered.connect(self.exportImage)
        exit_action = QtWidgets.QAction("&Exit", self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.closeEvent)
        newmodel = mfile.addMenu("&New")
        newmodel.addAction(new2dci_action)
        newmodel.addAction(new3dsi_action)
        newmodel.addAction(new3dbci_action)
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
        phases_action = QtWidgets.QAction("&Colors per slice...", self)
        phases_action.triggered.connect(self.colorsPerSlice)
        seive_action = QtWidgets.QAction("&Seive...", self)
        seive_action.triggered.connect(self.seives)
        mcomp.addAction(replc_action)
        mcomp.addAction(remov_action)
        mcomp.addAction(binar_action)
        mcomp.addAction(phases_action)
        mcomp.addAction(seive_action)
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
    def closeEvent(self, event):
        result = QtWidgets.QMessageBox.question(self,"Exit","Are you sure you want to exit the program?",QtWidgets.QMessageBox.Yes| QtWidgets.QMessageBox.No)
        if result == QtWidgets.QMessageBox.Yes:
            QtWidgets.qApp.quit()
        else:
            event.ignore()

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
        if not self.buttonPlot.isEnabled():
            self.loadImageData(filename,True)
            self.labelSliceId.setText("Slice = "+str(_value+1))
            return
        if not self.buttonHist.isEnabled():
            self.loadImageData(filename,True)
            self.labelSliceId.setText("Slice = "+str(_value+1))
            self.plotHistogram()
            return

    # @Slot()
    def slideMoveUp(self):
        if not self.buttonPlot.isEnabled():
            self.slideBar.setValue(self.slideBar.value()+1)
            return
        if not self.buttonHist.isEnabled():
            self.slideBar.setValue(self.slideBar.value()+1)
            self.plotHistogram()
            return

    # @Slot()
    def slideMoveDown(self):
        if not self.buttonPlot.isEnabled():
            self.slideBar.setValue(self.slideBar.value()-1)
            return
        if not self.buttonHist.isEnabled():
            self.slideBar.setValue(self.slideBar.value()-1)
            self.plotHistogram()
            return

    # @Slot()
    def newImage2D_CI(self):
        nx, ok1 = QtWidgets.QInputDialog.getInt(self,"Size","Length:", 100, 1, 2024, 1)
        if ok1:
            f, ok2 = QtWidgets.QInputDialog.getDouble(self,"Fraction","Inclusion fraction 0 < x < 1:", 0.5, 0, 1, 3, QtCore.Qt.WindowFlags(), 0.05)
            if ok2:
                ny = nx
                r = np.sqrt(4*f/np.pi)*nx*.5
                self.m_data = np.zeros([nx, ny, 1])
                for ii in range(nx):
                    for jj in range(ny):
                        val = (ii+0.5-nx/2)**2+(jj+0.5-ny/2)**2 - r**2
                        if val < 0:
                            self.m_data[ii, jj, 0] = 255

                if len(self.m_map) > 0:
                    self.removeTempImagens()
                self.m_map.clear()  # remove all items
                im = np.uint8(self.m_data)
                bytesPerLine = im.shape[1]
                image = QtGui.QImage(
                    im, im.shape[1], im.shape[0], bytesPerLine, QtGui.QImage.Format_Grayscale8)
                imgNumber = '{:04d}'.format(0)
                filepath = "temp_image_"+imgNumber+".tif"
                image.save(filepath)
                self.m_map.append(filepath)
                self.loadImageData(self.m_map[self.slideBar.value()], True)
                self.buttonPlus.setEnabled(True)
                self.buttonMinus.setEnabled(True)
                self.slideBar.setMaximum(len(self.m_map)-1)
                self.slideBar.setValue(0)
                self.slideBar.setEnabled(True)
                self.labelSliceId.setText("Slice = 1")

    # @Slot()
    def newImage3D_SI(self):
        nx, ok1 = QtWidgets.QInputDialog.getInt(self,"Size","Length:", 100, 1, 2024, 1)
        if ok1:
            f, ok2 = QtWidgets.QInputDialog.getDouble(self,"Fraction","Inclusion fraction 0 < x < 1:", 0.5, 0, 1, 3, QtCore.Qt.WindowFlags(), 0.05)
            if ok2:
                ny = nx
                nz = nx
                r = np.sqrt(4*f/np.pi)*nx*.5
                self.m_data = np.zeros([nx, ny, nz])
                for ii in range(nx):
                    for jj in range(ny):
                        for kk in range(nz):
                            val = (ii+0.5-nx/2)**2+(jj+0.5-ny/2)**2+(kk+0.5-nz/2)**2 - r**2
                            if val < 0:
                                self.m_data[ii, jj, kk] = 255

                if len(self.m_map) > 0:
                    self.removeTempImagens()
                self.m_map.clear()  # remove all items
                for kk in range(nz):
                    im = np.uint8(self.m_data[:,:,kk])
                    bytesPerLine = im.shape[1]
                    image = QtGui.QImage(im, im.shape[1], im.shape[0], bytesPerLine, QtGui.QImage.Format_Grayscale8)
                    filepath = "temp_image_"+str(kk)+".tif"
                    image.save(filepath)
                    self.m_map.append(filepath)
                self.loadImageData(self.m_map[self.slideBar.value()], True)
                self.buttonPlus.setEnabled(True)
                self.buttonMinus.setEnabled(True)
                self.slideBar.setMaximum(len(self.m_map)-1)
                self.slideBar.setValue(0)
                self.slideBar.setEnabled(True)
                self.labelSliceId.setText("Slice = 1")

    # @Slot()
    def newImage3D_BCI(self):
        nx, ok1 = QtWidgets.QInputDialog.getInt(self,"Size","Nx:", 200, 1, 2024, 1)
        if ok1:
            ny, ok2 = QtWidgets.QInputDialog.getInt(self,"Size","Ny:", 100, 1, 2024, 1)
            if ok2:
                nz, ok3 = QtWidgets.QInputDialog.getInt(self,"Size","Nz:", 100, 1, 2024, 1)
                if ok3:
                    r, ok4 = QtWidgets.QInputDialog.getInt(self,"Radius","Radius:", 18, 0, 2024, 1)
                    if ok4:
                        self.m_data = np.zeros([nx, ny, nz])
                        for ii in range(int(nx/2)):
                            for jj in range(ny):
                                val_bot = (ii+0.5-nx/4)**2+(jj+0.5-ny/2)**2 - r**2
                                for kk in range(nz):
                                    val_top = (kk+0.5-nz/2)**2+(ii+0.5-nx/4)**2 - r**2
                                    if val_bot < 0:
                                        self.m_data[ii + int(nx/2), jj, kk] = 255
                                    if val_top < 0:
                                        self.m_data[ii, jj, kk] = 255

                        if len(self.m_map) > 0:
                            self.removeTempImagens()
                        self.m_map.clear()  # remove all items
                        for kk in range(nz):
                            im = np.uint8(self.m_data[:,:,kk])
                            bytesPerLine = im.shape[1]
                            image = QtGui.QImage(im, im.shape[1], im.shape[0], bytesPerLine, QtGui.QImage.Format_Grayscale8)
                            filepath = "temp_image_"+str(kk)+".tif"
                            image.save(filepath)
                            self.m_map.append(filepath)
                        self.loadImageData(self.m_map[0], True)
                        self.buttonPlus.setEnabled(True)
                        self.buttonMinus.setEnabled(True)
                        self.slideBar.setMaximum(len(self.m_map)-1)
                        self.slideBar.setValue(0)
                        self.slideBar.setEnabled(True)
                        self.labelSliceId.setText("Slice = 1")

    # @Slot()
    def openImage(self):
        options = QtWidgets.QFileDialog.Options()
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self,"Open Tomo", "","Image Files (*.tif);;Image Files (*.tiff)", options=options)
        if files:
            if len(self.m_map) > 0:
                self.removeTempImagens()
            self.m_map.clear() # remove all items
            for filepath in files:
                img = Image.open(filepath)
                if img.n_frames > 1:
                    dirname = os.path.dirname(filepath)
                    filename = os.path.basename(filepath)
                    split_tiff_stack(input_filepath=filepath, output_filepath=os.path.join(dirname, filename + "_split", filename))
                    filepaths = os.listdir(os.path.join(dirname, filename + "_split"))
                    for filepath in filepaths:
                        self.m_map.append(os.path.join(dirname, filename + "_split", filepath))
                else:
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
    def seives(self):
       # check if there is at least one image open, and then proceed:
        if len(self.m_map) == 0:
            return
        diameter, ok = QtWidgets.QInputDialog.getInt(self,"Sieve Choice","Diameter:", 1, 1, 2024, 1)
        if ok:
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
            #print(imgLabels)
            count = 0
            numRemoved = 0
            for loc in ndimage.find_objects(imgLabels):
                count+=1
                #print(imgLabels[loc].shape)
                #print(imgLabels[loc])
                d2 = imgLabels[loc].shape[0]*imgLabels[loc].shape[0]+imgLabels[loc].shape[1]*imgLabels[loc].shape[1]+imgLabels[loc].shape[2]*imgLabels[loc].shape[2]
                #print(d2)
                if d2 < diameter*diameter:
                    #print("Remove")
                    imgLabels[imgLabels==count] = 0
                    numRemoved+=1
            #print(imgLabels)
            imgLabels[imgLabels>0] = -1
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
            sm = "Number of Grains: "+str(numLabels)+"  \nNumber of regions that were removed: "+str(numRemoved)
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
    def colorsPerSlice(self):
        # check if there is at least one image open, and then proceed:
        if len(self.m_map) == 0:
            return
        materials = {}
        materials_slices = defaultdict(list)
        vol = self.m_data.shape[1]*self.m_data.shape[0]
        vol_total = vol*len(self.m_map)
        for filepath in self.m_map:
            self.loadImageData(filepath, False)
            mat_i, cmat_i = np.unique(self.m_data, return_counts=True)
            for i in range(len(mat_i)):
                if mat_i[i] in materials:
                    materials[mat_i[i]] += cmat_i[i]
                else:
                    materials[mat_i[i]] = cmat_i[i]
        for i in materials:
            materials_slices[i] = []
        for filepath in self.m_map:
            self.loadImageData(filepath, False)
            mat_i, cmat_i = np.unique(self.m_data, return_counts=True)
            for i in range(len(mat_i)):
                materials_slices[mat_i[i]].append(cmat_i[i] * 100.0/vol)
            if len(mat_i) < len(materials):
                for j in materials_slices.keys():
                    if j not in mat_i:
                        materials_slices[j].append(0.0)
        materials = dict(sorted(materials.items(), key=lambda x: x[0]))
        materials_slices = dict(sorted(materials_slices.items(), key=lambda x: x[0]))
        mat = np.array(list(materials.keys()))
        cmat = np.array(list(materials.values()))
        cmat = cmat*100.0/vol_total
        # plot colors per slice
        for f in range(mat.shape[0]):
            plt.figure(f)
            plt.title('%% of material %s per slice' % mat[f])
            plt.axis([0, len(materials_slices[mat[f]])-1, 0.0, 100.0])
            plt.xlabel('Slice')
            plt.ylabel('%% of material %s' % mat[f])
            x_slices = list(range(len(materials_slices[mat[f]])))
            y_percent = materials_slices[mat[f]]
            maxpercent = max(y_percent)
            minpercent = min(y_percent)
            maxpercentslice = x_slices[y_percent.index(maxpercent)]
            minpercentslice = x_slices[y_percent.index(minpercent)]
            maxtext = 'max = {:.2f}%\nslice nº {:.0f}'.format(maxpercent, maxpercentslice)
            mintext = 'min = {:.2f}%\nslice nº {:.0f}'.format(minpercent, minpercentslice)
            bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
            slicepctg, = plt.plot(x_slices, y_percent, 'g', label = 'slice %')
            avrgpctg, = plt.plot(range(len(materials_slices[mat[f]])), [cmat[f]]*len(materials_slices[mat[f]]), 'r--', label = 'average (%.2f%%)' % cmat[f])
            x_maxbox = maxpercentslice/len(materials_slices[mat[f]]) + 0.2
            x_minbox = minpercentslice/len(materials_slices[mat[f]]) + 0.2
            y_maxbox = maxpercent/100 + 0.1
            y_minbox = minpercent/100 - 0.1
            arrowprops_max=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
            arrowprops_min=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=120")
            kw_max = dict(xycoords='data', textcoords='axes fraction', arrowprops=arrowprops_max, bbox=bbox_props, ha="left", va="bottom")
            kw_min = dict(xycoords='data', textcoords='axes fraction', arrowprops=arrowprops_min, bbox=bbox_props, ha="left", va="top")
            if maxpercent > 80 and maxpercentslice < len(materials_slices[mat[f]])/2:
                y_maxbox = maxpercent/100 - 0.1
                arrowprops_max=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=120")
                kw_max = dict(xycoords='data', textcoords='axes fraction', arrowprops=arrowprops_max, bbox=bbox_props, ha="left", va="top")
            elif maxpercent < 80 and maxpercentslice > len(materials_slices[mat[f]])/2:
                x_maxbox = maxpercentslice/len(materials_slices[mat[f]]) - 0.1
                arrowprops_max=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=120")
                kw_max = dict(xycoords='data', textcoords='axes fraction', arrowprops=arrowprops_max, bbox=bbox_props, ha="right", va="bottom")
            elif maxpercent > 80 and maxpercentslice > len(materials_slices[mat[f]])/2:
                x_maxbox = maxpercentslice/len(materials_slices[mat[f]]) - 0.1
                y_maxbox = maxpercent/100 - 0.1
                kw_max = dict(xycoords='data', textcoords='axes fraction', arrowprops=arrowprops_max, bbox=bbox_props, ha="right", va="top")
            if minpercent < 20 and minpercentslice < len(materials_slices[mat[f]])/2:
                y_minbox = minpercent/100 + 0.1
                arrowprops_min=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
                kw_min = dict(xycoords='data', textcoords='axes fraction', arrowprops=arrowprops_min, bbox=bbox_props, ha="left", va="bottom")
            elif minpercent > 20 and minpercentslice > len(materials_slices[mat[f]])/2:
                x_minbox = minpercentslice/len(materials_slices[mat[f]]) - 0.1
                arrowprops_min=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
                kw_min = dict(xycoords='data', textcoords='axes fraction', arrowprops=arrowprops_min, bbox=bbox_props, ha="right", va="top")
            elif minpercent < 20 and minpercentslice > len(materials_slices[mat[f]])/2:
                x_minbox = minpercentslice/len(materials_slices[mat[f]]) - 0.1
                y_minbox = minpercent/100 + 0.1
                kw_min = dict(xycoords='data', textcoords='axes fraction', arrowprops=arrowprops_min, bbox=bbox_props, ha="right", va="bottom")
            plt.annotate(maxtext, xy=(maxpercentslice, maxpercent), xytext=(x_maxbox, y_maxbox), **kw_max)
            plt.annotate(mintext, xy=(minpercentslice, minpercent), xytext=(x_minbox, y_minbox), **kw_min)
            stddev = np.std(materials_slices[mat[f]])
            varcoef = (stddev / cmat[f]) * 100
            stddev_leg = mpatches.Patch(color='white', label='Standard deviation: %.2f (%%)' % stddev)
            varcoef_leg = mpatches.Patch(color='white', label='Coefficient of variation: %.2f%%' % varcoef)
            plt.legend(handles=[slicepctg, avrgpctg, stddev_leg, varcoef_leg], loc='best')
            plt.show()

    # @Slot()
    def aboutDlg(self):
        sm = """<center>pyTomoViewer</center><center>Version 1.0.0</center><center>2020</center><center>License GPL 3.0</center><br><center>The authors and the involved Institutions are not responsible for the use or bad use of the program and their results. The authors have no legal dulty or responsability for any person or company for the direct or indirect damage caused resulting from the use of any information or usage of the program available here. The user is responsible for all and any conclusion made with the program. There is no warranty for the program use.</center>"""
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


def run():
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


def split_tiff_stack(input_filepath, output_filepath):
    img = io.imread(input_filepath)
    if img.ndim == 3:
        if not os.path.exists(os.path.dirname(output_filepath)):
            os.makedirs(os.path.dirname(output_filepath))
        for i in range(img.shape[2]):
            io.imsave(output_filepath + "_" + str(i) + ".tif", img[:, :, i], check_contrast=False)


if __name__ == '__main__':
    run()
