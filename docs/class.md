# pytomoviewer

A simple GUI that allows to load and visualize a stack of images (slices) from micro-computed tomography converted to 2D (or 3D) Numpy arrays.
Basic current capabilities:
- Visualize the image stack;
- Plot image histogram;
- Resample the data;
- Export to RAW (JSON) format;
- Identify unconnected pores;
- Replicate a single image;

# Classes

```pytomoviewer``` has only one class

class TomoViewer (QtWidgets.QMainWindow)

## Methods

Displays a message box with the software information

def aboutDlg (self)
```
def aboutDlg(self):
    sm = """pyTomoViewer\nVersion 1.0.0\n2020\nLicense GPL 3.0\n\nThe authors and the involved Institutions are not responsible for the use or bad use of the program and their results. The authors have no legal dulty or responsability for any person or company for the direct or indirect damage caused resulting from the use of any information or usage of the program available here. The user is responsible for all and any conclusion made with the program. There is no warranty for the program use. """
    msg = QtWidgets.QMessageBox()
    msg.setText(sm)
    msg.setWindowTitle("About")
    msg.exec_()
```

Change the image of the TIFF file stack

def changeValue (self, _value)
```
def changeValue(self, _value):
    filename = self.m_map[_value]
    print(filename)
    self.loadImageData(filename,True)
    self.labelSliceId.setText("Slice = "+str(_value+1))
```

This method opens an auxiliary dialog for users to enter a threshold (between 0 and 255) to perform a global segmentation (binarization) using the histogram data, and then generating binary images. This is useful to convert a full gray-scale image in a binary image, very practical in fluid analysis software.

def convertToBinary(self)
```
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
```

This method opens an auxiliary dialog for users to browse for a directory on which three output files will be created and placed. In addition to generating a binary RAW file that represents the original image, read from a TIFF file, or a stack of TIFF files, ```pytomoviewer``` provides a JSON and a Neutral file, that can be used as input for numerical analysis software. This button does nothing if no image has been read.

def exportImage(self)
```
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
```

Create a list of filepaths, convert Grayscale8 and call ```plotImage()```

def loadImageData(self, _filepath, _updateWindow)
```
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
```

This method opens an auxiliary dialog for users to browse for a TIFF file, or a stack of TIFF files, that represent, respectively, a 2D or 3D image.

def openImage(self)
```
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
```

Plot the histogram of the current image

def plotHistogram(self)
```
def plotHistogram(self):
    self.figure.clear()
    ax = self.figure.add_subplot(111)
    ax.hist(self.m_data.flatten(), bins=256, fc='k', ec='k')
    ax.set(xlim=(-4, 259))
    ax.figure.canvas.draw()
    self.buttonPlot.setEnabled(True)          
    self.buttonHist.setEnabled(False) 
```

Plot the current image

def plotImage(self)
```
def plotImage(self):
    self.figure.clear()
    ax = self.figure.add_subplot(111)
    img = ax.imshow(self.m_data,vmin=0,vmax=255)      
    self.figure.colorbar(img)
    ax.figure.canvas.draw()
    self.buttonPlot.setEnabled(False)          
    self.buttonHist.setEnabled(True)  
```

This method remove the old list of filepath

def removeTempImagens(self)
```
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
```

This method runs a filter that removes unconnected pores from the current image (2D), or stack of images (3D). This is useful to generate input for fluid analysis software.

def removeUnconnected(self)
```
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
```

This method opens an auxiliary dialog for users to enter a number of copies to generate from the image being displayed, creating a stack of images. This is useful to expand 2D images to 3D. This button does nothing if no image has been read.

def replicateImage(self)
```
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
```

Change the image of the TIFF file stack

def slideMoveDown(self)
```
def slideMoveDown(self):
    self.slideBar.setValue(self.slideBar.value()-1)
```

Change the image of the TIFF file stack

def slideMoveUp(self)
```
def slideMoveUp(self):
    self.slideBar.setValue(self.slideBar.value()+1)
```