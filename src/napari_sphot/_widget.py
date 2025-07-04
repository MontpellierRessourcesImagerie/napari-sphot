from typing import TYPE_CHECKING
import math
import matplotlib.pyplot as plt
import numpy as np
from PIL.ImageOps import scale
from scipy.ndimage import median_filter
from qtpy.QtWidgets import QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QFormLayout, QGroupBox
from napari.layers import Image
from napari.utils.events import Event
from napari.qt.threading import create_worker
from sphot.image import Segmentation
from sphot.image import SpotDetection
from sphot.image import SpotPerCellAnalyzer
from sphot.image import Correlator
from sphot.measure import TableTool
from napari_sphot.qtutil import WidgetTool
from napari_sphot.napari_util import NapariUtil
from napari_sphot.image import TiffFileTags
from napari_sphot.qtutil import TableView
if TYPE_CHECKING:
    import napari



class SpatialHeterogenityOfTranscriptionWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        self.scale = 50 #@TODO: get the scale from the input image
        self.fieldWidth = 300
        self.labelOfNucleus = 1
        self.medianFilterSize = 5
        self.medianFilterSizeInput = None
        self.segmentation = None
        self.layer = None
        self.cropImageLabelsCombo = None
        self.cropImageCombo = None
        self.cropLabel = 1
        self.cropLabelInput = None
        self.ccInputACombo = None
        self.ccInputBCombo = None
        self.spotsLayer = None
        self.gFunctionInput = None
        self.gFunctionSpotsCombo = None
        self.gFunctionLabelsCombo = None
        self.measurements = {}
        self.table = TableView(self.measurements)
        self.napariUtil = NapariUtil(self.viewer)
        self.pointsLayers = self.napariUtil.getPointsLayers()
        self.labelLayers = self.napariUtil.getLabelLayers()
        self.imageLayers = self.napariUtil.getImageLayers()
        self.viewer = viewer
        self.createLayout()
        self.viewer.layers.events.inserted.connect(self.onLayerAddedOrRemoved)
        self.viewer.layers.events.removed.connect(self.onLayerAddedOrRemoved)
        self.tableDockWidget = self.viewer.window.add_dock_widget(self.table,
                                                                  area='right', name='measurements', tabify=False)


    def createLayout(self):
        medianFilterLayout = QHBoxLayout()
        medianFilterLabel, self.medianFilterSizeInput = WidgetTool.getLineInput(self, "Filter Size: ",
                                                                      self.medianFilterSize,
                                                                      self.fieldWidth,
                                                                      self.medianFilterSizeChanged)
        medianFilterButton = QPushButton("Median Filter")
        medianFilterButton.clicked.connect(self._onMedianFilterButtonClicked)
        medianFilterLayout.addWidget(medianFilterLabel)
        medianFilterLayout.addWidget(self.medianFilterSizeInput)
        medianFilterLayout.addWidget(medianFilterButton)
        segmentImageButton = QPushButton("Segment Image")
        segmentImageButton.clicked.connect(self._onSegmentImageButtonClicked)
        detectSpotsButton = QPushButton("Detect Spots")
        detectSpotsButton.clicked.connect(self._onDetectSpotsButtonClicked)
        gFunctionGroupBox = self.getSpatialStatsWidget()
        measureGroupBox = self.getMeasurementsWidget()
        crossCorrelationGroupBox = self.getCrossCorrelationWidget()
        mainLayout = QVBoxLayout()
        mainLayout.addLayout(medianFilterLayout)
        mainLayout.addWidget(segmentImageButton)
        mainLayout.addWidget(detectSpotsButton)
        mainLayout.addWidget(gFunctionGroupBox)
        mainLayout.addWidget(measureGroupBox)
        mainLayout.addWidget(crossCorrelationGroupBox)
        self.setLayout(mainLayout)


    def getCrossCorrelationWidget(self):
        crossCorrelationGroupBox = QGroupBox("Cross Correlation")
        ccMainLayout = QVBoxLayout()
        ccCropImageLabelsLayout = QHBoxLayout()
        ccCropImageLayout = QHBoxLayout()
        ccCropLabelLayout = QHBoxLayout()
        cropImageLabelsLabel, self.cropImageLabelsCombo = WidgetTool.getComboInput(self, "Labels: ", self.labelLayers)
        cropImageLabel, self.cropImageCombo = WidgetTool.getComboInput(self, "Image: ", self.imageLayers)
        cropLabelLabel, self.cropLabelInput = WidgetTool.getLineInput(self, "Label: ", self.cropLabel,
                                                                            self.fieldWidth, self.cropLabelInputChanged)
        cropButton = QPushButton("Crop")
        cropButton.clicked.connect(self._onCropButtonPressed)
        ccCropImageLabelsLayout.addWidget(cropImageLabelsLabel)
        ccCropImageLabelsLayout.addWidget(self.cropImageLabelsCombo)
        ccCropImageLayout.addWidget(cropImageLabel)
        ccCropImageLayout.addWidget(self.cropImageCombo)
        ccCropLabelLayout.addWidget(cropLabelLabel)
        ccCropLabelLayout.addWidget(self.cropLabelInput)
        ccCropLabelLayout.addWidget(cropButton)

        ccInputALabel, self.ccInputACombo = WidgetTool.getComboInput(self, "Input A: ", self.imageLayers)
        ccInputBLabel, self.ccInputBCombo = WidgetTool.getComboInput(self, "Input B: ", self.imageLayers)
        correlationButton = QPushButton("Correlate")
        correlationButton.clicked.connect(self._onCorrelationButtonPressed)
        inputALayout = QHBoxLayout()
        inputALayout.addWidget(ccInputALabel)
        inputALayout.addWidget(self.ccInputACombo)
        inputBLayout = QHBoxLayout()
        inputBLayout.addWidget(ccInputBLabel)
        inputBLayout.addWidget(self.ccInputBCombo)
        correlationButtonLayout = QHBoxLayout()
        correlationButtonLayout.addWidget(correlationButton)
        ccMainLayout.addLayout(ccCropImageLabelsLayout)
        ccMainLayout.addLayout(ccCropImageLayout)
        ccMainLayout.addLayout(ccCropLabelLayout)
        ccMainLayout.addLayout(inputALayout)
        ccMainLayout.addLayout(inputBLayout)
        ccMainLayout.addLayout(correlationButtonLayout)
        crossCorrelationGroupBox.setLayout(ccMainLayout)
        return crossCorrelationGroupBox


    def getSpatialStatsWidget(self):
        gFunctionGroupBox = QGroupBox("Spatial-Statistics")
        gFunctionLabel, self.gFunctionInput = WidgetTool.getLineInput(self, "Label of nucleus: ",
                                                                      self.labelOfNucleus,
                                                                      self.fieldWidth,
                                                                      self.gFunctionInputChanged)
        fFunctionButton = QPushButton("F-Function")
        gFunctionButton = QPushButton("G-Function")
        hFunctionButton = QPushButton("H-Function")
        fFunctionButton.clicked.connect(self._onFFunctionButtonClicked)
        gFunctionButton.clicked.connect(self._onGFunctionButtonClicked)
        hFunctionButton.clicked.connect(self._onHFunctionButtonClicked)
        gFunctionMainLayout = QVBoxLayout()
        gFunctionGroupBox.setLayout(gFunctionMainLayout)
        gFunctionSpotsLabel, self.gFunctionSpotsCombo = WidgetTool.getComboInput(self, "Spots: ", self.pointsLayers)
        gFunctionLabelsLabel, self.gFunctionLabelsCombo = WidgetTool.getComboInput(self, "Cell labels: ",
                                                                                   self.labelLayers)
        gFunctionLayersLayout = QHBoxLayout()
        gFunctionLayersLayout.addWidget(gFunctionSpotsLabel)
        gFunctionLayersLayout.addWidget(self.gFunctionSpotsCombo)
        gFunctionLayersLayout.addWidget(gFunctionLabelsLabel)
        gFunctionLayersLayout.addWidget(self.gFunctionLabelsCombo)
        FGHLayout = QVBoxLayout()
        gFunctionCellLayout = QHBoxLayout()
        gFunctionCellLayout.addWidget(gFunctionLabel)
        gFunctionCellLayout.addWidget(self.gFunctionInput)
        FGHLayout.addLayout(gFunctionCellLayout)
        buttonsLayout = QHBoxLayout()
        buttonsLayout.addWidget(fFunctionButton)
        buttonsLayout.addWidget(gFunctionButton)
        buttonsLayout.addWidget(hFunctionButton)
        FGHLayout.addLayout(buttonsLayout)
        gFunctionMainLayout.addLayout(gFunctionLayersLayout)
        gFunctionMainLayout.addLayout(FGHLayout)
        return gFunctionGroupBox


    def getMeasurementsWidget(self):
        measurementsGroupBox = QGroupBox("Measurements")
        measureButton = QPushButton("Measure")
        measureButton.clicked.connect(self._onMeasureButtonClicked)
        mainLayout = QVBoxLayout()
        measurementsGroupBox.setLayout(mainLayout)
        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(measureButton)
        convexHullButton = QPushButton("Convex Hull")
        convexHullButton.clicked.connect(self._onConvexHullButtonClicked)
        delaunayButton = QPushButton("Delaunay")
        delaunayButton.clicked.connect(self._onDelaunayButtonClicked)
        voronoiButton = QPushButton("Voronoi")
        voronoiButton.clicked.connect(self._onVoronoiButtonClicked)
        displayButtonsLayout = QHBoxLayout()
        displayButtonsLayout.addWidget(convexHullButton)
        displayButtonsLayout.addWidget(delaunayButton)
        displayButtonsLayout.addWidget(voronoiButton)
        mainLayout.addLayout(buttonLayout)
        mainLayout.addLayout(displayButtonsLayout)
        return measurementsGroupBox


    def _onMedianFilterButtonClicked(self):
        layer = self.getActiveLayer()
        if not layer or not type(layer) is Image:
            return
        self.medianFilterSize = int(self.medianFilterSizeInput.text().strip())
        filteredImage = median_filter(layer.data, self.medianFilterSize)
        self.viewer.add_image(filteredImage, name=layer.name + "_median_" + str(self.medianFilterSize ))


    def _onMeasureButtonClicked(self):
        text = self.gFunctionSpotsCombo.currentText()
        spots = self.napariUtil.getDataOfLayerWithName(text)
        text = self.gFunctionLabelsCombo.currentText()
        labels = self.napariUtil.getDataOfLayerWithName(text)
        analyzer = SpotPerCellAnalyzer(spots, labels, self.scale)
        baseMeasurements = analyzer.getBaseMeasurements()
        nnMeasurements = analyzer.getNNMeasurements()
        nnMeasurements.pop('label')
        TableTool.addColumnsTableAToB(nnMeasurements, baseMeasurements)
        convexHullMeasurements = analyzer.getConvexHullMeasurements()
        convexHullMeasurements.pop('label')
        TableTool.addColumnsTableAToB(convexHullMeasurements, baseMeasurements)
        delaunayMeasurements = analyzer.getDelaunayMeasurements()
        delaunayMeasurements.pop('label')
        TableTool.addColumnsTableAToB(delaunayMeasurements, baseMeasurements)
        self.measurements = baseMeasurements
        self.tableDockWidget.close()
        self.table = TableView(self.measurements)
        self.tableDockWidget = self.viewer.window.add_dock_widget(self.table, area='right', name='measurements',
                                                                  tabify=False)

    def _onConvexHullButtonClicked(self):
        label = int(self.gFunctionInput.text().strip())
        if not label:
            return
        text = self.gFunctionSpotsCombo.currentText()
        spots = self.napariUtil.getDataOfLayerWithName(text)
        text = self.gFunctionLabelsCombo.currentText()
        labels = self.napariUtil.getDataOfLayerWithName(text)
        analyzer = SpotPerCellAnalyzer(spots, labels, self.scale)
        hull = analyzer.getConvexHull(label)
        self.viewer.add_points(hull.points[hull.vertices], scale=(self.scale, self.scale, self.scale))
        self.viewer.add_shapes(hull.points[hull.simplices], shape_type='polygon', scale=(self.scale, self.scale, self.scale))


    def _onDelaunayButtonClicked(self):
        label = int(self.gFunctionInput.text().strip())
        if not label:
            return
        text = self.gFunctionSpotsCombo.currentText()
        spots = self.napariUtil.getDataOfLayerWithName(text)
        text = self.gFunctionLabelsCombo.currentText()
        labels = self.napariUtil.getDataOfLayerWithName(text)
        analyzer = SpotPerCellAnalyzer(spots, labels, self.scale)
        tess = analyzer.getDelaunay(label)
        self.viewer.add_shapes(tess.points[tess.simplices], scale=(self.scale, self.scale, self.scale), shape_type='polygon')


    def _onVoronoiButtonClicked(self):
        label = int(self.gFunctionInput.text().strip())
        if not label:
            return
        text = self.gFunctionSpotsCombo.currentText()
        spots = self.napariUtil.getDataOfLayerWithName(text)
        text = self.gFunctionLabelsCombo.currentText()
        labels = self.napariUtil.getDataOfLayerWithName(text)
        analyzer = SpotPerCellAnalyzer(spots, labels, self.scale)
        regions = analyzer.getVoronoiRegions(label)
        self.viewer.add_shapes(regions, scale=(self.scale, self.scale, self.scale), shape_type='polygon')


    def _onSegmentImageButtonClicked(self):
        self.layer = self.getActiveLayer()
        if not self.layer or not type(self.layer) is Image:
            return
        self.segmentation = Segmentation(self.layer.data)
        worker = create_worker(self.segmentation.run,
                               _progress={'total': 5, 'desc': 'Segmenting cells...'})
        worker.finished.connect(self.onSegmentationFinished)
        worker.start()


    def _onDetectSpotsButtonClicked(self):
        self.spotsLayer = self.getActiveLayer()
        if not self.spotsLayer or not type(self.spotsLayer) is Image:
            return
        self.detection = SpotDetection(self.spotsLayer.data)
        worker = create_worker(self.detection.run,
                               _progress={'total': 2, 'desc': 'Detecting spots...'})
        worker.finished.connect(self.onDetectionFinished)
        worker.start()


    def _onGFunctionButtonClicked(self):
        label = int(self.gFunctionInput.text().strip())
        if not label:
            return
        self.labelOfNucleus = label
        text = self.gFunctionSpotsCombo.currentText()
        spots = self.napariUtil.getDataOfLayerWithName(text)
        text = self.gFunctionLabelsCombo.currentText()
        labels = self.napariUtil.getDataOfLayerWithName(text)
        analyzer = SpotPerCellAnalyzer(spots, labels, self.scale)
        analyzer.calculateGFunction()
        envelop = analyzer.getEnvelopForNNDistances(label, 100)
        ax = plt.subplot()
        analyzer.nnEcdfs[label].cdf.plot(ax)
        ax.set_xlabel('distances [nm]')
        ax.set_ylabel('Empirical CDF')
        maxDist = np.max(analyzer.nnDistances[label][0])
        xValues = np.array(list(range(0, math.floor(maxDist + 1), analyzer.scale)))
        plt.plot(xValues, envelop[0], 'r--')
        plt.plot(xValues, envelop[1], 'g--')
        plt.plot(xValues, envelop[2], 'g--')
        plt.plot(xValues, envelop[3], 'r--')
        plt.show()


    def _onHFunctionButtonClicked(self):
        label = int(self.gFunctionInput.text().strip())
        if not label:
            return
        self.labelOfNucleus = label
        text = self.gFunctionSpotsCombo.currentText()
        spots = self.napariUtil.getDataOfLayerWithName(text)
        text = self.gFunctionLabelsCombo.currentText()
        labels = self.napariUtil.getDataOfLayerWithName(text)
        analyzer = SpotPerCellAnalyzer(spots, labels, self.scale)
        analyzer.calculateHFunction()
        envelop = analyzer.getEnvelopForAllDistances(label, 100)
        ax = plt.subplot()
        analyzer.adEcdfs[label].cdf.plot(ax)
        ax.set_xlabel('distances [nm]')
        ax.set_ylabel('Empirical CDF')
        maxDist = np.max(analyzer.allDistances[label][0])
        xValues = np.array(list(range(0, math.floor(maxDist + 1), analyzer.scale)))
        plt.plot(xValues, envelop[0], 'r--')
        plt.plot(xValues, envelop[1], 'g--')
        plt.plot(xValues, envelop[2], 'g--')
        plt.plot(xValues, envelop[3], 'r--')
        plt.show()


    def _onFFunctionButtonClicked(self):
        label = int(self.gFunctionInput.text().strip())
        if not label:
            return
        self.labelOfNucleus = label
        text = self.gFunctionSpotsCombo.currentText()
        spots = self.napariUtil.getDataOfLayerWithName(text)
        text = self.gFunctionLabelsCombo.currentText()
        labels = self.napariUtil.getDataOfLayerWithName(text)
        analyzer = SpotPerCellAnalyzer(spots, labels, self.scale)
        analyzer.calculateFFunction()
        envelop = analyzer.getEnvelopForEmptySpaceDistances(label, 100)
        ax = plt.subplot()
        analyzer.esEcdfs[label].cdf.plot(ax)
        ax.set_xlabel('distances [nm]')
        ax.set_ylabel('Empirical CDF')
        maxDist = np.max(analyzer.emptySpaceDistances[label][0])
        xValues = np.array(list(range(0, math.floor(maxDist + 1), analyzer.scale)))
        plt.plot(xValues, envelop[0], 'r--')
        plt.plot(xValues, envelop[1], 'g--')
        plt.plot(xValues, envelop[2], 'g--')
        plt.plot(xValues, envelop[3], 'r--')
        plt.show()


    def _onCropButtonPressed(self):
        text = self.cropImageLabelsCombo.currentText()
        labels = self.napariUtil.getDataOfLayerWithName(text)
        text = self.cropImageCombo.currentText()
        image = self.napariUtil.getDataOfLayerWithName(text)
        self.cropLabel = int(self.cropLabelInput.text().strip())
        if not self.cropLabel:
            self.cropLabel = 1
            return
        analyzer = SpotPerCellAnalyzer(None, labels, 1)
        croppedImage = analyzer.cropImageForLabel(image, self.cropLabel)
        self.viewer.add_image(croppedImage, name=text + "_c" + str(self.cropLabel))


    def _onCorrelationButtonPressed(self):
        text1 = self.ccInputACombo.currentText()
        text2 = self.ccInputBCombo.currentText()
        if not text1 or not text2:
            return
        imageA = self.napariUtil.getDataOfLayerWithName(text1)
        imageB = self.napariUtil.getDataOfLayerWithName(text2)
        correlator = Correlator(imageA, imageB)
        correlator.calculateCrossCorrelationProfile()
        self.viewer.add_image(correlator.correlationImage, name="corr.: " + text1 + "-" + text2,
                                                           colormap='inferno', blending='additive')
        plt.plot(correlator.correlationProfile[0], correlator.correlationProfile[1])
        data = np.asarray([correlator.correlationProfile[0], correlator.correlationProfile[1]])
        np.savetxt("corr.: " + text1 + "-" + text2 + ".csv", data, delimiter=",")
        plt.show()


    def getActiveLayer(self):
        if len(self.viewer.layers) == 0:
            return None
        if len(self.viewer.layers) == 1:
            layer = self.viewer.layers[0]
        else:
            layer = self.viewer.layers.selection.active
        return layer


    def onSegmentationFinished(self):
        self.viewer.add_labels(self.segmentation.labels, scale=self.layer.scale, blending='additive')


    def onDetectionFinished(self):
        self.viewer.add_points(self.detection.spots, scale=self.spotsLayer.scale, blending='additive', size=1)


    def gFunctionInputChanged(self):
        pass


    def medianFilterSizeChanged(self):
        pass


    def cropLabelInputChanged(self):
        pass


    def onLayerAddedOrRemoved(self, event: Event):
        self.updateLayerSelectionComboBoxes()


    def updateLayerSelectionComboBoxes(self):
        labelComboBoxes = [self.gFunctionLabelsCombo, self.cropImageLabelsCombo]
        spotComboBoxes = [self.gFunctionSpotsCombo]
        imageComboBoxes = [self.cropImageCombo, self.ccInputACombo, self.ccInputBCombo]
        labelLayers = self.napariUtil.getLabelLayers()
        spotLayers = self.napariUtil.getPointsLayers()
        imageLayers = self.napariUtil.getImageLayers()
        for comboBox in labelComboBoxes:
            WidgetTool.replaceItemsInComboBox(comboBox, labelLayers)
        for comboBox in spotComboBoxes:
            WidgetTool.replaceItemsInComboBox(comboBox, spotLayers)
        for comboBox in imageComboBoxes:
            WidgetTool.replaceItemsInComboBox(comboBox, imageLayers)