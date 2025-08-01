from typing import TYPE_CHECKING
import math
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from napari.utils import notifications
from napari_bigfish.bigfishapp import BigfishApp
from sphot.filter import MedianFilter
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QGroupBox, QCheckBox
from qtpy.QtWidgets import QFormLayout
from napari.layers import Image
from napari.layers import Labels
from napari.utils.events import Event
from napari.qt.threading import create_worker
from sphot.image import Segmentation
from sphot.image import SpotDetection
from sphot.image import DecomposeDenseRegions
from sphot.image import Correlator
from sphot.image import FFunctionTask
from sphot.image import GFunctionTask
from sphot.image import HFunctionTask
from sphot.image import ConvexHullTask
from sphot.image import DelaunayTask
from sphot.image import VoronoiTask
from sphot.image import MeasureTask
from sphot.image import CropLabelTask
from sphot.measure import TableTool
from napari_sphot.qtutil import WidgetTool
from napari_sphot.napari_util import NapariUtil
from napari_sphot.qtutil import TableView
from napari_sphot.options import Options
if TYPE_CHECKING:
    import napari



class SpatialHeterogeneityOfTranscriptionWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        self.bigFishApp = None
        self.fieldWidth = 50
        self.comboMaxWidth = 150
        self.labelOfNucleus = 1
        self.medianFilterSize = 2
        self.medianFilterSizeInput = None
        self.medianFilter = None
        self.backgroundSigmaXYInput = None
        self.backgroundSigmaZInput = None
        self.backgroundSigmaXY = 2.3
        self.backgroundSigmaZ = 2.3
        self.segmentation = None
        self.keepLabelsText = ""
        self.keepLabelsInput = None
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
        self.fFunctionTask = None
        self.gFunctionTask = None
        self.hFunctionTask = None
        self.convexHullTask = None
        self.delaunayTask = None
        self.voronoiTask = None
        self.measureTask = None
        self.correlator = None
        self.cropLabelTask = None
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
        self.decomposeDense = None
        self.detection = None



    @classmethod
    def getOptionsButton(cls, callback):
        resourcesPATH = os.path.join(Path(__file__).parent.resolve(), "resources", "gear.png")
        gearIcon = QIcon(resourcesPATH)
        optionsButton = QPushButton()
        optionsButton.setIcon(gearIcon)
        optionsButton.clicked.connect(callback)
        return optionsButton


    def createLayout(self):
        preProcessingGroupBox = self.getPreProcessingWidget()
        segmentationGroupBox = self.getSegmentationWidget()
        gFunctionGroupBox = self.getSpatialStatsWidget()
        measureGroupBox = self.getMeasurementsWidget()
        crossCorrelationGroupBox = self.getCrossCorrelationWidget()
        mainLayout = QVBoxLayout()
        mainLayout.addWidget(preProcessingGroupBox)
        mainLayout.addWidget(segmentationGroupBox)
        mainLayout.addWidget(gFunctionGroupBox)
        mainLayout.addWidget(measureGroupBox)
        mainLayout.addWidget(crossCorrelationGroupBox)
        self.setLayout(mainLayout)


    def getSegmentationWidget(self):
        groupBoxLayout = QGroupBox("Segmentation/Detection")
        mainLayout = QVBoxLayout()
        segmentImageButton = QPushButton("Segment Image")
        segmentImageButton.clicked.connect(self._onSegmentImageButtonClicked)
        segmentImageOptionsButton = self.getOptionsButton(self._onSegmentImageOptionsClicked)
        segmentImageOptionsButton.setMaximumWidth(50)

        keepLabelsLabel, self.keepLabelsInput = WidgetTool.getLineInput(self, "Keep labels: ",
                                                                                self.keepLabelsText,
                                                                                self.fieldWidth*2,
                                                                                self.keepLabelsChanged)
        keepLabelsButton = QPushButton("Keep Labels")
        keepLabelsButton.clicked.connect(self._onKeepLabelsButtonClicked)

        detectSpotsButton = QPushButton("Detect Spots")
        detectSpotsButton.clicked.connect(self._onDetectSpotsButtonClicked)
        detectSpotsOptionsButton = self.getOptionsButton(self._onDetectSpotsOptionsClicked)
        detectSpotsOptionsButton.setMaximumWidth(50)
        segmentationLayout = QHBoxLayout()
        keepLabelsLayout = QHBoxLayout()
        detectionLayout = QHBoxLayout()
        segmentationLayout.addWidget(segmentImageButton)
        segmentationLayout.addWidget(segmentImageOptionsButton)
        keepLabelsLayout.addWidget(keepLabelsLabel)
        keepLabelsLayout.addWidget(self.keepLabelsInput)
        keepLabelsLayout.addWidget(keepLabelsButton)
        detectionLayout.addWidget(detectSpotsButton)
        detectionLayout.addWidget(detectSpotsOptionsButton)
        mainLayout.addLayout(segmentationLayout)
        mainLayout.addLayout(keepLabelsLayout)
        mainLayout.addLayout(detectionLayout)
        groupBoxLayout.setLayout(mainLayout)
        return groupBoxLayout


    def getPreProcessingWidget(self):
        preProcessingGroupBox = QGroupBox("Pre-Processing")
        preProcessingMainLayout = QVBoxLayout()
        preProcessingGroupBox.setLayout(preProcessingMainLayout)

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
        preProcessingMainLayout.addLayout(medianFilterLayout)

        subtractBackgroundLayout = QHBoxLayout()
        sigmaLayout = QVBoxLayout()
        sigmaXYLayout = QHBoxLayout()
        sigmaZLayout = QHBoxLayout()
        sigmaLayout.addLayout(sigmaXYLayout)
        sigmaLayout.addLayout(sigmaZLayout)
        subtractBackgroundLabelXY, self.backgroundSigmaXYInput = WidgetTool.getLineInput(self, "Sigma XY: ",
                                                                                self.backgroundSigmaXY,
                                                                                self.fieldWidth,
                                                                                self.backgroundSigmaXYChanged)
        subtractBackgroundLabelZ, self.backgroundSigmaZInput = WidgetTool.getLineInput(self, "Sigma Z: ",
                                                                                       self.backgroundSigmaZ,
                                                                                       self.fieldWidth,
                                                                                       self.backgroundSigmaZChanged)
        subtractBackgroundButton = QPushButton("Subtract\nBackground")
        subtractBackgroundButton.clicked.connect(self._onSubtractBackgroundButtonClicked)
        sigmaXYLayout.addWidget(subtractBackgroundLabelXY)
        sigmaXYLayout.addWidget(self.backgroundSigmaXYInput)
        sigmaZLayout.addWidget(subtractBackgroundLabelZ)
        sigmaZLayout.addWidget(self.backgroundSigmaZInput)

        subtractBackgroundLayout.addLayout(sigmaLayout)
        subtractBackgroundLayout.addWidget(subtractBackgroundButton)
        preProcessingMainLayout.addLayout(subtractBackgroundLayout)

        return preProcessingGroupBox


    def getCrossCorrelationWidget(self):
        crossCorrelationGroupBox = QGroupBox("Cross Correlation")
        ccMainLayout = QVBoxLayout()
        ccCropImageLabelsLayout = QHBoxLayout()
        ccCropImageLayout = QHBoxLayout()
        ccCropLabelLayout = QHBoxLayout()
        cropImageLabelsLabel, self.cropImageLabelsCombo = WidgetTool.getComboInput(self, "Labels: ", self.labelLayers)
        self.cropImageLabelsCombo.setMaximumWidth(self.comboMaxWidth)
        cropImageLabel, self.cropImageCombo = WidgetTool.getComboInput(self, "Image: ", self.imageLayers)
        self.cropImageCombo.setMaximumWidth(self.comboMaxWidth)
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
        self.ccInputACombo.setMaximumWidth(self.comboMaxWidth)
        ccInputBLabel, self.ccInputBCombo = WidgetTool.getComboInput(self, "Input B: ", self.imageLayers)
        self.ccInputBCombo.setMaximumWidth(self.comboMaxWidth)
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
        self.gFunctionSpotsCombo.setMaximumWidth(150)
        gFunctionLabelsLabel, self.gFunctionLabelsCombo = WidgetTool.getComboInput(self, "Cell labels: ",
                                                                                   self.labelLayers)
        self.gFunctionLabelsCombo.setMaximumWidth(150)
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


    def _onSegmentImageOptionsClicked(self):
        segmentationOptionsWidget = SegmentationOptionsWidget(self.viewer)
        self.viewer.window.add_dock_widget(segmentationOptionsWidget, area='right', name='Options of Segment Image',
                                                                      tabify = True)


    def _onDetectSpotsOptionsClicked(self):
        detectionOptionsWidget = DetectionOptionsWidget(self.viewer)
        self.viewer.window.add_dock_widget(detectionOptionsWidget, area='right', name='Options of Detect Spots',
                                                                   tabify=True)


    def _onMedianFilterButtonClicked(self):
        self.layer = self.getActiveLayer()
        if not self.layer or not type(self.layer) is Image:
            return
        self.medianFilterSize = int(self.medianFilterSizeInput.text().strip())
        self.medianFilter = MedianFilter(self.layer.data, radius=self.medianFilterSize, name=self.layer.name)
        worker = create_worker(self.medianFilter.run,
                               _progress={'desc': 'Median Filter Running...'})
        worker.finished.connect(self.onMedianFilterFinished)
        worker.start()


    def _onSubtractBackgroundButtonClicked(self):
        self.backgroundSigmaXY = float(self.backgroundSigmaXYInput.text().strip())
        self.backgroundSigmaZ = float(self.backgroundSigmaZInput.text().strip())
        self.bigFishApp = BigfishApp()
        activeLayer =  self.getActiveLayer()
        if not activeLayer:
            notifications.show_error("Subtract background needs an image!")
            return
        message = \
            "Running background subtraction with sigma xy = {}, sigma z = {} on {}."
        notifications.show_info(
            message.format(
                self.backgroundSigmaXY,
                self.backgroundSigmaZ,
                activeLayer.name))
        self.layer = activeLayer
        self.bigFishApp.setData(self.layer.data)
        self.bigFishApp.setSigmaXY(self.backgroundSigmaXY)
        self.bigFishApp.setSigmaZ(self.backgroundSigmaZ)
        worker = create_worker(self.bigFishApp.subtractBackground,
                               _progress={'desc': 'Subtracting Background...'})
        worker.finished.connect(self.onBackgroundSubtractionFinished)
        worker.start()


    def onMedianFilterFinished(self):
        layer = self.viewer.add_image(self.medianFilter.getResult(), name=self.medianFilter.getName()
                                                                  + "_median_" + str(self.medianFilterSize),
                                                             scale=self.layer.scale,
                                                             colormap=self.layer.colormap,
                                                             units=self.layer.units,
                                                             blending=self.layer.blending
                              )
        NapariUtil.copyOriginalPath(self.layer, layer)


    def onBackgroundSubtractionFinished(self):
        layer = self.viewer.add_image(self.bigFishApp.getResult(),
                              name=self.layer.name +
                                     "_background_" +
                                     str(self.backgroundSigmaZ) + "-"
                                     + str(self.backgroundSigmaXY),
                              scale=self.layer.scale,
                              colormap=self.layer.colormap,
                              units=self.layer.units,
                              blending=self.layer.blending
                              )
        NapariUtil.copyOriginalPath(self.layer, layer)


    def _onMeasureButtonClicked(self):
        text = self.gFunctionSpotsCombo.currentText()
        self.layer = self.napariUtil.getLayerWithName(text)
        spots, scale = self.napariUtil.getDataAndScaleOfLayerWithName(text)
        units = self.layer.units
        text = self.gFunctionLabelsCombo.currentText()
        labels = self.napariUtil.getDataOfLayerWithName(text)
        self.measureTask = MeasureTask(spots, labels, scale, units)
        worker = create_worker(self.measureTask.run,
                               _progress={'desc': 'Measuring Features...'})
        worker.finished.connect(self.onMeasureTaskFinished)
        worker.start()


    def onMeasureTaskFinished(self):
        path = NapariUtil.getOriginalPath(self.layer)
        filename = os.path.basename(path)
        dirname = os.path.dirname(path)
        self.measureTask.table['image'] = [filename] * len(self.measureTask.table['label'])
        self.measureTask.table['folder'] = [dirname] * len(self.measureTask.table['label'])
        self.tableDockWidget.close()
        if self.measurements:
            TableTool.addTableAToB(self.measureTask.table, self.measurements)
        else:
            self.measurements = self.measureTask.table
        self.table = TableView(self.measurements)
        self.tableDockWidget = self.viewer.window.add_dock_widget(self.table,
                                                                  area='right',
                                                                  name='measurements',
                                                                  tabify=False)


    def _onConvexHullButtonClicked(self):
        label = int(self.gFunctionInput.text().strip())
        if not label:
            return
        text = self.gFunctionSpotsCombo.currentText()
        spots, scale = self.napariUtil.getDataAndScaleOfLayerWithName(text)
        self.spotsLayer = self.napariUtil.getLayerWithName(text)
        text = self.gFunctionLabelsCombo.currentText()
        labels = self.napariUtil.getDataOfLayerWithName(text)
        self.convexHullTask = ConvexHullTask(spots, labels, scale, label)
        worker = create_worker(self.convexHullTask.run,
                               _progress={'desc': 'Calculating Convex Hull...'})
        worker.finished.connect(self.onConvexHullTaskFinished)
        worker.start()


    def onConvexHullTaskFinished(self):
        hull = self.convexHullTask.result
        scale = self.convexHullTask.scale
        units = self.spotsLayer.units
        self.viewer.add_points(hull.points[hull.vertices], scale=scale, units=units)
        self.viewer.add_shapes(hull.points[hull.simplices], shape_type='polygon', scale=scale, units=units)


    def _onDelaunayButtonClicked(self):
        label = int(self.gFunctionInput.text().strip())
        if not label:
            return
        text = self.gFunctionSpotsCombo.currentText()
        spots, scale = self.napariUtil.getDataAndScaleOfLayerWithName(text)
        text = self.gFunctionLabelsCombo.currentText()
        labels = self.napariUtil.getDataOfLayerWithName(text)
        self.layer = self.napariUtil.getLayerWithName(text)
        self.delaunayTask = DelaunayTask(spots, labels, scale, label)
        worker = create_worker(self.delaunayTask.run,
                               _progress={'desc': 'Calculating Delaunay Tesselation...'})
        worker.finished.connect(self.onDelaunayTaskFinished)
        worker.start()


    def onDelaunayTaskFinished(self):
        tess = self.delaunayTask.result
        units = self.layer.units
        self.viewer.add_shapes(tess.points[tess.simplices],
                               scale=self.delaunayTask.scale,
                               shape_type='path',
                               units=units)


    def _onVoronoiButtonClicked(self):
        label = int(self.gFunctionInput.text().strip())
        if not label:
            return
        text = self.gFunctionSpotsCombo.currentText()
        spots,scale = self.napariUtil.getDataAndScaleOfLayerWithName(text)
        text = self.gFunctionLabelsCombo.currentText()
        labels = self.napariUtil.getDataOfLayerWithName(text)
        self.layer = self.napariUtil.getLayerWithName(text)
        self.voronoiTask = VoronoiTask(spots, labels, scale, label)
        worker = create_worker(self.voronoiTask.run,
                               _progress={'desc': 'Calculating Voronoi Tesselation...'})
        worker.finished.connect(self.onVoronoiTaskFinished)
        worker.start()


    def onVoronoiTaskFinished(self):
        regions = self.voronoiTask.result
        units = self.layer.units
        self.viewer.add_shapes(regions, scale=self.voronoiTask.scale, shape_type='polygon', units=units)


    def _onSegmentImageButtonClicked(self):
        self.layer = self.getActiveLayer()
        if not self.layer or not type(self.layer) is Image:
            return
        options = SegmentationOptionsWidget(None).options
        self.segmentation = Segmentation(self.layer.data)
        self.segmentation.clearBorder = options.get('remove_border_objects')
        self.segmentation.minSize = options.get('min_size')
        self.segmentation.flowThreshold = options.get("flow_threshold")
        self.segmentation.cellProbabilityThreshold = options.get("cellprob_threshold")
        self.segmentation.diameter = options.get('diameter')
        self.segmentation.resampleDynamics = True
        worker = create_worker(self.segmentation.run,
                               _progress={'total': 5, 'desc': 'Segmenting cells...'})
        worker.finished.connect(self.onSegmentationFinished)
        worker.start()


    def _onKeepLabelsButtonClicked(self):
        self.layer = self.getActiveLayer()
        if not self.layer or not type(self.layer) is Labels:
            return
        text = self.keepLabelsInput.text().strip()
        labelTextList = text.split(',')
        if not labelTextList:
            return
        labelList = [int(labelText) for labelText in labelTextList]
        newLabels = Segmentation.keepLabels(self.layer.data, labelList)
        layer = self.viewer.add_labels(newLabels, scale=self.layer.scale, units=self.layer.units, blending='additive')
        NapariUtil.copyOriginalPath(self.layer, layer)


    def _onDetectSpotsButtonClicked(self):
        self.spotsLayer = self.getActiveLayer()
        if not self.spotsLayer or not type(self.spotsLayer) is Image:
            return
        options = DetectionOptionsWidget(None).options
        self.detection = SpotDetection(self.spotsLayer.data)
        self.detection.scale = (self.spotsLayer.scale[0].item(),
                                self.spotsLayer.scale[1].item(),
                                self.spotsLayer.scale[2].item())
        self.detection.threshold = options.get("threshold")
        self.detection.spotRadius = (options.get("radius_z"), options.get("radius_xy"), options.get("radius_xy"))
        self.detection.shallRemoveDuplicates = options.get("remove_duplicates")
        message = \
            ("Running spot detection with scale = {}, threshold = {}, spot radius = {}, "
             " remove duplicates = {}, find threshold = {} on {}.")
        notifications.show_info(
            message.format(
                self.detection.scale,
                self.detection.threshold,
                self.detection.spotRadius,
                self.detection.shallRemoveDuplicates,
                self.detection.shallFindThreshold,
                self.spotsLayer.name))
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
        spots, scale = self.napariUtil.getDataAndScaleOfLayerWithName(text)
        text = self.gFunctionLabelsCombo.currentText()
        labels = self.napariUtil.getDataOfLayerWithName(text)
        self.gFunctionTask = GFunctionTask(spots, labels, scale, label)
        self.gFunctionTask.nrOfSamples = 100
        worker = create_worker(self.gFunctionTask.run,
                      _progress={'desc': 'Calculating G-Function...'}
                      )
        worker.finished.connect(self.ongFunctionTaskFinished)
        worker.start()


    def _onHFunctionButtonClicked(self):
        label = int(self.gFunctionInput.text().strip())
        if not label:
            return
        self.labelOfNucleus = label
        text = self.gFunctionSpotsCombo.currentText()
        spots, scale = self.napariUtil.getDataAndScaleOfLayerWithName(text)
        text = self.gFunctionLabelsCombo.currentText()
        labels = self.napariUtil.getDataOfLayerWithName(text)
        self.hFunctionTask = HFunctionTask(spots, labels, scale, label)
        self.hFunctionTask.nrOfSamples = 100
        worker = create_worker(self.hFunctionTask.run,
                               _progress={'desc': 'Calculating H-Function...'}
                               )
        worker.finished.connect(self.onhFunctionTaskFinished)
        worker.start()


    def _onFFunctionButtonClicked(self):
        label = int(self.gFunctionInput.text().strip())
        if not label:
            return
        self.labelOfNucleus = label
        text = self.gFunctionSpotsCombo.currentText()
        spots, scale = self.napariUtil.getDataAndScaleOfLayerWithName(text)
        text = self.gFunctionLabelsCombo.currentText()
        labels = self.napariUtil.getDataOfLayerWithName(text)
        self.fFunctionTask = FFunctionTask(spots, labels, scale, label)
        self.fFunctionTask.nrOfSamples = 100
        worker = create_worker(self.fFunctionTask.run,
                      _progress={'desc': 'Calculating F-Function...'}
                      )
        worker.finished.connect(self.onfFunctionTaskFinished)
        worker.start()


    def onfFunctionTaskFinished(self):
        ax = plt.subplot()
        analyzer = self.fFunctionTask.analyzer
        analyzer.esEcdfs[self.fFunctionTask.label].cdf.plot(ax)
        ax.set_xlabel('distances [nm]')
        ax.set_ylabel('Empirical CDF')
        maxDist = np.max(analyzer.emptySpaceDistances[self.fFunctionTask.label][0])
        xValues = np.array(list(np.arange(0, math.floor(maxDist + 1), analyzer.scale[1])))
        envelop = self.fFunctionTask.envelop
        plt.plot(xValues, envelop[0], 'r--')
        plt.plot(xValues, envelop[1], 'g--')
        plt.plot(xValues, envelop[2], 'g--')
        plt.plot(xValues, envelop[3], 'r--')
        plt.show()


    def ongFunctionTaskFinished(self):
        ax = plt.subplot()
        analyzer = self.gFunctionTask.analyzer
        analyzer.nnEcdfs[self.gFunctionTask.label].cdf.plot(ax)
        ax.set_xlabel('distances [nm]')
        ax.set_ylabel('Empirical CDF')
        maxDist = np.max(analyzer.nnDistances[self.gFunctionTask.label][0])
        xValues = np.array(list(np.arange(0, math.floor(maxDist + 1), analyzer.scale[1])))
        envelop = self.gFunctionTask.envelop
        plt.plot(xValues, envelop[0], 'r--')
        plt.plot(xValues, envelop[1], 'g--')
        plt.plot(xValues, envelop[2], 'g--')
        plt.plot(xValues, envelop[3], 'r--')
        plt.show()


    def onhFunctionTaskFinished(self):
        ax = plt.subplot()
        analyzer = self.hFunctionTask.analyzer
        analyzer.adEcdfs[self.hFunctionTask.label].cdf.plot(ax)
        ax.set_xlabel('distances [nm]')
        ax.set_ylabel('Empirical CDF')
        maxDist = np.max(analyzer.allDistances[self.hFunctionTask.label][0])
        xValues = np.array(list(np.arange(0, math.floor(maxDist + 1), analyzer.scale[1])))
        envelop = self.hFunctionTask.envelop
        plt.plot(xValues, envelop[0], 'r--')
        plt.plot(xValues, envelop[1], 'g--')
        plt.plot(xValues, envelop[2], 'g--')
        plt.plot(xValues, envelop[3], 'r--')
        plt.show()


    def _onCropButtonPressed(self):
        text = self.cropImageLabelsCombo.currentText()
        labels = self.napariUtil.getDataOfLayerWithName(text)
        text = self.cropImageCombo.currentText()
        self.layer = self.napariUtil.getLayerWithName(text)
        image = self.layer.data
        self.cropLabel = int(self.cropLabelInput.text().strip())
        if not self.cropLabel:
            self.cropLabel = 1
            return
        self.cropLabelTask = CropLabelTask(labels, image, self.cropLabel)
        worker = create_worker(self.cropLabelTask.run,
                               _progress={'desc': 'Cropping image...'}
                               )
        worker.finished.connect(self.onCropLabelTaskFinished)
        worker.start()


    def onCropLabelTaskFinished(self):
        text = self.cropImageLabelsCombo.currentText()
        layer = self.viewer.add_image(self.cropLabelTask.result,
                              name=text + "_c" + str(self.cropLabel),
                              scale=self.layer.scale,
                              colormap=self.layer.colormap,
                              units=self.layer.units,
                              blending=self.layer.blending
                              )
        NapariUtil.copyOriginalPath(self.layer, layer)


    def _onCorrelationButtonPressed(self):
        text1 = self.ccInputACombo.currentText()
        text2 = self.ccInputBCombo.currentText()
        if not text1 or not text2:
            return
        self.layer = self.napariUtil.getLayerWithName(text1)
        imageA = self.napariUtil.getDataOfLayerWithName(text1)
        imageB = self.napariUtil.getDataOfLayerWithName(text2)
        self.correlator = Correlator(imageA, imageB)
        worker = create_worker(self.correlator.calculateCrossCorrelationProfile,
                               _progress={'desc': 'Calculating Cross-Correlation...'}
                               )
        worker.finished.connect(self.onCrossCorrelationFinished)
        worker.start()


    def onCrossCorrelationFinished(self):
        text1 = self.ccInputACombo.currentText()
        text2 = self.ccInputBCombo.currentText()
        self.correlator.calculateCrossCorrelationProfile()
        layer = self.viewer.add_image(self.correlator.correlationImage, name="corr.: " + text1 + "-" + text2,
                                                           colormap='inferno',
                                                           blending='additive',
                                                           scale=self.layer.scale,
                                                           units=self.layer.units,
                              )
        NapariUtil.copyOriginalPath(self.layer, layer)
        plt.plot(self.correlator.correlationProfile[0], self.correlator.correlationProfile[1])
        data = np.asarray([self.correlator.correlationProfile[0], self.correlator.correlationProfile[1]])
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
        layer = self.viewer.add_labels(self.segmentation.labels,
                               scale=self.layer.scale,
                               units = self.layer.units,
                               blending='additive')
        NapariUtil.copyOriginalPath(self.layer, layer)



    def onDetectionFinished(self):
        options = DetectionOptionsWidget(None).options
        doDecomposeDense = options.get("decompose_dense")
        if not doDecomposeDense:
            layer = self.viewer.add_points(self.detection.spots,
                                           scale=self.spotsLayer.scale,
                                           units=self.spotsLayer.units,
                                           blending='additive', size=2)
            NapariUtil.copyOriginalPath(self.spotsLayer, layer)
            return
        self.decomposeDense = DecomposeDenseRegions(self.spotsLayer.data, self.detection.spots)
        self.decomposeDense.voxelSize = tuple(self.spotsLayer.scale)
        self.decomposeDense.spotRadius = (options.get("radius_z"), options.get("radius_xy"), options.get("radius_xy"))
        self.decomposeDense.alpha = options.get("alpha")
        self.decomposeDense.beta = options.get("beta")
        self.decomposeDense.gamma = options.get("gamma")
        worker = create_worker(self.decomposeDense.run,
                               _progress={'total': 2, 'desc': 'Decomposing dense regions...'})
        worker.finished.connect(self.onDecomposeFinished)
        worker.start()


    def onDecomposeFinished(self):
        options = DetectionOptionsWidget(None).options
        layer = self.viewer.add_points(self.decomposeDense.decomposedSpots,
                                       scale=tuple(self.spotsLayer.scale),
                                       units=self.spotsLayer.units,
                                       blending='additive', size=2)
        NapariUtil.copyOriginalPath(self.spotsLayer, layer)
        if options.get('display_avg_spot') and not self.decomposeDense.referenceSpot is None:
            layer = self.viewer.add_image(self.decomposeDense.referenceSpot,
                                  scale=tuple(self.spotsLayer.scale),
                                  units=self.spotsLayer.units,
                                  name="reference spot",
                                  colormap=self.spotsLayer.colormap,
                                  blending=self.spotsLayer.blending
                                  )
            NapariUtil.copyOriginalPath(self.spotsLayer, layer)


    def gFunctionInputChanged(self):
        pass


    def medianFilterSizeChanged(self):
        pass


    def cropLabelInputChanged(self):
        pass


    def backgroundSigmaXYChanged(self):
        pass


    def backgroundSigmaZChanged(self):
        pass


    def keepLabelsChanged(self):
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



class OptionsWidget(QWidget):


    def __init__(self, viewer, app, name):
        super().__init__()
        self.viewer = viewer
        self.application = app
        self.name = name
        self.options = Options(self.application, self.name)
        self.fieldWidth = 50


    def _onOKButtonClicked(self):
        self.transferValues()
        self.options.save()
        self.options.load()
        self.shut()


    def _onCancelButtonClicked(self):
        self.shut()


    def shut(self):
        self.viewer.window.remove_dock_widget(self)
        self.close()


    def transferValues(self):
        self.subclassResponsability()


    def ignoreChange(self):
        pass



class DetectionOptionsWidget(OptionsWidget):


    def __init__(self, viewer):
        super().__init__(viewer, "napari-sphot", "detection")
        self.options.setDefaultValues(
            {
                'threshold': 0.01,
                'radius_xy': 2.5,
                'radius_z': 2.5,
                'remove_duplicates': True,
                'decompose_dense': True,
                'alpha': 0.5,
                'beta': 1.0,
                'gamma': 5.0,
                'display_avg_spot': True
            }
        )
        self.options.load()
        self.thresholdInput = None
        self.radiusXYInput = None
        self.radiusZInput = None
        self.removeDuplicatesCheckBox = None
        self.decomposeDenseCheckBox = None
        self.decomposeDenseCheckBox = None
        self.alphaInput = None
        self.betaInput = None
        self.gammaInput = None
        self.displayMeanSpotCheckBox = None
        self.createLayout()


    def createLayout(self):
        mainLayout = QVBoxLayout()
        formLayout = QFormLayout()
        buttonsLayout = QHBoxLayout()
        mainLayout.addLayout(formLayout)
        mainLayout.addLayout(buttonsLayout)
        thresholdLabel, self.thresholdInput = WidgetTool.getLineInput(self, "Threshold: ",
                                                                    self.options.get('threshold'),
                                                                    self.fieldWidth,
                                                                    self.ignoreChange)
        radiusXYLabel, self.radiusXYInput = WidgetTool.getLineInput(self, "Radius xy: ",
                                                                      self.options.get('radius_xy'),
                                                                      self.fieldWidth,
                                                                      self.ignoreChange)
        radiusZLabel, self.radiusZInput = WidgetTool.getLineInput(self, "Radius z: ",
                                                                    self.options.get('radius_z'),
                                                                    self.fieldWidth,
                                                                    self.ignoreChange)
        self.removeDuplicatesCheckBox = QCheckBox(text="Remove Duplicates")
        self.removeDuplicatesCheckBox.setChecked(self.options.get('remove_duplicates'))
        self.decomposeDenseCheckBox = QCheckBox(text="Decompose Dense")
        self.decomposeDenseCheckBox.setChecked(self.options.get('decompose_dense'))
        alphaLabel, self.alphaInput = WidgetTool.getLineInput(self, "Alpha: ",
                                                                  self.options.get('alpha'),
                                                                  self.fieldWidth,
                                                                  self.ignoreChange)
        betaLabel, self.betaInput = WidgetTool.getLineInput(self, "Beta: ",
                                                              self.options.get('beta'),
                                                              self.fieldWidth,
                                                              self.ignoreChange)
        gammaLabel, self.gammaInput = WidgetTool.getLineInput(self, "Gamma: ",
                                                              self.options.get('gamma'),
                                                              self.fieldWidth,
                                                              self.ignoreChange)
        self.displayMeanSpotCheckBox = QCheckBox(text="Display Avg. Spot")
        self.displayMeanSpotCheckBox.setChecked(self.options.get('display_avg_spot'))
        okButton = QPushButton("&OK")
        okButton.clicked.connect(self._onOKButtonClicked)
        cancelButton = QPushButton("&Cancel")
        cancelButton.clicked.connect(self._onCancelButtonClicked)
        buttonsLayout.addWidget(okButton)
        buttonsLayout.addWidget(cancelButton)

        formLayout.addRow(thresholdLabel, self.thresholdInput)
        formLayout.addRow(radiusXYLabel, self.radiusXYInput)
        formLayout.addRow(radiusZLabel, self.radiusZInput)
        formLayout.addWidget(self.removeDuplicatesCheckBox)
        formLayout.addWidget(self.decomposeDenseCheckBox)
        formLayout.addRow(alphaLabel, self.alphaInput)
        formLayout.addRow(betaLabel, self.betaInput)
        formLayout.addRow(gammaLabel, self.gammaInput)
        formLayout.addWidget(self.displayMeanSpotCheckBox)
        self.setLayout(mainLayout)


    def transferValues(self):
        self.options.set('threshold', float(self.thresholdInput.text().strip()))
        self.options.set('radius_xy', float(self.radiusXYInput.text().strip()))
        self.options.set('radius_z', float(self.radiusZInput.text().strip()))
        self.options.set('remove_duplicates', (self.removeDuplicatesCheckBox.isChecked()))
        self.options.set('decompose_dense', (self.decomposeDenseCheckBox.isChecked()))
        self.options.set('alpha', float(self.alphaInput.text().strip()))
        self.options.set('beta', float(self.betaInput.text().strip()))
        self.options.set('gamma', float(self.gammaInput.text().strip()))
        self.options.set('display_avg_spot', (self.displayMeanSpotCheckBox.isChecked()))


class SegmentationOptionsWidget(OptionsWidget):


    def __init__(self, viewer):
        super().__init__(viewer, "napari-sphot", "segmentation")
        self.options.setDefaultValues(
            {
                'diameter': 90.0,
                'cellprob_threshold': 0.0,
                'flow_threshold': 0.4,
                'min_size': 0.0,
                'remove_border_objects': True
            }
        )
        self.options.load()
        self.diameterInput = None
        self.cellprobeThresholdInput = None
        self.flowThresholdInput = None
        self.minSizeInput = None
        self.removeCheckbox = None
        self.createLayout()


    def createLayout(self):
        mainLayout = QVBoxLayout()
        formLayout = QFormLayout()
        buttonsLayout = QHBoxLayout()
        mainLayout.addLayout(formLayout)
        mainLayout.addLayout(buttonsLayout)
        diameterLabel, self.diameterInput = WidgetTool.getLineInput(self, "Diameter: ",
                                                                                self.options.get('diameter'),
                                                                                self.fieldWidth,
                                                                                self.diameterChanged)
        cellprobeLabel, self.cellprobeThresholdInput = WidgetTool.getLineInput(self, "Cellprob Threshold: ",
                                                                                self.options.get('cellprob_threshold'),
                                                                                self.fieldWidth,
                                                                                self.cellprobThresholdChanged)
        flowLabel, self.flowThresholdInput = WidgetTool.getLineInput(self, "Flow Threshold: ",
                                                                               self.options.get('flow_threshold'),
                                                                               self.fieldWidth,
                                                                               self.flowThresholdChanged)
        minSizeLabel, self.minSizeInput = WidgetTool.getLineInput(self, "Min. Size: ",
                                                                               self.options.get('min_size'),
                                                                               self.fieldWidth,
                                                                               self.minSizeChanged)
        self.removeCheckbox = QCheckBox(text="Remove Edge")
        self.removeCheckbox.setChecked(self.options.get('remove_border_objects'))
        okButton = QPushButton("&OK")
        okButton.clicked.connect(self._onOKButtonClicked)
        cancelButton = QPushButton("&Cancel")
        cancelButton.clicked.connect(self._onCancelButtonClicked)
        buttonsLayout.addWidget(okButton)
        buttonsLayout.addWidget(cancelButton)
        formLayout.addRow(diameterLabel, self.diameterInput)
        formLayout.addRow(cellprobeLabel, self.cellprobeThresholdInput)
        formLayout.addRow(flowLabel, self.flowThresholdInput)
        formLayout.addRow(minSizeLabel, self.minSizeInput)
        formLayout.addWidget(self.removeCheckbox)
        self.setLayout(mainLayout)



    def diameterChanged(self):
        pass


    def cellprobThresholdChanged(self):
        pass


    def flowThresholdChanged(self):
        pass


    def minSizeChanged(self):
        pass


    def transferValues(self):
        self.options.set('diameter', float(self.diameterInput.text().strip()))
        self.options.set('cellprob_threshold', float(self.cellprobeThresholdInput.text().strip()))
        self.options.set('flow_threshold', float(self.flowThresholdInput.text().strip()))
        self.options.set('min_size', float(self.minSizeInput.text().strip()))
        self.options.set('remove_border_objects', (self.removeCheckbox.isChecked()))


