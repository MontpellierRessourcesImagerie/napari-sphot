from typing import TYPE_CHECKING
import math
import matplotlib.pyplot as plt
import numpy as np
from PIL.ImageOps import scale
from qtpy.QtWidgets import QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QFormLayout, QGroupBox
from napari.layers import Image
from napari.utils.events import Event
from napari.qt.threading import create_worker
from sphot.image import Segmentation
from sphot.image import SpotDetection
from sphot.image import SpotPerCellAnalyzer
from napari_sphot.qtutil import WidgetTool
from napari_sphot.napari_util import NapariUtil
from napari_sphot.image import TiffFileTags




if TYPE_CHECKING:
    import napari



class ExampleQWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        self.scale = 50 #@TODO: get the scale from the input image
        self.fieldWidth = 300
        self.labelOfNucleus = 1
        self.segmentation = None
        self.layer = None
        self.spotsLayer = None
        self.gFunctionInput = None
        self.gFunctionSpotsCombo = None
        self.gFunctionLabelsCombo = None
        self.napariUtil = NapariUtil(self.viewer)
        self.pointsLayers = self.napariUtil.getPointsLayers()
        self.labelLayers = self.napariUtil.getLabelLayers()
        self.viewer = viewer
        self.createLayout()
        self.viewer.layers.events.inserted.connect(self.onLayerAddedOrRemoved)
        self.viewer.layers.events.removed.connect(self.onLayerAddedOrRemoved)


    def createLayout(self):
        segmentImageButton = QPushButton("Segment Image")
        segmentImageButton.clicked.connect(self._onSegmentImageButtonClicked)
        detectSpotsButton = QPushButton("Detect Spots")
        detectSpotsButton.clicked.connect(self._onDetectSpotsButtonClicked)

        gFunctionGroupBox = QGroupBox("G-Function")
        gFunctionLabel, self.gFunctionInput = WidgetTool.getLineInput(self, "Label of nucleus: ",
                                                                            self.labelOfNucleus,
                                                                            self.fieldWidth,
                                                                            self.gFunctionInputChanged)
        gFunctionButton = QPushButton("G-Function")
        gFunctionButton.clicked.connect(self._onGFunctionButtonClicked)
        gFunctionMainLayout = QVBoxLayout()
        gFunctionGroupBox.setLayout(gFunctionMainLayout)
        gFunctionSpotsLabel, self.gFunctionSpotsCombo = WidgetTool.getComboInput(self, "Spots: ", self.pointsLayers)
        gFunctionLabelsLabel, self.gFunctionLabelsCombo = WidgetTool.getComboInput(self, "Cell labels: ", self.labelLayers)
        gFunctionLayersLayout = QHBoxLayout()
        gFunctionLayersLayout.addWidget(gFunctionSpotsLabel)
        gFunctionLayersLayout.addWidget(self.gFunctionSpotsCombo)
        gFunctionLayersLayout.addWidget(gFunctionLabelsLabel)
        gFunctionLayersLayout.addWidget(self.gFunctionLabelsCombo)
        gFunctionCellLayout = QHBoxLayout()
        gFunctionCellLayout.addWidget(gFunctionLabel)
        gFunctionCellLayout.addWidget(self.gFunctionInput)
        gFunctionCellLayout.addWidget(gFunctionButton)
        gFunctionMainLayout.addLayout(gFunctionLayersLayout)
        gFunctionMainLayout.addLayout(gFunctionCellLayout)
        mainLayout = QVBoxLayout()
        mainLayout.addWidget(segmentImageButton)
        mainLayout.addWidget(detectSpotsButton)
        mainLayout.addWidget(gFunctionGroupBox)
        self.setLayout(mainLayout)


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
        analyzer.run()
        cdf = analyzer.ecdfs[label]
        envelop = analyzer.getEnvelopFor(label, 100)
        ax = plt.subplot()
        analyzer.ecdfs[label].cdf.plot(ax)
        ax.set_xlabel('distances [nm]')
        ax.set_ylabel('Empirical CDF')
        maxDist = np.max(analyzer.nnDistances[label][0])
        xValues = np.array(list(range(0, math.floor(maxDist + 1), analyzer.scale)))
        plt.plot(xValues, envelop[0], 'r--')
        plt.plot(xValues, envelop[1], 'g--')
        plt.plot(xValues, envelop[2], 'g--')
        plt.plot(xValues, envelop[3], 'r--')
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


    def onLayerAddedOrRemoved(self, event: Event):
        self.updateLayerSelectionComboBoxes()


    def updateLayerSelectionComboBoxes(self):
        labelComboBoxes = [self.gFunctionLabelsCombo]
        labelLayers = self.napariUtil.getLabelLayers()
        for comboBox in labelComboBoxes:
            WidgetTool.replaceItemsInComboBox(comboBox, labelLayers)
        spotComboBoxes = [self.gFunctionSpotsCombo]
        spotLayers = self.napariUtil.getPointsLayers()
        for comboBox in spotComboBoxes:
            WidgetTool.replaceItemsInComboBox(comboBox, spotLayers)