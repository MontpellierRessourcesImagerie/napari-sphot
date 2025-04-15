from typing import TYPE_CHECKING

from PIL.ImageOps import scale
from qtpy.QtWidgets import QVBoxLayout, QHBoxLayout, QPushButton, QWidget
from napari.layers import Image
from napari.qt.threading import create_worker

from sphot.image import Segmentation
from sphot.image import SpotDetection

if TYPE_CHECKING:
    import napari



class ExampleQWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.segmentation = None
        self.layer = None
        self.spotsLayer = None
        self.viewer = viewer

        self.createLayout()


    def createLayout(self):
        segmentImageButton = QPushButton("Segment Image")
        segmentImageButton.clicked.connect(self._onSegmentImageButtonClicked)
        detectSpotsButton = QPushButton("Detect Spots")
        detectSpotsButton.clicked.connect(self._onDetectSpotsButtonClicked)
        mainLayout = QVBoxLayout()
        mainLayout.addWidget(segmentImageButton)
        mainLayout.addWidget(detectSpotsButton)
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