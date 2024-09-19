# Network Tables
HOSTNAME = "visiontwhs"
VISION_TABLE_NAME = "vision"


# Camera stream arguments
STREAM_WIDTH = 480
STREAM_HEIGHT = 360
FPS = 15


# Processing image template
IMG_TEMPLATE = np.zeros(shape = (STREAM_HEIGHT, STREAM_WIDTH, 3), dtype = np.uint8)


# Camera dev paths and names
USB1_PATH = "/dev/v4l/by-path/platform-fd500000.pcie-pci-0000:01:00.0-usb-0:1.1:1.0-video-index0" # Launcher Cam
USB2_PATH = "/dev/v4l/by-path/platform-fd500000.pcie-pci-0000:01:00.0-usb-0:1.2:1.0-video-index0" # Left Cam
USB3_PATH = "/dev/v4l/by-path/platform-fd500000.pcie-pci-0000:01:00.0-usb-0:1.4:1.0-video-index0" # Right Cam
USB4_PATH = "/dev/v4l/by-path/platform-fd500000.pcie-pci-0000:01:00.0-usb-0:1.3:1.0-video-index0" # Intake Cam

CAM1_NAME = "Launcher Cam"
CAM2_NAME = "Left Cam"
CAM3_NAME = "Right Cam"
CAM4_NAME = "Intake Cam"


# Output stream name
OUTPUT_STREAM_NAME = "Cam Stream"


# AprilTag constants
DETECTION_MARGIN_THRESHOLD = 90
TAG_SIZE_m = 0.1651

TAG_PTS = np.array(
    [[-tag_size_m / 2, -tag_size_m / 2, 0],
    [tag_size_m / 2,   -tag_size_m / 2, 0],
    [tag_size_m / 2,    tag_size_m / 2, 0],
    [-tag_size_m / 2,   tag_size_m / 2, 0]],
    dtype=np.float32
)


# Lens intrinsics

# Intrinsics matrix structure
# [fx, 0, cx]
# [0, fy, cy]
# [0, 0,  1 ]
fx = 699.3778103158814
fy = 677.7161226393544
cx = 345.6059345433618
cy = 207.12741326228522

INTRINSICS_MATRIX = np.array(
    [[fx, 0,  cx],
    [0,   fy, cy],
    [0,   0,  1]],
    dtype = np.float64
)

DISTORTIONS = np.array(
    [0.14382207979312617,
    -0.9851192814987014,
    -0.018168751047242335,
    0.011034504043795105,
    1.9833437176538498],
    dtype = np.float64
)


# Gamepieces

# Adjust below based on color
LOWER_HSV = [0, 0 ,0]
UPPER_HSV = [180, 50, 50]

# If red color, need two upper and lower HSV arrays because red is as 0 degrees
LOWER_HSV2 = [0, 0 ,0]
UPPER_HSV2 = [180, 50, 50]

MIN_CONTOUR_AREA = 100

# Yolo model stuff
YOLO_MODEL_PATH = "/home/vision/Documents/Code/best.pt"
YOLO_CONF = 0.9

#TODO:: add gamepiece solvepnp stuff potentially
#TODO:: potentially add yolo stuff