
from roboflow import Roboflow
rf = Roboflow(api_key="AJdTIeCtYbAhwapDs0hu")
project = rf.workspace("palles").project("age-gander")
version = project.version(2)
dataset = version.download("yolov8")
