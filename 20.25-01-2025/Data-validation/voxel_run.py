import fiftyone as fo

dataset_name = "test_harhat_2"

dataset_dir = "C:/Monal/Work/AllLight/Krish-sir/Project/Data-validation/Hard Hat"
splits = ["train", "valid", "test"]
# splits = ["test"]
dataset = fo.Dataset(dataset_name)

for split in splits:
    dataset.add_dir(dataset_dir=dataset_dir,dataset_type=fo.types.YOLOv5Dataset,split=split,tags=split,)

session = fo.launch_app(dataset, port=5151)

session.wait()
