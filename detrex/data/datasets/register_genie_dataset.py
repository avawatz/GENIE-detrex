import os
import json
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog


def load_genie_dataset(data_ids, project_dir):
    assert os.path.isdir(project_dir), "Project Directory doesn't exist."
    if data_ids.get("augmented_set") is not None:
        assert os.path.isdir(os.path.join(project_dir, "augmented_set")), "Augmented Set Directory doesn't exist."
        data_ids["src_aug"] = os.listdir(os.path.join(project_dir, "augmented_set"))
        assert data_ids["src_aug"] != 0, "Augmented Set is there but augmentation images in augmented_set folder is zero."
    
    all_anns = list()
    ann_dir = os.path.join(project_dir, "annotations")
    
    for set_name, files in data_ids.items():
        for name in files:
            instance = dict()
            instance["file_name"] = instance["image_id"] = name
            instance["augmented"] = False if set_name != "src_aug" else True
            with open(os.path.join(ann_dir, f"{os.path.splitext(name)[0]}.json")) as j_file:
                data = json.load(j_file)
            instance["height"], instance["width"] = data["size"]
            instance["annotations"] = list()
            if data.get("bboxes") is not None:
                while len(data["bboxes"]):
                    instance["annotations"].append(dict(bbox=data["bboxes"].pop(0),
                                                        bbox_mode=BoxMode.XYXY_ABS,
                                                        category_id=data["class_ids"].pop(0)))
            elif data.get("segments") is not None:
                while len(data["segments"]):
                    if len(data["segments"][0]) >= 6 and len(data["segments"][0]) % 2 == 0:
                        instance["annotations"].append(dict(bbox=data["bboxes"].pop(0),
                                                            bbox_mode=BoxMode.XYXY_ABS,
                                                            category_id=data["class_ids"].pop(0),
                                                            segmentation=[data["segments"].pop(0)]))
                    else:
                        del data["segments"][0]
                        del data["class_ids"][0]
            else:
                raise ValueError("Invalid JSON file passed.")
            all_anns.append(instance)
    
    return all_anns


            
def register_genie_dataset(name: str,
                           data_ids: dict,
                           project_dir: str,
                           metadata: dict):
    
    assert os.path.isdir(project_dir), "Project Directory doesn't exist."
    if data_ids.get("augmented_set") is not None:
        assert os.path.isdir(os.path.join(project_dir, "augmented_set")), "Augmented Set Directory doesn't exist."

    DatasetCatalog.register(name, lambda: load_genie_dataset(data_ids=data_ids, 
                                                             project_dir=project_dir))
    
    with open(os.path.join(project_dir, "metadata.json")) as j_file:
        classes = json.load(j_file)["classes"]

    MetadataCatalog.get(name).set(
        thing_classes=classes, evaluator_type="coco", **metadata
    )
