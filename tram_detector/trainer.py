
import argparse
import json
import os
from pathlib import Path
from ultralytics import YOLO

def download_roboflow_dataset(api_key, workspace, project_name, version_num, export_format="yolov11"):
	from roboflow import Roboflow

	rf = Roboflow(api_key=api_key)
	proj = rf.workspace(workspace).project(project_name)
	ver = proj.version(version_num)
	print(f"Downloading dataset export '{export_format}' from Roboflow...")
	dataset_dir = ver.download(export_format).location
	print(f"Downloaded to: {dataset_dir}")
	return Path(dataset_dir)


def train(model_weights, data_yaml, epochs=50, batch=16, imgsz=720, project="runs/train", name=None):
	

	print(f"Starting training with model={model_weights}, data={data_yaml}")
	model = YOLO(model_weights)
	run_name = name or f"roboflow_train_{Path(data_yaml).stem}"
	model.train(data=data_yaml, epochs=epochs, batch=batch, imgsz=imgsz, 
			 	project=project, name=run_name, device='cuda', 
				box=1.0, cls=2.0, dfl=1.0)

	print("Training complete. Check the runs folder for weights and logs.")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--api_key", required=True, default="WwWi5GIvzTkGby6RpB6v")
	parser.add_argument("--workspace", default="new-workspace-5mbpa")
	parser.add_argument("--project", default="tram-detection-ebsat")
	parser.add_argument("--version", type=int, default=1)
	parser.add_argument("--export_format", default="yolov11")
	parser.add_argument("--model", default="yolo11n.pt")
	parser.add_argument("--epochs", type=int, default=200)
	parser.add_argument("--batch", type=int, default=64)
	parser.add_argument("--project_out", default="runs/train")
	parser.add_argument("--name", default=None)

	args = parser.parse_args()

	ds_dir = download_roboflow_dataset(args.api_key, args.workspace, args.project, args.version, args.export_format)
	data_yaml = os.path.join(ds_dir, "data.yaml")

	# Try to train with ultralytics (works for yolov8 family). If true yolov11 training
	# requires a dedicated repo, the user can change `--model_weights` to a yolov11 checkpoint
	train(args.model, data_yaml, epochs=args.epochs, batch=args.batch, project=args.project_out, name=args.name)


if __name__ == "__main__":
	main()
