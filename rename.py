from pathlib import Path


def rename_sequential(folder: Path) -> None:
	if not folder.exists() or not folder.is_dir():
		raise FileNotFoundError(f"Folder not found: {folder}")

	files = [p for p in folder.iterdir() if p.is_file()]
	files.sort(key=lambda p: p.name.lower())

	for idx, path in enumerate(files, start=1):
		new_name = f"{idx}{path.suffix.lower()}"
		new_path = path.with_name(new_name)
		if new_path.exists():
			continue
		path.rename(new_path)


if __name__ == "__main__":
	rename_sequential(Path("cropped_images"))
