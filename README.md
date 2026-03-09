## box-extractor

### Files:

- `build_dataset.py`: creates a test, train, val dataset into a new directory from an existing dataset directory which contains only "marked" and "unmarked" subdirectories. Has configurable options.

- `extractor.py`: takes a scannable worksheet file (corner tags have to be valid), and crops out the 80 bubbles into new files.

- `rename.py`: renames all files in a folder into sequential order.

- `image_service.py`, `models.py`: files taken from `paperplus_server` for reference.