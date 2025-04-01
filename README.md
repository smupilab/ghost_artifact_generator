python ghost.py --folder "input_images" --offset 15 --ratio 0.2 --alpha 0.4

## Options

* If the `--folder` option is provided, the `get_image_files_from_folder()` function collects all image files (supported extensions) within that folder.
* The `--folder` option is mutually exclusive with the `--input` option, allowing only one of them to be specified.
* The `--offset_pixels` option adjusts how many pixels the effect will be offset by.
* The `--ratio` option determines the ratio of edge ghost artifacts.
* The `--alpha` option controls the blur ratio of the ghost artifacts.
