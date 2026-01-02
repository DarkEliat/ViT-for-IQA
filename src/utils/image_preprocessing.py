from PIL import Image
import matplotlib.pyplot as plt


def resize(image: Image.Image,
           target_size: tuple[int, int],
           keep_aspect_ratio: bool,
           padding_color: tuple[int, int, int] = (0, 0, 0)
) -> Image.Image:
    original_width, original_height = image.size

    target_width, target_height = target_size

    scale_factors = (0, 0)

    if keep_aspect_ratio:
        lower_scale_factor = min(
            target_width / original_width,
            target_height / original_height
        )

        scale_factors = (lower_scale_factor, lower_scale_factor)
    else:
        scale_factors = (
            target_width / original_width,
            target_height / original_height
        )

    resized_width = int(round(original_width * scale_factors[0]))
    resized_height = int(round(original_height * scale_factors[1]))

    resized_image = image.resize(
        size=(resized_width, resized_height),
        resample=Image.BICUBIC
    )

    padding_width_total  = target_width - resized_width
    padding_height_total  = target_height - resized_height

    padding_left = padding_width_total // 2
    padding_top = padding_height_total // 2

    padded_image = Image.new(
        mode='RGB',
        size=(target_width, target_height),
        color=padding_color
    )

    padded_image.paste(
        resized_image,
        box=(padding_left, padding_top)
    )

    return padded_image
