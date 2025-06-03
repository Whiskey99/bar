# Cocktail Ingredient Identifier

This simple command-line application uses a pretrained image classification model
from `torchvision` to recognize ingredients in images and suggest cocktails that
can be made with those ingredients.

## Requirements

- Python 3.8+
- `torch`
- `torchvision`
- `Pillow`

## Usage

```
python cocktail_app.py path/to/image1.jpg path/to/image2.jpg
```

The script prints the top predicted labels for each image and lists cocktail
suggestions for any recognized ingredients.
