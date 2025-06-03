# Cocktail Ingredient Identifier and Suggestion App
# Requires PyTorch, torchvision, and Pillow

import argparse
from pathlib import Path
from typing import List, Dict

from PIL import Image
import torch
import torchvision.transforms as T
from torchvision import models

# Simple mapping from recognized ingredients to cocktail suggestions
COCKTAIL_SUGGESTIONS: Dict[str, List[str]] = {
    "lemon": ["Whiskey Sour", "Margarita"],
    "lime": ["Mojito", "Gin and Tonic"],
    "orange": ["Screwdriver", "Tequila Sunrise"],
    "strawberry": ["Strawberry Daiquiri"],
    "vodka": ["Bloody Mary", "Cosmopolitan"],
    "gin": ["Martini", "Tom Collins"],
    "rum": ["Mojito", "Daiquiri"],
}


def load_model() -> torch.nn.Module:
    """Load a pretrained ResNet50 model."""
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.eval()
    return model


def predict_image(model: torch.nn.Module, image_path: Path, topk: int = 5) -> List[str]:
    """Run image classification and return top predicted labels."""
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    top_probs, top_idxs = probabilities.topk(topk)
    labels = [models.ResNet50_Weights.DEFAULT.meta["categories"][i] for i in top_idxs]
    return labels


def suggest_cocktails(labels: List[str]) -> Dict[str, List[str]]:
    """Return a mapping of recognized ingredients to suggested cocktails."""
    suggestions = {}
    for label in labels:
        if label in COCKTAIL_SUGGESTIONS:
            suggestions[label] = COCKTAIL_SUGGESTIONS[label]
    return suggestions


def main() -> None:
    parser = argparse.ArgumentParser(description="Identify ingredients and suggest cocktails")
    parser.add_argument("images", nargs="+", type=Path, help="Path(s) to ingredient images")
    args = parser.parse_args()

    model = load_model()
    for img_path in args.images:
        labels = predict_image(model, img_path)
        suggestions = suggest_cocktails(labels)
        print(f"\nImage: {img_path}")
        print("Top predictions:", ", ".join(labels))
        if suggestions:
            for ingredient, drinks in suggestions.items():
                print(f"\nIngredient '{ingredient}' found. Try these cocktails:")
                for drink in drinks:
                    print(f" - {drink}")
        else:
            print("No cocktail suggestions found for recognized ingredients.")


if __name__ == "__main__":
    main()
