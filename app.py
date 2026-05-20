import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import gradio as gr

from myMetroProcessing import load_models, processOneMetroImage

# Load models once at startup
load_models()

def predict_metro_lines(image: Image.Image, resize_factor: float = 1.0):
    if image is None:
        return None, pd.DataFrame(columns=["image_idx", "y1", "y2", "x1", "x2", "line"])

    # Convert to numpy (RGB)
    im = np.array(image.convert("RGB"))

    # Call project processing function
    im_resized, bd = processOneMetroImage("uploaded", im, 0, resize_factor)

    # Draw predictions and bounding boxes
    fig, ax = plt.subplots()
    ax.imshow(im_resized)
    ax.axis("off")

    if bd is not None and bd.shape[0] > 0:
        for row in bd:
            _, y1, y2, x1, x2, metro_line = row
            rect = Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                linewidth=2
            )
            ax.add_patch(rect)
            ax.text(
                x1,
                y1 - 5,
                f"Line {metro_line}",
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.7)
            )

    # Convert matplotlib figure to PIL image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    out_image = Image.open(buf)

    # Results table for display in Gradio
    if bd is not None and bd.shape[0] > 0:
        df = pd.DataFrame(bd, columns=["image_idx", "y1", "y2", "x1", "x2", "line"])
    else:
        df = pd.DataFrame(columns=["image_idx", "y1", "y2", "x1", "x2", "line"])

    return out_image, df


demo = gr.Interface(
    fn=predict_metro_lines,
    inputs=[
        gr.Image(type="pil", label="Metro sign image"),
        gr.Slider(0.5, 1.5, value=1.0, step=0.1, label="Resize factor")
    ],
    outputs=[
        gr.Image(label="Annotated image"),
        gr.Dataframe(label="Detected lines")
    ],
    title="Paris Metro Line Detection",
    description="Upload a photo of a metro sign. The model detects metro pictograms and predicts the corresponding line.",
    examples=[
        ["IM_(1).jpg", 1.0],
        ["IM_(2).jpg", 1.0],
        ["IM_(3).jpg", 1.0]
    ]
)

if __name__ == "__main__":
    demo.launch()
