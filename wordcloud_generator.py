from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from wordcloud import WordCloud, ImageColorGenerator

def generate_wordcloud(reviews, mask_path="D:\SEM PROJECTS\SEM 2\EOC-2 and MFC-2\ScreenShots\wordcloud.webp"):
    """
    Generate a word cloud image from a list of reviews.

    Args:
    reviews (list): List of review texts.
    mask_path (str): Path to the mask image.

    Returns:
    str: Base64-encoded image string for displaying in HTML.
    """
    if len(reviews) == 0:
        return None

    # Combine all reviews into a single text
    text = " ".join(reviews)

    # Load mask image (if available)
    try:
        mask = np.array(Image.open(mask_path))
    except:
        mask = None

    # Generate the word cloud
    wc = WordCloud(width=800, height=400, random_state=1, 
                   background_color="white", colormap="Set2", 
                   collocations=False, mask=mask).generate(text)

    # Apply colors from mask if available
    if mask is not None:
        image_colors = ImageColorGenerator(mask)
        wc.recolor(color_func=image_colors)

    # Convert to base64 for displaying in Flask
    img = BytesIO()
    plt.figure(figsize=(8, 4))
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.imshow(wc, interpolation="bilinear")
    plt.savefig(img, format="png")
    plt.close()

    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()
