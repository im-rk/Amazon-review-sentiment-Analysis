from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from wordcloud import WordCloud, ImageColorGenerator

def generate_wordcloud(reviews, mask_path="D:\SEM PROJECTS\SEM 2\EOC-2 and MFC-2\ScreenShots\wordcloud.webp"):

    if len(reviews) == 0:
        return None
    text = " ".join(reviews)

    try:
        mask = np.array(Image.open(mask_path))
    except:
        mask = None

    wc = WordCloud(width=800, height=400, random_state=1, 
                   background_color="white", colormap="Set2", 
                   collocations=False, mask=mask).generate(text)

    if mask is not None:
        image_colors = ImageColorGenerator(mask)
        wc.recolor(color_func=image_colors)

    img = BytesIO()
    plt.figure(figsize=(8, 4))
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.imshow(wc, interpolation="bilinear")
    plt.savefig(img, format="png")
    plt.close()

    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()
