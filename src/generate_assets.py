"""
Generate local image assets for Amazon India BI Dashboard
Creates high-res crisp PNG cards for Amazon Logo and each of the 5 dataset product categories
"""

import os
from PIL import Image, ImageDraw, ImageFont

os.makedirs("assets", exist_ok=True)

def create_amazon_logo():
    width, height = 300, 90
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Try default fonts or clean shapes
    # Background pill
    draw.rounded_rectangle([10, 5, 290, 85], radius=12, fill=(35, 47, 62, 255))
    
    # Text "amazon"
    try:
        font_large = ImageFont.truetype("arial.ttf", 36)
        font_small = ImageFont.truetype("arial.ttf", 16)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
        
    draw.text((35, 18), "amazon", fill=(255, 255, 255, 255), font=font_large)
    draw.text((170, 24), ".in", fill=(255, 153, 0, 255), font=font_large)
    
    # Amazon Orange Smile curve
    draw.arc([35, 42, 175, 78], start=20, end=160, fill=(255, 153, 0, 255), width=4)
    # Smile arrow tip
    draw.polygon([(170, 62), (160, 54), (166, 68)], fill=(255, 153, 0, 255))
    
    img.save("assets/amazon_logo.png")
    print("Saved assets/amazon_logo.png")

def create_category_card(filename, title, subtitle, bg_color, accent_color, emoji_text):
    width, height = 400, 160
    img = Image.new("RGBA", (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Accent border & top highlight
    draw.rectangle([0, 0, width, 8], fill=accent_color)
    draw.rounded_rectangle([0, 0, width-1, height-1], radius=10, outline=accent_color, width=2)
    
    try:
        font_emoji = ImageFont.truetype("seguiemj.ttf", 48)
        font_title = ImageFont.truetype("arialbd.ttf", 20)
        font_sub = ImageFont.truetype("arial.ttf", 14)
    except:
        font_emoji = ImageFont.load_default()
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()

    # Left visual badge circle
    draw.ellipse([25, 40, 95, 110], fill=(255, 255, 255, 240), outline=accent_color, width=2)
    draw.text((38, 48), emoji_text, fill=accent_color, font=font_emoji)
    
    # Text details
    draw.text((115, 42), title, fill=(255, 255, 255, 255), font=font_title)
    draw.text((115, 72), subtitle, fill=(220, 220, 220, 255), font=font_sub)
    draw.text((115, 100), "Amazon India Verified Catalog", fill=accent_color, font=font_sub)
    
    img.save(f"assets/{filename}")
    print(f"Saved assets/{filename}")

if __name__ == "__main__":
    create_amazon_logo()
    
    create_category_card(
        "cat_electronics.png",
        "Electronics & Mobiles",
        "Phones, Smartwatches, Audio & Gear",
        (20, 30, 48, 255),
        (255, 153, 0, 255),
        "📱"
    )
    create_category_card(
        "cat_apparel.png",
        "Apparel & Fashion",
        "Ethnic Sarees, Kurtas, Denim & Shoes",
        (35, 45, 60, 255),
        (0, 168, 225, 255),
        "👕"
    )
    create_category_card(
        "cat_home.png",
        "Home & Kitchen",
        "Purifiers, Cooktops, Mixers & Decor",
        (28, 40, 35, 255),
        (46, 204, 113, 255),
        "🍳"
    )
    create_category_card(
        "cat_beauty.png",
        "Beauty & Personal Care",
        "Skincare, Lipsticks, Oils & Trimmers",
        (45, 25, 40, 255),
        (231, 76, 60, 255),
        "💄"
    )
    create_category_card(
        "cat_pantry.png",
        "Pantry & Groceries",
        "Ghee, Basmati Rice, Coffee & Spices",
        (45, 40, 25, 255),
        (241, 196, 15, 255),
        "🌾"
    )
