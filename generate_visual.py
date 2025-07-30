# generate_visual.py
import os
import asyncio
from fal import FalClient

async def generate_image(prompt: str, output_dir: str = "results"):
    fal_client = FalClient(key=os.getenv("FAL_KEY"))
    
    try:
        result = await fal_client.run(
            "fal-ai/flux/dev",
            arguments={
                "prompt": prompt,
                "image_size": "landscape_4_3",
                "num_inference_steps": 28
            }
        )
        image_url = result["image"]["url"]
        
        # Сохраняем ссылку
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/latest_visual.txt", "w") as f:
            f.write(image_url)
        
        print(f"✅ Визуал сгенерирован: {image_url}")
        return image_url
    except Exception as e:
        print(f"❌ Ошибка при генерации визуала: {e}")
        return None

if __name__ == "__main__":
    # Для теста
    prompt = input("Введите промпт: ")
    asyncio.run(generate_image(prompt))