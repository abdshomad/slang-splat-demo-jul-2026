import asyncio
import os
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        print("Launching headless browser...")
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1440, "height": 900})
        
        print("Navigating to http://localhost:3071...")
        await page.goto("http://localhost:3071")
        
        # Wait for pre-initialization and rendering
        print("Waiting 5 seconds for scene to load and render...")
        await asyncio.sleep(5)
        
        output_path = "screenshots/01_slang_splat/01_web_demo/01_after_launch.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        await page.screenshot(path=output_path)
        print(f"Screenshot successfully saved to: {output_path}")
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
