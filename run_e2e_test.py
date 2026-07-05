import asyncio
import os
import sys
from playwright.async_api import async_playwright

async def run_test():
    async with async_playwright() as p:
        print("Launching headless Chromium browser for E2E testing...")
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1440, "height": 900})
        
        url = "http://localhost:3071"
        print(f"Navigating to web interface at {url}...")
        await page.goto(url)
        
        # Base folder configuration
        base_dir = "screenshots/01_slang_splat/01_web_demo"
        os.makedirs(base_dir, exist_ok=True)
        
        # Wait for the initial scene configuration to load and render
        print("Waiting for page and default scene to render...")
        await asyncio.sleep(4)
        
        # 1. Capture before training start
        step1_path = f"{base_dir}/01_before_training_start.png"
        await page.screenshot(path=step1_path)
        print(f"Captured: {step1_path}")
        
        # 2. Click Start Train
        print("Clicking 'Train' button to start training...")
        await page.click("#btn-start")
        
        # Wait 3 seconds to let some training iterations progress
        await asyncio.sleep(3)
        
        # Capture training progress
        step2_path = f"{base_dir}/02_after_training_start.png"
        await page.screenshot(path=step2_path)
        print(f"Captured: {step2_path}")
        
        # 3. Click Pause
        print("Clicking 'Pause' button to halt training...")
        await page.click("#btn-stop")
        await asyncio.sleep(1) # wait for pause state to settle
        
        # Capture paused state
        step3_path = f"{base_dir}/03_after_training_pause.png"
        await page.screenshot(path=step3_path)
        print(f"Captured: {step3_path}")
        
        # 4. Drag Azimuth slider to rotate viewpoint
        print("Adjusting orbit azimuth slider to rotate camera...")
        # Evaluate to change azimuth to 180 degrees and dispatch input event
        await page.evaluate("""
            const slider = document.getElementById('orbit-azimuth-slider');
            slider.value = 180;
            slider.dispatchEvent(new Event('input'));
        """)
        await asyncio.sleep(1.5) # Wait for Vulkan render to complete and display
        
        # Capture rotated orbit view
        step4_path = f"{base_dir}/04_after_camera_orbit.png"
        await page.screenshot(path=step4_path)
        print(f"Captured: {step4_path}")
        
        # 5. Click Reset Scene
        print("Clicking 'Reset Scene' to restore original state...")
        await page.click("#btn-reset")
        await asyncio.sleep(2) # wait for reset and render
        
        # Capture reset state
        step5_path = f"{base_dir}/05_after_scene_reset.png"
        await page.screenshot(path=step5_path)
        print(f"Captured: {step5_path}")
        
        await browser.close()
        print("E2E testing complete. All screenshots generated successfully.")

if __name__ == "__main__":
    try:
        asyncio.run(run_test())
    except Exception as e:
        print(f"E2E Test encountered an error: {e}", file=sys.stderr)
        sys.exit(1)
