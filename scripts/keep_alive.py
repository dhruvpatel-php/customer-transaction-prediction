"""
Streamlit Community Cloud Keep-Alive & Auto Wake-Up Script
Visits the deployed Streamlit app using Playwright Headless Chromium.
- If the app is sleeping ('Zzzz'), automatically detects and clicks 'Yes, get this app back up!'.
- If the app is active, keeps the session alive to reset the 7-day inactivity timer.
"""

import sys
import time
from playwright.sync_api import sync_playwright

APP_URL = "https://customer-transaction-prediction.streamlit.app/"

def run_keep_alive():
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}] Starting keep-alive check for: {APP_URL}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 800}
        )
        page = context.new_page()

        try:
            print("Connecting to Streamlit application...")
            response = page.goto(APP_URL, wait_until="domcontentloaded", timeout=60000)
            status_code = response.status if response else "Unknown"
            print(f"HTTP Response Status: {status_code}")

            # Allow client scripts and web components to initialize
            time.sleep(6)

            page_title = page.title()
            page_content = page.content()
            print(f"Page Title: '{page_title}'")

            # Look for wake-up button variants
            wake_button = page.locator('button:has-text("Yes, get this app back up!")')
            if wake_button.count() == 0:
                wake_button = page.locator('button:has-text("get this app back up")')

            is_sleeping = (
                "This app has gone to sleep" in page_content or
                "Zzzz" in page_content or
                wake_button.count() > 0
            )

            if is_sleeping and wake_button.count() > 0:
                print(">>> [SLEEP MODE DETECTED] App is currently sleeping ('Zzzz - This app has gone to sleep').")
                print(">>> Clicking 'Yes, get this app back up!' button to wake up container...")
                wake_button.first.click()

                print(">>> Wake-up request dispatched. Waiting for boot sequence...")
                try:
                    wake_button.first.wait_for(state="detached", timeout=60000)
                    print(">>> Wake-up prompt dismissed. App container is now starting up!")
                except Exception as wait_err:
                    print(f">>> Detached wait info: {wait_err}")

                # Give time for the container to start booting and establish connection
                time.sleep(20)
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}] [SUCCESS] App wake-up successfully triggered!")
            else:
                print(">>> [ACTIVE MODE] App is awake and responding.")
                print(">>> Holding active session to initialize WebSocket connection and refresh 7-day inactivity timer...")

                try:
                    page.wait_for_selector('[data-testid="stAppViewContainer"], #root', timeout=30000)
                except Exception:
                    pass

                # Keep the WebSocket connection active for 15 seconds to register activity
                time.sleep(15)
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}] [SUCCESS] Active WebSocket session established. Inactivity timer reset!")

        except Exception as err:
            print(f"[ERROR] Keep-alive operation failed: {err}", file=sys.stderr)
            sys.exit(1)
        finally:
            browser.close()

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}] Keep-alive check completed.")

if __name__ == "__main__":
    run_keep_alive()
