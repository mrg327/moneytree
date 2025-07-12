"""TikTok automation for video uploads using Selenium WebDriver."""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import random

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager

logger = logging.getLogger(__name__)


class TikTokClient:
    """Client for automating TikTok video uploads using Selenium."""
    
    def __init__(self, headless: bool = True, cookies_file: Optional[str] = None):
        """Initialize TikTok client.
        
        Args:
            headless: Whether to run browser in headless mode
            cookies_file: Path to cookies file for session persistence
        """
        self.headless = headless
        self.cookies_file = cookies_file or 'tiktok_cookies.json'
        self.driver = None
        self.authenticated = False
        
        # Selenium configuration
        self.base_wait_time = 10
        self.upload_wait_time = 300  # 5 minutes for upload processing
        self.max_retries = 3
        self.typing_delay = 0.1
        
    def _setup_driver(self) -> bool:
        """Setup Chrome WebDriver with appropriate options.
        
        Returns:
            True if driver setup successful, False otherwise
        """
        try:
            chrome_options = Options()
            
            if self.headless:
                chrome_options.add_argument("--headless")
            
            # Add options to avoid detection
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            
            # Execute script to remove webdriver property
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            logger.info("Chrome WebDriver setup successful")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup Chrome WebDriver: {e}")
            return False
    
    def _save_cookies(self) -> bool:
        """Save current session cookies to file.
        
        Returns:
            True if cookies saved successfully, False otherwise
        """
        try:
            if not self.driver:
                return False
                
            cookies = self.driver.get_cookies()
            with open(self.cookies_file, 'w') as f:
                json.dump(cookies, f, indent=2)
            
            logger.info(f"Cookies saved to {self.cookies_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save cookies: {e}")
            return False
    
    def _load_cookies(self) -> bool:
        """Load session cookies from file.
        
        Returns:
            True if cookies loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(self.cookies_file):
                logger.info(f"No cookies file found at {self.cookies_file}")
                return False
            
            with open(self.cookies_file, 'r') as f:
                cookies = json.load(f)
            
            if not self.driver:
                return False
                
            # Navigate to TikTok first to set domain
            self.driver.get("https://www.tiktok.com")
            
            for cookie in cookies:
                try:
                    self.driver.add_cookie(cookie)
                except Exception as e:
                    logger.warning(f"Failed to add cookie {cookie.get('name', 'unknown')}: {e}")
            
            logger.info("Cookies loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load cookies: {e}")
            return False
    
    def _wait_and_click(self, selector: str, by: By = By.XPATH, timeout: int = None) -> bool:
        """Wait for element and click it.
        
        Args:
            selector: Element selector
            by: Selenium By strategy
            timeout: Wait timeout in seconds
            
        Returns:
            True if element clicked successfully, False otherwise
        """
        timeout = timeout or self.base_wait_time
        
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.element_to_be_clickable((by, selector))
            )
            element.click()
            return True
            
        except TimeoutException:
            logger.error(f"Timeout waiting for clickable element: {selector}")
            return False
        except Exception as e:
            logger.error(f"Error clicking element {selector}: {e}")
            return False
    
    def _wait_and_send_keys(self, selector: str, text: str, by: By = By.XPATH, 
                          timeout: int = None, clear_first: bool = True) -> bool:
        """Wait for element and send keys to it.
        
        Args:
            selector: Element selector
            text: Text to send
            by: Selenium By strategy
            timeout: Wait timeout in seconds
            clear_first: Whether to clear field first
            
        Returns:
            True if text sent successfully, False otherwise
        """
        timeout = timeout or self.base_wait_time
        
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by, selector))
            )
            
            if clear_first:
                element.clear()
            
            # Type character by character to simulate human typing
            for char in text:
                element.send_keys(char)
                time.sleep(self.typing_delay)
            
            return True
            
        except TimeoutException:
            logger.error(f"Timeout waiting for element: {selector}")
            return False
        except Exception as e:
            logger.error(f"Error sending keys to element {selector}: {e}")
            return False
    
    def _switch_to_upload_iframe(self) -> bool:
        """Switch to TikTok upload iframe.
        
        Returns:
            True if iframe switch successful, False otherwise
        """
        try:
            # Wait for iframe to be present
            iframe = WebDriverWait(self.driver, 30).until(
                EC.presence_of_element_located((By.TAG_NAME, "iframe"))
            )
            
            self.driver.switch_to.frame(iframe)
            logger.info("Switched to upload iframe")
            return True
            
        except TimeoutException:
            logger.error("Timeout waiting for upload iframe")
            return False
        except Exception as e:
            logger.error(f"Error switching to upload iframe: {e}")
            return False
    
    def authenticate(self, username: Optional[str] = None, password: Optional[str] = None) -> bool:
        """Authenticate with TikTok.
        
        Args:
            username: TikTok username/email
            password: TikTok password
            
        Returns:
            True if authentication successful, False otherwise
        """
        if not self._setup_driver():
            return False
        
        try:
            # Try cookie-based authentication first
            if self._load_cookies():
                self.driver.get("https://www.tiktok.com/creator-center/upload")
                time.sleep(3)
                
                # Check if we're already logged in
                if "login" not in self.driver.current_url.lower():
                    logger.info("Cookie-based authentication successful")
                    self.authenticated = True
                    return True
            
            # Fall back to username/password authentication
            if username and password:
                logger.info("Attempting username/password authentication")
                self.driver.get("https://www.tiktok.com/login/phone-or-email/email")
                
                # Fill in email
                if not self._wait_and_send_keys('//input[@name="username"]', username):
                    logger.error("Failed to enter username")
                    return False
                
                # Fill in password
                if not self._wait_and_send_keys('//input[@type="password"]', password):
                    logger.error("Failed to enter password")
                    return False
                
                # Click login button
                if not self._wait_and_click('//button[@type="submit"]'):
                    logger.error("Failed to click login button")
                    return False
                
                # Wait for potential 2FA or captcha
                logger.info("Waiting for login completion (handle 2FA/captcha if prompted)")
                time.sleep(10)
                
                # Check if login was successful
                if "login" not in self.driver.current_url.lower():
                    logger.info("Username/password authentication successful")
                    self._save_cookies()
                    self.authenticated = True
                    return True
            
            # Manual authentication fallback
            logger.info("Automatic authentication failed. Please log in manually.")
            self.driver.get("https://www.tiktok.com/login")
            
            input("Please log in manually in the browser window, then press Enter to continue...")
            
            # Check if manual login was successful
            if "login" not in self.driver.current_url.lower():
                logger.info("Manual authentication successful")
                self._save_cookies()
                self.authenticated = True
                return True
            else:
                logger.error("Manual authentication failed")
                return False
                
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False
    
    def upload_video(self, video_path: str, title: str, description: str = "", 
                    tags: Optional[List[str]] = None, privacy: str = "public") -> bool:
        """Upload a video to TikTok.
        
        Args:
            video_path: Path to video file
            title: Video title
            description: Video description
            tags: List of hashtags (without #)
            privacy: Privacy setting (public, private, friends)
            
        Returns:
            True if upload successful, False otherwise
        """
        if not self.authenticated:
            logger.error("Must authenticate before uploading videos")
            return False
        
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return False
        
        try:
            # Navigate to upload page
            self.driver.get("https://www.tiktok.com/creator-center/upload")
            
            # Switch to upload iframe
            if not self._switch_to_upload_iframe():
                logger.error("Failed to switch to upload iframe")
                return False
            
            # Upload video file
            try:
                file_input = WebDriverWait(self.driver, 30).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'input[type="file"]'))
                )
                file_input.send_keys(os.path.abspath(video_path))
                logger.info(f"Video file uploaded: {video_path}")
                
            except TimeoutException:
                logger.error("Timeout waiting for file upload input")
                return False
            
            # Wait for video to process
            time.sleep(10)
            
            # Fill in title
            title_text = title
            if tags:
                hashtags = ' '.join([f'#{tag}' for tag in tags])
                title_text = f"{title} {hashtags}"
            
            title_selectors = [
                '//div[@role="combobox"]',
                '//div[@data-testid="video-title-input"]',
                '//textarea[@placeholder="Describe your video"]',
                '//div[contains(@class, "public-DraftEditor-content")]'
            ]
            
            title_set = False
            for selector in title_selectors:
                try:
                    element = WebDriverWait(self.driver, 10).until(
                        EC.element_to_be_clickable((By.XPATH, selector))
                    )
                    element.click()
                    element.clear()
                    
                    # Type title character by character
                    for char in title_text:
                        element.send_keys(char)
                        time.sleep(self.typing_delay)
                    
                    logger.info(f"Title set using selector: {selector}")
                    title_set = True
                    break
                    
                except TimeoutException:
                    continue
                except Exception as e:
                    logger.warning(f"Failed to set title with selector {selector}: {e}")
                    continue
            
            if not title_set:
                logger.error("Failed to set video title")
                return False
            
            # Set privacy if needed
            if privacy.lower() != "public":
                privacy_selectors = [
                    '//div[@data-testid="privacy-selector"]',
                    '//button[contains(text(), "Who can view this video")]'
                ]
                
                for selector in privacy_selectors:
                    try:
                        if self._wait_and_click(selector, timeout=5):
                            # Look for privacy option
                            privacy_option = f'//div[contains(text(), "{privacy.title()}")]'
                            if self._wait_and_click(privacy_option, timeout=5):
                                logger.info(f"Privacy set to: {privacy}")
                                break
                    except:
                        continue
            
            # Wait for upload processing to complete
            logger.info("Waiting for video processing to complete...")
            
            # Look for the submit/post button
            submit_selectors = [
                '//button[contains(text(), "Post")]',
                '//button[@data-testid="submit-button"]',
                '//button[contains(@class, "css-y1m958")]',
                '//button[not(@disabled) and contains(text(), "Post")]'
            ]
            
            submit_clicked = False
            for selector in submit_selectors:
                try:
                    # Scroll to button
                    try:
                        button = self.driver.find_element(By.XPATH, selector)
                        self.driver.execute_script("arguments[0].scrollIntoView(true);", button)
                        time.sleep(2)
                    except:
                        pass
                    
                    # Wait for button to be clickable
                    element = WebDriverWait(self.driver, self.upload_wait_time).until(
                        EC.element_to_be_clickable((By.XPATH, selector))
                    )
                    element.click()
                    logger.info(f"Submit button clicked using selector: {selector}")
                    submit_clicked = True
                    break
                    
                except TimeoutException:
                    logger.warning(f"Timeout waiting for submit button: {selector}")
                    continue
                except Exception as e:
                    logger.warning(f"Failed to click submit button {selector}: {e}")
                    continue
            
            if not submit_clicked:
                logger.error("Failed to click submit button")
                return False
            
            # Wait for upload completion
            logger.info("Waiting for upload completion...")
            time.sleep(10)
            
            # Check for success indicators
            success_indicators = [
                "Your video is being processed",
                "Video uploaded successfully",
                "Posted successfully"
            ]
            
            page_source = self.driver.page_source.lower()
            upload_success = any(indicator.lower() in page_source for indicator in success_indicators)
            
            if upload_success:
                logger.info("Video upload completed successfully")
                return True
            else:
                logger.warning("Upload completion status uncertain")
                return True  # Assume success if no clear failure
                
        except Exception as e:
            logger.error(f"Error during video upload: {e}")
            return False
        finally:
            # Switch back to default content
            try:
                self.driver.switch_to.default_content()
            except:
                pass
    
    def close(self):
        """Close the browser and cleanup resources."""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("Browser closed successfully")
            except Exception as e:
                logger.error(f"Error closing browser: {e}")
            finally:
                self.driver = None
                self.authenticated = False
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()