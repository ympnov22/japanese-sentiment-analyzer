import pytest
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.common.exceptions import TimeoutException, WebDriverException

class TestBrowserCompatibility:
    """Browser compatibility testing for the Japanese sentiment analysis frontend"""
    
    FRONTEND_URL = "file:///home/ubuntu/japanese-sentiment-analyzer/frontend/index.html"
    
    def setup_method(self):
        """Setup test fixtures"""
        self.test_text = "この映画は素晴らしかった"
        self.drivers = []
    
    def teardown_method(self):
        """Cleanup after tests"""
        for driver in self.drivers:
            try:
                driver.quit()
            except:
                pass
    
    def create_chrome_driver(self, mobile=False):
        """Create Chrome WebDriver instance"""
        try:
            options = ChromeOptions()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            
            if mobile:
                mobile_emulation = {
                    "deviceMetrics": {"width": 375, "height": 667, "pixelRatio": 2.0},
                    "userAgent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15"
                }
                options.add_experimental_option("mobileEmulation", mobile_emulation)
            
            driver = webdriver.Chrome(options=options)
            self.drivers.append(driver)
            return driver
        except WebDriverException as e:
            pytest.skip(f"Chrome driver not available: {e}")
    
    def create_firefox_driver(self):
        """Create Firefox WebDriver instance"""
        try:
            options = FirefoxOptions()
            options.add_argument('--headless')
            
            driver = webdriver.Firefox(options=options)
            self.drivers.append(driver)
            return driver
        except WebDriverException as e:
            pytest.skip(f"Firefox driver not available: {e}")
    
    def test_chrome_desktop_compatibility(self):
        """Test Chrome desktop compatibility"""
        driver = self.create_chrome_driver()
        self._test_basic_functionality(driver, "Chrome Desktop")
    
    def test_chrome_mobile_compatibility(self):
        """Test Chrome mobile compatibility"""
        driver = self.create_chrome_driver(mobile=True)
        self._test_mobile_functionality(driver, "Chrome Mobile")
    
    def test_firefox_compatibility(self):
        """Test Firefox compatibility"""
        driver = self.create_firefox_driver()
        self._test_basic_functionality(driver, "Firefox")
    
    def _test_basic_functionality(self, driver, browser_name):
        """Test basic functionality across browsers"""
        print(f"Testing {browser_name} compatibility...")
        
        driver.get(self.FRONTEND_URL)
        
        wait = WebDriverWait(driver, 10)
        
        title_element = wait.until(EC.presence_of_element_located((By.TAG_NAME, "h1")))
        assert "日本語感情分析" in title_element.text
        
        text_input = wait.until(EC.presence_of_element_located((By.ID, "textInput")))
        assert text_input.is_displayed()
        
        analyze_button = driver.find_element(By.ID, "analyzeButton")
        assert analyze_button.is_displayed()
        
        result_div = driver.find_element(By.ID, "result")
        assert result_div.is_displayed()
        
        text_input.clear()
        text_input.send_keys(self.test_text)
        
        char_count = driver.find_element(By.ID, "charCount")
        assert str(len(self.test_text)) in char_count.text
        
        print(f"{browser_name} basic functionality test passed")
    
    def _test_mobile_functionality(self, driver, browser_name):
        """Test mobile-specific functionality"""
        print(f"Testing {browser_name} mobile compatibility...")
        
        driver.get(self.FRONTEND_URL)
        
        wait = WebDriverWait(driver, 10)
        
        container = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "container")))
        
        viewport_width = driver.execute_script("return window.innerWidth")
        assert viewport_width <= 375, f"Mobile viewport width {viewport_width}px exceeds 375px"
        
        text_input = driver.find_element(By.ID, "textInput")
        input_width = text_input.size['width']
        container_width = container.size['width']
        
        assert input_width <= container_width, "Text input exceeds container width on mobile"
        
        analyze_button = driver.find_element(By.ID, "analyzeButton")
        button_width = analyze_button.size['width']
        
        assert button_width <= container_width, "Button exceeds container width on mobile"
        
        print(f"{browser_name} mobile functionality test passed")
    
    def test_css_compatibility(self):
        """Test CSS compatibility across browsers"""
        driver = self.create_chrome_driver()
        driver.get(self.FRONTEND_URL)
        
        wait = WebDriverWait(driver, 10)
        
        container = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "container")))
        
        container_styles = driver.execute_script(
            "return window.getComputedStyle(arguments[0])", container
        )
        
        assert container_styles['max-width'] == '800px'
        assert 'auto' in container_styles['margin']
        
        text_input = driver.find_element(By.ID, "textInput")
        input_styles = driver.execute_script(
            "return window.getComputedStyle(arguments[0])", text_input
        )
        
        assert input_styles['width'] == '100%'
        assert input_styles['border-radius'] == '4px'
        
        print("CSS compatibility test passed")
    
    def test_javascript_compatibility(self):
        """Test JavaScript compatibility"""
        driver = self.create_chrome_driver()
        driver.get(self.FRONTEND_URL)
        
        wait = WebDriverWait(driver, 10)
        
        text_input = wait.until(EC.presence_of_element_located((By.ID, "textInput")))
        
        text_input.send_keys("テスト")
        
        char_count = driver.find_element(By.ID, "charCount")
        
        time.sleep(0.5)
        
        assert "3" in char_count.text
        
        text_input.clear()
        text_input.send_keys("a" * 1001)
        
        time.sleep(0.5)
        
        error_div = driver.find_element(By.ID, "error")
        error_text = error_div.text
        
        assert "1000文字以内" in error_text or error_div.is_displayed()
        
        print("JavaScript compatibility test passed")
    
    def test_responsive_design(self):
        """Test responsive design at different viewport sizes"""
        driver = self.create_chrome_driver()
        
        viewport_sizes = [
            (375, 667),   # Mobile
            (768, 1024),  # Tablet
            (1024, 768),  # Desktop small
            (1920, 1080)  # Desktop large
        ]
        
        for width, height in viewport_sizes:
            driver.set_window_size(width, height)
            driver.get(self.FRONTEND_URL)
            
            wait = WebDriverWait(driver, 10)
            container = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "container")))
            
            container_width = container.size['width']
            viewport_width = driver.execute_script("return window.innerWidth")
            
            if width <= 768:
                assert container_width <= viewport_width * 0.95, f"Container too wide for {width}px viewport"
            else:
                assert container_width <= 800, f"Container exceeds max-width at {width}px viewport"
            
            text_input = driver.find_element(By.ID, "textInput")
            assert text_input.is_displayed(), f"Text input not visible at {width}x{height}"
            
            analyze_button = driver.find_element(By.ID, "analyzeButton")
            assert analyze_button.is_displayed(), f"Button not visible at {width}x{height}"
            
            print(f"Responsive design test passed for {width}x{height}")
    
    def test_form_validation_compatibility(self):
        """Test form validation across browsers"""
        driver = self.create_chrome_driver()
        driver.get(self.FRONTEND_URL)
        
        wait = WebDriverWait(driver, 10)
        
        analyze_button = wait.until(EC.element_to_be_clickable((By.ID, "analyzeButton")))
        
        analyze_button.click()
        
        time.sleep(1)
        
        error_div = driver.find_element(By.ID, "error")
        assert error_div.is_displayed() or "入力してください" in error_div.text
        
        text_input = driver.find_element(By.ID, "textInput")
        text_input.send_keys("a" * 1001)
        
        analyze_button.click()
        
        time.sleep(1)
        
        assert error_div.is_displayed() or "1000文字以内" in error_div.text
        
        print("Form validation compatibility test passed")

class TestAccessibility:
    """Accessibility testing for the frontend"""
    
    FRONTEND_URL = "file:///home/ubuntu/japanese-sentiment-analyzer/frontend/index.html"
    
    def setup_method(self):
        """Setup test fixtures"""
        self.drivers = []
    
    def teardown_method(self):
        """Cleanup after tests"""
        for driver in self.drivers:
            try:
                driver.quit()
            except:
                pass
    
    def create_driver(self):
        """Create WebDriver instance"""
        try:
            options = ChromeOptions()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            
            driver = webdriver.Chrome(options=options)
            self.drivers.append(driver)
            return driver
        except WebDriverException as e:
            pytest.skip(f"Chrome driver not available: {e}")
    
    def test_keyboard_navigation(self):
        """Test keyboard navigation"""
        driver = self.create_driver()
        driver.get(self.FRONTEND_URL)
        
        wait = WebDriverWait(driver, 10)
        
        text_input = wait.until(EC.presence_of_element_located((By.ID, "textInput")))
        
        text_input.click()
        
        active_element = driver.switch_to.active_element
        assert active_element == text_input
        
        from selenium.webdriver.common.keys import Keys
        active_element.send_keys(Keys.TAB)
        
        active_element = driver.switch_to.active_element
        analyze_button = driver.find_element(By.ID, "analyzeButton")
        
        assert active_element == analyze_button or active_element.tag_name == "button"
        
        print("Keyboard navigation test passed")
    
    def test_aria_labels(self):
        """Test ARIA labels and accessibility attributes"""
        driver = self.create_driver()
        driver.get(self.FRONTEND_URL)
        
        wait = WebDriverWait(driver, 10)
        
        text_input = wait.until(EC.presence_of_element_located((By.ID, "textInput")))
        
        placeholder = text_input.get_attribute("placeholder")
        assert placeholder and len(placeholder) > 0
        
        analyze_button = driver.find_element(By.ID, "analyzeButton")
        button_text = analyze_button.text
        assert button_text and len(button_text) > 0
        
        print("ARIA labels test passed")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
