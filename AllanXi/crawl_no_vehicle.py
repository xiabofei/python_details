#coding=utf8
import selenium.webdriver as webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
import time
import os

def get_zero_vehicle_households(driver, zipcode):
    """imitate browser action to get 2015 zero vehicle households

    Parameters
    ----------
    driver:
        selenium.webdriver
        Driver which connects to a certain browser instance
    zipcode:
        str
        Input zip code

    Returns
    -------
    tuple
        e.g. (515,1201)
        '515' for 'Owner occupied No vehicle avaiable of 2015 year'
        '1201' for 'Renter occupied No vehicle avaiable of 2015 year'
    """
    # step1. Open the seed web page
    driver.get('https://factfinder.census.gov/faces/nav/jsf/pages/searchresults.xhtml?refresh=t')
    # step2. Wait until the  'topic' 'geo' 'GO' loaded in the target <form> element
    wait = WebDriverWait(driver, 30)
    wait.until(EC.visibility_of_element_located((By.ID,"searchTopicInput")))
    wait.until(EC.visibility_of_element_located((By.ID,"searchGeoInput")))
    wait.until(EC.element_to_be_clickable((By.ID,"searchGeoInput")))
    # step3. Fill in the form & click the 'GO' button
    driver.find_element_by_id('searchTopicInput').send_keys("B25044")
    driver.find_element_by_id('searchGeoInput').send_keys(str(zipcode))
    time.sleep(1)
    driver.find_element_by_id('refinesearchsubmit').click()
    # step4. Wait for the search result loaded in the target <table> element
    wait.until(EC.visibility_of_element_located((By.ID,"resulttable")))
    # step5. Focus & Click the specific position of the table to redirect to target table
    element_xpath = "//tr[@id='yui-rec0']/td/div/a"
    wait.until(EC.visibility_of_element_located((By.XPATH, element_xpath)))
    time.sleep(1)
    driver.find_element_by_xpath(element_xpath).click()
    # step6. Wait for the target inner table loaded 
    wait.until(EC.visibility_of_element_located((By.ID,"inner_table_container")))
    inner_table_container = driver.find_element_by_id("inner_table_container")
    no_vehicle_owner = inner_table_container.find_element_by_xpath("//tr[3]/td[1]") 
    no_vehicle_renter = inner_table_container.find_element_by_xpath("//tr[10]/td[1]")
    return (int(no_vehicle_owner.text.replace(',','')), int(no_vehicle_renter.text.replace(',','')))

if __name__ == '__main__':
    # step0. Create a certain type of web browser instance (e.g. Chrome FireFox)
    caps = DesiredCapabilities.FIREFOX
    caps["marionette"] = True
    assert os.path.isfile('/Applications/Firefox.app/Contents/MacOS/firefox-bin'), "firefox binary path does not match" 
    caps["binary"] = '/Applications/Firefox.app/Contents/MacOS/firefox-bin'
    geckodriver = './geckodriver'
    try:
        driver = webdriver.Firefox(capabilities=caps, executable_path=geckodriver)
        # If we choose Chrome as webdriver, it will probably occur "cannot focus element", which is a known bug
        # of chromedriver. (https://github.com/seleniumhq/selenium-google-code-issue-archive/issues/2328)
        # driver = webdriver.Chrome('./chromedriver')
        driver.set_page_load_timeout(30)
        ret = get_zero_vehicle_households(driver, 94577)
        print ret
    finally:
        if driver is not None:
            driver.quit()
