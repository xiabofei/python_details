#coding=utf8
import contextlib
import selenium.webdriver as webdriver
import selenium.webdriver.support.ui as ui

from ipdb import set_trace as st

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
        (515,1201)
        '515' for 'Owner occupied No vehicle avaiable of 2015 year'
        '1201' for 'Renter occupied No vehicle avaiable of 2015 year'
    """
    # step1. Open the seed web page
    driver.get('https://factfinder.census.gov/faces/nav/jsf/pages/searchresults.xhtml?refresh=t')
    # step2. Wait until the 'GO' button loaded in the target <form> element
    wait = ui.WebDriverWait(driver,100)
    wait.until(lambda driver: driver.find_element_by_id('refinesearchsubmit'))
    # step3. Fill in the form & click the 'GO' button
    topic = driver.find_element_by_id('searchTopicInput')
    geo = driver.find_element_by_id('searchGeoInput')
    go_button = driver.find_element_by_id('refinesearchsubmit')
    topic.send_keys("B25044")
    geo.send_keys(str(zipcode))
    go_button.click()
    # step4. Wait for the search result loaded in the target <table> element
    wait.until(lambda driver: driver.find_element_by_id('resulttable'))
    st(context=21)

if __name__ == '__main__':
    # step0. Create a certain type of web browser instance (e.g. Chrome FireFox)
    with contextlib.closing(webdriver.Chrome()) as driver:
        get_zero_vehicle_households(driver, 94577)

