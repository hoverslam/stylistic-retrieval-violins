import re
import json
import time
import requests
from pathlib import Path
from dataclasses import dataclass, asdict

from tqdm import tqdm
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By


@dataclass
class Instrument:
    """Represents an instrument with its attributes.

    Attributes:
        id (str): The unique identifier of the instrument.
        family (str): The family (MIMO 1st level, e.g. "Stringed instruments") to which the instrument belongs.
        group (str): The group (MIMO 2nd level, e.g. "Violins") to which the instrument belongs.
        instrument (str): The specific name (MIMO 3nd level, e.g. "Violin")  of the instrument.
        instrument_url (str): The permanent link to the instrument.
        maker (str): The author/creator/maker of the instrument.
        date_of_production (str): The date when the instrument was produced.
        place_of_production (str): The location where the instrument was produced.
        images (list): A list of image file names associated with the instrument.
        image_urls (list): A list of URLs from where the images are dowloaded from.
    """

    id: str
    family: str
    group: str
    instrument: str
    instrument_url: str
    maker: str
    date_of_production: str
    place_of_production: str
    images: list
    image_urls: list


class MIMOScraper:
    """A scraper for extracting instrument information from the MIMO (Musical Instrument Museums Online) website.

    This class provides methods to navigate the MIMO website, extract instrument details, download images, and store
    the data in a structured JSON format.
    """

    def __init__(self, root: str | Path, name: str) -> None:
        """Initializes the MIMOScraper with a root directory and a name for the instrument type.

        Args:
            root (str | Path): The root directory where the scraped data will be stored.
            name (str): The name of the instrument type being scraped. This will be used as a subdirectory name.
        """
        self.root = Path(root)
        self.name = name

        self.instruments = []

        img_folder = self.root / self.name / "images"
        img_folder.mkdir(parents=True, exist_ok=True)

        meta_folder = self.root / self.name / "meta"
        meta_folder.mkdir(parents=True, exist_ok=True)

    def start(
        self,
        family_name: str,
        group_name: str,
        instrument_name: str,
        request_interval: int = 5,
        download_images: bool = True,
    ) -> None:
        """Start the scraping process for a specific instrument, extracting its data.

        This method performs the following steps:
        1. Navigates the MIMO International website to select the specified instrument
           using the provided family, group, and instrument names.
        2. Retrieves the starting URL of the AJAX catalog for the selected instrument.
        3. Finds all instrument IDs within the catalog.
        4. Parses instrument information and downloads images for each ID.
        5. Writes the extracted data to a JSON file.

        Args:
            family_name (str): The exact title of the instrument family to select.
            group_name (str): The exact title of the instrument group to select.
            instrument_name (str): The exact title of the specific instrument to select.
            request_interval (int, optional): The time interval (in seconds) between requests to avoid overloading the server. Defaults to 5.
            download_images (bool, optional): Whether to download images associated with each instrument. Defaults to True.
        """
        # Step 1: Find all ids for the selected instrument type. This has to be done with Selenium and
        # a headless browser since MIMO uses AJAX to load its instrument overview.
        print("\nStep 1/3: Looking for available instruments ...")
        url = self._get_starting_url(family_name, group_name, instrument_name)
        instrument_ids = self.get_instrument_ids(url) if url is not None else []
        print(f"Found {len(instrument_ids)} unique instruments.")
        print("DONE!" + "\n")

        # Step 2: With these ids the information and all available images from the individual
        # instrument pages is extracted.
        print("Step 2/3: Extracting information and images of instruments ...")
        with tqdm(total=len(instrument_ids), unit="instrument") as pbar:
            for instrument_id in instrument_ids:
                pbar.set_description(f"{instrument_id}")
                self.instruments.append(self._parse_instrument(instrument_id, download_images))
                time.sleep(request_interval)
                pbar.update(1)
        print("DONE!" + "\n")

        # Step 3: For each instrument, write information (together with the image file names) to a JSON file.
        print("Step 3/3: Writing info to JSON file ...")
        instruments_data = [asdict(instr) for instr in self.instruments]
        file_path = self.root / self.name / "meta" / f"{self.name}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(instruments_data, f, indent=4, ensure_ascii=False)
        print("DONE!")

    def get_instrument_ids(self, url: str) -> set[str]:
        """Retrieves a list of instrument IDs from the given URL using Selenium and a headless browser.

        This method navigates the page, waits for the content to load (which is done via AJAX),
        and extracts the IDs from the HTML source. It also handles pagination by clicking the "next" button.

        Args:
            url (str): The starting URL of the instrument listing page.

        Returns:
            set[str]: A set of instrument IDs found on the page.
        """
        driver = webdriver.Firefox()
        driver.get(url)

        instrument_ids = set()
        try:
            while True:
                # Wait for the results container to load
                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "#resultats")))

                # Parse the current page's source with BeautifulSoup and extract instrument ids
                page_source = driver.page_source
                html = BeautifulSoup(page_source, "html.parser")
                results = html.find_all("li", class_="search-item")
                for parent in results:
                    notice = parent.find(class_="notice")
                    if notice and notice.has_attr("id"):
                        id = notice["id"].removeprefix("rscId_").removesuffix("_IFD")
                        instrument_ids.add(id)

                # Check if the "Next" button is available and click it
                try:
                    next_button = WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, ".btn-link-like.suivant"))
                    )
                    next_button.click()
                    time.sleep(2)
                    WebDriverWait(driver, 30).until(lambda driver: driver.execute_script("return jQuery.active") == 0)
                except Exception:
                    break
        finally:
            driver.quit()

        return instrument_ids

    def _parse_instrument(self, id: str, download_images: bool) -> Instrument | None:
        """Parses the instrument information from a single instrument page.

        Args:
            id (str): The ID of the instrument.
            download_images (bool): Whether to download images for the instrument.

        Returns:
            Instrument | None: An Instrument object containing the extracted data, or None if the page could not be accessed.
        """
        url = "https://mimo-international.com/MIMO/doc/IFD/" + id
        headers = {"User-Agent": "stylistic-retrieval/1.0"}
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            html = BeautifulSoup(response.content, "html.parser")  # type: ignore

            type_tag = html.find("div", class_="InstrumentTypeValue")
            type = type_tag.get_text(strip=True) if type_tag else "N/A"
            matches = re.split(r"\s*/\s*", type.strip())
            family = matches[0] if len(matches) > 0 else "N/A"
            group = matches[1] if len(matches) > 1 else "N/A"
            instrument = matches[2] if len(matches) > 2 else "N/A"

            author_tag = html.find("h3", class_="AuthorValue")
            author = author_tag.get_text(strip=True) if author_tag else "N/A"

            date_of_production_tag = html.find("p", class_="DateOfProductionValue")
            date_of_production = date_of_production_tag.get_text(strip=True) if date_of_production_tag else "N/A"

            place_of_production_tag = html.find("p", class_="PlaceOfProductionValue")
            place_of_production = place_of_production_tag.get_text(strip=True) if place_of_production_tag else "N/A"

            # Find all available images for the instrument and donwload them
            images = []
            img_urls = []
            img_tag = html.find("a", class_="modal-focus")
            img_urls = img_urls + [img_tag["href"]] if img_tag else img_urls  # type: ignore
            tab2_tag = html.find("div", id="tab2")  # tab2 can hold more images
            if tab2_tag:
                for element in tab2_tag.find_all("a", class_="fancybox"):  # type: ignore
                    img_urls.append(element["href"])
            if download_images:
                images = list(filter(None, (self._download_image(img_url, id) for img_url in img_urls)))

        else:
            return None

        return Instrument(
            id,
            family,
            group,
            instrument,
            url,
            author,
            date_of_production,
            place_of_production,
            images,
            img_urls,
        )

    def _download_image(self, url: str, instrument_id: str) -> str | None:
        """Downloads an image and saves it to a subfolder named after the instrument ID.

        Args:
            url (str): The URL of the image to download.
            instrument_id (str): The ID of the instrument, used as the subfolder name.

        Returns:
            str | None: The relative path (id/filename) of the downloaded image,
                or None if the download failed.
        """
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            file_name = url.rstrip("/").split("/")[-1]

            # Define and create the specific subfolder: root/name/img/id/
            subfolder_path = self.root / self.name / "images" / instrument_id
            subfolder_path.mkdir(parents=True, exist_ok=True)

            file_path = subfolder_path / file_name

            if not file_path.exists():
                with open(file_path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)

            # Return relative path for storage in the JSON metadata
            return f"images/{instrument_id}/{file_name}"
        else:
            return None

    def _get_starting_url(self, family_name: str, group_name: str, instrument_name: str) -> str | None:
        """Navigates the MIMO International instrument selection and returns the URL
        of the first page of the AJAX catalog for the specified instrument.

        Args:
            family_name (str): The exact title of the instrument family.
            group_name (str): The exact title of the instrument group.
            instrument_name (str): The exact title of the specific instrument.

        Returns:
            str: The URL of the first page of the instrument catalog, or None if not found.
        """
        driver = webdriver.Firefox()
        driver.get("https://mimo-international.com/MIMO/instrument-families.aspx")

        try:
            wait = WebDriverWait(driver, 10)

            # 1. Click on the Family
            family_list = wait.until(EC.presence_of_element_located((By.ID, "InstrumentTypeLevel2_exact")))
            family_element = None
            for li in family_list.find_elements(By.TAG_NAME, "li"):
                if li.get_attribute("title") == family_name:
                    family_element = li
                    break
            if not family_element:
                print(f"Family '{family_name}' not found.")
                return None
            family_element.click()

            # 2. Wait for and Click on the Group
            group_list = wait.until(EC.visibility_of_element_located((By.ID, "InstrumentTypeLevel3_exact")))
            group_element = None
            for li in group_list.find_elements(By.TAG_NAME, "li"):
                if li.get_attribute("title") == group_name:
                    group_element = li
                    break
            if not group_element:
                print(f"Group '{group_name}' not found under '{family_name}'.")
                return None
            group_element.click()

            # 3. Wait for and Click on the Instrument
            instrument_list = wait.until(EC.visibility_of_element_located((By.ID, "InstrumentTypeLevel4_exact")))
            instrument_element = None
            for li in instrument_list.find_elements(By.TAG_NAME, "li"):
                if li.get_attribute("title") == instrument_name:
                    instrument_element = li
                    break
            if not instrument_element:
                print(f"Instrument '{instrument_name}' not found under '{family_name}' -> '{group_name}'.")
                return None
            instrument_element.click()

            # 4. Wait for the catalog to load and get the URL
            wait.until(EC.presence_of_element_located((By.ID, "criteres_recherche")))

            return driver.current_url

        finally:
            driver.quit()
