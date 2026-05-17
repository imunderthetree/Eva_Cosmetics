from __future__ import annotations

import argparse
import csv
import html
import json
import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SELENIUM_IMPORT_ERROR: ModuleNotFoundError | None = None

try:
    from selenium import webdriver
    from selenium.common.exceptions import (
        ElementClickInterceptedException,
        ElementNotInteractableException,
        NoSuchElementException,
        StaleElementReferenceException,
        TimeoutException,
    )
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.common.by import By
    from selenium.webdriver.remote.webdriver import WebDriver
    from selenium.webdriver.remote.webelement import WebElement
    from selenium.webdriver.support.ui import WebDriverWait
except ModuleNotFoundError as exc:
    SELENIUM_IMPORT_ERROR = exc
    webdriver = None
    ChromeOptions = None
    By = None
    WebDriverWait = None

    class MissingSeleniumException(Exception):
        pass

    ElementClickInterceptedException = MissingSeleniumException
    ElementNotInteractableException = MissingSeleniumException
    NoSuchElementException = MissingSeleniumException
    StaleElementReferenceException = MissingSeleniumException
    TimeoutException = MissingSeleniumException


START_URL = "https://www.shop.eva-cosmetics.com/"
LEGACY_START_URL_HOSTS = {"eva-cosmetics.com", "www.eva-cosmetics.com"}
CSV_FIELDS = [
    "name of product",
    "price",
    "how many in stock",
    "description",
    "type",
    "features",
]
PRODUCT_LIMIT = 451
UNKNOWN_STOCK_VALUE = "unknown"

PRODUCT_CARD_SELECTOR = "div.product-card"
FILTER_CONTAINER_SELECTOR = ".accordion-details__content.parent-display.flex.flex-col.gap-3"
PROMOTION_LINK_SELECTOR = (
    ".promotion__item.promotion__item-image a[href], "
    ".promotionitem.promotion__item-image a[href]"
)

CARD_DATA_SCRIPT = """
const card = arguments[0];
const text = (node) => node ? (node.innerText || node.textContent || '').replace(/\\s+/g, ' ').trim() : '';
const link =
  card.querySelector('.product-card__title a[href]') ||
  card.querySelector('a[href*="/products/"]');
const sale = card.querySelector('.f-price__sale .f-price-item--sale');
const regular =
  card.querySelector('.f-price__regular .f-price-item--regular') ||
  card.querySelector('.f-price-item--regular');
const button = card.querySelector('.product-card__atc, button[name="add"]');
const soldOutBadge = card.querySelector('.f-badge--soldout');
const soldOutText = /sold\\s*out/i.test(text(card));
const disabledAdd = button ? button.disabled || button.hasAttribute('disabled') : false;
return {
  name: text(link) || (link ? link.getAttribute('aria-label') : ''),
  url: link ? link.href : '',
  price: text(sale) || text(regular),
  sold_out: Boolean(soldOutBadge || soldOutText || disabledAdd)
};
"""

PRODUCT_PAGE_DETAILS_SCRIPT = """
const normalize = (value) => (value || '').toString().replace(/\\s+/g, ' ').trim();
const text = (node) => normalize(node ? (node.innerText || node.textContent || '') : '');
const htmlToText = (value) => normalize((value || '').toString().replace(/<[^>]+>/g, ' '));
const firstText = (selectors) => {
    for (const selector of selectors) {
        const node = document.querySelector(selector);
        const value = text(node);
        if (value) return value;
    }
    return '';
};
const parseJsonFromScripts = (selectors) => {
    for (const selector of selectors) {
        const node = document.querySelector(selector);
        if (!node || !node.textContent) continue;
        try {
            return JSON.parse(node.textContent);
        } catch (_) {
            continue;
        }
    }
    return null;
};
const parseLdProduct = () => {
    const nodes = Array.from(document.querySelectorAll('script[type="application/ld+json"]'));
    for (const node of nodes) {
        if (!node.textContent) continue;
        try {
            const data = JSON.parse(node.textContent);
            const items = Array.isArray(data) ? data : [data];
            for (const item of items) {
                if (item && item['@type'] === 'Product') return item;
            }
        } catch (_) {
            continue;
        }
    }
    return null;
};

let description = firstText([
    '.product__description',
    '.product__description .rte',
    '.product__description.rte',
    '.product__info .rte',
    '.product__info [class*="description"]',
    '[data-product-description]',
    '.product-info__description',
    '.product__accordion .rte'
]);

if (!description) {
    const productJson = parseJsonFromScripts([
        'script[type="application/json"][data-product-json]',
        'script[type="application/json"][id^="ProductJson"]',
        'script[type="application/json"][id*="product-json"]'
    ]);
    if (productJson && productJson.body_html) {
        description = htmlToText(productJson.body_html);
    }
}

if (!description) {
    const meta = document.querySelector('meta[property="og:description"], meta[name="description"]');
    if (meta) {
        description = normalize(meta.getAttribute('content'));
    }
}

const productLd = parseLdProduct();
if (!description && productLd && productLd.description) {
    description = normalize(productLd.description);
}

let soldOut = false;
const addButton = document.querySelector(
    'button[name="add"], form[action*="/cart"] button[type="submit"], .product-form__submit'
);
const addText = text(addButton);
if (/sold\\s*out|out\\s*of\\s*stock/i.test(addText)) {
    soldOut = true;
}

const inventoryText = firstText([
    '.product__inventory-text',
    '.product__inventory'
]);
const inventoryNumberMatch = inventoryText.match(/(\\d[\\d,]*)/);
const inventoryNumber = inventoryNumberMatch ? inventoryNumberMatch[1].replace(/,/g, '') : '';
let inventoryCount = null;
if (inventoryNumber) {
    const parsed = parseInt(inventoryNumber, 10);
    if (!Number.isNaN(parsed)) {
        inventoryCount = parsed;
    }
}
if (/sold\\s*out|out\\s*of\\s*stock/i.test(inventoryText)) {
    soldOut = true;
}
if (inventoryCount !== null) {
    soldOut = inventoryCount <= 0;
} else if (
    inventoryText
    && /in\\s*stock|left/i.test(inventoryText)
    && !/sold\\s*out|out\\s*of\\s*stock/i.test(inventoryText)
) {
    soldOut = false;
}

const stockBadge = firstText([
    '.product__inventory',
    '.product__stock',
    '.product__availability',
    '.product__badge',
    '.badge--soldout',
    '.f-badge--soldout',
    '[data-product-stock]'
]);
if (/sold\\s*out|out\\s*of\\s*stock/i.test(stockBadge)) {
    soldOut = true;
}

let availabilityText = '';
if (productLd && productLd.offers) {
    const offers = Array.isArray(productLd.offers) ? productLd.offers : [productLd.offers];
    availabilityText = offers
        .map((offer) => offer && offer.availability)
        .filter(Boolean)
        .join(' ');
}
if (/OutOfStock/i.test(availabilityText)) {
    soldOut = true;
}
if (/InStock/i.test(availabilityText) && !/OutOfStock/i.test(availabilityText)) {
    soldOut = false;
}

let stock = '';
if (inventoryNumber) {
    stock = inventoryNumber;
} else if (soldOut) {
    stock = '0';
} else if (inventoryText && /in\\s*stock|left/i.test(inventoryText)) {
    stock = 'in stock';
} else if (addButton || availabilityText) {
    stock = 'in stock';
}

return { description, stock, sold_out: soldOut };
"""

FILTER_OPTIONS_SCRIPT = """
const text = (node) => node ? (node.innerText || node.textContent || '').replace(/\\s+/g, ' ').trim() : '';
function labelFor(input) {
  if (!input.id) return null;
  return document.querySelector('label[for="' + input.id.replace(/"/g, '\\\\"') + '"]');
}
function cleanLabel(label) {
  if (!label) return '';
  const clone = label.cloneNode(true);
  clone.querySelectorAll('.text-sm, .text-subtext, .facets__count').forEach((node) => node.remove());
  return text(clone).replace(/\\s*\\(\\d+\\)\\s*$/, '').trim();
}
let details = Array.from(document.querySelectorAll('details[data-index*="filter.p.tag"]'));
let preferred = details.filter((detail) => {
  const index = detail.getAttribute('data-index') || '';
  return index.startsWith('vertical-') && !detail.closest('.drawer');
});
if (preferred.length === 0) {
  preferred = details.filter((detail) => !detail.closest('.drawer'));
}
if (preferred.length === 0) {
  preferred = details;
}
const seen = new Set();
const options = [];
for (const detail of preferred) {
  detail.open = true;
  for (const input of detail.querySelectorAll('input[name="filter.p.tag"]')) {
    const label = labelFor(input);
    const labelText = cleanLabel(label) || input.value || input.id;
    const value = input.value || labelText;
    const key = value.toLowerCase();
    if (!labelText || seen.has(key) || input.disabled) continue;
    seen.add(key);
    options.push({ id: input.id, label: labelText, value });
  }
}
return options;
"""


@dataclass
class ProductRecord:
    name: str
    price: str = ""
    stock: str = ""
    description: str = ""
    product_type: str = ""
    features: set[str] = field(default_factory=set)
    url: str = ""


class ProductStore:
    def __init__(self, output_path: Path, fresh: bool = False) -> None:
        self.output_path = output_path
        self.records: OrderedDict[str, ProductRecord] = OrderedDict()
        self.name_index: dict[str, str] = {}
        self.url_index: dict[str, str] = {}

        if fresh:
            self.save()
            return

        if output_path.exists() and output_path.stat().st_size > 0:
            self._load()

    def _load(self) -> None:
        with self.output_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                name = clean_text(row.get("name of product", ""))
                if not name:
                    continue
                key = f"name:{normalize_name(name)}"
                record = ProductRecord(
                    name=name,
                    price=clean_text(row.get("price", "")),
                    stock=clean_text(row.get("how many in stock", "")),
                    description=clean_text(row.get("description", "")),
                    product_type=clean_text(row.get("type", "")),
                    features=parse_features(row.get("features", "")),
                )
                self.records[key] = record
                self.name_index[normalize_name(name)] = key

    def count(self) -> int:
        return len(self.records)

    def upsert(self, product: dict[str, Any], features: list[str] | None = None) -> bool:
        name = clean_text(product.get("name", ""))
        if not name:
            return False

        url = clean_text(product.get("url", ""))
        key = self._find_key(name, url) or self._new_key(name, url)
        is_new = key not in self.records

        if is_new:
            self.records[key] = ProductRecord(name=name, url=url)

        record = self.records[key]
        record.name = record.name or name
        record.url = record.url or url
        record.price = clean_text(product.get("price", "")) or record.price
        record.stock = merge_stock(record.stock, product)
        description = clean_text(product.get("description", ""))
        if description:
            record.description = description
        record.product_type = record.product_type or infer_product_type(record.name)

        for feature in features or []:
            feature = clean_feature(feature)
            if feature:
                record.features.add(feature)

        self._index_record(key, record)
        return is_new

    def add_feature_to_existing(self, product: dict[str, Any], feature: str) -> bool:
        name = clean_text(product.get("name", ""))
        url = clean_text(product.get("url", ""))
        key = self._find_key(name, url)
        feature = clean_feature(feature)
        if not key or not feature:
            return False
        self.records[key].features.add(feature)
        return True

    def save(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for record in self.records.values():
                writer.writerow(
                    {
                        "name of product": record.name,
                        "price": record.price,
                        "how many in stock": record.stock,
                        "description": record.description,
                        "type": record.product_type,
                        "features": "; ".join(sorted(record.features, key=str.casefold)),
                    }
                )

    def _find_key(self, name: str, url: str) -> str | None:
        url_key = normalize_product_url(url)
        if url_key and url_key in self.url_index:
            return self.url_index[url_key]

        name_key = normalize_name(name)
        if name_key and name_key in self.name_index:
            return self.name_index[name_key]

        return None

    def _new_key(self, name: str, url: str) -> str:
        url_key = normalize_product_url(url)
        if url_key:
            return f"url:{url_key}"
        return f"name:{normalize_name(name)}"

    def _index_record(self, key: str, record: ProductRecord) -> None:
        name_key = normalize_name(record.name)
        if name_key:
            self.name_index[name_key] = key

        url_key = normalize_product_url(record.url)
        if url_key:
            self.url_index[url_key] = key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape Eva Cosmetics collection products into a CSV file."
    )
    parser.add_argument("--start-url", default=START_URL)
    parser.add_argument("--output", default="eva_products.csv")
    parser.add_argument("--limit", type=int, default=PRODUCT_LIMIT)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--delay", type=float, default=0.6)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="Overwrite the CSV before scraping.")
    parser.add_argument("--skip-line-pass", action="store_true")
    return parser.parse_args()


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def clean_feature(value: Any) -> str:
    text = clean_text(value)
    text = re.sub(r"\s*\(\d+\)\s*$", "", text).strip()
    return text


def parse_features(value: str | None) -> set[str]:
    if not value:
        return set()
    return {clean_feature(part) for part in value.split(";") if clean_feature(part)}


def normalize_name(value: str) -> str:
    value = clean_text(value).casefold()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_product_url(url: str) -> str:
    if not url:
        return ""
    path = urlparse(url).path.rstrip("/")
    match = re.search(r"/products/([^/]+)$", path)
    return match.group(1).casefold() if match else ""


def infer_product_type(name: str) -> str:
    text = clean_text(name).casefold()
    rules = [
        ("shaving", ["after shave", "shave", "shaving", "razor"]),
        ("body splash", ["body splash", "body fragrance", "body mist", "fragrance", "splash"]),
        ("gel", [" gel", "gel ", "gel-"]),
        ("oil", [" oil", "oil ", "oil-"]),
        ("cream", ["cream", "lotion", "balsam", "moisturizer", "moisturising", "moisturizing"]),
    ]
    padded = f" {text} "
    for product_type, keywords in rules:
        if any(keyword in padded for keyword in keywords):
            return product_type
    return ""


def merge_stock(current: str, product: dict[str, Any]) -> str:
    detailed = clean_text(product.get("stock", ""))
    if detailed:
        return detailed
    if product.get("sold_out"):
        return "0"
    if current and current != UNKNOWN_STOCK_VALUE:
        return current
    return UNKNOWN_STOCK_VALUE


def is_numeric_stock(value: str) -> bool:
    return bool(re.fullmatch(r"\d+", clean_text(value)))


def html_to_text(value: str) -> str:
    if not value:
        return ""
    stripped = re.sub(r"<[^>]+>", " ", value)
    return clean_text(html.unescape(stripped))


def product_json_url(product_url: str) -> str:
    url = clean_text(product_url)
    if not url:
        return ""
    parsed = urlparse(url)
    if "/products/" not in parsed.path:
        return ""
    path = parsed.path.rstrip("/")
    if not path.endswith(".json"):
        path = f"{path}.json"
    return urlunparse(parsed._replace(path=path, query=""))


def extract_stock_from_product(product: dict[str, Any]) -> str:
    variants = product.get("variants") or []
    quantities = [
        variant.get("inventory_quantity")
        for variant in variants
        if isinstance(variant, dict) and isinstance(variant.get("inventory_quantity"), int)
    ]
    if quantities:
        return str(sum(quantities))
    available = any(
        isinstance(variant, dict) and variant.get("available") is True for variant in variants
    )
    if available:
        return "in stock"
    if variants:
        return "0"
    return UNKNOWN_STOCK_VALUE


def fetch_product_details(product_url: str, timeout: int) -> dict[str, str]:
    json_url = product_json_url(product_url)
    if not json_url:
        return {}
    request = Request(json_url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except (HTTPError, URLError, ValueError) as exc:
        logging.debug("Failed to fetch product JSON: %s (%s)", json_url, exc)
        return {}

    try:
        data = json.loads(payload.decode("utf-8", "ignore"))
    except json.JSONDecodeError:
        logging.debug("Failed to parse product JSON: %s", json_url)
        return {}

    product = data.get("product") or {}
    return {
        "description": html_to_text(product.get("body_html", "")),
        "stock": extract_stock_from_product(product),
    }


def scrape_product_page_details(
    driver: WebDriver, product_url: str, timeout: int
) -> dict[str, str]:
    url = clean_text(product_url)
    if not url:
        return {}

    try:
        driver.get(url)
        wait_for_ready(driver, timeout)
        details = driver.execute_script(PRODUCT_PAGE_DETAILS_SCRIPT) or {}
    except Exception as exc:
        logging.debug("Failed to read product page: %s (%s)", url, exc)
        return {}

    description = clean_text(details.get("description", ""))
    stock = clean_text(details.get("stock", ""))
    sold_out = bool(details.get("sold_out"))
    result: dict[str, str] = {}
    if description:
        result["description"] = description
    if stock:
        result["stock"] = stock
    elif sold_out:
        result["stock"] = "0"
    return result


def build_driver(headless: bool) -> WebDriver:
    if SELENIUM_IMPORT_ERROR is not None:
        raise SystemExit(
            "Selenium is not installed. Run `python -m pip install -r requirements.txt` "
            "from this folder, then run the scraper again."
        )

    options = ChromeOptions()
    options.add_argument("--window-size=1440,1200")
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-dev-shm-usage")
    if headless:
        options.add_argument("--headless=new")
    return webdriver.Chrome(options=options)


def wait_for_ready(driver: WebDriver, timeout: int) -> None:
    WebDriverWait(driver, timeout).until(
        lambda browser: browser.execute_script("return document.readyState") == "complete"
    )
    time.sleep(0.25)


def scroll_to_load_products(driver: WebDriver, delay: float) -> None:
    last_height = 0
    for _ in range(4):
        height = driver.execute_script("return document.body.scrollHeight || 0")
        if height == last_height:
            break
        last_height = height
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(delay)
    driver.execute_script("window.scrollTo(0, 0);")


def safe_click(driver: WebDriver, element: WebElement) -> None:
    driver.execute_script(
        "arguments[0].scrollIntoView({block: 'center', inline: 'center'});",
        element,
    )
    time.sleep(0.1)
    try:
        element.click()
    except (ElementClickInterceptedException, ElementNotInteractableException):
        driver.execute_script("arguments[0].click();", element)


def fallback_shop_url(start_url: str) -> str:
    parsed = urlparse(start_url)
    if (parsed.hostname or "").casefold() in LEGACY_START_URL_HOSTS:
        return START_URL
    return ""


def collect_collection_links(driver: WebDriver, start_url: str, timeout: int) -> list[tuple[str, str]]:
    logging.info("Opening homepage: %s", start_url)
    driver.get(start_url)
    wait_for_ready(driver, timeout)

    links: OrderedDict[str, str] = OrderedDict()
    selectors = [
        "nav.header__menu ul.flex.flex-wrap a[href]",
        "ul.flex.flex-wrap a[href]",
    ]
    for selector in selectors:
        for element in driver.find_elements(By.CSS_SELECTOR, selector):
            href = clean_text(element.get_attribute("href"))
            label = clean_text(element.text) or clean_text(element.get_attribute("aria-label"))
            if "/collections/" in href:
                links[href] = label or href.rsplit("/", 1)[-1].replace("-", " ").title()

    for element in driver.find_elements(By.CSS_SELECTOR, "ul.flex.flex-wrap summary[data-link]"):
        data_link = clean_text(element.get_attribute("data-link"))
        if "/collections/" not in data_link:
            continue
        href = urljoin(driver.current_url, data_link)
        label = clean_text(element.text) or href.rsplit("/", 1)[-1].replace("-", " ").title()
        links[href] = label

    if not links:
        fallback_url = fallback_shop_url(start_url)
        if fallback_url:
            logging.warning(
                "No collection links found at %s; retrying shop homepage: %s",
                start_url,
                fallback_url,
            )
            return collect_collection_links(driver, fallback_url, timeout)

    return list(links.items())


def collect_filter_options(driver: WebDriver) -> list[dict[str, str]]:
    options = driver.execute_script(FILTER_OPTIONS_SCRIPT)
    return [
        {
            "id": clean_text(option.get("id", "")),
            "label": clean_feature(option.get("label", "")),
            "value": clean_text(option.get("value", "")),
        }
        for option in options
        if clean_feature(option.get("label", ""))
    ]


def apply_filter(driver: WebDriver, collection_url: str, option: dict[str, str], timeout: int) -> None:
    driver.get(collection_url)
    wait_for_ready(driver, timeout)
    driver.execute_script(
        """
        const input = document.getElementById(arguments[0]);
        if (input) {
          const details = input.closest('details');
          if (details) details.open = true;
        }
        """,
        option["id"],
    )

    old_url = driver.current_url
    try:
        label = driver.find_element(By.CSS_SELECTOR, f'label[for="{option["id"]}"]')
        safe_click(driver, label)
        wait_for_filter_change(driver, old_url, timeout)
    except (NoSuchElementException, TimeoutException):
        fallback_url = build_filter_url(collection_url, option["value"])
        logging.debug("Falling back to filter URL: %s", fallback_url)
        driver.get(fallback_url)
        wait_for_ready(driver, timeout)


def wait_for_filter_change(driver: WebDriver, old_url: str, timeout: int) -> None:
    try:
        WebDriverWait(driver, timeout).until(
            lambda browser: browser.current_url != old_url
            or len(browser.find_elements(By.CSS_SELECTOR, ".facets-active a, .facets-active button")) > 0
        )
    finally:
        time.sleep(0.8)


def build_filter_url(collection_url: str, value: str) -> str:
    parsed = urlparse(collection_url)
    query = [
        (key, val)
        for key, val in parse_qsl(parsed.query, keep_blank_values=True)
        if key not in {"filter.p.tag", "page"}
    ]
    query.append(("filter.p.tag", value))
    return urlunparse(parsed._replace(query=urlencode(query)))


def extract_product_cards(driver: WebDriver, delay: float) -> list[dict[str, Any]]:
    scroll_to_load_products(driver, delay)
    products: list[dict[str, Any]] = []
    for card in driver.find_elements(By.CSS_SELECTOR, PRODUCT_CARD_SELECTOR):
        try:
            product = driver.execute_script(CARD_DATA_SCRIPT, card)
        except StaleElementReferenceException:
            continue
        product["name"] = clean_text(product.get("name", ""))
        product["price"] = clean_text(product.get("price", ""))
        product["url"] = clean_text(product.get("url", ""))
        if product["name"]:
            products.append(product)
    return products


def next_page_url(driver: WebDriver) -> str:
    return clean_text(
        driver.execute_script(
            """
            const relNext = document.querySelector('a[rel="next"][href]');
            if (relNext) return relNext.href;

            const current = Number(new URL(location.href).searchParams.get('page') || '1');
            const candidates = Array.from(document.querySelectorAll('a[href]'))
              .map((link) => {
                try {
                  const url = new URL(link.href);
                  const page = Number(url.searchParams.get('page') || '0');
                  return { href: link.href, page };
                } catch (_) {
                  return null;
                }
              })
              .filter((item) => item && item.page > current)
              .sort((a, b) => a.page - b.page);
            return candidates.length ? candidates[0].href : '';
            """
        )
    )


def scrape_products_from_current_collection(
    driver: WebDriver,
    store: ProductStore,
    feature: str,
    limit: int,
    timeout: int,
    delay: float,
) -> None:
    visited_pages: set[str] = set()

    while store.count() < limit:
        wait_for_ready(driver, timeout)
        current_url = driver.current_url
        if current_url in visited_pages:
            break
        visited_pages.add(current_url)

        products = extract_product_cards(driver, delay)
        next_url = next_page_url(driver)
        new_count = 0
        for product in products:
            if store.count() >= limit and not store._find_key(product["name"], product["url"]):
                break
            if store.upsert(product, [feature]):
                new_count += 1

            key = store._find_key(product["name"], product["url"])
            record = store.records.get(key) if key else None
            needs_description = bool(record and not record.description)
            needs_stock = bool(
                record
                and (
                    not record.stock
                    or record.stock == UNKNOWN_STOCK_VALUE
                    or not is_numeric_stock(record.stock)
                )
            )
            if (needs_description or needs_stock) and product.get("url"):
                details = scrape_product_page_details(driver, product["url"], timeout)
                details_description = clean_text(details.get("description", ""))
                details_stock = clean_text(details.get("stock", ""))
                stock_is_precise = is_numeric_stock(details_stock)
                needs_fallback = (
                    (needs_description and not details_description)
                    or (needs_stock and not stock_is_precise)
                )
                if needs_fallback:
                    fallback = fetch_product_details(product["url"], timeout)
                    fallback_description = clean_text(fallback.get("description", ""))
                    fallback_stock = clean_text(fallback.get("stock", ""))
                    if fallback_description and not details_description:
                        details_description = fallback_description
                    if fallback_stock:
                        if not details_stock:
                            details_stock = fallback_stock
                        elif not stock_is_precise and is_numeric_stock(fallback_stock):
                            details_stock = fallback_stock
                if details_description:
                    record.description = details_description
                if details_stock:
                    record.stock = details_stock
                if details_description or details_stock:
                    time.sleep(delay)
        store.save()

        logging.info(
            "Feature '%s': scraped %s cards, %s new, %s total",
            feature,
            len(products),
            new_count,
            store.count(),
        )

        if not next_url:
            break
        driver.get(next_url)


def scrape_navigation_collections(
    driver: WebDriver,
    store: ProductStore,
    collection_links: list[tuple[str, str]],
    args: argparse.Namespace,
) -> None:
    for index, (collection_url, collection_label) in enumerate(collection_links, start=1):
        if store.count() >= args.limit:
            break

        logging.info("[%s/%s] Collection: %s", index, len(collection_links), collection_label)
        driver.get(collection_url)
        wait_for_ready(driver, args.timeout)
        options = collect_filter_options(driver)

        if not options:
            feature = clean_feature(collection_label)
            scrape_products_from_current_collection(
                driver, store, feature, args.limit, args.timeout, args.delay
            )
            continue

        logging.info("Found %s tag filters in %s", len(options), collection_label)
        for option in options:
            if store.count() >= args.limit:
                break
            feature = option["label"]
            logging.info("Applying filter: %s", feature)
            apply_filter(driver, collection_url, option, args.timeout)
            scrape_products_from_current_collection(
                driver, store, feature, args.limit, args.timeout, args.delay
            )


def collect_promotion_links(driver: WebDriver, start_url: str, timeout: int) -> list[tuple[str, str]]:
    logging.info("Returning to homepage for promotion line links")
    driver.get(start_url)
    wait_for_ready(driver, timeout)

    links: OrderedDict[str, str] = OrderedDict()
    for element in driver.find_elements(By.CSS_SELECTOR, PROMOTION_LINK_SELECTOR):
        href = clean_text(element.get_attribute("href"))
        label = clean_text(element.get_attribute("aria-label")) or clean_text(element.text)
        if href and "/collections/" in href:
            links[href] = label or href.rsplit("/", 1)[-1].replace("-", " ").title()

    if not links:
        fallback_url = fallback_shop_url(start_url)
        if fallback_url:
            logging.warning(
                "No promotion line links found at %s; retrying shop homepage: %s",
                start_url,
                fallback_url,
            )
            return collect_promotion_links(driver, fallback_url, timeout)

    return list(links.items())


def collection_title(driver: WebDriver, fallback_url: str) -> str:
    title = clean_text(
        driver.execute_script(
            """
            const selectors = [
              'main h1',
              '.collection-hero__title',
              '.collection__title',
              '.section__heading',
              'h1'
            ];
            for (const selector of selectors) {
              const node = document.querySelector(selector);
              const text = node ? (node.innerText || node.textContent || '').replace(/\\s+/g, ' ').trim() : '';
              if (text) return text;
            }
            const meta = document.querySelector('meta[property="og:title"], meta[name="twitter:title"]');
            return meta ? meta.getAttribute('content') : '';
            """
        )
    )
    if title:
        return title
    return fallback_url.rsplit("/", 1)[-1].replace("-", " ").title()


def update_line_features(
    driver: WebDriver,
    store: ProductStore,
    line_links: list[tuple[str, str]],
    args: argparse.Namespace,
) -> None:
    for index, (line_url, fallback_label) in enumerate(line_links, start=1):
        logging.info("[%s/%s] Line collection: %s", index, len(line_links), fallback_label)
        driver.get(line_url)
        wait_for_ready(driver, args.timeout)
        line_name = clean_feature(collection_title(driver, line_url)) or clean_feature(fallback_label)

        visited_pages: set[str] = set()
        updated = 0
        while True:
            wait_for_ready(driver, args.timeout)
            current_url = driver.current_url
            if current_url in visited_pages:
                break
            visited_pages.add(current_url)

            for product in extract_product_cards(driver, args.delay):
                if store.add_feature_to_existing(product, line_name):
                    updated += 1

            next_url = next_page_url(driver)
            if not next_url:
                break
            driver.get(next_url)

        store.save()
        logging.info("Line '%s': updated %s matching product rows", line_name, updated)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    store = ProductStore(Path(args.output), fresh=args.fresh)
    driver = build_driver(args.headless)

    try:
        collection_links = collect_collection_links(driver, args.start_url, args.timeout)
        logging.info("Found %s navigation collection links", len(collection_links))
        scrape_navigation_collections(driver, store, collection_links, args)

        if store.count() < args.limit:
            logging.warning(
                "Only %s unique products were found before the line pass; target was %s",
                store.count(),
                args.limit,
            )

        if not args.skip_line_pass:
            line_links = collect_promotion_links(driver, args.start_url, args.timeout)
            logging.info("Found %s promotion line links", len(line_links))
            update_line_features(driver, store, line_links, args)

    finally:
        store.save()
        driver.quit()

    logging.info("Done. Wrote %s products to %s", store.count(), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
