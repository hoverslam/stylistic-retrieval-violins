from stylistic_retrieval.scraper import MIMOScraper
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A scraper for extracting instrument information from the MIMO (Musical Instrument Museums Online) website."
    )
    parser.add_argument(
        "--root",
        type=str,
        help="The root directory where the scraped data will be stored.",
    )
    parser.add_argument(
        "--name",
        type=str,
        help="The name of the instrument type being scraped. This will be used as a subdirectory name.",
    )
    parser.add_argument(
        "--family",
        type=str,
        help="The exact title of the instrument family to scrape.",
    )
    parser.add_argument(
        "--group",
        type=str,
        help="The exact title of the instrument group to scrape.",
    )
    parser.add_argument(
        "--instrument",
        type=str,
        help="The exact title of the specific instrument to scrape.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="The time interval (in seconds) between requests.",
    )
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download images associated with each instrument.",
    )
    args = parser.parse_args()
    mimo = MIMOScraper(args.root, args.name)
    mimo.start(
        family_name=args.family,
        group_name=args.group,
        instrument_name=args.instrument,
        request_interval=args.interval,
        download_images=args.download,
    )
