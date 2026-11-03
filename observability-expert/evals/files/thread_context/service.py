import logging
from concurrent.futures import ThreadPoolExecutor

from obslog.context import log_context, new_request_id

logger = logging.getLogger(__name__)
pool = ThreadPoolExecutor(max_workers=8)


def resize_image(path: str) -> None:
    logger.info("resizing %s", path)


def handle_upload(paths: list[str]) -> None:
    with log_context(request_id=new_request_id()):
        logger.info("upload received: %d files", len(paths))
        futures = [pool.submit(resize_image, p) for p in paths]
        for f in futures:
            f.result()
