from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ThreadedExecutor:
    """Wrapper around ThreadPoolExecutor for safe, resumable batch operations."""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers

    def execute_batch(
        self,
        items: List[Any],
        fn: Callable[[Any], Any],
        task_name: str = "Batch",
        skip_fn: Optional[Callable[[Any], bool]] = None,
    ) -> List[Any]:
        """
        Execute function on list of items with multiple threads.

        Args:
            items: List of items to process
            fn: Function to apply to each item
            task_name: Name for logging
            skip_fn: Optional function to check if item should be skipped (resume logic)

        Returns:
            List of results (in order)
        """
        results = []
        to_process = [item for item in items if not (skip_fn and skip_fn(item))]

        if not to_process:
            logger.info(f"{task_name}: No items to process (all skipped)")
            return results

        logger.info(f"{task_name}: Processing {len(to_process)}/{len(items)} items with {self.max_workers} threads")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(fn, item): item for item in to_process}

            completed = 0
            total = len(to_process)
            log_interval = max(1, total // 100)  # Log every ~1%
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    if completed % log_interval == 0 or completed == total:
                        pct = completed * 100 // total
                        logger.info(f"{task_name}: {completed}/{total} ({pct}%)")
                except Exception as e:
                    item = futures[future]
                    logger.error(f"{task_name}: Error processing {item}: {e}")

        logger.info(f"{task_name}: Complete. {completed} items processed")
        return results
