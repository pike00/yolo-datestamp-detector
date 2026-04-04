import os
import logging
from apprise import Apprise

logger = logging.getLogger(__name__)


def send_telegram_notification(message: str, title: str = "🎯 Dedup Pipeline") -> bool:
    """
    Send Telegram notification via Apprise.

    Args:
        message: Message body
        title: Notification title

    Returns:
        True if sent successfully, False otherwise
    """
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        logger.warning("Telegram not configured (TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing)")
        return False

    try:
        apprise_obj = Apprise()
        apprise_obj.add(f"tgram://{bot_token}/{chat_id}")
        apprise_obj.notify(body=message, title=title)
        logger.info("✓ Telegram notification sent")
        return True
    except Exception as e:
        logger.error(f"Failed to send Telegram notification: {e}")
        return False
