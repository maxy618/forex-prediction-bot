import os
import time
from telegram import InputMediaPhoto, InputMediaAnimation
from telegram.error import BadRequest
from logging_util import setup_logging
from telegram_utils.state import get_chat_lock, get_chat_state, save_chat_state, CHAT_STATE
from telegram_utils.keyboards import markup_repr

logger = setup_logging(name=__name__)


def is_protected_asset(path, protected_set):
    if isinstance(path, str):
        if "assets" in os.path.abspath(path).split(os.sep):
            return True
        if path in protected_set:
            return True
    return False


def is_generated_media(path, temp_folder):
    try:
        if not isinstance(path, str) or not temp_folder:
            return False
        return os.path.abspath(path).startswith(os.path.abspath(temp_folder))
    except Exception:
        return False


def _update_state_media(chat_id, media_path, is_gif, msg_id=None, caption=None, reply_markup=None, temp_folder=None, protected_set=None):
    with get_chat_lock(chat_id):
        s = get_chat_state(chat_id)
        s["last_media"] = media_path
        
        # Сохраняем путь к файлу или file_id
        is_file_on_disk = isinstance(media_path, str) and os.path.exists(media_path)
        is_protected = is_protected_asset(media_path, protected_set or set())
        
        if is_gif:
            s["last_media_gif"] = media_path
            if not is_file_on_disk and not is_protected:
                s["last_media_gif_file_id"] = media_path
        else:
            s["last_media_png"] = media_path
            if not is_file_on_disk and not is_protected:
                s["last_media_png_file_id"] = media_path
        
        if msg_id:
            s["msg_id"] = msg_id
        if caption is not None:
            s["last_caption"] = caption
        if reply_markup is not None:
            s["last_markup"] = markup_repr(reply_markup)
        CHAT_STATE[chat_id] = s


def edit_with_retries(action_callable, bot, chat_id, message_id, max_attempts=3, delay=1.0):
    lock = get_chat_lock(chat_id)
    with lock:
        for attempt in range(1, max_attempts + 1):
            try:
                action_callable()
                return True
            except BadRequest as be:
                last_exc = be
                msg = str(be)
                if "Message is not modified" in msg:
                    return True
                if "not found" in msg.lower():
                    break 
            except Exception as e:
                last_exc = e
            if attempt < max_attempts:
                time.sleep(delay)
    return False


def replace_with_logo(bot, chat_id, message_id, logo_path, caption=None, reply_markup=None):
    def try_edit_media():
        with open(logo_path, "rb") as f:
            media = InputMediaPhoto(f, caption=caption)
            bot.edit_message_media(media=media, chat_id=chat_id, message_id=message_id, reply_markup=reply_markup)
    
    success = edit_with_retries(try_edit_media, bot, chat_id, message_id)
    if success:
        save_chat_state(chat_id, message_id, True, last_caption=caption or "", last_markup=markup_repr(reply_markup))
    return success


def send_replace_media(bot, chat_id, message_id, media_path, is_gif, caption, reply_markup, protected_set, temp_folder):
    def _is_file_id(m):
        return isinstance(m, str) and (not os.path.exists(m)) and (not is_protected_asset(m, protected_set))
    
    media_obj = None
    file_handle = None

    try:
        if _is_file_id(media_path):
            media_obj = InputMediaAnimation(media=media_path, caption=caption) if is_gif else InputMediaPhoto(media=media_path, caption=caption)
        elif is_protected_asset(media_path, protected_set) or os.path.exists(media_path):
            file_handle = open(media_path, "rb")
            media_obj = InputMediaAnimation(file_handle, caption=caption) if is_gif else InputMediaPhoto(file_handle, caption=caption)
        
        if media_obj:
            try:
                bot.edit_message_media(media=media_obj, chat_id=chat_id, message_id=message_id, reply_markup=reply_markup)
                _update_state_media(chat_id, media_path, is_gif, caption=caption, reply_markup=reply_markup, temp_folder=temp_folder, protected_set=protected_set)
                return True
            except BadRequest as be:
                if "Message is not modified" in str(be):
                    return True
                raise be
    except Exception:
        try:
            bot.delete_message(chat_id=chat_id, message_id=message_id)
        except Exception:
            pass
        
        if file_handle:
            file_handle.close()
            file_handle = open(media_path, "rb")
        
        try:
            if is_gif:
                new_msg = bot.send_animation(chat_id=chat_id, animation=file_handle or media_path, caption=caption, reply_markup=reply_markup)
            else:
                new_msg = bot.send_photo(chat_id=chat_id, photo=file_handle or media_path, caption=caption, reply_markup=reply_markup)
            
            _update_state_media(chat_id, media_path, is_gif, msg_id=new_msg.message_id, caption=caption, reply_markup=reply_markup, temp_folder=temp_folder, protected_set=protected_set)
            return True
        except Exception as e:
            return False
    finally:
        if file_handle:
            file_handle.close()
    return False