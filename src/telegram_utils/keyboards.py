from telegram import InlineKeyboardButton, InlineKeyboardMarkup


def _make_rows(pairs):
    keyboard = []
    for row in pairs:
        keyboard.append([InlineKeyboardButton(text, callback_data=cb) for text, cb in row])
    return InlineKeyboardMarkup(keyboard)


def kb_first(ui_config, currencies):
    codes = ui_config["buttons"].get("currency_codes", currencies)
    pairs = []
    chunk_size = 3
    for i in range(0, len(codes), chunk_size):
        chunk = codes[i : i + chunk_size]
        pairs.append([(c, f"first:{c}") for c in chunk])
    return _make_rows(pairs)


def kb_second(ui_config, currencies, first):
    codes = ui_config["buttons"].get("currency_codes", currencies)
    buttons = [(c, f"second:{first}:{c}") for c in codes if c != first]
    pairs = []
    chunk_size = 3
    for i in range(0, len(buttons), chunk_size):
        pairs.append(buttons[i : i + chunk_size])
    pairs.append([(ui_config["buttons"]["back_label"], "back:first")])
    return _make_rows(pairs)


def kb_days(ui_config, first, second):
    rows = []
    row = []
    day_labels = ui_config["buttons"].get("days", [str(i) for i in range(1, 10)])
    for i, label in enumerate(day_labels, start=1):
        row.append((label, f"days:{first}:{second}:{i}"))
        if len(row) == 3:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([(ui_config["buttons"]["back_label"], f"back:second:{first}")])
    return _make_rows(rows)


def kb_confirm(ui_config, first, second, days):
    confirm_label = ui_config["buttons"].get("confirm_label", "Все верно")
    back_label = ui_config["buttons"]["back_label"]
    return _make_rows([[(confirm_label, f"confirm:{first}:{second}:{days}")], [(back_label, f"back:second:{first}")]])


def kb_for_state(ui_config, state):
    awaiting = bool(state.get("awaiting_question", False))
    asked = int(state.get("asked_count", 0))
    asked_at_pred = int(state.get("asked_count_at_prediction", 0)) if state.get("asked_count_at_prediction") is not None else asked
    show_toggle = (not awaiting) and (asked == asked_at_pred)
    media_format = state.get("media_format", "gif")
    
    toggle_label = ui_config["buttons"]["png"] if media_format == "gif" else ui_config["buttons"]["gif"]
    first = state.get("first")
    second = state.get("second")
    days = int(state.get("days", 1)) if state.get("days") is not None else 1

    rows = []
    if show_toggle:
        rows.append([(toggle_label, "toggle")])

    if first and second and (asked > asked_at_pred):
        back_cb = "back:restore"
    else:
        back_cb = "back:first"

    if first and second:
        ask_label = ui_config["buttons"]["ask_label_first"] if asked == 0 else ui_config["buttons"]["ask_label_more"]
        rows.append([(ui_config["buttons"]["back_label"], back_cb), (ask_label, f"ask:{first}:{second}:{days}")])
    else:
        rows.append([(ui_config["buttons"]["back_label"], "back:first")])

    return _make_rows(rows)


def markup_repr(reply_markup):
    if not reply_markup:
        return None
    try:
        rows = []
        for row in getattr(reply_markup, "inline_keyboard", []):
            cols = []
            for b in row:
                cols.append((getattr(b, "text", None), getattr(b, "callback_data", None)))
            rows.append(tuple(cols))
        return tuple(rows)
    except Exception:
        return str(reply_markup)