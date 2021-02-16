import os
import logging
import string

logger = logging.getLogger(__name__)

def set_root_logger():
    """Configures root logger. This function may only be called once to avoid duplicate logging output

    :return: None
    """
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    log_formatter = logging.Formatter('%(asctime)s; %(name)s; %(levelname)s; %(message)s')
    console_handler.setFormatter(log_formatter)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    logger.info('Root logger is set up')

def make_sure_dir_exists(file_path):
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logger.info(f'Created directory ``{dir_path}``')


def clean_ticket(ticket, debug=False):
    for char in string.punctuation:
        if char in ticket:
            ticket = ticket.replace(char, '')
    ticket = ticket.lower()
    if debug:
        print(ticket)
    return ticket


def get_ticket_prefix(ticket, sep=' ', debug=False):
    lead = ticket.split(sep)[0]
    if debug:
        print(lead)
    if lead.isalpha():
        return lead
    else:
        return 'no_prefix'


def get_ticket_number(ticket, sep=' ', debug=False):
    ticket_number = ticket.split(sep)[-1]
    if debug:
        print(ticket_number)
    return int(ticket_number)


def get_ticket_number_digit_len(ticket_number, debug=False):
    ticket_no_digit_len = len(str(ticket_number))
    if debug:
        print(ticket_no_digit_len)
    return ticket_no_digit_len


def get_leading_ticket_number_digit(ticket_number, debug=False):
    leading_ticket_no_digit = int(str(ticket_number)[0])
    if debug:
        print(leading_ticket_no_digit)
    return leading_ticket_no_digit


def get_ticket_group(ticket_number, debug=False):
    ticket_group = ticket_number//10
    if debug:
        print(ticket_group)
    return ticket_group

