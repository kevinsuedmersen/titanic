import os
import logging
import string

logger = logging.getLogger(__name__)

def set_root_logger():
    """Configures the root logger
    """
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    log_formatter = logging.Formatter('%(asctime)s; %(name)s; %(levelname)s; %(message)s')
    console_handler.setFormatter(log_formatter)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    logger.info('Root logger is set up')

def make_sure_dir_exists(file_path: str):
    """Makes sure the dir of file_path exists

    :param file_path: Path to some file
    :type file_path: str
    """
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logger.info(f'Created directory ``{dir_path}``')


def clean_ticket(ticket: str, debug: bool=False):
    """Cleans a ticket entry

    :param ticket: Ticket
    :type ticket: str
    :param debug: Whether or not to print ticket to console, defaults to False
    :type debug: bool, optional
    :return: Ticket
    :rtype: str
    """
    for char in string.punctuation:
        if char in ticket:
            ticket = ticket.replace(char, '')
    ticket = ticket.lower()
    if debug:
        print(ticket)
    return ticket


def get_ticket_prefix(ticket: str, sep: str=' ', debug: bool=False):
    """Gets the ticket prefix

    :param ticket: Ticket
    :type ticket: str
    :param sep: Prefix separator, defaults to ' '
    :type sep: str, optional
    :param debug: Whether or not to print ticket prefix to console, defaults to False
    :type debug: bool, optional
    :return: Ticket prefix
    :rtype: str
    """
    lead = ticket.split(sep)[0]
    if debug:
        print(lead)
    if lead.isalpha():
        return lead
    else:
        return 'no_prefix'


def get_ticket_number(ticket: str, sep: str=' ', debug: bool=False):
    """Gets the ticket number

    :param ticket: Ticket
    :type ticket: str
    :param sep: Separator between ticket and number, defaults to ' '
    :type sep: str, optional
    :param debug: Whether or not to print ticket number to console, defaults to False
    :type debug: bool, optional
    :return: Ticket number
    :rtype: int
    """
    ticket_number = ticket.split(sep)[-1]
    if debug:
        print(ticket_number)
    return int(ticket_number)


def get_ticket_number_digit_len(ticket_number: int, debug: bool=False):
    """Gets the number of digits in a ticket number

    :param ticket_number: Ticket number
    :type ticket_number: int
    :param debug: Whether or not to print the intermediate output to console, defaults to False
    :type debug: bool, optional
    :return: Number of digits in ticket number
    :rtype: int
    """
    ticket_no_digit_len = len(str(ticket_number))
    if debug:
        print(ticket_no_digit_len)
    return ticket_no_digit_len


def get_leading_ticket_number_digit(ticket_number: int, debug: bool=False):
    """Gets the leading number from the ticket number

    :param ticket_number: Ticket number
    :type ticket_number: int
    :param debug: Whether or not to print the intermediate output to console, defaults to False
    :type debug: bool, optional
    :return: Leading digit in the ticket number
    :rtype: int
    """
    leading_ticket_no_digit = int(str(ticket_number)[0])
    if debug:
        print(leading_ticket_no_digit)
    return leading_ticket_no_digit


def get_ticket_group(ticket_number: int, debug: bool=False):
    """Gets all but the last digit of the ticket number

    :param ticket_number: Ticket number
    :type ticket_number: int
    :param debug: Whether or not to print the output to console, defaults to False
    :type debug: bool, optional
    :return: All but the last digits of the ticke number
    :rtype: int
    """
    ticket_group = ticket_number//10
    if debug:
        print(ticket_group)
    return ticket_group

