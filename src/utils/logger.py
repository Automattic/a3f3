# pylint: disable=invalid-name

import logging.handlers
import os
import time
import traceback
from functools import wraps

from bottle import request, HTTPError

logger = logging.getLogger('a3f3')

# set up the logger From http://stackoverflow.com/a/31093434/17523
logger.setLevel(logging.DEBUG)
LOG_DIRECTORIES = ['/home/a3f3/log/', './']
FILE_HANDLER = None

for ld in LOG_DIRECTORIES:
    ld = os.path.abspath(ld)
    if os.path.exists(ld):
        FILE_HANDLER = logging.handlers.RotatingFileHandler(
            os.path.join(ld, 'a3f3.log'), maxBytes=1e7, backupCount=10)
        break
assert FILE_HANDLER is not None
FORMATTER = logging.Formatter('%(asctime)15s %(name)s %(levelname)s %(message)s')
FILE_HANDLER.setLevel(logging.DEBUG)
FILE_HANDLER.setFormatter(FORMATTER)
logger.addHandler(FILE_HANDLER)


def log_to_logger(function):
    """Wrap a Bottle request so that a log line is emitted after it's handled.

    This decorator can be extended to take the desired logger as a param.
    """

    @wraps(function)
    def _log_to_logger(*args, **kwargs):
        time_start = time.time()
        try:
            actual_response = function(*args, **kwargs)
            if hasattr(actual_response, 'status'):
                response_status = actual_response.status
            else:
                response_status = 200
        except Exception as exception:  # pylint: disable=broad-except
            stacktrace = traceback.format_exc()
            for line in stacktrace.splitlines():
                logger.error(line)
            actual_response = HTTPError(500, 'Internal Server Error', exception, stacktrace)
            response_status = 500
        time_end = time.time()
        msg = '%7.3f %s %s %s.' % (time_end - time_start, request.method, request.url, response_status)
        logger.info(msg)
        return actual_response

    return _log_to_logger
