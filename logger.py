import pandas as pd
"""
LOGGER
"""
LOG_QUEUE = []
LOGGING = True
def log(msg=None,start=True,end_prev=True,new_level=False,end_level=False):
    if not LOGGING: return
    tag = ''
    now = pd.to_datetime('now') - pd.Timedelta(5, unit='h')
    if end_level: end_log_level()
    if end_prev and len(LOG_QUEUE) > 0:
        level = LOG_QUEUE.pop()
        if len(level) > 0:
            start_dt = level.pop()
            tag = '<<'*(len(LOG_QUEUE)) + ' --- '
            __print_entry(now,tag,'({}s)'.format((now-start_dt).total_seconds()))
        LOG_QUEUE.append(level)
    if msg is None: return
    if start:
        if len(LOG_QUEUE) == 0: LOG_QUEUE.append([])
        tag = '>>'*(len(LOG_QUEUE)-1) + ' '
        LOG_QUEUE[-1].append(now)
        if new_level: LOG_QUEUE.append([])
    else: tag = ''
    __print_entry(now,tag,msg)

def __print_entry(dt,tag,msg):
    print '({}){}{}'.format(dt,tag,msg)


def end_log_level():
    if len(LOG_QUEUE) == 0: return
    while len(LOG_QUEUE[-1]) > 0 : log()
    LOG_QUEUE.pop()
    log()

def end_log():
    while len(LOG_QUEUE) > 0 : end_log_level()

def stop_logging():
    LOGGING = False
